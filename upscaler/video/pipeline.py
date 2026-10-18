"""Upscale a whole video to 4K.

Frames stream from one ffmpeg decoder through the model and the finishing step into an ffmpeg
encoder, which is restarted every 10 seconds of video (a segment). Each finished segment is recorded
in `job_dir/manifest.json`, so a job interrupted by a crash, restart or cancel picks up where it left off.

A video that runs out of disk space (before a segment, while one is written, or while joining them)
raises NotEnoughDiskSpace, and one whose joining fails for another reason raises JoinFailed. Either
way the finished segments stay in `job_dir`, so the caller can keep that folder and carry on from them.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np
import torch
import torch.nn.functional as F

from upscaler.config import SPACE_RESERVE_BYTES, Cancelled, ProgressFn
from upscaler.finishing import finish
from upscaler.models import MODELS, LoadedModel, ModelSpec, ensure_model, load_model
from upscaler.tiling import upscale_tiled
from upscaler.video import ffmpeg
from upscaler.video.ffmpeg import concat_and_mux, decode_cmd, encode_cmd
from upscaler.video.probe import (
    OUTPUT_LONG_EDGE,
    NotEnoughDiskSpace,
    VideoError,
    VideoInfo,
    output_size_4k,
    probe_video,
)

log = logging.getLogger(__name__)

VIDEO_MODES = {"fast": "fast", "general": "video_general", "faithful": "faithful"}  # mode -> MODELS key
SEGMENT_SECONDS = 10

ALREADY_4K = "This video is already 4K or bigger, so there's nothing to upscale."
_READ_FAILED = "Couldn't read part of this video. The file may be damaged; try exporting it again as an MP4."
_WRITE_FAILED = "Couldn't save the upscaled video. Check that the computer has free disk space, then try again."
_NO_SPACE = "There isn't enough free disk space to finish this video."
# The caller adds how to carry on (Retry in the app, the same command again on the command line).
_JOIN_FAILED = "Couldn't put the finished video together. Check that the computer has free disk space."
_SAVE_FAILED = (
    "Couldn't save the finished video in the {folder} folder. Check that its drive is connected and has free space."
)
_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})
_DISK_FULL_LOG = "No space left on device"  # what ffmpeg prints when a write fails for that reason

# Light models run a whole frame up to this size in one go, which is faster than tiles.
# The DAT2 transformer (tile 256) needs ~10 GB for a 512 px tile, so it always keeps its tiles.
_WHOLE_FRAME_MAX_PIXELS = 2_100_000
_LIGHT_MODEL_MIN_TILE = 512
_SMALL_SOURCE_FACTOR = 5  # enlarging more than this looks soft; say so

# 4K HEVC comes out at up to about this rate, whatever the source (the server's upload check uses it too).
OUTPUT_BITS_PER_SECOND = 60_000_000
_OUTPUT_BYTES_PER_SECOND = OUTPUT_BITS_PER_SECOND / 8
# Joining writes the whole video again while the segments still exist. On top of the segments it
# needs room for the sound (1 Mbit/s covers AAC, AC-3 and most ALAC) and the MP4 index; like every
# other write it leaves SPACE_RESERVE_BYTES free.
_JOIN_OVERHEAD = 1.05
_SOUND_BYTES_PER_SECOND = 1_000_000 / 8
_SPARE_BYTES = 200_000_000  # less free than this after a failed write: it failed because the disk was full

_PREPARE_SHARE = 0.02  # progress bar share for downloading/loading the model
_FRAMES_END = 0.97  # frames fill the bar up to here; joining the segments takes the rest
_ETA_WINDOW = 30  # frames in the moving average behind the time estimate
_ETA_MIN_FRAMES = 3


class JoinFailed(VideoError):
    """Every segment is finished, but joining them into the video (or saving it) failed. The segments
    are kept, so carrying on only has to join them again."""


@dataclass
class VideoOptions:
    mode: str = "fast"
    preview_seconds: float | None = None
    strength: float = 1.0


@dataclass
class VideoResult:
    output_path: Path
    width: int
    height: int
    seconds: float
    notes: list[str]


def upscale_video(
    src: Path, out_dir: Path, job_dir: Path, opts: VideoOptions, *, progress: ProgressFn, cancel: threading.Event
) -> VideoResult:
    started = time.monotonic()
    spec = _model_spec(opts)
    info = probe_video(src)
    if max(info.display_width, info.display_height) >= OUTPUT_LONG_EDGE:
        raise VideoError(ALREADY_4K)
    if info.hdr and not ffmpeg.has_zscale():
        raise VideoError(ffmpeg.HDR_UNSUPPORTED)
    out_size = output_size_4k(info)
    plan = _plan_segments(info, opts.preview_seconds)
    encoder = ffmpeg.pick_encoder()  # before loading the model: a computer that can't make videos says so at once

    progress(0.0, "Getting ready")
    ensure_model(spec, progress=lambda fraction, message: progress(_PREPARE_SHARE * fraction, message), cancel=cancel)
    progress(_PREPARE_SHARE, f"Loading the {spec.label} model")
    model = load_model(spec)
    upscaler = _FrameUpscaler(model, out_size, opts.strength, _frame_tile(spec, info))

    with _disk_full_keeps_the_work():
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest = _Manifest(job_dir, _manifest_key(src, info, opts, out_size, plan, encoder))
        reporter = _Reporter(progress, plan.total_frames, manifest.frames_done())
        _upscale_segments(src, info, plan, manifest, upscaler, reporter, cancel)

        progress(_FRAMES_END, "Putting the video together and adding the sound")
        segments = manifest.finished_segments()
        if not segments:  # the decoder gave no frames at all
            raise VideoError(_READ_FAILED)
        preview_duration = float(plan.preview_frames / info.fps) if plan.preview_frames else None
        _check_space_to_join(segments, job_dir, out_dir, seconds=preview_duration or info.duration)
        output = _join(segments, src, job_dir, out_dir, _output_stem(src, opts), info.has_audio, preview_duration)
        manifest.remove()
    progress(1.0, "Finished")
    return VideoResult(output, *out_size, time.monotonic() - started, _notes(info, plan, out_size, opts))


@contextlib.contextmanager
def _disk_full_keeps_the_work() -> Iterator[None]:
    """The disk filling up anywhere along the way (a log file, the list of finished segments, the
    joined video) is NotEnoughDiskSpace, which tells the caller to keep the finished segments."""
    try:
        yield
    except OSError as exc:
        if exc.errno not in _DISK_FULL_ERRNOS:
            raise
        raise NotEnoughDiskSpace(f"{_NO_SPACE} Free up some space.") from exc


def _model_spec(opts: VideoOptions) -> ModelSpec:
    if opts.mode not in VIDEO_MODES:
        raise ValueError(f"Unknown video mode {opts.mode!r}; choose one of {', '.join(VIDEO_MODES)}")
    if not 0 <= opts.strength <= 1:
        raise ValueError(f"strength must be between 0 and 1, got {opts.strength}")
    if opts.preview_seconds is not None and opts.preview_seconds <= 0:
        raise ValueError(f"preview_seconds must be positive, got {opts.preview_seconds}")
    return MODELS[VIDEO_MODES[opts.mode]]


def _frame_tile(spec: ModelSpec, info: VideoInfo) -> int | None:
    w, h = info.display_width, info.display_height
    if w * h <= _WHOLE_FRAME_MAX_PIXELS and spec.tile >= _LIGHT_MODEL_MIN_TILE:
        return max(w, h)
    return None  # the model's own tile size


# --- Segments ------------------------------------------------------------------


@dataclass(frozen=True)
class _Segment:
    index: int
    start_frame: int
    frames: int | None  # None: read to the end of the video


@dataclass(frozen=True)
class _SegmentPlan:
    segments: list[_Segment]
    segment_frames: int
    total_frames: int  # an estimate when the last segment reads to the end
    preview_frames: int | None


def _plan_segments(info: VideoInfo, preview_seconds: float | None) -> _SegmentPlan:
    per_segment = max(1, round(info.fps * SEGMENT_SECONDS))
    total, preview = info.frames, None
    if preview_seconds is not None and round(info.fps * preview_seconds) < total:
        total = preview = max(1, round(info.fps * preview_seconds))
    segments = [_Segment(i, i * per_segment, per_segment) for i in range(math.ceil(total / per_segment))]
    last = segments[-1]
    # The frame count of a whole video is an estimate, so the last segment reads until the video ends.
    segments[-1] = _Segment(last.index, last.start_frame, total - last.start_frame if preview else None)
    return _SegmentPlan(segments, per_segment, total, preview)


def _upscale_segments(
    src: Path,
    info: VideoInfo,
    plan: _SegmentPlan,
    manifest: _Manifest,
    upscaler: _FrameUpscaler,
    reporter: _Reporter,
    cancel: threading.Event,
) -> None:
    """Upscale the segments that aren't finished yet.

    One decoder reads the video in order and its frames are cut into segments by count. Finished
    segments are always the first ones, so a resumed job's decoder skips that many frames.
    """
    todo = [segment for segment in plan.segments if segment.index not in manifest.done]
    if not todo:
        return
    first = manifest.frames_done()
    if first:
        reporter.resuming()
    limit = plan.preview_frames - first if plan.preview_frames is not None else None
    decode_log = manifest.folder / "decode.log"
    with decode_log.open("wb") as decode_err:
        decoder = _spawn(
            decode_cmd(src, info, first, limit), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=decode_err
        )
        frames = _DecodedFrames(decoder, info.display_width, info.display_height, decode_log)
        try:
            for segment in todo:
                _check_space_for_segment(manifest, plan, segment, info.fps)
                path = manifest.path_for(segment.index)
                written = _encode_segment(frames, info.fps, segment, upscaler, path, reporter, cancel)
                manifest.mark_done(segment.index, written)
            frames.finish()
        except BaseException:
            _stop(decoder)
            raise
        finally:
            decoder.stdout.close()


class _DecodedFrames:
    """The frames coming out of an ffmpeg decoder, one at a time."""

    def __init__(self, process: subprocess.Popen, w: int, h: int, log_path: Path):
        self._process, self._w, self._h, self._log = process, w, h, log_path
        self._ended = False

    def read(self) -> np.ndarray | None:
        """The next frame, or None once the video has ended."""
        if self._ended:
            return None
        frame = _read_frame(self._process.stdout, self._w, self._h)
        if frame is None:
            self._ended = True
            self.finish()  # it may have stopped early because part of the file is damaged
        return frame

    def finish(self) -> None:
        _wait(self._process, self._log, _READ_FAILED)


def _encode_segment(
    frames: _DecodedFrames,
    fps: Fraction,
    segment: _Segment,
    upscaler: _FrameUpscaler,
    out_path: Path,
    reporter: _Reporter,
    cancel: threading.Event,
) -> int:
    """Upscale the segment's frames into `out_path`; returns the number of frames written (0 writes no file).

    The GPU enlarges frame n+1 while a helper thread finishes frame n on the CPU and feeds it to
    the encoder, so the finishing step costs almost no extra time.
    """
    part = out_path.with_name(out_path.name + ".part")
    encode_log = out_path.parent / "encode.log"
    taken, written, encoder, in_flight = 0, 0, None, None

    def collect_finished_frame() -> None:
        nonlocal written
        in_flight.result()  # re-raises anything that went wrong in the helper thread
        written += 1
        reporter.frame_done()

    with encode_log.open("wb") as encode_err:
        helper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-finish")
        try:
            while (segment.frames is None or taken < segment.frames) and (frame := frames.read()) is not None:
                taken += 1
                if cancel.is_set():
                    raise Cancelled()
                enlarged = upscaler.enlarge(frame, cancel)
                if in_flight is not None:
                    collect_finished_frame()
                if encoder is None:  # started on the first frame, so an empty segment leaves no file
                    out_w, out_h = upscaler.out_size
                    encoder = _spawn(
                        encode_cmd(part, out_w, out_h, fps),
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_err,
                    )
                in_flight = helper.submit(_finish_and_write, upscaler, frame, enlarged, encoder, encode_log)
            if in_flight is not None:
                collect_finished_frame()
            if encoder is not None:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    log.error("ffmpeg exited with %s: %s", encoder.returncode, _tail(encode_log))
                    raise _write_failure(out_path.parent, encode_log)
        except BaseException:
            # Stop ffmpeg before waiting for the helper: a write it's blocked in then fails at once.
            _stop(encoder)
            helper.shutdown()
            part.unlink(missing_ok=True)
            raise
        finally:
            helper.shutdown()
    if written:
        _flush_to_disk(part)  # before it counts as finished, so a power cut can't leave half of it
        os.replace(part, out_path)
    return written


def _finish_and_write(
    upscaler: _FrameUpscaler, frame: np.ndarray, enlarged: np.ndarray, encoder: subprocess.Popen, encode_log: Path
) -> None:
    _write_frame(encoder, upscaler.finish(frame, enlarged), encode_log)


def _spawn(cmd: list[str], **pipes) -> subprocess.Popen:
    return subprocess.Popen(cmd, **ffmpeg.process_options(), **pipes)


def _read_frame(stream: BinaryIO, w: int, h: int) -> np.ndarray | None:
    size = w * h * 3
    data = stream.read(size)  # a buffered pipe read returns `size` bytes unless the video ended
    if len(data) < size:
        if data:
            log.warning("Ignoring %d stray bytes at the end of the decoded video", len(data))
        return None
    return np.frombuffer(data, np.uint8).reshape(h, w, 3)


def _write_frame(encoder: subprocess.Popen, frame: np.ndarray, encode_log: Path) -> None:
    try:
        encoder.stdin.write(np.ascontiguousarray(frame).data)
    except (BrokenPipeError, OSError) as e:
        encoder.wait()
        log.warning("Video encoder stopped (exit code %s): %s", encoder.returncode, _tail(encode_log))
        raise _write_failure(encode_log.parent, encode_log) from e


def _write_failure(folder: Path, encode_log: Path | None = None) -> VideoError:
    """Why the encoder stopped, as people should hear it. When the disk has filled up (it's nearly
    full now, or ffmpeg said so), it's NotEnoughDiskSpace: the segments finished so far are kept, and
    the job carries on from them once there's room again. Otherwise the plain "couldn't save" message."""
    full = _disk_is_full(folder) or (encode_log is not None and _DISK_FULL_LOG in _tail(encode_log))
    return NotEnoughDiskSpace(f"{_NO_SPACE} Free up some space.") if full else VideoError(_WRITE_FAILED)


def _disk_is_full(folder: Path) -> bool:
    try:
        return _free_bytes(folder) < _SPARE_BYTES
    except OSError:
        return False


def _wait(process: subprocess.Popen, log_path: Path, message: str) -> None:
    if process.wait() != 0:
        log.error("ffmpeg exited with %s: %s", process.returncode, _tail(log_path))
        raise VideoError(message)


def _stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.kill()
        process.wait()
    if process.stdin:
        with contextlib.suppress(OSError):  # flushing into a dead process fails; nothing left to save
            process.stdin.close()


def _flush_to_disk(path: Path) -> None:
    with path.open("r+b") as f:  # Windows can only flush a file that's open for writing
        os.fsync(f.fileno())


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(errors="replace").strip()[-limit:]
    except OSError:
        return ""


# --- Per-frame work --------------------------------------------------------------


class _FrameUpscaler:
    """enlarge: model, then resize to the output size on the GPU.
    finish: tone lock and detail strength on the CPU."""

    def __init__(self, model: LoadedModel, out_size: tuple[int, int], strength: float, tile: int | None):
        self.model = model
        self.out_size = out_size
        self.strength = strength
        self.tile = tile

    def enlarge(self, frame: np.ndarray, cancel: threading.Event) -> np.ndarray:
        upscaled = upscale_tiled(frame, self.model, tile=self.tile, cancel=cancel)
        return _resize(upscaled, self.out_size, self.model.device)

    def finish(self, frame: np.ndarray, enlarged: np.ndarray) -> np.ndarray:
        return finish(frame, enlarged, strength=self.strength, tone_lock=True)


def _resize(img: np.ndarray, size: tuple[int, int], device: torch.device) -> np.ndarray:
    """Bicubic resize with antialiasing (most 4x results shrink to 4K, which aliases without it)."""
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    # Half precision holds 0..255 to within 1/8 of a level, plenty before rounding to uint8.
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    with torch.inference_mode():
        x = torch.from_numpy(img).to(device).permute(2, 0, 1).unsqueeze(0).to(dtype)
        y = F.interpolate(x, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
        pixels = y.clamp_(0, 255).round_().to(torch.uint8).squeeze(0).permute(1, 2, 0)
        return pixels.contiguous().cpu().numpy()


# --- Progress ------------------------------------------------------------------------


class _Reporter:
    def __init__(self, progress: ProgressFn, total: int, done: int):
        self._progress = progress
        self._total = max(1, total)
        self._done = done
        self._finished_at: deque[float] = deque(maxlen=_ETA_WINDOW + 1)

    def resuming(self) -> None:
        """Before a resumed job's first new frame: ffmpeg first reads through the frames already done."""
        self._progress(self._fraction(min(self._done, self._total)), "Picking up where it left off")

    def frame_done(self) -> None:
        self._done += 1
        self._finished_at.append(time.monotonic())
        shown = min(self._done, self._total)
        message = f"Upscaling frame {shown:,} of {self._total:,}"
        if (seconds := self._seconds_left()) is not None:
            message += f", {_time_left(seconds)}"
        self._progress(self._fraction(shown), message)

    def _fraction(self, frames: int) -> float:
        return min(_PREPARE_SHARE + (_FRAMES_END - _PREPARE_SHARE) * frames / self._total, _FRAMES_END)

    def _seconds_left(self) -> float | None:
        if len(self._finished_at) <= _ETA_MIN_FRAMES:
            return None
        per_frame = (self._finished_at[-1] - self._finished_at[0]) / (len(self._finished_at) - 1)
        return per_frame * max(0, self._total - self._done)


def _time_left(seconds: float) -> str:
    if seconds < 60:
        return "less than a minute left"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} minute{'s' if minutes != 1 else ''} left"
    hours, minutes = divmod(minutes, 60)
    text = f"about {hours} hour{'s' if hours != 1 else ''}"
    if minutes:
        text += f" {minutes} minute{'s' if minutes != 1 else ''}"
    return text + " left"


# --- Resume manifest ----------------------------------------------------------------


def _manifest_key(
    src: Path, info: VideoInfo, opts: VideoOptions, out_size: tuple[int, int], plan: _SegmentPlan, encoder: list[str]
) -> dict:
    """Everything that shapes the segments. If any of it changes, the old segments can't be reused.

    That includes the encoder: segments from two encoders can't be joined without re-encoding, and
    which one works can change between runs (e.g. NVIDIA's after a driver update)."""
    key = {
        "version": 2,  # 2: frames are counted from the start of the file instead of found by seeking
        "source": src.name,
        "source_bytes": src.stat().st_size,
        "mode": opts.mode,
        "strength": opts.strength,
        "size": list(out_size),
        "fps": str(info.fps),
        "segment_frames": plan.segment_frames,
        "frames": plan.total_frames,
        "preview": plan.preview_frames,
        "encoder": encoder,
        "ffmpeg": ffmpeg.ffmpeg_version(),
    }
    return json.loads(json.dumps(key))  # the same types it will have when read back


class _Manifest:
    """Which segments are finished: {segment index: frames written}, saved after every segment."""

    def __init__(self, job_dir: Path, key: dict):
        self._path = job_dir / "manifest.json"
        self.folder = job_dir / "segments"
        self._key = key
        self.done = self._load()

    def _load(self) -> dict[int, int]:
        saved = self._read_matching()
        if saved is None:  # a different job's leftovers, or none at all
            shutil.rmtree(self.folder, ignore_errors=True)
            saved = {}
        self.folder.mkdir(parents=True, exist_ok=True)
        for leftover in self.folder.glob("*.part"):  # a segment that was cut off mid-way
            leftover.unlink()
        # Keep the finished segments from the start up to the first one that's missing or damaged (a power
        # cut can leave one cut short). Frames are read in order, so everything after that is made again.
        done: dict[int, int] = {}
        while (index := len(done)) in saved and self._intact(index, saved[index]):
            done[index] = saved[index]
        for index in saved.keys() - done.keys():
            self.path_for(index).unlink(missing_ok=True)
        return done

    def _intact(self, index: int, frames: int) -> bool:
        return frames == 0 or _frames_in(self.path_for(index)) == frames

    def _read_matching(self) -> dict[int, int] | None:
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
            if saved.get("key") != self._key:
                return None
            return {int(i): int(n) for i, n in saved["done"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def path_for(self, index: int) -> Path:
        return self.folder / f"segment-{index:05d}.mp4"

    def frames_done(self) -> int:
        return sum(self.done.values())

    def mark_done(self, index: int, frames: int) -> None:
        self.done[index] = frames
        temporary = self._path.with_name(self._path.name + ".tmp")
        state = {"key": self._key, "done": {str(i): n for i, n in sorted(self.done.items())}}
        with temporary.open("w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self._path)  # atomic, so a crash never leaves half a manifest

    def finished_segments(self) -> list[Path]:
        return [self.path_for(i) for i, frames in sorted(self.done.items()) if frames]

    def remove(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)
        self._path.unlink(missing_ok=True)


def _frames_in(segment: Path) -> int | None:
    """How many frames a finished segment holds, or None if it's missing or can't be read."""
    try:
        with av.open(str(segment)) as container:
            return container.streams.video[0].frames
    except (av.FFmpegError, OSError, IndexError):
        return None


# --- Free space -----------------------------------------------------------------------


def _check_space_for_segment(manifest: _Manifest, plan: _SegmentPlan, segment: _Segment, fps: Fraction) -> None:
    """Before each segment: stop (NotEnoughDiskSpace, so the finished segments are kept) when one more
    segment wouldn't leave SPACE_RESERVE_BYTES free, so the disk keeps that much rather than running
    down to nothing. The space was checked when the video was added, but other jobs, other people's
    uploads and other apps may have used it since. The message says how much to free for the rest of
    the video and its joining."""
    free = _free_bytes(manifest.folder)
    segment_bytes = plan.segment_frames / fps * _OUTPUT_BYTES_PER_SECOND
    if free >= segment_bytes + SPACE_RESERVE_BYTES:
        return
    rest = max(plan.total_frames - segment.start_frame, plan.segment_frames) / fps * _OUTPUT_BYTES_PER_SECOND
    done = sum(path.stat().st_size for path in manifest.finished_segments())
    needed = rest + (done + rest) * _JOIN_OVERHEAD + SPACE_RESERVE_BYTES
    raise NotEnoughDiskSpace(f"{_NO_SPACE} Free up about {_gigabytes(needed - free)} GB.")


def _check_space_to_join(segments: list[Path], job_dir: Path, out_dir: Path, *, seconds: float) -> None:
    """Joining writes the finished video next to the segments, so it needs about their size again
    (and again on the output drive, if that's a different one), and still leaves SPACE_RESERVE_BYTES
    free. Say so now rather than fail part-way."""
    video = sum(segment.stat().st_size for segment in segments) * _JOIN_OVERHEAD
    needed = video + seconds * _SOUND_BYTES_PER_SECOND + SPACE_RESERVE_BYTES
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = [job_dir] if _same_drive(job_dir, out_dir) else [job_dir, out_dir]
    short = max(needed - _free_bytes(folder) for folder in folders)
    if short > 0:
        # How to carry on (Retry, or the same command again) is for the caller to add.
        raise NotEnoughDiskSpace(f"{_NO_SPACE} Free up about {_gigabytes(short)} GB.")


def _gigabytes(num_bytes: float) -> int:
    return max(1, math.ceil(num_bytes / 1e9))


# --- Joining --------------------------------------------------------------------------


def _join(
    segments: list[Path],
    src: Path,
    job_dir: Path,
    out_dir: Path,
    stem: str,
    has_audio: bool,
    duration: float | None,
) -> Path:
    """Join the finished segments, add the sound and move the video into `out_dir`. If that fails, the
    half-made video goes (freeing its space) but the segments stay: NotEnoughDiskSpace when the disk
    filled up, JoinFailed otherwise, so carrying on only has to join them again."""
    joined = job_dir / "output.part.mp4"
    try:
        concat_and_mux(segments, src, joined, has_audio, duration=duration)
    except VideoError as exc:
        full = _disk_is_full(job_dir)  # looked at before the half-made video goes and frees its space
        joined.unlink(missing_ok=True)
        raise (NotEnoughDiskSpace(f"{_NO_SPACE} Free up some space.") if full else JoinFailed(_JOIN_FAILED)) from exc
    output = _unique_path(out_dir, stem, ".mp4")
    try:
        shutil.move(joined, output)  # a rename, unless the output folder is on another drive
    except OSError as exc:
        full = exc.errno in _DISK_FULL_ERRNOS or _disk_is_full(out_dir)
        if joined.exists():
            output.unlink(missing_ok=True)  # half a copy on the other drive
        joined.unlink(missing_ok=True)
        if full:
            raise NotEnoughDiskSpace(f"{_NO_SPACE} Free up some space.") from exc
        raise JoinFailed(_SAVE_FAILED.format(folder=out_dir.name or out_dir)) from exc
    return output


def _same_drive(a: Path, b: Path) -> bool:
    return a.stat().st_dev == b.stat().st_dev


def _free_bytes(folder: Path) -> int:
    return shutil.disk_usage(folder).free


# --- Output ---------------------------------------------------------------------------


def _output_stem(src: Path, opts: VideoOptions) -> str:
    stem = f"{src.stem}-4K-{opts.mode}"
    if opts.preview_seconds is not None:
        stem += f"-preview{opts.preview_seconds:g}s"
    return stem


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate, n = directory / f"{stem}{suffix}", 2
    while candidate.exists():
        candidate, n = directory / f"{stem}-{n}{suffix}", n + 1
    return candidate


def _notes(info: VideoInfo, plan: _SegmentPlan, out_size: tuple[int, int], opts: VideoOptions) -> list[str]:
    notes = []
    if plan.preview_frames is not None:
        seconds = f"{opts.preview_seconds:g}"
        notes.append(f"This is a preview of the first {seconds} second{'' if seconds == '1' else 's'}.")
    if info.hdr:
        notes.append(
            "This was an HDR video, so it was converted to standard colour. "
            "It may look a little less bright than on a phone."
        )
    if info.interlaced:
        notes.append(
            "This video was interlaced (common with old camcorders and TV recordings), "
            "so it was converted to smooth whole frames."
        )
    if max(out_size) > _SMALL_SOURCE_FACTOR * max(info.display_width, info.display_height):
        notes.append("This video is quite small, so it was enlarged a lot and may look a little soft.")
    return notes
