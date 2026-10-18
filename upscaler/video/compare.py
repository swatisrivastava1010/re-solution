"""Side-by-side video comparison: browser-friendly copies of a video job's original and its result.

The upscaled result is 10-bit HEVC, which many browsers can't play, and the original can be anything
a camera or phone writes. So the first time someone opens the comparison, two copies are made into
the job's folder (`<job>/compare/<n>/`) and kept there with the job:

- original.mp4: the original exactly as the upscaler saw it (upright, square pixels, deinterlaced,
  standard range, at its constant frame rate, from ffmpeg.decode_cmd), as 8-bit H.264. No sound.
- upscaled.mp4: the result as 8-bit H.264 at its full size, with its sound.

Both have the same frame rate and the same number of frames, starting at zero, so frame n of one is
frame n of the other and the two players can be kept in step. Each copy also gets a small .json
saying what it is; the comparison is ready when both copies and both notes are there.

Apple's hardware H.264 encoder is used on Macs and NVIDIA's on PCs that have one, otherwise x264.
`Comparisons` runs the copies in the background, one comparison at a time, with progress for the
page and a way to stop.

A very large original (a camcorder or ProRes file of many GB) is swapped for its small copy as soon
as its job finishes (keep_compact_original), so the job's folder doesn't keep gigabytes for days.

The result's card on the page shows a still from it (poster.jpg, made on request and kept in the
same folder) and plays the upscaled copy when asked, so the page never decodes the 10-bit HEVC file
itself.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av

from upscaler.config import Cancelled, ProgressFn
from upscaler.video import ffmpeg
from upscaler.video.probe import VideoError, VideoInfo, audio_codecs, probe_video

log = logging.getLogger(__name__)

ORIGINAL = "original"
UPSCALED = "upscaled"
COMPARE_FOLDER = "compare"

IDLE, WAITING, PREPARING, READY, FAILED, CANCELLED = "idle", "waiting", "preparing", "ready", "failed", "cancelled"

ORIGINAL_GONE = "The original video isn't kept any more, so this one can't be compared. Upscale it again to compare."
RESULT_GONE = "The upscaled video isn't here any more, so it can't be compared."
COPY_FAILED = "Couldn't get the comparison ready. Please try again."
STOPPED = "Stopped. Press Try again to get the comparison ready."

# Very large originals become their small copy once their job finishes: bigger than this, and with a
# bit rate well above what the copy needs (a camcorder's DV or a ProRes file, not a phone's HEVC).
COMPACT_OVER_BYTES = 2 * 1024**3
COMPACT_BITRATE_FACTOR = 3
_COPY_BITS_PER_PIXEL = 0.12  # about what a good-looking H.264 copy needs, per pixel per frame

_H264_ENCODERS: dict[str, tuple[str, ...]] = {
    "h264_videotoolbox": ("-c:v", "h264_videotoolbox", "-q:v", "70", "-profile:v", "high"),
    "h264_nvenc": ("-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0", "-profile:v", "high"),
    "libx264": ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high"),
}
_SOFTWARE = "libx264"
_BT709 = (
    "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_TAGS = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv")
_BROWSER_AUDIO = {"aac", "mp3"}  # plays in every browser straight from an MP4
_AAC = ("-c:a", "aac", "-b:a", "192k")
_PROBE_SIZE = 256
_PROBE_TIMEOUT_SECONDS = 60

POSTER = "poster.jpg"
POSTER_EDGE = 960  # the still's longer side, in pixels: sharp on a card, small to send
POSTER_FAILED = "Couldn't make a still from this video."
_POSTER_TIMEOUT_SECONDS = 60
_poster_guard = threading.Lock()
_poster_locks: dict[Path, threading.Lock] = {}


def compare_folder(job_dir: Path, n: int) -> Path:
    """Where the comparison copies of a job's result n live."""
    return Path(job_dir) / COMPARE_FOLDER / str(int(n))


def _paths(folder: Path, which: str) -> tuple[Path, Path]:
    return folder / f"{which}.mp4", folder / f"{which}.json"


def copy_path(folder: Path, which: str) -> Path:
    """The finished copy `which` (ORIGINAL or UPSCALED) in `folder`."""
    if which not in (ORIGINAL, UPSCALED):
        raise ValueError(f"Not a comparison copy: {which!r}")
    return _paths(Path(folder), which)[0]


def _read_note(folder: Path, which: str) -> dict | None:
    video, note = _paths(folder, which)
    if not video.is_file():
        return None
    try:
        data = json.loads(note.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def ready_info(folder: Path) -> dict | None:
    """What the page needs once both copies are made, or None while they aren't."""
    folder = Path(folder)
    original, upscaled = _read_note(folder, ORIGINAL), _read_note(folder, UPSCALED)
    if original is None or upscaled is None:
        return None
    return {
        "fps": upscaled["fps"],
        "frames": min(original["frames"], upscaled["frames"]),
        "duration": upscaled["duration"],
        "has_audio": upscaled["has_audio"],
        "original": {"width": original["width"], "height": original["height"]},
        "upscaled": {"width": upscaled["width"], "height": upscaled["height"]},
    }


# --- A still for the result's card ------------------------------------------------------------


def poster_time(duration: float) -> float:
    """Where the still is taken: a second in, or a quarter of the way through a shorter clip (the
    very first frame is often black or a fade)."""
    if not duration or duration <= 0:
        return 0.0
    return round(min(1.0, duration * 0.25), 3)


def poster(folder: Path, upscaled: Path) -> Path:
    """A JPEG still from the result, at most POSTER_EDGE pixels on its longer side, made once and kept
    in the comparison's folder (made again if the result is newer). Raises VideoError."""
    folder, upscaled = Path(folder), Path(upscaled)
    target = folder / POSTER
    if not upscaled.is_file():
        raise VideoError(RESULT_GONE)
    if _poster_current(target, upscaled):
        return target
    with _poster_guard:
        lock = _poster_locks.setdefault(folder.resolve(), threading.Lock())
    with lock:  # one still at a time per result, however many cards ask at once
        if _poster_current(target, upscaled):
            return target
        folder.mkdir(parents=True, exist_ok=True)
        try:
            duration = _probe(upscaled).duration
        except VideoError:
            duration = 0.0
        part = folder / "poster.part.jpg"
        for at in dict.fromkeys((poster_time(duration), 0.0)):  # the start, if the clip is shorter than it says
            cmd = [
                *_base(), "-y", "-ss", f"{at:.3f}", "-i", str(upscaled), "-frames:v", "1", "-an", "-sn",
                "-vf", f"scale={POSTER_EDGE}:{POSTER_EDGE}:force_original_aspect_ratio=decrease",
                "-pix_fmt", "yuvj420p", "-q:v", "4", "-f", "image2", str(part),
            ]  # fmt: skip
            try:
                done = subprocess.run(
                    cmd,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=_POSTER_TIMEOUT_SECONDS,
                    creationflags=ffmpeg.popen_flags(),
                )
            except (OSError, subprocess.SubprocessError):
                part.unlink(missing_ok=True)
                continue
            if done.returncode == 0 and part.is_file() and part.stat().st_size > 0:
                os.replace(part, target)
                return target
            part.unlink(missing_ok=True)
        log.warning("Couldn't make a still from %s", upscaled)
        raise VideoError(POSTER_FAILED)


def _poster_current(target: Path, source: Path) -> bool:
    try:
        made = target.stat()
        return made.st_size > 0 and made.st_mtime_ns >= source.stat().st_mtime_ns
    except OSError:
        return False


# --- Making the copies -------------------------------------------------------------------


def prepare(
    folder: Path, original: Path | None, upscaled: Path, *, progress: ProgressFn, cancel: threading.Event
) -> dict:
    """Make whichever copies aren't there yet. Returns ready_info.

    Raises VideoError (a message for people) when a copy can't be made, and Cancelled."""
    folder = Path(folder)
    if not Path(upscaled).is_file():
        raise VideoError(RESULT_GONE)
    need_original = _read_note(folder, ORIGINAL) is None
    if need_original and (original is None or not Path(original).is_file()):
        raise VideoError(ORIGINAL_GONE)
    folder.mkdir(parents=True, exist_ok=True)
    result = _probe(upscaled)
    frames = _frame_count(upscaled) or result.frames
    source = probe_video(original) if need_original else None
    # The result was made at the original's frame rate; when the original is gone (a compact copy
    # stands in for it), the result's own rate is the same one.
    fps = source.fps if source is not None else _note_fps(folder) or result.fps
    # Each copy's share of the bar, by how many pixels it encodes.
    original_pixels = source.display_width * source.display_height if source is not None else 0
    upscaled_pixels = result.display_width * result.display_height if _read_note(folder, UPSCALED) is None else 0
    total = max(1, original_pixels + upscaled_pixels)
    share = original_pixels / total
    if source is not None:
        progress(0.0, "Making a copy of the original your browser can play")
        _original_copy(original, source, folder, frames, fps, _span(progress, 0.0, share), cancel)
    if upscaled_pixels:
        progress(share, "Making a copy of the upscaled video your browser can play")
        _upscaled_copy(upscaled, result, folder, frames, fps, _span(progress, share, 1.0), cancel)
    progress(1.0, "Ready")
    ready = ready_info(folder)
    if ready is None:  # a note couldn't be read back
        raise VideoError(COPY_FAILED)
    return ready


def _original_copy(
    source: Path,
    info: VideoInfo,
    folder: Path,
    frames: int | None,
    fps: Fraction,
    progress: ProgressFn,
    cancel: threading.Event,
) -> None:
    """The original's frames exactly as the upscaler decoded them (ffmpeg.decode_cmd: upright, square
    pixels, deinterlaced, standard range, constant rate), piped into an H.264 encoder. Odd sides lose
    their last row or column, which H.264 can't hold."""
    width, height = info.display_width, info.display_height
    even_w, even_h = width - width % 2, height - height % 2
    if even_w < 2 or even_h < 2:
        raise VideoError(COPY_FAILED)
    video, note = _paths(folder, ORIGINAL)
    decoder = ffmpeg.decode_cmd(Path(source), info, 0, frames)

    def encoder(h264: tuple[str, ...], target: Path) -> list[str]:
        return [
            *_base(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", _rate(fps), "-i", "-",
            "-vf", f"crop={even_w}:{even_h}:0:0,{_BT709}", *h264, "-g", str(_gop(fps)), *_TAGS, "-an",
            "-movflags", "+faststart", *_PROGRESS, "-f", "mp4", str(target),
        ]  # fmt: skip

    written = _encode_with_fallback(folder, ORIGINAL, encoder, decoder, frames, progress, cancel)
    _write_note(
        note, {"width": width, "height": height, "fps": _rate(fps), "frames": written, "duration": written / fps}
    )
    log.info("Comparison copy of the original: %s (%d frames)", video, written)


def _upscaled_copy(
    upscaled: Path,
    info: VideoInfo,
    folder: Path,
    frames: int | None,
    fps: Fraction,
    progress: ProgressFn,
    cancel: threading.Event,
) -> None:
    """The result at its full size as 8-bit H.264 at the same constant rate, with its sound (as AAC
    when it isn't a kind every browser plays)."""
    video, note = _paths(folder, UPSCALED)
    codecs = _audio_codecs(upscaled)
    audio = ["-map", "0:a:0?", *(["-c:a", "copy"] if codecs and codecs[0] in _BROWSER_AUDIO else list(_AAC))]
    limit = ["-frames:v", str(frames)] if frames else []

    def encoder(h264: tuple[str, ...], target: Path) -> list[str]:
        return [
            *_base(), "-y", "-i", str(upscaled), "-map", "0:v:0", *audio,
            "-vf", f"fps=fps={_rate(fps)}:start_time=0,format=yuv420p", "-fps_mode", "passthrough", *limit,
            *h264, "-g", str(_gop(fps)), *_TAGS, "-movflags", "+faststart", *_PROGRESS, "-f", "mp4", str(target),
        ]  # fmt: skip

    written = _encode_with_fallback(folder, UPSCALED, encoder, None, frames, progress, cancel)
    _write_note(
        note,
        {
            "width": info.display_width,
            "height": info.display_height,
            "fps": _rate(fps),
            "frames": written,
            "duration": written / fps,
            "has_audio": bool(codecs),
        },
    )
    log.info("Comparison copy of the result: %s (%d frames)", video, written)


_PROGRESS = ("-progress", "pipe:1", "-nostats", "-stats_period", "0.25")


def _encode_with_fallback(
    folder: Path,
    which: str,
    encoder: Callable[[tuple[str, ...], Path], list[str]],
    decoder: list[str] | None,
    frames: int | None,
    progress: ProgressFn,
    cancel: threading.Event,
) -> int:
    """Run the encode into <which>.part.mp4 and move it into place. A hardware encoder that fails
    (some can't take every size) gets one more go with x264. Returns the frames written."""
    video, _ = _paths(folder, which)
    part = video.with_name(f"{which}.part.mp4")
    choices = [h264_encoder()]
    if choices[0] != _SOFTWARE:
        choices.append(_SOFTWARE)
    for n, name in enumerate(choices):
        try:
            written = _run(
                folder / f"{which}.log", encoder(_H264_ENCODERS[name], part), decoder, frames, progress, cancel
            )
        except _EncodeFailed:
            part.unlink(missing_ok=True)
            if n + 1 < len(choices):
                log.warning("The %s encoder couldn't make the %s copy; trying %s", name, which, choices[n + 1])
                continue
            raise VideoError(COPY_FAILED) from None
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        if written < 1:
            part.unlink(missing_ok=True)
            raise VideoError(COPY_FAILED)
        os.replace(part, video)
        return written
    raise VideoError(COPY_FAILED)  # not reached


class _EncodeFailed(Exception):
    pass


def _run(
    log_path: Path,
    encode: list[str],
    decode: list[str] | None,
    frames: int | None,
    progress: ProgressFn,
    cancel: threading.Event,
) -> int:
    """Run ffmpeg (fed by a decoder when `decode` is given), reporting progress from its frame count.
    Returns the frames it wrote. Raises Cancelled (after stopping ffmpeg) or _EncodeFailed."""
    if cancel.is_set():
        raise Cancelled()
    processes: list[subprocess.Popen] = []
    written = 0
    with log_path.open("wb") as errors:
        try:
            stdin = subprocess.DEVNULL
            if decode is not None:
                decoder = subprocess.Popen(
                    decode,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    creationflags=ffmpeg.popen_flags(),
                )
                processes.append(decoder)
                stdin = decoder.stdout
            encoder = subprocess.Popen(
                encode, stdin=stdin, stdout=subprocess.PIPE, stderr=errors, creationflags=ffmpeg.popen_flags()
            )
            processes.append(encoder)
            if decode is not None:
                processes[0].stdout.close()  # the encoder holds it now; the decoder stops if the encoder does
            watcher = threading.Thread(target=_stop_when_cancelled, args=(processes, cancel), daemon=True)
            watcher.start()
            for raw in encoder.stdout:
                key, _, value = raw.decode("ascii", "replace").strip().partition("=")
                if key == "frame" and value.isdigit():
                    written = int(value)
                    if frames:
                        progress(min(written / frames, 1.0), "")
            encoder.wait()
            for process in processes[:-1]:
                process.wait()
        finally:
            for process in processes:
                _stop(process)
            if processes and processes[-1].stdout:
                processes[-1].stdout.close()
    if cancel.is_set():
        raise Cancelled()
    if any(process.returncode != 0 for process in processes):
        log.error("ffmpeg couldn't make a comparison copy: %s", _tail(log_path))
        raise _EncodeFailed()
    log_path.unlink(missing_ok=True)
    return written


def _stop_when_cancelled(processes: list[subprocess.Popen], cancel: threading.Event) -> None:
    while any(process.poll() is None for process in processes):
        if cancel.wait(0.2):
            for process in processes:
                _stop(process)
            return


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        with contextlib.suppress(OSError):
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(5)


# --- Encoder choice -------------------------------------------------------------------------


@functools.cache
def h264_encoder() -> str:
    """Apple's hardware H.264 encoder on Macs and NVIDIA's on PCs that have one, otherwise x264.
    Each is tried on a real frame first: an encoder can be built in but unusable (no NVIDIA card)."""
    candidates = ["h264_videotoolbox"] if sys.platform == "darwin" else ["h264_nvenc"]
    for name in candidates:
        if _encoder_works(_H264_ENCODERS[name]):
            log.info("Comparison copies use %s", name)
            return name
    return _SOFTWARE


def _encoder_works(args: tuple[str, ...]) -> bool:
    cmd = [
        *_base(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{_PROBE_SIZE}x{_PROBE_SIZE}", "-r", "25", "-i", "-",
        "-frames:v", "1", "-vf", _BT709, *args, "-f", "null", "-",
    ]  # fmt: skip
    try:
        result = subprocess.run(
            cmd,
            check=False,
            input=bytes(_PROBE_SIZE * _PROBE_SIZE * 3),
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            creationflags=ffmpeg.popen_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


# --- Very large originals ------------------------------------------------------------------


def worth_compacting(source: Path) -> bool:
    """Whether a finished job's original is so big that the comparison's small copy should be kept
    instead: over COMPACT_OVER_BYTES, at a bit rate several times what the copy needs."""
    try:
        size = Path(source).stat().st_size
    except OSError:
        return False
    if size <= COMPACT_OVER_BYTES:
        return False
    try:
        info = probe_video(source)
    except VideoError:
        return False
    if info.duration <= 0:
        return False
    bitrate = size * 8 / info.duration
    copy_bitrate = info.display_width * info.display_height * float(info.fps) * _COPY_BITS_PER_PIXEL
    return bitrate > COMPACT_BITRATE_FACTOR * copy_bitrate


def keep_compact_original(source: Path, folder: Path, upscaled: Path, *, cancel: threading.Event | None = None) -> bool:
    """Make the comparison's copy of the original now, and delete the original if the copy is
    smaller. Returns True if the original was swapped for its copy. Stopping (`cancel`) keeps the
    original and doesn't raise."""
    source, folder = Path(source), Path(folder)
    cancel = cancel or threading.Event()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        info = probe_video(source)
        frames = _frame_count(upscaled) or None
        _original_copy(source, info, folder, frames, info.fps, lambda fraction, message: None, cancel)
    except Cancelled:
        return False
    video, note = _paths(folder, ORIGINAL)
    copy_bytes, original_bytes = video.stat().st_size, source.stat().st_size
    if copy_bytes >= original_bytes:
        video.unlink(missing_ok=True)  # not smaller after all: the comparison makes it again when asked
        note.unlink(missing_ok=True)
        return False
    source.unlink()
    log.info(
        "Kept a %d MB copy of %s instead of the %d MB original",
        copy_bytes // 10**6,
        source.name,
        original_bytes // 10**6,
    )
    return True


# --- In the background, for the page ----------------------------------------------------------


@dataclass
class _Task:
    status: str = WAITING
    progress: float = 0.0
    message: str = "Waiting for another comparison to finish"
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.status in (WAITING, PREPARING)

    def view(self) -> dict:
        view = {"status": self.status, "progress": round(self.progress, 4), "message": self.message}
        if self.error:
            view["error"] = self.error
        return view


class Comparisons:
    """Comparison copies made in the background, one comparison at a time (they share the
    computer's video encoder with any video job that is running)."""

    def __init__(self, at_once: int = 1):
        self._lock = threading.Lock()
        self._tasks: dict[Path, _Task] = {}
        self._turn = threading.BoundedSemaphore(at_once)

    def status(self, folder: Path) -> dict:
        folder = Path(folder)
        with self._lock:
            task = self._tasks.get(folder)
            if task is not None and task.active:
                return task.view()
        ready = ready_info(folder)
        if ready is not None:
            return {"status": READY, "progress": 1.0, "message": "Ready", **ready}
        if task is not None and task.status != READY:  # made once, but gone since (the job was tidied away)
            return task.view()
        return {"status": IDLE, "progress": 0.0, "message": ""}

    def start(self, folder: Path, original: Path | None, upscaled: Path) -> dict:
        """Start making the copies, unless they're made or being made (so asking twice is fine)."""
        folder = Path(folder)
        with self._lock:
            task = self._tasks.get(folder)
            if task is not None and task.active:
                return task.view()
            if ready_info(folder) is None and original is None and _read_note(folder, ORIGINAL) is None:
                task = self._tasks[folder] = _Task(status=FAILED, message="Didn't work", error=ORIGINAL_GONE)
                return task.view()
            if ready_info(folder) is None:
                task = self._tasks[folder] = _Task()
                threading.Thread(
                    target=self._make, args=(task, folder, original, upscaled), name="video-compare", daemon=True
                ).start()
        return self.status(folder)

    def cancel(self, folder: Path) -> bool:
        with self._lock:
            task = self._tasks.get(Path(folder))
            if task is None or not task.active:
                return False
            task.cancel.set()
            task.message = "Stopping…"
            return True

    def stop_all(self) -> None:
        with self._lock:
            for task in self._tasks.values():
                task.cancel.set()

    def _make(self, task: _Task, folder: Path, original: Path | None, upscaled: Path) -> None:
        try:
            while not self._turn.acquire(timeout=0.2):
                if task.cancel.is_set():
                    raise Cancelled()
            try:
                task.status, task.message = PREPARING, "Preparing the comparison"

                def report(fraction: float, message: str) -> None:
                    task.progress = min(max(float(fraction), 0.0), 1.0)
                    if message and not task.cancel.is_set():
                        task.message = message

                prepare(folder, original, upscaled, progress=report, cancel=task.cancel)
            finally:
                self._turn.release()
        except Cancelled:
            task.status, task.message, task.error = CANCELLED, STOPPED, None
        except VideoError as exc:
            task.status, task.message, task.error = FAILED, "Didn't work", str(exc) or COPY_FAILED
        except Exception:
            log.exception("Couldn't make the comparison copies in %s", folder)
            task.status, task.message, task.error = FAILED, "Didn't work", COPY_FAILED
        else:
            task.status, task.progress, task.message = READY, 1.0, "Ready"


COMPARISONS = Comparisons()
atexit.register(COMPARISONS.stop_all)


# --- Helpers ---------------------------------------------------------------------------------


def _base() -> list[str]:
    return [ffmpeg.ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error"]


def _rate(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def _gop(fps: Fraction) -> int:
    """A keyframe every second, so scrubbing and stepping frame by frame stay quick."""
    return max(1, round(float(fps)))


def _probe(path: Path) -> VideoInfo:
    return probe_video(Path(path))


def _frame_count(path: Path) -> int:
    """The frames a finished MP4 holds (from its index), or 0 if it doesn't say."""
    try:
        with av.open(str(path)) as container:
            return int(container.streams.video[0].frames or 0)
    except (av.FFmpegError, OSError, IndexError):
        return 0


def _audio_codecs(path: Path) -> list[str]:
    try:
        return audio_codecs(Path(path))
    except (av.FFmpegError, OSError):
        return []


def _note_fps(folder: Path) -> Fraction | None:
    note = _read_note(folder, ORIGINAL)
    try:
        return Fraction(note["fps"]) if note else None
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def _write_note(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps({**data, "fps": str(data["fps"]), "duration": float(data["duration"])}), "utf-8")
    os.replace(temporary, path)


def _span(progress: ProgressFn, start: float, end: float) -> ProgressFn:
    def report(fraction: float, message: str) -> None:
        progress(start + (end - start) * min(max(fraction, 0.0), 1.0), message)

    return report


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(errors="replace").strip()[-limit:]
    except OSError:
        return ""
