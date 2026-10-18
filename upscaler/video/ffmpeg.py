"""Command lines for the bundled ffmpeg: decode to raw RGB, encode HEVC segments, join them with the sound."""

from __future__ import annotations

import functools
import logging
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import imageio_ffmpeg

from upscaler.video.probe import VideoError, VideoInfo, audio_codecs

log = logging.getLogger(__name__)

HDR_UNSUPPORTED = "This is an HDR video; HDR isn't supported yet. Export it as SDR from Photos first."
NO_ENCODER = (
    "This computer couldn't create video files, so videos can't be upscaled here. "
    "Try restarting the computer; if that doesn't help, run the installer again."
)

_ENCODERS: dict[str, tuple[str, ...]] = {
    "hevc_videotoolbox": (
        "-c:v", "hevc_videotoolbox", "-q:v", "65", "-profile:v", "main10", "-pix_fmt", "p010le", "-tag:v", "hvc1",
    ),
    "hevc_nvenc": (
        "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
        "-profile:v", "main10", "-pix_fmt", "p010le", "-tag:v", "hvc1",
    ),
    "libx265": ("-c:v", "libx265", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"),
}

# RGB -> BT.709 limited-range YUV. ffmpeg 7.1 takes the stream's colour tags from the frames, so
# setparams is what actually tags primaries and transfer; the -color_* flags alone leave them unset.
_TO_BT709 = (
    "scale=out_color_matrix=bt709:out_range=tv,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)
_BT709_TAGS = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv")

# HDR (PQ or HLG) -> SDR BT.709: linear light, BT.709 primaries, Hable curve, BT.709 transfer.
_TONE_MAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv444p16le"
)

# Audio QuickTime plays straight from an MP4; anything else is converted to AAC.
_QUICKTIME_AUDIO = {"aac", "alac", "mp3", "ac3", "eac3"}
_AAC = ("-c:a", "aac", "-b:a", "256k")

_PROBE_SIZE = 256
_PROBE_TIMEOUT_SECONDS = 60


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _base() -> list[str]:
    return [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error"]


def _rate(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


def popen_flags() -> int:
    """Keeps Windows from flashing a console window for every ffmpeg run."""
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def process_options() -> dict:
    """Popen arguments for the long ffmpeg runs: no console window, and a process group of their own.

    Ctrl+C in a Terminal window goes to its whole process group. If ffmpeg got it, it would quit, and a
    job that should pause (to carry on after a restart) would fail instead. The app stops ffmpeg itself.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


@functools.cache
def ffmpeg_version() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_version()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --- Encoder choice -------------------------------------------------------------


@functools.cache
def _working_encoder() -> tuple[str, ...]:
    candidates = ["hevc_videotoolbox", "libx265"] if sys.platform == "darwin" else ["hevc_nvenc", "libx265"]
    for name in candidates:
        if _encoder_works(_ENCODERS[name]):
            log.info("Video encoder: %s", name)
            return _ENCODERS[name]
        log.info("Video encoder %s isn't usable here; trying the next one", name)
    log.error("No working HEVC encoder in ffmpeg (tried %s)", ", ".join(candidates))
    raise VideoError(NO_ENCODER)


def pick_encoder() -> list[str]:
    """Encoder arguments: Apple's hardware encoder on Macs, NVENC on NVIDIA, otherwise x265.

    Each candidate must encode a real frame first, because an encoder can be compiled in
    but unusable (no NVIDIA card, an old driver, a VM)."""
    return list(_working_encoder())


def _encoder_works(encoder_args: tuple[str, ...]) -> bool:
    cmd = [
        *_base(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{_PROBE_SIZE}x{_PROBE_SIZE}", "-r", "25", "-i", "-",
        "-frames:v", "1", "-vf", _TO_BT709, *encoder_args, *_BT709_TAGS, "-f", "null", "-",
    ]
    frame = bytes(_PROBE_SIZE * _PROBE_SIZE * 3)
    try:
        result = subprocess.run(
            cmd, input=frame, capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS, creationflags=popen_flags()
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.info("Encoder probe failed to run: %s", e)
        return False
    if result.returncode != 0:
        log.info("Encoder probe said: %s", result.stderr.decode(errors="replace").strip()[-500:])
    return result.returncode == 0


@functools.cache
def has_zscale() -> bool:
    """zscale (zimg) is what converts HDR to SDR; not every ffmpeg build has it."""
    try:
        result = subprocess.run(
            [*_base(), "-filters"], capture_output=True, text=True, timeout=30, creationflags=popen_flags()
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(r"^\s*\S+\s+zscale\s", result.stdout, re.MULTILINE) is not None


# --- Decode / encode ---------------------------------------------------------------


def decode_cmd(src: Path, info: VideoInfo, start_frame: int = 0, frames: int | None = None) -> list[str]:
    """Decode upright, square-pixel, progressive SDR frames as raw RGB to stdout, at a constant rate.

    ffmpeg turns rotated phone video upright by itself (autorotate is on by default).
    `start_frame` skips that many frames (a resumed job); `frames=None` reads to the end of the video.

    It always decodes from the start of the file and counts frames, so frame n is the same frame
    whether or not the frames before it are skipped. Seeking with -ss isn't safe for this: after a
    seek in MPEG-TS or DVD files (.mts, .m2ts, .ts, .mpg) the timestamps don't restart at the seek
    point, so a constant-rate decode repeats the first frame, then runs behind the sound.
    """
    cmd = [*_base(), "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn"]
    cmd += ["-vf", ",".join(_decode_filters(info, start_frame)), "-fps_mode", "passthrough"]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    return cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _decode_filters(info: VideoInfo, start_frame: int = 0) -> list[str]:
    filters = []
    if info.interlaced:
        filters.append("bwdif=mode=send_frame")
    # The constant rate comes before scaling: there it still knows each frame's length, so it keeps the
    # last frame (later in the chain it drops it), and skipped frames aren't scaled for nothing.
    filters.append(f"fps=fps={_rate(info.fps)}:start_time=0")
    if start_frame > 0:
        filters += [f"trim=start_frame={start_frame}", "setpts=PTS-STARTPTS"]
    if info.hdr:
        if not has_zscale():
            raise VideoError(HDR_UNSUPPORTED)
        filters.append(_TONE_MAP)
    to_rgb = f"scale={info.display_width}:{info.display_height}:flags=lanczos+accurate_rnd+full_chroma_int"
    if not info.color_tagged and not info.hdr and (info.width >= 1280 or info.height >= 720):
        to_rgb += ":in_color_matrix=bt709"  # untagged HD is BT.709 in practice; ffmpeg would assume BT.601
    filters += [to_rgb, "setsar=1"]
    return filters


def encode_cmd(dst: Path, w: int, h: int, fps: Fraction) -> list[str]:
    """Raw RGB frames on stdin -> a 10-bit HEVC MP4 segment, tagged BT.709."""
    return [
        *_base(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", _rate(fps), "-i", "-",
        "-vf", _TO_BT709, *pick_encoder(), *_BT709_TAGS, "-an", "-f", "mp4", str(dst),
    ]


# --- Join -----------------------------------------------------------------------------


def concat_and_mux(
    segments: list[Path], src: Path, dst: Path, has_audio: bool, *, duration: float | None = None
) -> None:
    """Join the segments without re-encoding and add the source's sound (and date and other metadata).

    `duration` cuts the sound to the length of the video, e.g. for a preview of the first seconds.
    """
    if not segments:
        raise ValueError("No video segments to join")
    playlist = dst.with_name(dst.name + ".concat.txt")
    playlist.write_text("".join(f"file '{_concat_quote(s)}'\n" for s in segments), encoding="utf-8")

    cmd = [*_base(), "-y", "-f", "concat", "-safe", "0", "-i", str(playlist), "-i", str(src)]
    cmd += ["-map", "0:v:0", "-c:v", "copy", "-tag:v", "hvc1", "-map_metadata", "1"]
    if has_audio:
        cmd += ["-map", "1:a?", *_audio_args(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-movflags", "+faststart", "-f", "mp4", str(dst)]
    try:
        result = subprocess.run(cmd, capture_output=True, **process_options())
    finally:
        playlist.unlink(missing_ok=True)
    if result.returncode != 0:
        log.error("ffmpeg join failed: %s", result.stderr.decode(errors="replace").strip()[-2000:])
        raise VideoError("Couldn't put the finished video together. Please try again.")


def _concat_quote(path: Path) -> str:
    """Escape a path for a single-quoted line in an ffmpeg concat list."""
    return str(path.resolve()).replace("'", "'\\''")


def _audio_args(src: Path) -> list[str]:
    codecs = audio_codecs(src)
    return ["-c:a", "copy"] if codecs and all(c in _QUICKTIME_AUDIO for c in codecs) else list(_AAC)
