"""What we need to know about a video before upscaling it. PyAV only reads here; ffmpeg does the work."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

OUTPUT_LONG_EDGE = 3840

# FFmpeg's AVColorTransferCharacteristic / AVColorSpace / AVFieldOrder values.
_TRC_PQ = 16
_TRC_HLG = 18
_COLORSPACE_UNSPECIFIED = 2
_FIELD_ORDER_UNKNOWN = 0
_FIELD_ORDER_PROGRESSIVE = 1

_STANDARD_RATES = [
    Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30), Fraction(48),
    Fraction(50), Fraction(60000, 1001), Fraction(60), Fraction(100), Fraction(120), Fraction(240),
]
_SNAP_TOLERANCE = 0.02  # an average rate this close to a standard one is that rate with a few dropped frames
_DEFAULT_RATE = Fraction(30)


class VideoError(Exception):
    """A video can't be processed. The message is written for the person using the app."""


class NotEnoughDiskSpace(VideoError):
    """There isn't room on the disk to finish a video. The finished segments are kept in the job's
    folder, so the same job carries on from them once there's space (if the caller keeps that folder)."""


@dataclass
class VideoInfo:
    width: int  # stored size, before rotation
    height: int
    display_width: int  # what people see: square pixels, rotation applied
    display_height: int
    fps: Fraction  # the constant rate frames are decoded at
    frames: int  # frames at `fps` (an estimate for variable-frame-rate sources)
    duration: float  # seconds
    rotation: int  # degrees clockwise to show it upright: 0, 90, 180 or 270
    interlaced: bool
    hdr: bool
    has_audio: bool
    codec: str
    color_tagged: bool = True  # False when the file doesn't say which YUV colour matrix it uses


def probe_video(path: Path) -> VideoInfo:
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as e:
        raise VideoError("This file isn't a video the app can read. Try exporting it again as an MP4.") from e
    with container:
        if not container.streams.video:
            raise VideoError("This file has no picture in it, so there's nothing to upscale.")
        stream = container.streams.video[0]
        cc = stream.codec_context
        if not cc.width or not cc.height:
            raise VideoError("This file isn't a video the app can read. Try exporting it again as an MP4.")

        rotation = _rotation(stream)
        display_w, display_h = _display_size(cc.width, cc.height, _fraction(stream.sample_aspect_ratio), rotation)
        fps = _frame_rate(_fraction(stream.average_rate), _fraction(stream.guessed_rate))
        duration = _duration(container, stream)
        return VideoInfo(
            width=cc.width,
            height=cc.height,
            display_width=display_w,
            display_height=display_h,
            fps=fps,
            frames=max(1, round(duration * fps)),
            duration=duration,
            rotation=rotation,
            interlaced=_is_interlaced(container, stream),
            hdr=cc.color_trc in (_TRC_PQ, _TRC_HLG),
            has_audio=bool(container.streams.audio),
            codec=cc.name,
            color_tagged=cc.colorspace != _COLORSPACE_UNSPECIFIED,
        )


def audio_codecs(path: Path) -> list[str]:
    with av.open(str(path)) as container:
        return [s.codec_context.name for s in container.streams.audio]


def output_size_4k(info: VideoInfo, long_edge: int = OUTPUT_LONG_EDGE) -> tuple[int, int]:
    """Output size with a `long_edge` long side and even sides (HEVC 4:2:0 needs them), never below the source."""
    w, h = info.display_width, info.display_height
    if max(w, h) >= long_edge:
        return w + w % 2, h + h % 2
    scale = Fraction(long_edge, max(w, h))
    return _nearest_even(w * scale), _nearest_even(h * scale)


def _fraction(value: object) -> Fraction | None:
    """PyAV gives rates as its own rational type; Fraction is easier to compute with."""
    if value is None or not getattr(value, "denominator", 0):
        return None
    return Fraction(value.numerator, value.denominator)


def _nearest_even(x: Fraction) -> int:
    return max(2, 2 * math.floor(x / 2 + Fraction(1, 2)))


def _rotation(stream: av.video.stream.VideoStream) -> int:
    matrix = stream.codec_context.coded_side_data.get("display_matrix")
    if matrix:
        return _clockwise_from_display_matrix(matrix)
    tag = stream.metadata.get("rotate", "0")  # written by older tools instead of a display matrix
    return int(tag) % 360 if tag.lstrip("-").isdigit() else 0


def _clockwise_from_display_matrix(raw: bytes) -> int:
    """Same maths as FFmpeg's av_display_rotation_get, which gives counter-clockwise degrees."""
    m = struct.unpack("=9i", raw[:36])
    a, b, c, d = (v / 65536 for v in (m[0], m[1], m[3], m[4]))  # 16.16 fixed point
    scale_x, scale_y = math.hypot(a, c), math.hypot(b, d)
    if not scale_x or not scale_y:
        return 0
    counter_clockwise = -math.degrees(math.atan2(b / scale_y, a / scale_x))
    return round(-counter_clockwise / 90) * 90 % 360


def _display_size(w: int, h: int, sar: Fraction | None, rotation: int) -> tuple[int, int]:
    if sar and sar > 0 and sar != 1:
        w = max(1, math.floor(w * sar + Fraction(1, 2)))
    return (h, w) if rotation in (90, 270) else (w, h)


def _frame_rate(average: Fraction | None, nominal: Fraction | None) -> Fraction:
    """A constant rate for decoding. Phones record a variable rate; their average sits a little
    below the nominal rate (dropped frames), so the nominal standard rate wins when it is close."""
    average = average if average and average > 0 else None
    nominal = nominal if nominal and nominal > 0 else None
    if average is None:
        return _snap(nominal) if nominal else _DEFAULT_RATE
    # Up to a fifth of frames dropped still counts as "the nominal rate"; beyond that the nominal
    # rate is more likely a timestamp artefact (e.g. 600), and doubling the work would be wasted.
    if nominal and _is_standard(nominal) and average * Fraction(99, 100) <= nominal <= average * Fraction(5, 4):
        return _snap(nominal)
    return _snap(average)


def _is_standard(rate: Fraction) -> bool:
    return any(abs(rate / s - 1) <= 0.005 for s in _STANDARD_RATES)


def _snap(rate: Fraction) -> Fraction:
    closest = min(_STANDARD_RATES, key=lambda s: abs(rate / s - 1))
    return closest if abs(rate / closest - 1) <= _SNAP_TOLERANCE else rate.limit_denominator(1001)


def _duration(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float:
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / av.time_base
    return _duration_from_packets(container, stream)


def _duration_from_packets(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float:
    end = 0.0
    for packet in container.demux(stream):
        if packet.pts is not None and packet.time_base:
            end = max(end, float((packet.pts + (packet.duration or 0)) * packet.time_base))
    container.seek(0)
    if end <= 0:
        raise VideoError("This video seems to be empty or damaged. Try exporting it again as an MP4.")
    return end


def _is_interlaced(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> bool:
    field_order = stream.codec_context.field_order
    if field_order != _FIELD_ORDER_UNKNOWN:
        return field_order != _FIELD_ORDER_PROGRESSIVE
    try:  # the container doesn't say; ask the first decoded frame
        frame = next(container.decode(stream))
    except (StopIteration, av.FFmpegError):
        return False
    return bool(frame.interlaced_frame)
