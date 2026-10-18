"""Output sizes for each preset, how many AI passes it takes to get there, and what a job will cost
in time, disk space and print size.

Plain maths on the standard library (and the app's small shared types) only, so anything can use
it cheaply (torch is loaded only when an estimate needs to know whether Photo-real runs on this
computer).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from upscaler.config import SPACE_RESERVE_BYTES, SPACE_RESERVE_TEXT, UserError

if TYPE_CHECKING:
    from upscaler.hardware import Hardware

LONG_EDGE_PRESETS = {"4k": 3840, "6k": 6144, "8k": 8192, "16k": 16384}
TV_PRESETS = {"8k_tv": (7680, 4320), "16k_tv": (15360, 8640)}  # landscape box; portrait pictures turn it
SCALE_PRESETS = {"2x": 2, "3x": 3, "4x": 4}

PRESETS: dict[str, str] = {
    "4k": "4K (3840 px)",
    "6k": "6K (6144 px)",
    "8k": "8K (8192 px)",
    "8k_tv": "8K TV (7680 × 4320)",
    "16k": "16K (16384 px)",
    "16k_tv": "16K TV (15360 × 8640)",
    "2x": "2× bigger",
    "3x": "3× bigger",
    "4x": "4× bigger",
}
SIXTEEN_K_PRESETS = ("16k", "16k_tv")

# Photo-real detail levels (docs/DESIGN.md, "Photo-real detail levels"). All of them use SeedVR2:
# balanced: SeedVR2 2x, then the Fast model straight to the size (the default: natural, crisp edges)
# rich:     SeedVR2 4x (the most new detail; the Fast model adds the rest beyond about 5x), or 2x
#           when a plain resize can cover the rest (up to 2.5x; 2.6x for a big photo 4x would shrink)
# gentle:   SeedVR2 2x, then a plain resize (the softest; the Fast model only beyond about 5x)
# A picture so small that 2x and the Fast model's 4x can't reach the size (more than 10x) gets
# SeedVR2 4x at every level, so nothing is left to a blurry stretch.
PHOTOREAL_LEVELS = ("balanced", "rich", "gentle")
DEFAULT_PHOTOREAL_LEVEL = "balanced"

DEFAULT_DPI = 300
DPI_CHOICES = (72, 150, 300, 600)

SMALL_IMAGE_WARNING = "Your image is small, so it needs two AI passes; the result may look a little painted."
TWO_PASS_NOTE = (
    "16K takes two AI passes: your chosen mode does the first 4×, then the Fast model takes it to full size."
)

# One model pass may be stretched this much more by the final Lanczos resize
# before a second pass is worth it (4x model -> up to 5x total).
MAX_LANCZOS_STRETCH = Fraction(5, 4)
_FAST_MODEL_SCALE = 4  # the Fast model, which takes a first pass's result the rest of the way
# SeedVR2 works on every real pixel of a picture, even when that makes a bigger picture than needed
# (the final resize brings it down), as long as the result is no longer than this or the final size.
# Beyond that, its time and memory grow for nothing, so it gets a smaller copy (photoreal_route).
SEEDVR2_MAX_LONG_EDGE = LONG_EDGE_PRESETS["8k"]  # about 20 minutes of SeedVR2 on an M4
# Rich runs 2x on the whole picture, with a slightly longer final resize than usual, rather than 4x
# on a copy that had to be shrunk first (3200 px going to 8K: 2x and 1.28x, not 4x on 2048 px).
_RICH_2X_STRETCH = Fraction(13, 10)
# A picture that can't reach 8K in one pass is "small"; bigger ones only need two passes for 16K.
_SMALL_REACH = LONG_EDGE_PRESETS["8k"]

# --- Time and size estimates ---------------------------------------------------------
# Seconds per input megapixel for one 4x pass on an Apple M4 with 16 GB (docs/RESEARCH.md §2, §4a).
# Photo-real: about 22 min for a 2048x1539 picture and 10 min for 941x1672, including tile overlap.
SECONDS_PER_INPUT_MP = {"photoreal": 400.0, "faithful": 45.0, "faithful_sharp": 45.0, "fast": 0.5, "illustration": 3.5}
CLOUD_SECONDS_PER_INPUT_MP = 20.0  # mostly upload, queue and download, so the same on every computer
SECOND_PASS_SECONDS_PER_MP = 0.5  # the Fast model, per megapixel of the first pass's result
PHOTOREAL_SCALES = (2, 4)  # SeedVR2 runs 2x as well as 4x
_PHOTOREAL_2X_WORK = 0.25  # a 2x pass is about a quarter of the work of a 4x pass (§4a)
_HARDWARE_SPEED = {"mps": 1.0, "cuda": 1 / 4, "cpu": 15.0}  # multiplies the M4 time
# Finishing, previews and saving, per output megapixel (measured at 16K on the M4; numpy and
# Pillow work on the processor, so this isn't scaled for the GPU).
_FINISH_SECONDS_PER_OUTPUT_MP = {"jpeg": 0.035, "png": 0.07, "tiff": 0.06}
BYTES_PER_PIXEL = {"jpeg": 0.2, "png": 1.6, "tiff": 1.8}  # typical file size per output pixel
_ALPHA_SIZE_FACTOR = 4 / 3  # a kept alpha channel is a fourth channel next to R, G and B
_FORMAT_NAMES = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF"}

_LONG_JOB_SECONDS = 600
_VERY_LARGE_PIXELS = 100_000_000
_FREE_SPACE_FACTOR = 3  # file sizes vary a lot with the picture, so warn well before the disk is full

_CLOUD_PREFIX = "cloud:"
_FALLBACK_MODE = "faithful"
_CM_PER_INCH = 2.54


def _round_half_up(x: Fraction) -> int:
    return math.floor(x + Fraction(1, 2))


def _scale_factor(w: int, h: int, preset: str) -> Fraction:
    if preset in SCALE_PRESETS:
        return Fraction(SCALE_PRESETS[preset])
    if preset in LONG_EDGE_PRESETS:
        return Fraction(LONG_EDGE_PRESETS[preset], max(w, h))
    if preset in TV_PRESETS:
        box_w, box_h = TV_PRESETS[preset] if w >= h else TV_PRESETS[preset][::-1]
        return min(Fraction(box_w, w), Fraction(box_h, h))
    raise UserError(f"Unknown size {preset!r}; use one of {', '.join(PRESETS)}.")


def target_size(w: int, h: int, preset: str, *, exact: bool = False) -> tuple[int, int]:
    """Output size for `preset`. Never smaller than the input, unless `exact`: a vector, which is
    drawn at any size, gets the preset's own size even when its declared size is bigger."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Image size must be positive, got {w}x{h}")
    factor = _scale_factor(w, h, preset)
    if factor == 1 or (factor < 1 and not exact):
        return w, h
    return max(1, _round_half_up(w * factor)), max(1, _round_half_up(h * factor))


@dataclass
class PassPlan:
    passes: int  # 0: already at the target size; 1: one pass, then an exact resize; 2: see below
    second_pass_scale: float | None  # with 2 passes: how much the Fast model enlarges the first pass's result
    warning: str | None


def plan_passes(src: tuple[int, int], dst: tuple[int, int], model_scale: int = 4) -> PassPlan:
    """One model pass, then an exact resize, unless that resize would have to stretch too far.

    Then the chosen mode runs once at `model_scale`, and the Fast model takes its result straight
    to `dst` (tiling.upscale_tiled_to), so no bigger-than-needed picture is ever made.
    """
    factor = Fraction(max(dst), max(src))
    if factor <= 1:
        return PassPlan(0, None, None)
    if factor <= model_scale * MAX_LANCZOS_STRETCH:
        return PassPlan(1, None, None)
    small = max(src) * model_scale * MAX_LANCZOS_STRETCH < _SMALL_REACH
    return PassPlan(2, float(factor / model_scale), SMALL_IMAGE_WARNING if small else TWO_PASS_NOTE)


def first_pass_scale(src: tuple[int, int], dst: tuple[int, int], scales: tuple[int, ...]) -> int:
    """The smallest model scale that gets close enough to `dst` for the final resize, else the biggest."""
    factor = Fraction(max(dst), max(src))
    return next((s for s in sorted(scales) if s * MAX_LANCZOS_STRETCH >= factor), max(scales))


@dataclass(frozen=True)
class PhotorealRoute:
    """How Photo-real takes a picture to its size at one detail level."""

    seedvr2_input: tuple[int, int]  # the picture SeedVR2 gets: the original, or a smaller copy (see below)
    seedvr2_scale: int  # 2 or 4
    fast_pass: bool  # the Fast model takes SeedVR2's result straight to the size; otherwise a plain resize
    warning: str | None


def photoreal_route(src: tuple[int, int], dst: tuple[int, int], level: str = DEFAULT_PHOTOREAL_LEVEL) -> PhotorealRoute:
    """The SeedVR2 scale for `level`, and whether the Fast model or a plain resize does the rest.

    SeedVR2 keeps every real pixel of the picture, even when its result is a little bigger than
    needed (a 3000 px photo going to 4K becomes 6000 px, and the final resize brings it down), as
    long as that result is no longer than SEEDVR2_MAX_LONG_EDGE or the final size. Only beyond
    that does it get a copy shrunk just enough: a 5000 px photo going to 16K at Rich is shrunk to
    4096 px for its 4x, instead of SeedVR2 making a 20000 px picture to throw away (which would
    take hours and several GB of memory). Rich avoids that shrink with 2x when 2x gets within
    _RICH_2X_STRETCH of the size.
    """
    if level not in PHOTOREAL_LEVELS:
        raise UserError(f"Unknown Photo-real detail level {level!r}; use one of {', '.join(PHOTOREAL_LEVELS)}.")
    factor = Fraction(max(dst), max(src))
    plan = plan_passes(src, dst, model_scale=max(PHOTOREAL_SCALES))
    limit = max(max(dst), SEEDVR2_MAX_LONG_EDGE)
    if level == "rich":
        scale = first_pass_scale(src, dst, PHOTOREAL_SCALES)
        if scale == 4 and max(src) * scale > limit and factor <= 2 * _RICH_2X_STRETCH:
            scale = 2  # keeps every real pixel of a big photo (see _RICH_2X_STRETCH)
        fast_pass = plan.passes == 2
        warning = plan.warning if fast_pass else None
    else:
        # 2x, unless the Fast model's 4x after it can't reach the size (a phone screenshot going to
        # 16K): 4x then, rather than a blurry stretch at the end. On a picture that small, SeedVR2's
        # 4x takes minutes (about 11 for 1500 px on an M4), not hours.
        reach = _FAST_MODEL_SCALE * MAX_LANCZOS_STRETCH
        scale = next((s for s in sorted(PHOTOREAL_SCALES) if s * reach >= factor), max(PHOTOREAL_SCALES))
        fast_pass = factor > scale if level == "balanced" else plan.passes == 2
        # Two steps are what these levels are, so only a small picture gets a word of warning.
        warning = SMALL_IMAGE_WARNING if fast_pass and plan.warning == SMALL_IMAGE_WARNING else None
    seedvr2_input = src
    made = max(src) * scale
    if made > limit:
        shrink = Fraction(limit, made)
        seedvr2_input = (max(1, _round_half_up(src[0] * shrink)), max(1, _round_half_up(src[1] * shrink)))
    return PhotorealRoute(seedvr2_input, scale, fast_pass, warning)


def print_size(w: int, h: int, dpi: int) -> dict:
    """How big a w×h picture prints at `dpi`, in inches and centimetres (to 0.1)."""
    if dpi <= 0:
        raise ValueError(f"DPI must be positive, got {dpi}")
    inches = (w / dpi, h / dpi)
    return {
        "inches": tuple(round(side, 1) for side in inches),
        "cm": tuple(round(side * _CM_PER_INCH, 1) for side in inches),
        "dpi": dpi,
    }


def estimated_file_bytes(w: int, h: int, fmt: str, *, alpha: bool = False) -> int:
    """A typical file size for a w×h picture saved as `fmt`; `alpha` if the file keeps an alpha channel."""
    if fmt not in BYTES_PER_PIXEL:
        raise UserError(f"Unknown output format {fmt!r}; use one of {', '.join(BYTES_PER_PIXEL)}.")
    return round(w * h * BYTES_PER_PIXEL[fmt] * (_ALPHA_SIZE_FACTOR if alpha else 1))


def free_disk_bytes(directory: Path) -> int:
    """Free space on the disk that holds `directory`, which doesn't have to exist yet.

    Raises OSError when the disk can't be read (an unplugged drive, say).
    """
    directory = Path(directory).absolute()
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    return shutil.disk_usage(directory).free


def describe_size(num_bytes: int) -> str:
    """'363 MB', '1.2 GB': file sizes the way Finder and Explorer show them."""
    if num_bytes >= 10**9:
        return f"{num_bytes / 1e9:.1f} GB"
    return f"{max(1, round(num_bytes / 1e6))} MB"


def mode_for_hardware(mode: str, hw: Hardware) -> tuple[str, str | None]:
    """The mode that will run on `hw` (Photo-real falls back to Faithful), and the reason if it changed."""
    from upscaler.hardware import photoreal_support  # here, so importing sizing doesn't load torch

    if mode == "photoreal":
        ok, reason = photoreal_support(hw)
        if not ok:
            return _FALLBACK_MODE, reason
    elif mode not in SECONDS_PER_INPUT_MP and not mode.startswith(_CLOUD_PREFIX):
        raise UserError(f"Unknown quality mode {mode!r}.")
    return mode, None


def pass_seconds(
    src: tuple[int, int], dst: tuple[int, int], mode: str, hw: Hardware, *, level: str = DEFAULT_PHOTOREAL_LEVEL
) -> tuple[float, float]:
    """Estimated seconds for the first pass (in `mode`, as it runs on `hw`) and the second (Fast) pass.

    `level` is the Photo-real detail level; other modes ignore it.
    """
    plan = plan_passes(src, dst)
    if plan.passes == 0:
        return 0.0, 0.0
    hardware = _HARDWARE_SPEED.get(hw.device, _HARDWARE_SPEED["cpu"])
    if mode == "photoreal":
        route = photoreal_route(src, dst, level)
        input_mp = route.seedvr2_input[0] * route.seedvr2_input[1] / 1e6
        first = SECONDS_PER_INPUT_MP[mode] * input_mp * hardware
        if route.seedvr2_scale == 2:
            first *= _PHOTOREAL_2X_WORK
        second = 0.0
        if route.fast_pass:
            second = SECOND_PASS_SECONDS_PER_MP * input_mp * route.seedvr2_scale**2 * hardware
        return first, second
    src_mp = src[0] * src[1] / 1e6
    if mode.startswith(_CLOUD_PREFIX):
        first = CLOUD_SECONDS_PER_INPUT_MP * src_mp
    else:
        first = SECONDS_PER_INPUT_MP[mode] * src_mp * hardware
    second = 0.0
    if plan.passes == 2:
        second = SECOND_PASS_SECONDS_PER_MP * src_mp * 16 * hardware  # the first pass made it 4x bigger
    return first, second


def ai_passes(src: tuple[int, int], dst: tuple[int, int], mode: str, *, level: str = DEFAULT_PHOTOREAL_LEVEL) -> int:
    """How many AI passes `mode` makes on the way to `dst`: 0 (already big enough), 1 or 2."""
    plan = plan_passes(src, dst)
    if plan.passes == 0 or mode != "photoreal":
        return plan.passes
    return 2 if photoreal_route(src, dst, level).fast_pass else 1


def estimate(
    w: int,
    h: int,
    size: str,
    mode: str,
    fmt: str,
    hw: Hardware,
    *,
    dpi: int = DEFAULT_DPI,
    out_dir: Path | None = None,
    level: str = DEFAULT_PHOTOREAL_LEVEL,
    vector: bool = False,
) -> dict:
    """What upscaling a w×h picture will give and cost, for the readout before a job starts.

    `out_dir` is where the result will be saved (default: the disk with the home folder).
    `level` is the Photo-real detail level; other modes ignore it. `vector`: an SVG, drawn straight
    at exactly the size asked for, with no AI and so no fallback.
    Seconds come from speeds measured on an M4, scaled for `hw`, plus finishing and saving.
    """
    if level not in PHOTOREAL_LEVELS:
        raise UserError(f"Unknown Photo-real detail level {level!r}; use one of {', '.join(PHOTOREAL_LEVELS)}.")
    out = target_size(w, h, size, exact=vector)
    file_bytes = estimated_file_bytes(*out, fmt)
    used, fallback_reason = mode_for_hardware(mode, hw)
    if vector:
        fallback_reason = None
    first, second = (0.0, 0.0) if vector else pass_seconds((w, h), out, used, hw, level=level)
    seconds = round(first + second + _FINISH_SECONDS_PER_OUTPUT_MP[fmt] * out[0] * out[1] / 1e6, 1)

    warnings: list[str] = []
    if fallback_reason:
        warnings.append(f"{fallback_reason}; Faithful will be used instead.")
    if seconds > _LONG_JOB_SECONDS:
        warnings.append(f"This will take about {round(seconds / 60)} minutes.")
    if out[0] * out[1] > _VERY_LARGE_PIXELS:
        what = "16K files are" if size in SIXTEEN_K_PRESETS else "Pictures this big are"
        warnings.append(
            f"{what} very large: about {describe_size(file_bytes)} as {_FORMAT_NAMES[fmt]}. "
            "They mainly help big prints; on a screen they look the same as 8K."
        )
    try:
        free = free_disk_bytes(out_dir if out_dir is not None else Path.home())
    except OSError:  # e.g. the output folder is on a drive that isn't plugged in; saving will say so
        free = None
    if free is not None and free < _FREE_SPACE_FACTOR * file_bytes + SPACE_RESERVE_BYTES:
        warnings.append(
            f"Not enough free space: this picture needs about {describe_size(_FREE_SPACE_FACTOR * file_bytes)} "
            f"free to save safely, and Re:Solution always leaves {SPACE_RESERVE_TEXT} free for the computer's "
            f"other apps. The disk has {describe_size(free)}. Free up some space first"
            + (", or choose JPEG." if fmt != "jpeg" else ".")
        )

    return {
        "out": out,
        "megapixels": round(out[0] * out[1] / 1e6, 1),
        "seconds": seconds,
        "file_mb": round(file_bytes / 1e6, 1),
        "print": print_size(*out, dpi),
        "passes": 0 if vector else ai_passes((w, h), out, used, level=level),
        "warnings": warnings,
        # What results can take up before the disk is down to the space the app always leaves free, for
        # a warning about a whole batch (None when the disk can't be read).
        "room_mb": None if free is None else round(max(0, free - SPACE_RESERVE_BYTES) / 1e6, 1),
    }
