"""Upscale one picture: open it, run the chosen engine, resize to the exact size, finish, save.

Engines: Photo-real (SeedVR2, the default), the local spandrel models, or a cloud service.
Photo-real falls back to Faithful, with a note, whenever it can't run on this computer.
Photo-real has three detail levels (sizing.photoreal_route): Balanced (SeedVR2 2x, then the Fast
model straight to the size; the default), Rich (SeedVR2 4x) and Gentle (SeedVR2 2x, then a plain
resize). When one 4x pass isn't enough (16K, or a small picture), the Fast model takes the first
pass's result straight to the exact size, tile by tile. The local models always work that way, so
a big picture going to 16K never makes the model's 4x picture (32K) first.

Vector pictures (SVG) skip the engines: they're drawn straight at the size asked for, whatever the
mode, then finished and saved like any other result.

`upscale_array` runs just the engine part on pixels in memory, for the "Preview styles" crops.

Given a `viewer_dir`, upscale_image also writes the compare viewer's tile pyramid there while the
finished picture is still in memory (imageio_utils.write_tile_pyramid), so the viewer can show the
whole picture and every pixel without opening the saved file again.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from upscaler.cloud import CloudProvider, get_provider, keys
from upscaler.config import Cancelled, ProgressFn, UserError
from upscaler.finishing import GRAIN_LEVELS, finish
from upscaler.hardware import Hardware, detect_hardware, photoreal_support
from upscaler.imageio_utils import (
    FORMATS_WITH_ALPHA,
    OUTPUT_SUFFIXES,
    SAVE_FORMATS,
    LoadedImage,
    check_dpi,
    compare_crops,
    ensure_free_space,
    load_image,
    preview_jpeg,
    render_vector,
    resolve_format,
    save_image,
    unique_path,
    write_tile_pyramid,
)
from upscaler.models import MODELS, LoadedModel, ensure_model, load_model, unload_model
from upscaler.seedvr2 import SeedVR2Unavailable, get_seedvr2_engine
from upscaler.sizing import (
    DEFAULT_DPI,
    DEFAULT_PHOTOREAL_LEVEL,
    MAX_LANCZOS_STRETCH,
    PHOTOREAL_LEVELS,
    PRESETS,
    estimated_file_bytes,
    pass_seconds,
    photoreal_route,
    plan_passes,
    print_size,
    target_size,
)
from upscaler.tiling import upscale_tiled, upscale_tiled_to

log = logging.getLogger(__name__)

PHOTOREAL = "photoreal"
DEFAULT_MODE = PHOTOREAL  # SeedVR2 is the default for pictures: the family's choice
FALLBACK_MODE = "faithful"
LOCAL_MODES = ("faithful", "faithful_sharp", "fast", "illustration")  # also their MODELS keys
CLOUD_PREFIX = "cloud:"
ORIGINAL = "original"  # mode_used when the picture was already big enough and was saved as it is
VECTOR = "vector"  # mode_used for an SVG, drawn at full size instead of upscaled
VECTOR_FORMAT = "SVG"  # LoadedImage.format of a vector picture
VECTOR_NOTE = "Vector image: drawn directly at full size, no AI needed"
SECOND_PASS_MODE = "fast"  # when one pass isn't enough, the Fast model always does the second
# What upscale_array can run: each Photo-real level, and every local model.
UPSCALE_ROUTES = (*(f"{PHOTOREAL}:{level}" for level in PHOTOREAL_LEVELS), *LOCAL_MODES)
SOFTNESS_PER_CREATIVITY = 0.5  # Creativity 100% is SeedVR2 softness 0.5; beyond that it loses the picture

PRESET_TAGS = {
    "4k": "4K",
    "6k": "6K",
    "8k": "8K",
    "8k_tv": "8KTV",
    "16k": "16K",
    "16k_tv": "16KTV",
    "2x": "2x",
    "3x": "3x",
    "4x": "4x",
}

# About 4 GB of working memory at the end; 4x of a 12 MP phone photo (195 MP) still fits.
MAX_OUTPUT_PIXELS = 400_000_000

# When a cloud job needs two passes, the service runs at 4x (the cost of a normal upscale)
# and the Fast model does the rest.
_CLOUD_FIRST_PASS_SCALE = 4
# A one-pass cloud job's share of the progress bar; the rest is for the Fast model, in case the
# service sends back a smaller picture than asked for.
_CLOUD_ONE_PASS_SHARE = 0.9
_MAX_SOURCE_STEM_BYTES = 180  # leaves room in the file name for "-8KTV-faithful_sharp-2.jpg"

# Progress: opening 2%, engine 85%, finishing and saving 13%.
_OPENED_AT = 0.01
_ENGINE_START = 0.02
_ENGINE_END = 0.87
_PREVIEWS_AT = 0.94
_SAVING_AT = 0.97

_TRANSPARENT_NOTE = "This picture has see-through areas, so it was saved as a PNG to keep them."
# Lossy formats sizing has no file-size figures for; their files come out about the size of a JPEG.
_SIZED_LIKE_JPEG = ("webp", "heic", "avif")


@dataclass
class ImageOptions:
    mode: str = DEFAULT_MODE
    size: str = "8k"
    strength: float = 1.0  # detail strength (0..1) for every mode except Photo-real
    grain: str = "off"
    output_format: str = "jpeg"  # "jpeg" | "png" | "tiff" | "webp" | "original" (the picture's own format)
    dpi: int = DEFAULT_DPI  # written into the file, so it opens at the right print size
    photoreal_level: str = DEFAULT_PHOTOREAL_LEVEL  # "balanced" | "rich" | "gentle"
    creativity: float = 0.0  # Photo-real (0..1): how much new detail SeedVR2 may invent
    resemblance: float = 1.0  # Photo-real (0..1): how much of its result is kept over a plain enlargement
    guest_key: str | None = field(default=None, repr=False)  # never in logs


@dataclass
class ImageResult:
    output_path: Path
    width: int
    height: int
    mode_used: str
    notes: list[str]
    seconds: float
    preview: bytes
    compare_before: bytes
    compare_after: bytes
    dpi: int = DEFAULT_DPI
    print: dict = field(default_factory=dict)  # sizing.print_size of the result at `dpi`
    file_bytes: int = 0
    source_width: int = 0  # the picture as it was opened (upright), before upscaling
    source_height: int = 0
    viewer: Path | None = None  # the compare viewer's tile pyramid, when upscale_image was asked for one


def resolve_mode(mode: str, hw: Hardware) -> tuple[str, list[str]]:
    """The mode that will actually run here, plus a note if it had to change.

    Raises UserError (a ValueError) for a mode the app doesn't know.
    """
    if mode == PHOTOREAL:
        ok, reason = photoreal_support(hw)
        if not ok:
            return FALLBACK_MODE, [f"{reason}; used Faithful instead."]
        return mode, []
    if mode in LOCAL_MODES:
        return mode, []
    if mode.startswith(CLOUD_PREFIX):
        provider_id, _ = _parse_cloud_mode(mode)
        _provider(provider_id)
        return mode, []
    raise UserError(f"Unknown quality mode {mode!r}.")


def upscale_image(
    path: Path,
    opts: ImageOptions,
    out_dir: Path,
    *,
    progress: ProgressFn,
    cancel: threading.Event,
    viewer_dir: Path | None = None,
) -> ImageResult:
    """Upscale the picture at `path` into `out_dir`. With `viewer_dir`, the compare viewer's tile
    pyramid of the result is written there too (if that fails, the picture is still saved)."""
    started = time.monotonic()
    _check_options(opts)
    mode, mode_notes = resolve_mode(opts.mode, detect_hardware())

    progress(0.0, "Opening the picture")
    image = load_image(path)
    src = image.rgb
    src_size = (src.shape[1], src.shape[0])
    vector = image.format == VECTOR_FORMAT
    dst = target_size(*src_size, opts.size, exact=vector)  # a vector is drawn at any size, even a smaller one
    _check_output_size(dst, opts.size)
    notes = list(image.notes)  # e.g. "only the first page is upscaled"
    if vector:  # no AI, whatever the mode: drawn at full size, which takes seconds
        progress(_OPENED_AT, "Drawing the picture at full size")
        up, alpha = _draw_vector(Path(path), dst)
    else:
        alpha = image.alpha
    fmt, format_notes = _output_format(opts.output_format, image.format, dst, alpha)
    # Before a long upscale, not after, and leaving the app's reserve free: other jobs, other people's
    # pictures and other apps may have used the space since the picture was added.
    needed = estimated_file_bytes(*dst, estimate_format(fmt), alpha=alpha is not None)
    ensure_free_space(out_dir, needed, keep_reserve=True)
    _raise_if_cancelled(cancel)

    if vector:
        mode_used, notes = VECTOR, [VECTOR_NOTE, *notes]
        progress(_ENGINE_END, "Adding the finishing touches")
        up = finish(src, up, strength=1.0, tone_lock=False, grain=GRAIN_LEVELS[opts.grain], in_place=True)
    elif dst == src_size:
        up, mode_used = src, ORIGINAL
        notes.append(f"This picture is already {PRESETS[opts.size]} or bigger, so it was kept at its original size.")
    else:
        progress(_OPENED_AT, "Getting ready")
        up, mode_used, engine_notes = _run_engine(image, dst, mode, opts, progress, cancel)
        notes += mode_notes + engine_notes
        _raise_if_cancelled(cancel)
        progress(_ENGINE_END, "Adding the finishing touches")
        up = _resize(up, dst)  # lets go of the engine's own-size picture before finishing
        # Photo-real's "stay close to the original" is its detail strength.
        strength = opts.resemblance if mode_used == PHOTOREAL else opts.strength
        up = finish(src, up, strength=strength, tone_lock=True, grain=GRAIN_LEVELS[opts.grain], in_place=True)
        alpha = None if image.alpha is None else _resize(image.alpha, dst)
    notes += format_notes

    _raise_if_cancelled(cancel)
    progress(_PREVIEWS_AT, "Making the previews")
    preview, before, after, viewer = _previews(image, up, alpha, viewer_dir, cancel)

    _raise_if_cancelled(cancel)
    progress(_SAVING_AT, "Saving")
    target = unique_path(out_dir, _output_stem(image.name, opts.size, mode_used), OUTPUT_SUFFIXES[fmt])
    output = save_image(up, alpha, target, fmt, image.icc, dpi=opts.dpi, source_format=image.format)
    progress(1.0, "Done")
    return ImageResult(
        output_path=output,
        width=dst[0],
        height=dst[1],
        mode_used=mode_used,
        notes=notes,
        seconds=time.monotonic() - started,
        preview=preview,
        compare_before=before,
        compare_after=after,
        dpi=opts.dpi,
        print=print_size(*dst, opts.dpi),
        file_bytes=output.stat().st_size,
        source_width=src_size[0],
        source_height=src_size[1],
        viewer=viewer,
    )


def estimate_format(fmt: str, source_format: str | None = None) -> str:
    """The format to estimate the file size of a result saved as `fmt` with, among sizing's
    (JPEG, PNG, TIFF): "original" becomes what it will really be (PNG when the source isn't known),
    and WebP, HEIC and AVIF files come out about the size of a JPEG."""
    fmt, _ = resolve_format(fmt, source_format)
    return "jpeg" if fmt in _SIZED_LIKE_JPEG else fmt


def upscale_array(
    rgb: np.ndarray,
    route: str,
    out_size: tuple[int, int],
    *,
    progress: ProgressFn,
    cancel: threading.Event,
    creativity: float = 0.0,
) -> np.ndarray:
    """`rgb` enlarged to exactly `out_size` = (width, height) by one of UPSCALE_ROUTES.

    Only the engine part of upscale_image: no finishing, no saving, and no fallback, so a preview
    always shows what really ran. Raises SeedVR2Unavailable when Photo-real can't run here, and
    UserError (a ValueError) for an unknown route.
    """
    mode, level = parse_route(route)
    if not 0 <= creativity <= 1:
        raise UserError(f"Creativity must be between 0 and 1, not {creativity}.")
    _raise_if_cancelled(cancel)
    if mode == PHOTOREAL:
        ok, reason = photoreal_support(detect_hardware())
        if not ok:
            raise SeedVR2Unavailable(f"{reason}.")
        up, _ = _run_photoreal(rgb, out_size, level, _softness(creativity), progress, cancel)
    else:
        up, _ = _run_local(mode, rgb, out_size, lambda fraction, message: progress(0.0, message), progress, cancel)
    _raise_if_cancelled(cancel)
    return _resize(up, tuple(out_size))


def parse_route(route: str) -> tuple[str, str]:
    """A route's mode and Photo-real level, e.g. "photoreal:rich" -> ("photoreal", "rich"); "photoreal"
    alone means the default level, and other modes get the default level too (they ignore it).
    Raises UserError (a ValueError) for anything that isn't one of UPSCALE_ROUTES."""
    mode, _, level = route.partition(":") if isinstance(route, str) else ("", "", "")
    if mode == PHOTOREAL and (level or DEFAULT_PHOTOREAL_LEVEL) in PHOTOREAL_LEVELS:
        return mode, level or DEFAULT_PHOTOREAL_LEVEL
    if route in LOCAL_MODES:
        return route, DEFAULT_PHOTOREAL_LEVEL
    raise UserError(f"Unknown route {route!r}; use one of {', '.join(UPSCALE_ROUTES)}.")


# --- Engines ----------------------------------------------------------------------


def _run_engine(
    image: LoadedImage,
    dst: tuple[int, int],
    mode: str,
    opts: ImageOptions,
    progress: ProgressFn,
    cancel: threading.Event,
) -> tuple[np.ndarray, str, list[str]]:
    """Enlarge the picture to at least about `dst`. Returns the pixels, the mode used and any notes."""
    src = image.rgb
    engine_progress = _span(progress, _ENGINE_START, _ENGINE_END)
    if mode.startswith(CLOUD_PREFIX):
        provider_id, _ = _parse_cloud_mode(mode)
        up, warning = _run_cloud(image, mode, dst, opts.guest_key, engine_progress, cancel)
        return up, f"cloud-{provider_id}", [warning] if warning else []

    notes: list[str] = []
    if mode == PHOTOREAL:
        try:
            softness = _softness(opts.creativity)
            up, warning = _run_photoreal(src, dst, opts.photoreal_level, softness, engine_progress, cancel)
            return up, PHOTOREAL, [warning] if warning else []
        except SeedVR2Unavailable as exc:
            log.warning("Photo-real couldn't run (%s); using Faithful", exc.reason, exc_info=True)
            notes.append(f"{exc.reason} Used Faithful instead.")
            mode = FALLBACK_MODE

    up, warning = _run_local(mode, src, dst, _span(progress, _OPENED_AT, _ENGINE_START), engine_progress, cancel)
    return up, mode, notes + ([warning] if warning else [])


def _run_photoreal(
    src: np.ndarray,
    dst: tuple[int, int],
    level: str,
    softness: float,
    progress: ProgressFn,
    cancel: threading.Event,
) -> tuple[np.ndarray, str | None]:
    """SeedVR2 at the level's scale, then the Fast model or (by the caller) a plain resize.

    Returns the enlarged pixels (exactly `dst` after the Fast model) and the route's warning.
    """
    hardware = detect_hardware()
    engine = get_seedvr2_engine(hardware)
    if engine is None:
        raise SeedVR2Unavailable("Photo-real can't run on this computer.")
    if hardware.device == "cuda":
        # SeedVR2 runs as a program of its own on NVIDIA, and needs the graphics memory this app's
        # Fast or Faithful model would otherwise go on holding.
        unload_model()
    src_size = (src.shape[1], src.shape[0])
    route = photoreal_route(src_size, dst, level)
    share = _first_pass_share(src_size, dst, PHOTOREAL, level) if route.fast_pass else 1.0
    # The whole picture, or a smaller copy when SeedVR2's result would be longer than both 8K and
    # the size asked for (sizing.photoreal_route).
    pixels = _resize(src, route.seedvr2_input)
    up = engine.upscale(
        pixels, route.seedvr2_scale, progress=_span(progress, 0.0, share), cancel=cancel, softness=softness
    )
    if not route.fast_pass:
        return up, route.warning
    _raise_if_cancelled(cancel)
    return _second_pass(up, dst, _span(progress, share, 1.0), cancel), route.warning


def _softness(creativity: float) -> float:
    return SOFTNESS_PER_CREATIVITY * creativity


def _run_local(
    mode: str,
    src: np.ndarray,
    dst: tuple[int, int],
    prepare_progress: ProgressFn,
    progress: ProgressFn,
    cancel: threading.Event,
) -> tuple[np.ndarray, str | None]:
    spec = MODELS[mode]
    ensure_model(spec, progress=prepare_progress, cancel=cancel)  # downloads it the first time
    prepare_progress(1.0, f"Loading the {spec.label} model")
    model = load_model(spec)
    src_size = (src.shape[1], src.shape[0])
    plan = plan_passes(src_size, dst, model_scale=model.scale)
    if plan.passes <= 1:
        return _upscale_to(src, model, dst, progress, cancel), None

    # Too far for one pass: the model's own 4x, then the Fast model straight to `dst`.
    share = _first_pass_share(src_size, dst, mode)
    first = upscale_tiled(src, model, progress=_span(progress, 0.0, share), cancel=cancel)
    del model
    _raise_if_cancelled(cancel)
    return _second_pass(first, dst, _span(progress, share, 1.0), cancel), plan.warning


def _upscale_to(
    pixels: np.ndarray, model: LoadedModel, dst: tuple[int, int], progress: ProgressFn, cancel: threading.Event
) -> np.ndarray:
    """The model's result at exactly `dst`, tile by tile.

    Only the result is ever held, never the model's whole 4x picture: for an 8192 x 6144 photo
    going to 16K, that picture alone would be 2.4 GB, and resizing it would take several GB more.
    """
    if (pixels.shape[1] * model.scale, pixels.shape[0] * model.scale) == dst:
        return upscale_tiled(pixels, model, progress=progress, cancel=cancel)  # already the exact size
    return upscale_tiled_to(pixels, model, dst, progress=progress, cancel=cancel)


def _second_pass(pixels: np.ndarray, dst: tuple[int, int], progress: ProgressFn, cancel: threading.Event) -> np.ndarray:
    """The Fast model takes the first pass's result to exactly `dst`, tile by tile, so the 4x-of-that
    picture (32K for a 16K result) is never made."""
    spec = MODELS[SECOND_PASS_MODE]
    # Downloads it the first time.
    ensure_model(spec, progress=lambda fraction, message: progress(0.0, message), cancel=cancel)
    progress(0.0, f"Loading the {spec.label} model")
    model = load_model(spec)
    return _upscale_to(pixels, model, dst, progress, cancel)


def _first_pass_share(
    src: tuple[int, int], dst: tuple[int, int], mode: str, level: str = DEFAULT_PHOTOREAL_LEVEL
) -> float:
    """The first pass's share of the engine's progress bar, by expected time."""
    first, second = pass_seconds(src, dst, mode, detect_hardware(), level=level)
    return first / (first + second) if first + second > 0 else 0.5


def _run_cloud(
    image: LoadedImage,
    mode: str,
    dst: tuple[int, int],
    guest_key: str | None,
    progress: ProgressFn,
    cancel: threading.Event,
) -> tuple[np.ndarray, str | None]:
    """The service's result (at its own size), and a note or warning for the person, if any.

    The service gets the picture as the app opened it (cloud.base.prepare_upload): the upright 8-bit
    RGB pixels, so files only the app can open (EXR, HDR, RAW, CMYK with its profile) arrive right,
    and nothing is decoded or compressed twice. When it sends back a picture too small for the final
    resize to finish (fal's Topaz stops at 4x, Magnific Creative at about 25 megapixels), the Fast
    model takes it the rest of the way rather than stretching it soft. The service is never asked twice.
    """
    provider_id, model_id = _parse_cloud_mode(mode)
    provider = _provider(provider_id)
    key = guest_key or keys.get_key(provider_id)
    if not key:
        raise UserError(f"To use this mode, connect {provider.name} in Settings first.")
    src_size = (image.rgb.shape[1], image.rgb.shape[0])
    plan = plan_passes(src_size, dst)
    scale = cloud_scale(src_size, dst)
    if plan.passes <= 1:
        share, note = _CLOUD_ONE_PASS_SHARE, _short_result_note(provider)
    else:
        share, note = _first_pass_share(src_size, dst, mode), plan.warning
    up = provider.upscale(image, model_id, scale, key, progress=_span(progress, 0.0, share), cancel=cancel)
    if max(up.shape[:2]) * MAX_LANCZOS_STRETCH >= max(dst):
        return up, None  # big enough: the final resize covers the rest
    _raise_if_cancelled(cancel)
    return _second_pass(up, dst, _span(progress, share, 1.0), cancel), note


def cloud_scale(src: tuple[int, int], dst: tuple[int, int]) -> float:
    """The scale a cloud service is asked for to take a `src`-sized picture towards `dst`: all the way
    in one pass, or 4x when one pass isn't enough (the Fast model here does the rest). What the service
    makes, and so charges for, follows from it (CloudProvider.price_usd)."""
    if plan_passes(src, dst).passes <= 1:
        return max(dst) / max(src)
    return float(_CLOUD_FIRST_PASS_SCALE)


def _short_result_note(provider: CloudProvider) -> str:
    return f"{provider.name} sent back a smaller picture than needed, so the Fast model finished the enlargement."


def _parse_cloud_mode(mode: str) -> tuple[str, str]:
    parts = mode.split(":")
    if len(parts) != 3 or not all(parts):
        raise UserError(f"Unknown cloud mode {mode!r}; expected cloud:<service>:<model>.")
    return parts[1], parts[2]


def _provider(provider_id: str) -> CloudProvider:
    try:
        return get_provider(provider_id)
    except KeyError:
        raise UserError(f"Unknown cloud service {provider_id!r}.") from None


# --- Helpers ----------------------------------------------------------------------


def _check_options(opts: ImageOptions) -> None:
    if opts.size not in PRESETS:
        raise UserError(f"Unknown size {opts.size!r}; use one of {', '.join(PRESETS)}.")
    if opts.grain not in GRAIN_LEVELS:
        raise UserError(f"Unknown grain {opts.grain!r}; use one of {', '.join(GRAIN_LEVELS)}.")
    if opts.output_format not in SAVE_FORMATS:
        raise UserError(f"Unknown output format {opts.output_format!r}; use one of {', '.join(SAVE_FORMATS)}.")
    if opts.photoreal_level not in PHOTOREAL_LEVELS:
        raise UserError(
            f"Unknown Photo-real detail level {opts.photoreal_level!r}; use one of {', '.join(PHOTOREAL_LEVELS)}."
        )
    for label, value in (
        ("Detail strength", opts.strength),
        ("Creativity", opts.creativity),
        ("Stay close to the original", opts.resemblance),
    ):
        if not 0 <= value <= 1:
            raise UserError(f"{label} must be between 0 and 1, not {value}.")
    check_dpi(opts.dpi)


def _output_format(
    requested: str, source_format: str | None, size: tuple[int, int], alpha: np.ndarray | None
) -> tuple[str, list[str]]:
    """The format to save in, with notes when "Same as original" or see-through areas changed it.

    Raises UserError, worded for people, when the format can't hold a picture of `size` (WebP above
    16,383 px), so nobody waits for an upscale that can't be saved.
    """
    fmt, note = resolve_format(requested, source_format, size, alpha=alpha is not None)
    notes = [note] if note else []
    if alpha is not None and fmt not in FORMATS_WITH_ALPHA:
        return "png", [*notes, _TRANSPARENT_NOTE]
    return fmt, notes


def _draw_vector(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """An SVG drawn at exactly `size`: RGB, plus alpha when some of it is see-through."""
    rgba = render_vector(path, size)
    alpha = rgba[..., 3]
    rgb = np.ascontiguousarray(rgba[..., :3])
    return rgb, (None if alpha.min() == 255 else np.ascontiguousarray(alpha))


def output_size_problem(dst: tuple[int, int], size: str) -> str | None:
    """Why a `dst` result (for size preset `size`) is too big to make, or None if it's fine."""
    if dst[0] * dst[1] <= MAX_OUTPUT_PIXELS:
        return None
    return (
        f"Making this picture {PRESETS[size]} would give {dst[0]:,} × {dst[1]:,} pixels, which is too "
        "big for this app. Please choose a smaller size, such as 8K."
    )


def _check_output_size(dst: tuple[int, int], size: str) -> None:
    if problem := output_size_problem(dst, size):
        raise UserError(problem)


def _raise_if_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def _span(progress: ProgressFn, start: float, end: float) -> ProgressFn:
    """A progress callback that maps 0..1 onto start..end of `progress`."""

    def report(fraction: float, message: str) -> None:
        progress(start + (end - start) * min(max(fraction, 0.0), 1.0), message)

    return report


def _resize(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos resize of RGB (H, W, 3) or alpha (H, W) to `size` = (width, height)."""
    if (pixels.shape[1], pixels.shape[0]) == size:
        return pixels
    return np.asarray(Image.fromarray(pixels).resize(size, Image.LANCZOS))


def _previews(
    image: LoadedImage,
    up: np.ndarray,
    alpha: np.ndarray | None,
    viewer_dir: Path | None = None,
    cancel: threading.Event | None = None,
) -> tuple[bytes, bytes, bytes, Path | None]:
    """The whole-picture preview and the before/after crops, as JPEG bytes, and the viewer's tile
    pyramid when `viewer_dir` is given (None if there's none).

    A function of its own so the on-white copy of a transparent result (0.6 GB at 16K) is freed
    before saving.
    """
    shown = _on_white(up, alpha)
    before, after = compare_crops(_on_white(image.rgb, image.alpha), shown, icc=image.icc)
    viewer = None
    if viewer_dir is not None:
        viewer = _write_viewer(shown, image, Path(viewer_dir), cancel)
    return preview_jpeg(shown, icc=image.icc), before, after, viewer


def _write_viewer(shown: np.ndarray, image: LoadedImage, folder: Path, cancel: threading.Event | None) -> Path | None:
    """The compare viewer's tiles of the finished picture. They're only for looking: if they can't
    be written (a full disk, say), the picture is still saved, and the viewer makes them later."""
    try:
        source_size = (image.rgb.shape[1], image.rgb.shape[0])
        write_tile_pyramid(shown, folder, icc=image.icc, source_size=source_size, cancel=cancel)
    except Cancelled:
        raise
    except Exception:
        log.warning("Couldn't write the viewer's tiles for %s", image.name, exc_info=True)
        return None
    return folder


def _on_white(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """See-through areas shown on white, for previews (the colour under them is often junk)."""
    if alpha is None:
        return rgb
    out = np.empty_like(rgb)
    for y in range(0, rgb.shape[0], 1024):  # in bands, so an 8K picture needs little extra memory
        a = alpha[y : y + 1024, :, None].astype(np.uint16)
        out[y : y + 1024] = (rgb[y : y + 1024] * a + 255 * (255 - a) + 127) // 255
    return out


def _output_stem(name: str, size: str, mode_used: str) -> str:
    return f"{_source_stem(name)}-{PRESET_TAGS[size]}-{mode_used}"


def _source_stem(name: str) -> str:
    return Path(name).stem.encode()[:_MAX_SOURCE_STEM_BYTES].decode("utf-8", "ignore") or "picture"
