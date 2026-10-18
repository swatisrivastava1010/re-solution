"""Preview styles: one small crop through each style, so people can compare before choosing.

Photo-real Balanced, Rich and Gentle, Faithful and Fast each enlarge the same 256 px crop 4x (to
1024 px), the scale of a 2K picture going to 8K. The crop is the picture's most detailed spot,
unless the person clicked somewhere. "Original (enlarged)" is that crop with a plain Lanczos
enlargement, to compare against. Every style also gets the time the whole picture would take in
it, at the size the person chose.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from upscaler import sizing
from upscaler.config import Cancelled, ProgressFn, UserError
from upscaler.hardware import detect_hardware, photoreal_support
from upscaler.imageio_utils import load_image
from upscaler.models import MODELS, ModelDownloadError
from upscaler.seedvr2 import SeedVR2Unavailable
from upscaler.tiling import BrokenOutput

log = logging.getLogger(__name__)

PREVIEW_ROUTES = ["photoreal:balanced", "photoreal:rich", "photoreal:gentle", "faithful", "fast"]
ORIGINAL = "original"  # the route name of the plain enlargement
PREVIEW_SIZE = 256  # px of the picture
PREVIEW_SCALE = 4

_PHOTOREAL = "photoreal"
_LEVEL_NAMES = {"balanced": "Balanced", "rich": "Rich", "gentle": "Gentle"}
LABELS = {
    ORIGINAL: "Original (enlarged)",
    **{f"{_PHOTOREAL}:{level}": f"Photo-real · {name}" for level, name in _LEVEL_NAMES.items()},
    **{mode: MODELS[mode].label for mode in ("faithful", "faithful_sharp", "fast", "illustration")},
}

_DETAIL_MAP_EDGE = 1024  # the detail search looks at a copy this size
_NEAR_BEST = 0.999  # among spots (almost) as detailed as the best, take the most central
_CROP_QUALITY = 92
_STYLE_FAILED = "This style couldn't be previewed. Please try again."
_ESTIMATE_FORMAT = "jpeg"


# --- Where to look -------------------------------------------------------------------------


def pick_region(rgb: np.ndarray, size: int = PREVIEW_SIZE) -> tuple[int, int, int, int]:
    """The most detailed size x size box (by Laplacian variance) as (x, y, width, height).

    Among boxes nearly as detailed as the best, the most central wins, so a plain picture gets its
    middle. A picture smaller than the box is used whole.
    """
    h, w = rgb.shape[:2]
    box_w, box_h = min(size, w), min(size, h)
    if (box_w, box_h) == (w, h):
        return 0, 0, w, h
    scale = min(1.0, _DETAIL_MAP_EDGE / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    grey = small[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F).astype(np.float64)
    win_w = max(1, min(small.shape[1], round(box_w * scale)))
    win_h = max(1, min(small.shape[0], round(box_h * scale)))
    mean = _window_sums(cv2.integral(lap), win_w, win_h) / (win_w * win_h)
    variance = _window_sums(cv2.integral(lap * lap), win_w, win_h) / (win_w * win_h) - mean**2

    rows, cols = np.nonzero(variance >= _NEAR_BEST * variance.max() - 1e-6)
    centre_y, centre_x = (variance.shape[0] - 1) / 2, (variance.shape[1] - 1) / 2
    best = int(np.argmin((rows - centre_y) ** 2 + (cols - centre_x) ** 2))
    x = min(max(round(cols[best] / scale), 0), w - box_w)
    y = min(max(round(rows[best] / scale), 0), h - box_h)
    return int(x), int(y), box_w, box_h


def _window_sums(integral: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """Sums over every win_w x win_h window that fits, indexed by its top-left corner."""
    s = integral
    return s[win_h:, win_w:] - s[:-win_h, win_w:] - s[win_h:, :-win_w] + s[:-win_h, :-win_w]


def region_at(width: int, height: int, x: float, y: float, size: int = PREVIEW_SIZE) -> tuple[int, int, int, int]:
    """The size x size box centred on a spot the person clicked, kept inside the picture."""
    box_w, box_h = min(size, width), min(size, height)
    left = min(max(round(x - box_w / 2), 0), width - box_w)
    top = min(max(round(y - box_h / 2), 0), height - box_h)
    return int(left), int(top), box_w, box_h


def _inside(region: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """A requested (x, y, width, height) box, at most PREVIEW_SIZE square and moved inside the picture."""
    try:
        x, y, box_w, box_h = (int(v) for v in region)
    except (TypeError, ValueError):
        raise UserError("That spot on the picture wasn't understood. Click on the picture again.") from None
    if box_w <= 0 or box_h <= 0:
        raise UserError("That spot on the picture wasn't understood. Click on the picture again.")
    box_w, box_h = min(box_w, PREVIEW_SIZE, width), min(box_h, PREVIEW_SIZE, height)
    return min(max(x, 0), width - box_w), min(max(y, 0), height - box_h), box_w, box_h


# --- Running the styles --------------------------------------------------------------------


def run_preview(
    path: Path,
    routes: list[str] | None,
    region: tuple[int, int, int, int] | None,
    *,
    progress: ProgressFn,
    cancel: threading.Event,
    out_dir: Path,
    size: str = "8k",
    creativity: float = 0.0,
) -> list[dict]:
    """Enlarge one crop of the picture at `path` in each of `routes`, saving JPEGs in `out_dir`.

    `routes` defaults to PREVIEW_ROUTES, without the Photo-real ones when this computer can't run
    Photo-real. `region` is (x, y, width, height) in the picture, or None for pick_region. `size`
    is the size preset the whole picture would go to, for the time estimates, and `creativity` is
    passed to Photo-real. Returns the original first, then one item per route:
    {route, label, crop_path, seconds, est_full_seconds}. A style that can't run here has
    crop_path and est_full_seconds None and an `error` written for people; the other styles still run.
    """
    from upscaler import pipeline_image  # the engines; loaded only when a preview really runs

    if not 0 <= creativity <= 1:
        raise UserError("Creativity must be between 0 and 1.")
    hardware = detect_hardware()
    routes = _check_routes(routes, hardware)
    progress(0.0, "Opening the picture")
    image = load_image(Path(path))
    height, width = image.rgb.shape[:2]
    estimates = {route: _full_seconds(width, height, size, route, hardware) for route in routes}  # checks `size`
    x, y, box_w, box_h = pick_region(image.rgb) if region is None else _inside(region, width, height)
    window = (slice(y, y + box_h), slice(x, x + box_w))
    crop = _on_white(image.rgb[window], None if image.alpha is None else image.alpha[window])
    icc = image.icc
    del image  # up to about 600 MB for a 16K picture; the styles only need the crop
    out_size = (box_w * PREVIEW_SCALE, box_h * PREVIEW_SCALE)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enlarged = np.asarray(Image.fromarray(crop).resize(out_size, Image.LANCZOS))
    items = [_item(ORIGINAL, _save(enlarged, out_dir / f"0-{ORIGINAL}.jpg", icc), 0.0, None)]

    weights = [_crop_seconds((box_w, box_h), out_size, route, hardware) for route in routes]
    finished = 0.0
    for n, (route, weight) in enumerate(zip(routes, weights), start=1):
        _raise_if_cancelled(cancel)
        start, end = finished / sum(weights), (finished + weight) / sum(weights)
        report = _span(progress, start, end, LABELS[route])
        report(0.0, "Starting")
        started = time.monotonic()
        try:
            up = pipeline_image.upscale_array(
                crop,
                route,
                out_size,
                progress=report,
                cancel=cancel,
                creativity=creativity if route.startswith(_PHOTOREAL) else 0.0,
            )
        except Cancelled:
            raise
        except Exception as exc:
            items.append({**_item(route, None, time.monotonic() - started, None), "error": _friendly(exc)})
        else:
            _raise_if_cancelled(cancel)
            saved = _save(up, out_dir / f"{n}-{route.replace(':', '-')}.jpg", icc)
            items.append(_item(route, saved, time.monotonic() - started, estimates[route]))
        finished += weight
    progress(1.0, "Done")
    return items


def _check_routes(routes: list[str] | None, hardware) -> list[str]:
    if routes is None:
        can_photoreal, _ = photoreal_support(hardware)
        return [route for route in PREVIEW_ROUTES if can_photoreal or not route.startswith(_PHOTOREAL)]
    unknown = [route for route in routes if route not in LABELS or route == ORIGINAL]
    if unknown:
        raise UserError(f"These styles can't be previewed: {', '.join(map(str, unknown))}.")
    return list(dict.fromkeys(routes))  # each style once, in the order asked


def _split(route: str) -> tuple[str, str]:
    mode, _, level = route.partition(":")
    return mode, level or sizing.DEFAULT_PHOTOREAL_LEVEL


def _full_seconds(width: int, height: int, size: str, route: str, hardware) -> float | None:
    """sizing.estimate's time for the whole picture in this style, at `size`; None for Photo-real
    where it can't run, because sizing.estimate would time Faithful, its stand-in, instead."""
    mode, level = _split(route)
    seconds = sizing.estimate(width, height, size, mode, _ESTIMATE_FORMAT, hardware, level=level)["seconds"]
    return None if mode == _PHOTOREAL and not photoreal_support(hardware)[0] else seconds


def _crop_seconds(src: tuple[int, int], dst: tuple[int, int], route: str, hardware) -> float:
    """Expected seconds for the crop, to share out the progress bar (plus a little for loading a model)."""
    mode, level = _split(route)
    return sum(sizing.pass_seconds(src, dst, mode, hardware, level=level)) + 0.5


def _friendly(exc: Exception) -> str:
    """What a style's card says when it failed: only messages written for people are shown as they are."""
    if isinstance(exc, SeedVR2Unavailable):
        return exc.reason
    if isinstance(exc, ModelDownloadError):
        return str(exc)
    if isinstance(exc, BrokenOutput):  # trying again would break the same way
        return exc.user_message
    log.warning("A preview style failed", exc_info=exc)  # anything else is technical, even a ValueError
    return _STYLE_FAILED


def _item(route: str, crop_path: Path | None, seconds: float, est_full_seconds: float | None) -> dict:
    return {
        "route": route,
        "label": LABELS[route],
        "crop_path": None if crop_path is None else str(crop_path),
        "seconds": round(seconds, 1),
        "est_full_seconds": est_full_seconds,
    }


def _save(rgb: np.ndarray, path: Path, icc: bytes | None) -> Path:
    picture = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8))
    picture.save(path, "JPEG", quality=_CROP_QUALITY, icc_profile=icc)
    return path


def _on_white(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """See-through areas shown on white, as in the app's other previews."""
    if alpha is None:
        return np.ascontiguousarray(rgb)
    a = alpha[..., None].astype(np.uint16)
    return ((rgb * a + 255 * (255 - a) + 127) // 255).astype(np.uint8)


def _span(progress: ProgressFn, start: float, end: float, label: str) -> ProgressFn:
    def report(fraction: float, message: str) -> None:
        progress(start + (end - start) * min(max(fraction, 0.0), 1.0), f"{label}: {message}")

    return report


def _raise_if_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()
