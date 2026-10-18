"""Run a model over an image in tiles, so any image size fits in GPU memory."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable

import numpy as np
import torch

from upscaler.config import Cancelled, ProgressFn
from upscaler.models import LoadedModel

log = logging.getLogger(__name__)

BROKEN_OUTPUT_MESSAGE = "The AI model gave a broken result for this image. Please try a different quality mode."

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in input pixels
PlaceFn = Callable[[Box, Box, torch.Tensor], None]  # (tile, tile with context, model output for the latter)

_LANCZOS_LOBES = 3
# Target-canvas tiles are resized in blocks of at most this many output pixels a side,
# so even a tiny picture going to 16K needs only about 50 MB of float32 at a time.
_RESIZE_BLOCK = 2048


class BrokenOutput(RuntimeError):
    """The model gave a broken result even at full precision. The message is written for people:
    trying the same mode again would break the same way, so it suggests a different one."""

    user_message = BROKEN_OUTPUT_MESSAGE

    def __init__(self) -> None:
        super().__init__(BROKEN_OUTPUT_MESSAGE)


class _NonFiniteOutput(Exception):
    pass


def upscale_tiled(
    img: np.ndarray,
    model: LoadedModel,
    *,
    tile: int | None = None,
    pad: int = 16,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Upscale an RGB uint8 image (H, W, 3) to (H*scale, W*scale, 3).

    Each tile is run with `pad` pixels of surrounding context and only its centre is kept,
    so there are no seams. If reduced precision gives NaNs, the model is switched to float32
    for good and the image is redone.
    """
    _check_rgb_uint8(img)
    tile = _check_tiling(tile or model.tile, pad)
    h, w = img.shape[:2]
    s = model.scale
    out = np.empty((h * s, w * s, 3), np.uint8)

    def place(box: Box, padded: Box, result: torch.Tensor) -> None:
        x0, y0, x1, y1 = box
        px0, py0 = padded[:2]
        centre = result[0, :, (y0 - py0) * s : (y1 - py0) * s, (x0 - px0) * s : (x1 - px0) * s]
        if not torch.isfinite(centre).all():
            raise _NonFiniteOutput()
        out[y0 * s : y1 * s, x0 * s : x1 * s] = _to_uint8(centre)

    _run_tiles(img, model, tile, pad, place, "Upscaling", progress, cancel)
    return out


def upscale_tiled_to(
    img: np.ndarray,
    model: LoadedModel,
    out_size: tuple[int, int],
    *,
    tile: int | None = None,
    pad: int = 16,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Upscale an RGB uint8 image (H, W, 3) with the model, straight to `out_size` = (width, height).

    Target-canvas mode: each tile goes through the model and is then resized (Lanczos, on the GPU)
    to exactly its share of the output. Only the output and one tile ever exist, never the full
    model-sized picture (32K for a 16K result). The result matches a Lanczos resize of the whole
    model output, without seams: each output pixel is computed from the same input pixels and
    weights as in a whole-picture resize, and tiles get enough context to supply them.
    """
    _check_rgb_uint8(img)
    out_w, out_h = out_size
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Output size must be positive, got {out_w}x{out_h}")
    tile = _check_tiling(tile or model.tile, pad)
    h, w = img.shape[:2]
    s = model.scale
    cols = _LanczosAxis(w * s, out_w)
    rows = _LanczosAxis(h * s, out_h)
    pad = max(pad, math.ceil((max(cols.support, rows.support) + 2) / s))
    out = np.empty((out_h, out_w, 3), np.uint8)

    def place(box: Box, padded: Box, result: torch.Tensor) -> None:
        x0, y0, x1, y1 = box
        px0, py0 = padded[0] * s, padded[1] * s  # where the model output starts, in model-output pixels
        ox0, ox1 = _output_edge(x0, w, out_w), _output_edge(x1, w, out_w)
        oy0, oy1 = _output_edge(y0, h, out_h), _output_edge(y1, h, out_h)
        if ox0 == ox1 or oy0 == oy1:
            return  # only when shrinking: this tile's share of the output rounds to nothing
        xlo, xhi = cols.reach(ox0, ox1)
        ylo, yhi = rows.reach(oy0, oy1)
        if xlo < px0 or ylo < py0 or xhi > px0 + result.shape[3] or yhi > py0 + result.shape[2]:
            raise RuntimeError("Tile context is too small for the resize")  # a bug, not a user problem
        region = result[0, :, ylo - py0 : yhi - py0, xlo - px0 : xhi - px0].float()
        if not torch.isfinite(region).all():
            raise _NonFiniteOutput()
        region.clamp_(0, 1)  # resize the picture the model meant, as upscale_tiled would have saved it
        # Like Pillow: across first, clipped, then down; in blocks so a huge tile stays small in memory.
        for c in range(ox0, ox1, _RESIZE_BLOCK):
            d = min(c + _RESIZE_BLOCK, ox1)
            clo, chi = cols.reach(c, d)
            across = region[:, :, clo - xlo : chi - xlo] @ cols.weights(c, d, clo, chi, region.device).T
            across.clamp_(0, 1)  # (3, yhi-ylo, d-c)
            for a in range(oy0, oy1, _RESIZE_BLOCK):
                b = min(a + _RESIZE_BLOCK, oy1)
                rlo, rhi = rows.reach(a, b)
                block = rows.weights(a, b, rlo, rhi, region.device) @ across[:, rlo - ylo : rhi - ylo]
                out[a:b, c:d] = _to_uint8(block)  # (3, b-a, d-c) -> (b-a, d-c, 3)

    _run_tiles(img, model, tile, pad, place, "Upscaling to full size", progress, cancel)
    return out


def _check_rgb_uint8(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"Expected an RGB uint8 image of shape (H, W, 3), got {img.dtype} {img.shape}")


def _check_tiling(tile: int, pad: int) -> int:
    if tile <= 0 or pad < 0:
        raise ValueError(f"tile must be positive and pad non-negative, got tile={tile}, pad={pad}")
    return tile


def _tile_boxes(w: int, h: int, tile: int) -> list[Box]:
    return [
        (x0, y0, min(x0 + tile, w), min(y0 + tile, h))
        for y0 in range(0, h, tile)
        for x0 in range(0, w, tile)
    ]


def _run_tiles(
    img: np.ndarray,
    model: LoadedModel,
    tile: int,
    pad: int,
    place: PlaceFn,
    label: str,
    progress: ProgressFn | None,
    cancel: threading.Event | None,
) -> None:
    """Run the model on every tile (with `pad` pixels of context) and hand each result to `place`.

    If reduced precision gives NaNs, the model is switched to float32 for good and every tile is redone.
    """
    h, w = img.shape[:2]
    boxes = _tile_boxes(w, h, tile)
    try:
        with torch.inference_mode():
            # Arrays from np.asarray(PIL image) are read-only, which torch warns about; copy only then.
            x = torch.from_numpy(np.require(img, requirements=["C", "W"])).to(model.device)
            try:
                _each_tile(x, boxes, pad, model, place, label, progress, cancel)
            except _NonFiniteOutput:
                if model.dtype == torch.float32:
                    raise BrokenOutput() from None
                log.warning("Model gave non-finite output in %s; redoing in float32", model.dtype)
                model.net.float()
                model.dtype = torch.float32
                try:
                    _each_tile(x, boxes, pad, model, place, label, progress, cancel)
                except _NonFiniteOutput:
                    raise BrokenOutput() from None
    finally:
        _empty_gpu_cache(model.device)


def _each_tile(
    x: torch.Tensor,
    boxes: list[Box],
    pad: int,
    model: LoadedModel,
    place: PlaceFn,
    label: str,
    progress: ProgressFn | None,
    cancel: threading.Event | None,
) -> None:
    h, w = x.shape[:2]
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        padded = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h))
        place((x0, y0, x1, y1), padded, _run_model(x, padded, model))
        if progress:
            progress(i / len(boxes), label if len(boxes) == 1 else f"{label} (part {i} of {len(boxes)})")


def _run_model(x: torch.Tensor, box: Box, model: LoadedModel) -> torch.Tensor:
    """The model's output (1, 3, h*scale, w*scale) for `box` of the uint8 (H, W, 3) image `x`."""
    x0, y0, x1, y1 = box
    patch = x[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).float().div_(255)
    result = model.net(patch.to(model.dtype).contiguous())
    expected = (1, 3, (y1 - y0) * model.scale, (x1 - x0) * model.scale)
    if tuple(result.shape) != expected:
        raise RuntimeError(f"Model returned shape {tuple(result.shape)}, expected {expected}")
    return result


def _to_uint8(pixels: torch.Tensor) -> np.ndarray:
    """(3, h, w) in 0..1 to uint8 (h, w, 3) on the CPU."""
    # float32 before rounding: bfloat16 can't hold every value in 0..255 exactly.
    return pixels.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _output_edge(edge: int, size: int, out_size: int) -> int:
    """Where input column/row `edge` falls in the output, rounded half up. Neighbouring tiles share it."""
    return (2 * edge * out_size + size) // (2 * size)


class _LanczosAxis:
    """Lanczos-3 weights for resizing one axis from `in_size` to `out_size` samples.

    The same filter and pixel-centre convention as Pillow's Image.resize(..., LANCZOS), including
    the wider filter when shrinking (antialiasing) and the renormalised taps at the picture's edges.
    """

    def __init__(self, in_size: int, out_size: int):
        scale = in_size / out_size
        stretch = max(scale, 1.0)
        self.support = _LANCZOS_LOBES * stretch
        centre = (np.arange(out_size) + 0.5) * scale
        self._first = np.clip(np.trunc(centre - self.support + 0.5), 0, in_size).astype(np.int64)
        self._end = np.clip(np.trunc(centre + self.support + 0.5), 0, in_size).astype(np.int64)
        taps = self._first[:, None] + np.arange(int((self._end - self._first).max()))
        self._used = taps < self._end[:, None]
        weights = np.where(self._used, _lanczos((taps - centre[:, None] + 0.5) / stretch), 0.0)
        self._weights = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

    def reach(self, start: int, stop: int) -> tuple[int, int]:
        """The input samples that output samples start..stop are made from."""
        return int(self._first[start]), int(self._end[stop - 1])

    def weights(self, start: int, stop: int, lo: int, hi: int, device: torch.device) -> torch.Tensor:
        """Dense (stop - start, hi - lo) resize matrix for output samples start..stop from inputs lo..hi."""
        matrix = np.zeros((stop - start, hi - lo), np.float32)
        used = self._used[start:stop]
        rows = np.broadcast_to(np.arange(stop - start)[:, None], used.shape)
        taps = self._first[start:stop, None] - lo + np.arange(used.shape[1])
        matrix[rows[used], taps[used]] = self._weights[start:stop][used]
        return torch.from_numpy(matrix).to(device)


def _lanczos(x: np.ndarray) -> np.ndarray:
    return np.where((x >= -_LANCZOS_LOBES) & (x < _LANCZOS_LOBES), np.sinc(x) * np.sinc(x / _LANCZOS_LOBES), 0.0)


def _empty_gpu_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
