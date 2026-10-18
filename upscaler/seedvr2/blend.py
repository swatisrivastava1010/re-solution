"""Overlapping tiles for SeedVR2 and seam-free feathered blending of the upscaled tiles.

SeedVR2 can't take a whole 2K image on a 16 GB Mac, so the input is cut into overlapping tiles
of one size. Every tile then has the same shape for the model, and wide linear ramps across the
overlaps hide the places where neighbouring tiles invented slightly different detail.

Boxes are ``(x0, y0, x1, y1)`` in input pixels, end-exclusive.
"""

from __future__ import annotations

import math

import numpy as np

Box = tuple[int, int, int, int]

_BAND_ROWS = 512  # output rows blended at a time, which keeps an 8K blend to a few tens of MB


def _axis_tiles(length: int, tile: int, overlap: int) -> tuple[int, list[int]]:
    """Tile size and starts along one axis.

    Uses the fewest tiles of at most ``tile`` px, each made just big enough that neighbours
    overlap by at least ``overlap``. Full-size tiles would redo up to twice the image's pixels.
    """
    if length <= tile:
        return length, [0]
    count = math.ceil((length - overlap) / (tile - overlap))
    size = math.ceil((length + (count - 1) * overlap) / count)
    while size - math.ceil((length - size) / (count - 1)) < overlap:  # rounding of the starts
        size += 1
    span = length - size
    return size, sorted({i * span // (count - 1) for i in range(count)})


def tile_grid(w: int, h: int, tile: int, overlap: int) -> list[Box]:
    """Input-space boxes of one size, at most ``tile`` px, covering the image."""
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap must be in [0, {tile}), got {overlap}")
    tile_w, xs = _axis_tiles(w, tile, overlap)
    tile_h, ys = _axis_tiles(h, tile, overlap)
    return [(x0, y0, x0 + tile_w, y0 + tile_h) for y0 in ys for x0 in xs]


def _ramp(length: int, ramp: int, start: bool, end: bool) -> np.ndarray:
    """1-D weights: 1 in the middle, rising linearly from near 0 over ``ramp`` pixels at feathered ends."""
    weights = np.ones(length, np.float32)
    if ramp <= 0:
        return weights
    rising = np.minimum((np.arange(length, dtype=np.float32) + 0.5) / ramp, 1.0)
    if start:
        weights *= rising
    if end:
        weights *= rising[::-1]
    return weights


def feather_weights(h: int, w: int, ramp: int, edges: tuple[bool, bool, bool, bool]) -> np.ndarray:
    """Blend weights for one tile, shape (h, w), all > 0.

    ``edges`` says which sides are interior and get a ramp, in box order (left, top, right, bottom).
    Sides on the image border keep full weight, since no other tile covers them.
    """
    left, top, right, bottom = edges
    return np.outer(_ramp(h, ramp, top, bottom), _ramp(w, ramp, left, right))


def blend_tiles(tiles: list[tuple[Box, np.ndarray]], out_w: int, out_h: int, scale: int, ramp: int) -> np.ndarray:
    """Weighted average of upscaled tiles into one (out_h, out_w, 3) uint8 image.

    Each tile is ``(input box, upscaled pixels)``; its pixels must be exactly the box size times
    ``scale``. Weights are normalised per pixel, so any overlap pattern reproduces flat areas exactly.
    """
    placed = [_place(box, pixels, out_w, out_h, scale, ramp) for box, pixels in tiles]
    out = np.empty((out_h, out_w, 3), np.uint8)
    for band_y0 in range(0, out_h, _BAND_ROWS):
        band_y1 = min(band_y0 + _BAND_ROWS, out_h)
        total = np.zeros((band_y1 - band_y0, out_w, 3), np.float32)
        weight_sum = np.zeros((band_y1 - band_y0, out_w, 1), np.float32)
        for (x0, y0, x1, y1), pixels, wy, wx in placed:
            top, bottom = max(y0, band_y0), min(y1, band_y1)
            if top >= bottom:
                continue
            weights = np.outer(wy[top - y0 : bottom - y0], wx)[..., None]
            total[top - band_y0 : bottom - band_y0, x0:x1] += pixels[top - y0 : bottom - y0] * weights
            weight_sum[top - band_y0 : bottom - band_y0, x0:x1] += weights
        if not weight_sum.all():
            raise ValueError("The tiles don't cover the whole image.")
        out[band_y0:band_y1] = np.clip(np.rint(total / weight_sum), 0, 255).astype(np.uint8)
    return out


def _place(box: Box, pixels: np.ndarray, out_w: int, out_h: int, scale: int, ramp: int):
    """Output-space box and 1-D weight profiles for one tile."""
    x0, y0, x1, y1 = (v * scale for v in box)
    if pixels.shape[:2] != (y1 - y0, x1 - x0) or x1 > out_w or y1 > out_h:
        raise ValueError(f"Tile for box {box} has shape {pixels.shape[:2]}, expected {(y1 - y0, x1 - x0)}")
    wy = _ramp(y1 - y0, ramp, start=y0 > 0, end=y1 < out_h)
    wx = _ramp(x1 - x0, ramp, start=x0 > 0, end=x1 < out_w)
    return (x0, y0, x1, y1), pixels, wy, wx
