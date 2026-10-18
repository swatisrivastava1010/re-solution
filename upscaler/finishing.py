"""Finishing: blend AI detail over a clean resize, lock the original's tones, add fine grain.

Everything runs in horizontal strips so an 8192x8192 result needs only a few hundred MB
of working memory instead of several full-size float32 copies. With `in_place`, the strips are
written back into the AI result itself, so a 16K picture (0.6 GB) doesn't need a second canvas.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

GRAIN_LEVELS = {"off": 0.0, "subtle": 3.0, "medium": 6.0}

_STRIP_PIXELS = 2_000_000  # output pixels per strip; bounds the float32 working memory
_TONE_GRID_DIVISOR = 32
_TONE_GRID_MIN = 8
_LANCZOS_REACH = 3  # Lanczos-3 looks 3 output pixels each way
_GRAIN_KERNEL = (0.2, 0.6, 0.2)  # a slight blur so grain isn't single-pixel noise
_GRAIN_GAIN = 2.0


def finish(
    src: np.ndarray,
    up: np.ndarray,
    *,
    strength: float = 1.0,
    tone_lock: bool = True,
    grain: float = 0.0,
    seed: int = 0,
    in_place: bool = False,
) -> np.ndarray:
    """Finish an AI result `up` (already at the final size) using the original `src`.

    - strength: how much AI detail to keep over a Lanczos resize of `src` (0..1).
    - tone_lock: take low-frequency colour and tone from `src`, so only fine detail comes from the AI.
    - grain: film-grain amount in 0..255 levels (see GRAIN_LEVELS), strongest in the midtones.
    - in_place: write the result into `up` (when it is writable) instead of a new array.
    """
    _check_rgb_uint8(src, "src")
    _check_rgb_uint8(up, "up")
    if not 0 <= strength <= 1:
        raise ValueError(f"strength must be between 0 and 1, got {strength}")
    if grain < 0:
        raise ValueError(f"grain must not be negative, got {grain}")
    writable = in_place and up.flags.writeable
    if strength == 1 and not tone_lock and grain == 0:
        return up if writable else up.copy()

    h, w = up.shape[:2]
    src_channels = _float_channels(src) if strength < 1 or tone_lock else []
    correction = _tone_correction(src_channels, up, strength) if tone_lock else None
    noise = _GrainRows(seed, w) if grain > 0 else None

    # Safe in place: the tone correction has already read all of `up`, and each strip is read before
    # it is written.
    out = up if writable else np.empty_like(up)
    rows_per_strip = max(1, _STRIP_PIXELS // w)
    for y0 in range(0, h, rows_per_strip):
        y1 = min(y0 + rows_per_strip, h)
        strip = up[y0:y1].astype(np.float32)
        if strength < 1:
            base = _resize_rows(src_channels, (w, h), y0, y1, Image.LANCZOS)
            strip -= base
            strip *= strength
            strip += base
        if correction is not None:
            strip += _resize_rows(correction, (w, h), y0, y1, Image.BILINEAR)
        if noise is not None:
            _add_grain(strip, noise.next_rows(y1 - y0), grain)
        out[y0:y1] = _to_uint8(strip)
    return out


def _check_rgb_uint8(img: np.ndarray, name: str) -> None:
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"{name} must be an RGB uint8 image of shape (H, W, 3), got {img.dtype} {img.shape}")


def _float_channels(img: np.ndarray) -> list[Image.Image]:
    """One 32-bit float PIL image per channel, so resizing keeps full precision."""
    return [Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32)) for c in range(3)]


def _resize_rows(
    channels: list[Image.Image], size: tuple[int, int], y0: int, y1: int, resample: Image.Resampling
) -> np.ndarray:
    """Rows y0..y1 of `channels` resized to `size`, as float32 (rows, W, C).

    Pillow takes its filter taps from the whole source even with a `box`, so the rows
    match a full-size resize without ever making one.
    """
    w, h = size
    src_w, src_h = channels[0].size
    step = src_h / h
    box = (0, y0 * step, src_w, min(y1 * step, src_h))
    return np.stack([np.asarray(c.resize((w, y1 - y0), resample, box=box)) for c in channels], axis=-1)


def _tone_grid_size(w: int, h: int) -> tuple[int, int]:
    def side(n: int) -> int:
        return min(n, max(_TONE_GRID_MIN, n // _TONE_GRID_DIVISOR))

    return side(w), side(h)


def _tone_correction(src_channels: list[Image.Image], up: np.ndarray, strength: float) -> list[Image.Image]:
    """Low-resolution colour/tone difference between the original and the blended result.

    Resizing is linear, so low(base) - low(base + s*(up - base)) = s*(low(base) - low(up)).
    low(base) is taken straight from `src`: Lanczos-upscaling first changes it by well
    under 0.2 levels.
    """
    h, w = up.shape[:2]
    grid = _tone_grid_size(w, h)
    low_src = np.stack([np.asarray(c.resize(grid, Image.LANCZOS)) for c in src_channels], axis=-1)
    correction = strength * (low_src - _downscale(up, grid))
    return _float_channels(correction)


def _downscale(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos-downscale a large uint8 image to a small float32 one, a band of rows at a time."""
    h, w = img.shape[:2]
    out_w, out_h = size
    step = h / out_h  # input rows per output row
    reach = _LANCZOS_REACH * max(step, 1.0)
    out = np.empty((out_h, out_w, 3), np.float32)
    rows_per_band = max(1, int(_STRIP_PIXELS / (w * step)))
    for j0 in range(0, out_h, rows_per_band):
        j1 = min(j0 + rows_per_band, out_h)
        top, bottom = j0 * step, min(j1 * step, h)
        # Include every input row the filter can reach, so each band matches a full-image resize.
        r0 = max(0, math.floor(top - reach) - 1)
        r1 = min(h, math.ceil(bottom + reach) + 1)
        box = (0, top - r0, w, bottom - r0)
        for c in range(3):
            band = Image.fromarray(img[r0:r1, :, c].astype(np.float32))
            out[j0:j1, :, c] = np.asarray(band.resize((out_w, j1 - j0), Image.LANCZOS, box=box))
    return out


class _GrainRows:
    """Blurred noise produced row by row from one random stream, so strips join seamlessly."""

    def __init__(self, seed: int, width: int):
        self._rng = np.random.default_rng(seed)
        self._width = width
        self._carry = self._rng.standard_normal((2, width), dtype=np.float32)  # rows i-1 and i

    def next_rows(self, n: int) -> np.ndarray:
        raw = np.concatenate([self._carry, self._rng.standard_normal((n, self._width), dtype=np.float32)])
        self._carry = raw[-2:].copy()
        k0, k1, k2 = _GRAIN_KERNEL
        vertical = k0 * raw[:-2] + k1 * raw[1:-1] + k2 * raw[2:]
        padded = np.pad(vertical, ((0, 0), (1, 1)), mode="edge")
        return (k0 * padded[:, :-2] + k1 * padded[:, 1:-1] + k2 * padded[:, 2:]) * _GRAIN_GAIN


def _add_grain(strip: np.ndarray, noise: np.ndarray, amount: float) -> None:
    luminance = np.clip(strip.mean(axis=-1) / 255, 0, 1)
    midtones = 4 * luminance * (1 - luminance)
    strip += (amount * noise * midtones)[..., None]


def _to_uint8(strip: np.ndarray) -> np.ndarray:
    np.clip(strip, 0, 255, out=strip)
    strip += 0.5
    return strip.astype(np.uint8)
