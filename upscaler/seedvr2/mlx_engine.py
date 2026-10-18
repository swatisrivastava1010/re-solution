"""Photo-real on Apple Silicon: SeedVR2-3B through mflux (MLX), one overlapping tile at a time."""

from __future__ import annotations

import contextlib
import functools
import gc
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from upscaler.hardware import detect_hardware
from upscaler.seedvr2.base import (
    SEED,
    SeedVR2Unavailable,
    download_seedvr2_weights,
    fit_tile,
    weights_cached,
)
from upscaler.seedvr2.blend import blend_tiles, tile_grid

if TYPE_CHECKING:
    from upscaler.config import ProgressFn

TILE = 512  # input pixels; measured against 384 on the M4, see docs/RESEARCH.md §4a
OVERLAP = 64
CACHE_LIMIT_BYTES = 1_000_000_000
# The loaded model holds about 4.8 GB, and loading it again takes about 2.4 s (docs/RESEARCH.md
# §4a), so it's freed once Photo-real has been idle this long. Macs with less memory free it sooner,
# because Photoshop or Figma running alongside need that memory more.
IDLE_SECONDS = 300
IDLE_SECONDS_SMALL_MAC = 60
SMALL_MAC_GB = 24


class CancelCheck:
    """An mflux callback that stops a tile between the model's phases when Cancel is pressed.

    mflux calls it after the picture is encoded, at the model's step and before decoding, so Cancel
    waits for one phase of a tile rather than the whole tile (up to about 85 s at 4x).
    """

    def __init__(self) -> None:
        self.event: threading.Event | None = None  # the running job's cancel event

    def _check(self) -> None:
        from upscaler.config import Cancelled

        if self.event is not None and self.event.is_set():
            raise Cancelled()

    def call_before_loop(self, **_: Any) -> None:
        self._check()

    def call_in_loop(self, **_: Any) -> None:
        self._check()

    def call_after_loop(self, **_: Any) -> None:
        self._check()


class MLXSeedVR2Engine:
    name = "mlx"

    def __init__(self, *, tile: int = TILE, overlap: int = OVERLAP, idle_seconds: float | None = None):
        self.tile = tile
        self.overlap = overlap
        self._model: Any = None
        self._lock = threading.Lock()
        self._cancel_check = CancelCheck()
        self._idle_seconds = idle_seconds
        self._idle_timer: threading.Timer | None = None
        self._runs = 0  # tells an idle timer whether Photo-real has run since it was started

    def weights_present(self) -> bool:
        return weights_cached()

    def download_weights(self, progress: ProgressFn, cancel: threading.Event) -> None:
        download_seedvr2_weights(progress, cancel)

    def upscale(
        self, rgb: np.ndarray, scale: int, *, progress: ProgressFn, cancel: threading.Event, softness: float = 0.0
    ) -> np.ndarray:
        from upscaler.config import Cancelled

        if not self.weights_present():
            raise SeedVR2Unavailable("Photo-real hasn't been downloaded yet.")
        h, w = rgb.shape[:2]
        boxes = tile_grid(w, h, self.tile, self.overlap)
        tiles = []
        with self._lock:
            self._runs += 1
            self._cancel_check.event = cancel
            try:
                with tempfile.TemporaryDirectory(prefix="upscaler-seedvr2-") as workdir:
                    model = self._loaded_model(progress)
                    for i, (x0, y0, x1, y1) in enumerate(boxes):
                        if cancel.is_set():
                            raise Cancelled()
                        progress(i / len(boxes), f"Photo-real: part {i + 1} of {len(boxes)}")
                        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                        tiles.append(((x0, y0, x1, y1), self._run_tile(model, crop, scale, Path(workdir), softness)))
            finally:
                self._cancel_check.event = None
                self._free_when_idle()
        progress(1.0, "Photo-real: putting the parts together")
        return blend_tiles(tiles, w * scale, h * scale, scale, ramp=self.overlap * scale)

    def release(self) -> None:
        """Free the model's memory now. It loads again, in a few seconds, when Photo-real next runs."""
        with self._lock:
            self._drop_model()

    def _free_when_idle(self) -> None:
        """Start the countdown to freeing the model (called with ``_lock`` held, as a run ends)."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._model is None:
            return
        timer = threading.Timer(self._idle_delay(), self._free_if_still_idle, args=(self._runs,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_delay(self) -> float:
        if self._idle_seconds is not None:
            return self._idle_seconds
        return IDLE_SECONDS_SMALL_MAC if detect_hardware().memory_gb < SMALL_MAC_GB else IDLE_SECONDS

    def _free_if_still_idle(self, runs: int) -> None:
        if not self._lock.acquire(blocking=False):
            return  # Photo-real is running; it starts a new countdown when it ends
        try:
            if self._runs == runs:
                self._drop_model()
        finally:
            self._lock.release()

    def _drop_model(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._model is None:
            return
        self._model = None
        gc.collect()  # the model and its MemorySaver callback refer to each other
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.clear_cache()

    def _loaded_model(self, progress: ProgressFn) -> Any:
        if self._model is None:
            progress(0.0, "Starting Photo-real…")
            try:
                self._model = self._load_model()
            except ImportError as exc:
                raise SeedVR2Unavailable("Photo-real isn't installed in this copy of the app.") from exc
            except Exception as exc:
                raise SeedVR2Unavailable("Photo-real couldn't start on this Mac.") from exc
        return self._model

    def _run_tile(self, model: Any, crop: np.ndarray, scale: int, workdir: Path, softness: float) -> np.ndarray:
        from upscaler.config import Cancelled

        h, w = crop.shape[:2]
        try:
            pixels = self._upscale_tile(model, crop, scale, workdir, softness)
        except Cancelled:
            raise  # Cancel pressed inside the tile: stop, rather than fall back to Faithful
        except Exception as exc:
            raise SeedVR2Unavailable("Photo-real stopped partway through.") from exc
        return fit_tile(pixels, w * scale, h * scale)

    def _load_model(self) -> Any:
        from mflux.callbacks.instances.memory_saver import MemorySaver
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

        model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
        use_single_frame_convs(model.vae)
        model.tiling_config = TilingConfig()
        # keep_transformer=True: mflux's single-image low-RAM setting deletes the transformer after
        # one generate_image call, so the second tile would crash. Cap MLX's buffer cache instead.
        model.callbacks.register(
            MemorySaver(model=model, keep_transformer=True, cache_limit_bytes=CACHE_LIMIT_BYTES)
        )
        model.callbacks.register(self._cancel_check)
        return model

    def _upscale_tile(self, model: Any, crop: np.ndarray, scale: int, workdir: Path, softness: float) -> np.ndarray:
        import mlx.core as mx

        path = workdir / "tile.png"
        Image.fromarray(crop).save(path)
        h, w = crop.shape[:2]
        # An exact pixel size rather than ScaleFactor(scale): ScaleFactor rounds the short side
        # down to a multiple of 16, which would make most tiles slightly less than scale× bigger.
        with _quiet_mflux():
            result = model.generate_image(seed=SEED, image_path=path, resolution=min(w, h) * scale, softness=softness)
        pixels = np.asarray(result.image.convert("RGB"))
        mx.clear_cache()
        return pixels


@contextlib.contextmanager
def _quiet_mflux() -> Iterator[None]:
    """Keep the Terminal window calm while mflux works on a tile.

    For every tile mflux warns "Width and height should be multiples of 16. Rounding down." (fit_tile
    puts the exact size back) and draws a progress bar for its single step. The app shows its own
    progress, so for this call only, mflux's warnings are held back (its errors still show) and its
    bar is drawn into nothing. Only one tile runs at a time (the engine's lock), so changing mflux's
    settings for the call is safe.
    """
    try:
        from mflux.models.common.config import config as mflux_config
    except ImportError:  # a different mflux layout: let it talk rather than fail
        yield
        return
    logger = logging.getLogger("mflux")
    level = logger.level
    bar = getattr(mflux_config, "tqdm", None)
    with open(os.devnull, "w", encoding="utf-8") as nowhere:
        logger.setLevel(logging.ERROR)
        if bar is not None:
            mflux_config.tqdm = functools.partial(bar, file=nowhere)
        try:
            yield
        finally:
            logger.setLevel(level)
            if bar is not None:
                mflux_config.tqdm = bar


def use_single_frame_convs(vae: Any) -> None:
    """Run SeedVR2's video VAE as an image VAE: same pixels, about 3x faster, far less memory.

    For a single frame, every causal conv3d pads the frame with copies of itself, so its time
    kernel always sees identical frames. Summing the kernel's time taps gives one conv2d with the
    same result. On the M4 this took a 384 px tile's VAE work from 60 s to 21 s and its peak
    memory from 15.6 GB to 9.1 GB (docs/RESEARCH.md section 4a).
    """
    original, single_frame = _conv_classes()
    for module in vae.modules():
        if type(module) is original:
            module.__class__ = single_frame


@functools.cache
def _conv_classes() -> tuple[type, type]:
    import mlx.core as mx
    from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d

    class SingleFrameCausalConv3d(CausalConv3d):
        def __call__(self, x: mx.array) -> mx.array:
            if not self._folds(x):
                return super().__call__(x)
            weight = getattr(self, "_weight_2d", None)
            if weight is None:
                weight = self.weight.astype(mx.float32).sum(axis=1).astype(self.weight.dtype)
                self._weight_2d = weight  # underscore: MLX leaves it out of parameters()
            frame = x[:, :, 0].transpose(0, 2, 3, 1)
            out = mx.conv2d(frame, weight, stride=self.stride[1:], padding=self.padding[1:]) + self.bias
            return out.transpose(0, 3, 1, 2)[:, :, None]

        def _folds(self, x: mx.array) -> bool:
            """True when the output is a single frame built only from copies of the input frame."""
            kt, pt = self.kernel_size[0], self.padding[0]
            if x.shape[2] != 1:
                return False
            if kt == 1:
                return pt == 0
            causal_pad = 2 * pt if self.use_padding_causal else kt - 1
            return self.causal_temporal and causal_pad == kt - 1

    return CausalConv3d, SingleFrameCausalConv3d
