"""The local upscaling models: where to get them, how to download them safely, how to load them."""

from __future__ import annotations

import contextlib
import errno
import gc
import hashlib
import logging
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import spandrel
import torch
from torch import nn

from upscaler.config import Cancelled, ProgressFn, models_dir
from upscaler.hardware import detect_hardware

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    file: str
    url: str
    sha256: str
    size_bytes: int
    scale: int
    tile: int
    license: str
    license_url: str
    author: str
    source_url: str


_CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
_APACHE_2 = "https://www.apache.org/licenses/LICENSE-2.0"
_REAL_ESRGAN_LICENSE = "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE"
_OPENCV_ZOO = "https://github.com/opencv/opencv_zoo"
_YUNET_COMMIT = "f12e12798e8314f7c074a6656816c048dcc95b7a"
# The Hugging Face commit that added 4x-ClearRealityV1.safetensors, so a later re-upload to the
# repository's main branch can't change the file every new install downloads (and fail its SHA-256).
_CLEAR_REALITY_COMMIT = "56f9841c86d054ea19f820341c8f66f0fcaa16af"
_CLEAR_REALITY = f"https://huggingface.co/Kim2091/ClearRealityV1/resolve/{_CLEAR_REALITY_COMMIT}"

MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="faithful",
            label="Faithful",
            file="4xBHI_dat2_real.safetensors",
            url="https://github.com/Phhofm/models/releases/download/4xBHI_dat2_real/4xBHI_dat2_real.safetensors",
            sha256="1f651fe8e2b7ad978eee0db9c8de89b7bbfd9d2d671ffababb9baf6e2d5b38f6",
            size_bytes=139792588,
            scale=4,
            tile=256,
            license="CC-BY-4.0",
            license_url=_CC_BY_4,
            author="Philip Hofmann (Phhofm)",
            source_url="https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real",
        ),
        ModelSpec(
            id="faithful_sharp",
            label="Faithful for sharp photos",
            file="4xNomos2_hq_dat2.pth",
            url="https://github.com/Phhofm/models/releases/download/4xNomos2_hq_dat2/4xNomos2_hq_dat2.pth",
            sha256="1f2be2b4786b5031776aed90e1818564f5576ab1e44da9355c7f3788ea25bdea",
            size_bytes=140196334,
            scale=4,
            tile=256,
            license="CC-BY-4.0",
            license_url=_CC_BY_4,
            author="Philip Hofmann (Phhofm)",
            source_url="https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_dat2",
        ),
        ModelSpec(
            id="fast",
            label="Fast",
            file="4x-ClearRealityV1.safetensors",
            url=f"{_CLEAR_REALITY}/4x-ClearRealityV1.safetensors",
            sha256="83a9c8e279e3d07548e2b0d736dc2c73c1736a07072a748ed4fafbba99b44693",
            size_bytes=4492232,
            scale=4,
            tile=512,
            license="Apache-2.0",
            license_url=_APACHE_2,
            author="Kim2091",
            source_url="https://huggingface.co/Kim2091/ClearRealityV1",
        ),
        ModelSpec(
            id="illustration",
            label="Illustration",
            file="RealESRGAN_x4plus_anime_6B.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
            size_bytes=17938799,
            scale=4,
            tile=512,
            license="BSD-3-Clause",
            license_url=_REAL_ESRGAN_LICENSE,
            author="Xintao Wang",
            source_url="https://github.com/xinntao/Real-ESRGAN",
        ),
        ModelSpec(
            id="video_general",
            label="Balanced",
            file="realesr-general-x4v3.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            size_bytes=4885111,
            scale=4,
            tile=512,
            license="BSD-3-Clause",
            license_url=_REAL_ESRGAN_LICENSE,
            author="Xintao Wang",
            source_url="https://github.com/xinntao/Real-ESRGAN",
        ),
        # Not an upscaler: OpenCV's YuNet face detector, which auto-suggest uses (upscaler/suggest.py).
        # Loaded with cv2.FaceDetectorYN, never with load_model. Pinned to the opencv_zoo commit
        # that added this file (the 2023mar model, fixed input size, for OpenCV 4.x).
        ModelSpec(
            id="face_yunet",
            label="face finder",
            file="face_detection_yunet_2023mar.onnx",
            url=f"{_OPENCV_ZOO}/raw/{_YUNET_COMMIT}/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            size_bytes=232589,
            scale=1,
            tile=0,
            license="MIT",
            license_url=f"{_OPENCV_ZOO}/blob/{_YUNET_COMMIT}/models/face_detection_yunet/LICENSE",
            author="Shiqi Yu (YuNet), OpenCV Model Zoo",
            source_url=f"{_OPENCV_ZOO}/tree/main/models/face_detection_yunet",
        ),
    ]
}


# --- Download ---------------------------------------------------------------

_CHUNK_BYTES = 1 << 20
_DOWNLOAD_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2.0
_SPACE_MARGIN_BYTES = 50 * 2**20
_TIMEOUT = httpx.Timeout(30.0)
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_CONTENT_RANGE_START = re.compile(r"bytes (\d+)-")
_TURN_POLL_SECONDS = 0.25  # how often a download waiting its turn checks for Cancel

_download_lock = threading.Lock()


class ModelDownloadError(Exception):
    """A model couldn't be downloaded. The message is written for the person using the app."""


class _TryAgain(Exception):
    """A download attempt failed in a way that a fresh attempt may fix."""

    def __init__(self, reason: str, status: int | None = None):
        super().__init__(reason)
        self.status = status  # the HTTP status, when the server itself reported the problem


def _mb(num_bytes: int) -> int:
    return max(1, math.ceil(num_bytes / 2**20))


def ensure_model(
    spec: ModelSpec, progress: ProgressFn | None = None, cancel: threading.Event | None = None
) -> Path:
    """Return the path of the model file, downloading it first if needed.

    The file only appears under its real name after its SHA-256 matches, so the model
    cache never holds a damaged file. An interrupted download stays as `<file>.part`
    and is resumed next time. Downloads take turns; a model that's already here is
    returned at once, even while another model is downloading. Setting `cancel` stops a
    download within a chunk (raising Cancelled), keeping what arrived for next time.
    """
    target = models_dir() / spec.file
    if _is_full_size(target, spec):
        return target
    with _download_turn(cancel):
        if _is_full_size(target, spec):  # it may have arrived while this waited its turn
            return target
        target.unlink(missing_ok=True)  # an old or damaged copy
        part = target.with_name(target.name + ".part")

        if _is_full_size(part, spec) and not _hash_matches(part, spec):
            part.unlink()  # a damaged leftover from an earlier run: start over

        _download_with_retries(spec, part, progress, cancel)
        if not _hash_matches(part, spec):
            part.unlink()
            raise ModelDownloadError(_damaged_message(spec))
        os.replace(part, target)
        return target


@contextlib.contextmanager
def _download_turn(cancel: threading.Event | None) -> Iterator[None]:
    """Wait for any other download to finish first, giving up if `cancel` is set meanwhile."""
    while not _download_lock.acquire(timeout=_TURN_POLL_SECONDS):
        _raise_if_cancelled(cancel)
    try:
        yield
    finally:
        _download_lock.release()


def _raise_if_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _is_full_size(path: Path, spec: ModelSpec) -> bool:
    return path.is_file() and path.stat().st_size == spec.size_bytes


def _hash_matches(path: Path, spec: ModelSpec) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest() == spec.sha256


def _download_with_retries(
    spec: ModelSpec, part: Path, progress: ProgressFn | None, cancel: threading.Event | None
) -> None:
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        _raise_if_cancelled(cancel)
        try:
            _download(spec, part, progress, cancel)
            return
        except (httpx.TransportError, _TryAgain) as e:
            log.warning("Download of %s failed (attempt %d): %s", spec.file, attempt, e)
            if attempt == _DOWNLOAD_ATTEMPTS:
                if isinstance(e, _TryAgain) and e.status:
                    raise ModelDownloadError(_server_error_message(spec, e.status)) from e
                raise ModelDownloadError(
                    f"Couldn't download the {spec.label} model. Check that this computer is connected "
                    "to the internet, then try again."
                ) from e
            delay = _RETRY_DELAY_SECONDS * attempt
            if cancel is None:
                time.sleep(delay)
            elif cancel.wait(delay):
                raise Cancelled()


def _damaged_message(spec: ModelSpec) -> str:
    return f"The {spec.label} model didn't download correctly, so it was deleted. Please try again."


def _server_error_message(spec: ModelSpec, status: int) -> str:
    return (
        f"The download site for the {spec.label} model isn't working right now (error {status}). "
        "Please try again later."
    )


def _download(spec: ModelSpec, part: Path, progress: ProgressFn | None, cancel: threading.Event | None) -> None:
    have = part.stat().st_size if part.is_file() else 0
    if have > spec.size_bytes:
        part.unlink()
        have = 0
    if have == spec.size_bytes:
        return
    _check_free_space(part.parent, spec.size_bytes - have, spec)

    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
        with client.stream("GET", spec.url, headers=headers) as response:
            have = _resume_offset(response, have, part, spec)
            _write_body(response, part, have, spec, progress, cancel)

    if part.stat().st_size != spec.size_bytes:
        raise _TryAgain("the server stopped sending before the end of the file")


def _resume_offset(response: httpx.Response, have: int, part: Path, spec: ModelSpec) -> int:
    """Where in the file the response body starts: `have` when resuming, 0 when starting over."""
    status = response.status_code
    if status == 206 and have:
        match = _CONTENT_RANGE_START.match(response.headers.get("Content-Range", ""))
        if match and int(match.group(1)) == have:
            return have
        part.unlink()
        raise _TryAgain("the server resumed from the wrong place")
    if status == 200:
        return 0  # the server ignored the Range header; the body is the whole file
    if status == 416:
        part.unlink()
        raise _TryAgain("the server couldn't resume the download")
    if status in _RETRYABLE_STATUS:
        raise _TryAgain(f"server error {status}", status)
    raise ModelDownloadError(_server_error_message(spec, status))


def _write_body(
    response: httpx.Response,
    part: Path,
    offset: int,
    spec: ModelSpec,
    progress: ProgressFn | None,
    cancel: threading.Event | None,
) -> None:
    done = offset
    try:
        with part.open("ab" if offset else "wb") as f:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                done += len(chunk)
                if done > spec.size_bytes:
                    break
                f.write(chunk)
                if progress:
                    progress(
                        done / spec.size_bytes,
                        f"Downloading the {spec.label} model ({_mb(done)} of {_mb(spec.size_bytes)} MB)",
                    )
                _raise_if_cancelled(cancel)  # what arrived stays in the .part file, to resume from
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise ModelDownloadError(
                f"This computer's disk is full, so the {spec.label} model couldn't be saved. "
                "Free up some space and try again."
            ) from e
        raise ModelDownloadError(f"Couldn't save the {spec.label} model on this computer ({e.strerror}).") from e

    if done > spec.size_bytes:
        part.unlink()
        raise ModelDownloadError(_damaged_message(spec))


def _check_free_space(directory: Path, needed: int, spec: ModelSpec) -> None:
    free = shutil.disk_usage(directory).free
    if free < needed + _SPACE_MARGIN_BYTES:
        raise ModelDownloadError(
            f"There isn't enough free space on this computer for the {spec.label} model. "
            f"Free up about {_mb(needed + _SPACE_MARGIN_BYTES - free)} MB and try again."
        )


# --- Loading ----------------------------------------------------------------


@dataclass
class LoadedModel:
    net: torch.nn.Module  # takes (1, 3, H, W) in 0..1 at `dtype` on `device`; returns (1, 3, H*scale, W*scale)
    scale: int
    dtype: torch.dtype  # may switch to float32 if half precision gives broken output
    tile: int
    device: torch.device


class _SpandrelNet(nn.Module):
    """Runs a spandrel model through its descriptor, which pads inputs to the sizes the
    architecture needs (e.g. DAT2 windows) and clamps the output to 0..1."""

    def __init__(self, descriptor: spandrel.ImageModelDescriptor):
        super().__init__()
        self.model = descriptor.model  # registered, so .to()/.float() reach the weights
        self._descriptor = descriptor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._descriptor(x)


def choose_dtype(supports_half: bool, supports_bfloat16: bool, device: str) -> torch.dtype:
    if supports_half and device != "cpu":
        return torch.float16
    if supports_bfloat16 and device in ("mps", "cuda"):
        return torch.bfloat16
    return torch.float32


_cache_lock = threading.Lock()
_cached: tuple[tuple[str, str], LoadedModel] | None = None


def load_model(spec: ModelSpec, device: str | None = None) -> LoadedModel:
    """Load a model onto the GPU (or `device`). Only the most recent model stays loaded."""
    global _cached
    device = device or detect_hardware().device
    key = (spec.id, device)
    with _cache_lock:
        if _cached is not None and _cached[0] == key:
            return _cached[1]
        if _cached is not None:
            (_, old_device), _old = _cached
            _cached = _old = None  # free the previous model before loading the next
            _release_memory(old_device)

        descriptor = spandrel.ModelLoader().load_from_file(ensure_model(spec))
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise ValueError(f"{spec.file} is not an image upscaling model")
        dtype = choose_dtype(descriptor.supports_half, descriptor.supports_bfloat16, device)
        descriptor.to(torch.device(device)).to(dtype).eval()

        loaded = LoadedModel(
            net=_SpandrelNet(descriptor),
            scale=descriptor.scale,
            dtype=dtype,
            tile=spec.tile,
            device=torch.device(device),
        )
        _cached = (key, loaded)
        return loaded


def unload_model() -> None:
    """Let go of the loaded model and give its memory back to the GPU (or the system), for work that
    needs it more: Photo-real on an NVIDIA card runs as a program of its own. The next load_model
    loads it again."""
    global _cached
    with _cache_lock:
        if _cached is None:
            return
        (_, device), _old = _cached
        _cached = _old = None
        _release_memory(device)


def _release_memory(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
