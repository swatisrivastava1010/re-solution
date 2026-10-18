"""What every SeedVR2 engine shares: the interface, the weights and their download."""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from upscaler.config import ProgressFn

SEEDVR2_REPO = "numz/SeedVR2_comfyUI"
# The commit mflux 0.23 pins for this repo. Using the same one means the copy mflux already
# downloaded is found, and both engines always run the same weights.
SEEDVR2_REVISION = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"
SEED = 42


@dataclass(frozen=True)
class WeightFile:
    name: str
    size_bytes: int
    sha256: str


SEEDVR2_FILES = (
    WeightFile(
        "seedvr2_ema_3b_fp16.safetensors",
        6_783_018_808,
        "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304",
    ),
    WeightFile(
        "ema_vae_fp16.safetensors",
        501_324_814,
        "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
    ),
)
DOWNLOAD_BYTES = sum(f.size_bytes for f in SEEDVR2_FILES)
# Room kept free on top of what a download needs, so the disk is never filled to the last byte.
DISK_MARGIN_BYTES = 1_000_000_000
# A download in progress writes to its partial file every few seconds (every 10 MB chunk), so a
# partial file untouched for this long was left behind by an app that was closed mid-download.
STALE_PARTIAL_SECONDS = 10 * 60


class SeedVR2Unavailable(Exception):
    """Photo-real can't run right now; ``reason`` is a short plain-English explanation."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class SeedVR2Engine(Protocol):
    name: str  # "mlx" | "torch"

    def weights_present(self) -> bool: ...

    def download_weights(self, progress: ProgressFn, cancel: threading.Event) -> None: ...

    def upscale(
        self, rgb: np.ndarray, scale: int, *, progress: ProgressFn, cancel: threading.Event, softness: float = 0.0
    ) -> np.ndarray:
        """`rgb` enlarged exactly `scale` times. `softness` (0..1) is SeedVR2's own setting: higher
        values give the model a softer picture, so it invents more detail of its own."""
        ...

    def release(self) -> None:
        """Free memory kept between pictures now; the engine loads again when Photo-real next runs."""
        ...


def _cached_path(file: WeightFile) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(SEEDVR2_REPO, file.name, revision=SEEDVR2_REVISION)
    if not isinstance(found, str):
        return None
    path = Path(found)
    return path if path.is_file() and path.stat().st_size == file.size_bytes else None


def cached_weights_dir() -> Path | None:
    """The Hugging Face snapshot folder holding both weight files, if they're fully downloaded."""
    paths = [_cached_path(f) for f in SEEDVR2_FILES]
    if any(p is None for p in paths):
        return None
    return paths[0].parent


def weights_cached() -> bool:
    return cached_weights_dir() is not None


def missing_download_bytes() -> int:
    """How much of the weights still has to be downloaded."""
    return sum(f.size_bytes for f in SEEDVR2_FILES if _cached_path(f) is None)


def hub_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def check_disk_space(needs: Iterable[tuple[Path, int]]) -> None:
    """Raise SeedVR2Unavailable unless every disk has room for what is about to go onto it.

    ``needs`` pairs a folder (which may not exist yet) with the bytes it will take. Folders on the
    same disk share its free space, and each disk keeps DISK_MARGIN_BYTES spare.
    """
    per_disk: dict[int, list] = {}  # disk -> [a folder on it, bytes going onto it]
    try:
        for folder, size in needs:
            if size > 0:
                place = _existing_parent(folder)
                per_disk.setdefault(place.stat().st_dev, [place, 0])[1] += size
        short = [place for place, size in per_disk.values() if shutil.disk_usage(place).free < size + DISK_MARGIN_BYTES]
    except OSError:
        return  # can't tell; let the download try, and report a full disk if it happens
    if short:
        total = sum(size for _, size in per_disk.values()) + DISK_MARGIN_BYTES
        raise SeedVR2Unavailable(
            f"There isn't enough free disk space for Photo-real (it needs about {max(1, round(total / 1e9))} GB)."
        )


def _existing_parent(folder: Path) -> Path:
    folder = folder.absolute()
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return folder


def download_seedvr2_weights(progress: ProgressFn, cancel: threading.Event) -> Path:
    """Fetch both weight files into the Hugging Face cache, verify them, and return their folder.

    A file that finished earlier is kept, so an interrupted setup only repeats the unfinished file.
    """
    from huggingface_hub import hf_hub_download

    from upscaler.config import Cancelled

    remove_stale_partial_downloads()
    check_disk_space([(hub_cache_dir(), missing_download_bytes())])
    done = 0
    for file in SEEDVR2_FILES:
        if cancel.is_set():
            raise Cancelled()
        if _cached_path(file) is None:
            report = _byte_reporter(progress, done, file.name)
            try:
                with _plain_http_downloads():
                    path = hf_hub_download(
                        SEEDVR2_REPO,
                        file.name,
                        revision=SEEDVR2_REVISION,
                        tqdm_class=_progress_bar_class(report, cancel),
                    )
            except Cancelled:
                raise
            except OSError as exc:
                raise SeedVR2Unavailable(_download_failure_reason(exc)) from exc
            except Exception as exc:  # huggingface_hub raises its own HTTP/network error types
                raise SeedVR2Unavailable("The Photo-real download didn't finish. Check the internet connection and try again.") from exc
            _verify(Path(path), file, progress, done + file.size_bytes, cancel)
        done += file.size_bytes
    progress(1.0, "Photo-real is ready.")
    folder = cached_weights_dir()
    if folder is None:
        raise SeedVR2Unavailable("The Photo-real download didn't finish. Please try again.")
    return folder


def remove_stale_partial_downloads() -> None:
    """Delete half-downloaded weight files left behind when the app was closed mid-download.

    huggingface_hub gives every attempt its own partial file and deletes it only when Python gets
    to clean up. Closing the app's window (or a crash, or a power cut) doesn't allow that, so each
    cut-short attempt would leave up to 6.8 GB in a hidden folder that nobody would ever find.
    """
    from huggingface_hub.file_download import repo_folder_name

    blobs = hub_cache_dir() / repo_folder_name(repo_id=SEEDVR2_REPO, repo_type="model") / "blobs"
    cutoff = time.time() - STALE_PARTIAL_SECONDS
    try:
        partials = list(blobs.glob("*.incomplete"))
    except OSError:
        return
    for partial in partials:
        try:
            if partial.stat().st_mtime < cutoff:
                partial.unlink()
        except OSError:
            pass  # in use or already gone; the next setup tries again


def _download_failure_reason(exc: OSError) -> str:
    if getattr(exc, "errno", None) == 28:  # ENOSPC
        return "There isn't enough free disk space for Photo-real (it needs about 8 GB)."
    return "The Photo-real download didn't finish. Check the internet connection and try again."


def _byte_reporter(progress: ProgressFn, done_before: int, name: str) -> Callable[[int], None]:
    def report(file_bytes: int) -> None:
        got = done_before + file_bytes
        progress(min(got / DOWNLOAD_BYTES, 0.999), f"Downloading Photo-real: {got / 1e9:.1f} of {DOWNLOAD_BYTES / 1e9:.1f} GB")

    return report


def _progress_bar_class(report: Callable[[int], None], cancel: threading.Event):
    """A tqdm class that forwards byte counts to ``report`` and stops the download on cancel."""
    from tqdm.auto import tqdm

    from upscaler.config import Cancelled

    class _ReportingBar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # no terminal output; the app shows its own progress
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            if cancel.is_set():
                raise Cancelled()
            self.n += n or 0  # disabled bars don't count, so keep the total ourselves
            report(int(self.n))
            return None

    return _ReportingBar


@contextlib.contextmanager
def _plain_http_downloads() -> Iterator[None]:
    """Download over plain HTTP instead of hf_xet for the duration of the block.

    hf_xet swallows exceptions raised from progress callbacks, so Cancel would be ignored until the
    7 GB finished. The plain HTTP path stops on Cancel (checked every 10 MB chunk) and resumes by
    itself after brief network drops. A cancelled or crashed download restarts that file from zero:
    huggingface_hub 1.x deletes partial files instead of resuming them.
    """
    from huggingface_hub import constants

    before = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = before


def _verify(path: Path, file: WeightFile, progress: ProgressFn, done_after: int, cancel: threading.Event) -> None:
    """Check a downloaded file's SHA-256, removing it if it's damaged.

    The bar stays where the file's download ended. Cancel stops the check by removing only the
    snapshot's link to the file: the downloaded copy stays, and the next setup links it again in a
    moment and checks it from the start, so a file is never used unchecked. Where the snapshot holds
    the file itself (Windows without symlinks), the check finishes first instead.
    """
    from upscaler.config import Cancelled

    progress(min(done_after / DOWNLOAD_BYTES, 0.999), "Checking the Photo-real download…")
    can_stop_early = path.is_symlink()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            if can_stop_early and cancel.is_set():
                path.unlink(missing_ok=True)
                raise Cancelled()
            digest.update(chunk)
    if digest.hexdigest() != file.sha256:
        blob = path.resolve()
        path.unlink(missing_ok=True)
        blob.unlink(missing_ok=True)
        raise SeedVR2Unavailable("The Photo-real download was damaged, so it was removed. Please try again.")


def soften(pixels: np.ndarray, scale: int, softness: float) -> np.ndarray:
    """SeedVR2's `softness`, for a runner that doesn't have it (mflux's SeedVR2Util.preprocess_image).

    mflux shrinks the scale×-enlarged picture by 1 + 7 × softness and enlarges it back before the
    model sees it. Only shrinking below the original size takes detail away, so that part is done
    here, at the original size; otherwise the picture is returned as it is.
    """
    factor = 1.0 + 7.0 * min(max(softness, 0.0), 1.0)
    if scale / factor >= 1:
        return pixels
    h, w = pixels.shape[:2]
    small = (max(2, int(w * scale / factor)), max(2, int(h * scale / factor)))
    image = Image.fromarray(pixels).resize(small, Image.BICUBIC).resize((w, h), Image.BICUBIC)
    return np.asarray(image)


def fit_tile(pixels: np.ndarray, w: int, h: int) -> np.ndarray:
    """Make an upscaled tile exactly w×h: crop any padding, or resize the odd missing pixel row."""
    pixels = pixels[:h, :w]
    if pixels.shape[:2] != (h, w):
        pixels = np.asarray(Image.fromarray(pixels).resize((w, h), Image.LANCZOS))
    return np.ascontiguousarray(pixels)
