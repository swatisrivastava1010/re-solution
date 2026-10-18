"""Photo-real on NVIDIA: numz's SeedVR2 command-line runner, in an isolated environment.

The runner (Apache-2.0, https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) needs packages the
app doesn't (diffusers, opencv, peft, ...), so it lives in its own uv venv under
``<data_dir>/seedvr2-torch/`` and never touches the app's environment. All tiles of an image go
through one CLI run on a folder, so the model loads once per image (``--cache_dit/--cache_vae``).

Not tested on real NVIDIA hardware: this path ships as "should work".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
import zlib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import numpy as np
from PIL import Image

from upscaler.hardware import detect_hardware
from upscaler.seedvr2.base import (
    SEED,
    SEEDVR2_FILES,
    SeedVR2Unavailable,
    cached_weights_dir,
    check_disk_space,
    download_seedvr2_weights,
    fit_tile,
    hub_cache_dir,
    missing_download_bytes,
    soften,
)
from upscaler.seedvr2.blend import blend_tiles, tile_grid

if TYPE_CHECKING:
    from upscaler.config import ProgressFn

RUNNER_VERSION = "2.5.24"
# v2.5.24 was never tagged (the newest tag is v2.5.23). This is the main-branch commit that set the
# version to 2.5.24: v2.5.23 plus a fix for an Apple-GPU memory leak.
RUNNER_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
RUNNER_URL = f"https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/archive/{RUNNER_COMMIT}.zip"
# SHA-256 of the files inside that archive (runner_files_digest), not of the zip itself: GitHub builds
# these zips on request and says their compression may change, which changes the zip's own bytes.
RUNNER_FILES_SHA256 = "37b3dcb24296117e02e591a7da88db974e96d06181c35660bc6f98c6e175914b"
RUNNER_PYTHON = "3.12"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# The runner's own packages (diffusers, peft, transformers, ...) are installed as they were on PyPI
# on this date, so a later release can't break new installs or reach family PCs unvetted. Checked to
# resolve for Windows and Linux with Python 3.12 and torch 2.14.1. torch and torchvision are pinned to
# the app's exact versions instead: PyTorch's own index has no upload dates, so under a cutoff uv
# would find no torch at all.
RUNNER_PACKAGES_CUTOFF = "2026-10-17T00:00:00Z"
RUNNER_DISK_BYTES = 6_000_000_000  # the add-on's packages (CUDA torch and the rest), installed
# fp16 is the copy already in the Hugging Face cache; fp8 would be another 3.4 GB download.
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
# The runner's guide runs this fp16 model without BlockSwap only on 24 GB cards (which report about
# 23.6 GB). Below that, this many of the model's 32 blocks wait in system memory and move to the
# card only while they run (upstream's suggested starting point). Not yet measured on a 12 GB card.
BLOCKSWAP_BELOW_GB = 20
BLOCKS_TO_SWAP = 16
TILE = 512
OVERLAP = 64
# Tile folders untouched this long were left behind when the app was closed during Photo-real. A
# running Photo-real adds a finished tile every few minutes at most.
STALE_WORK_SECONDS = 60 * 60
LAUNCHER_NAME = "re_solution_launcher.py"
LAUNCHER_SOURCE = '''"""Starts numz's inference_cli.py with OpenCV reading and writing pictures through Python.

Written by Re:Solution. On Windows, OpenCV's imread and imwrite can't open a path with letters
outside the system's code page (a user name in Hindi, say), so every tile would fail. Python's own
file functions handle any path, so the pictures go through them instead.
"""

import os
import runpy
import sys


def use_python_file_access(cv2, np):
    def imread(path, flags=cv2.IMREAD_COLOR):
        try:
            data = np.fromfile(os.fspath(path), dtype=np.uint8)
        except OSError:
            return None  # what OpenCV's own imread returns for a file it can't open
        return cv2.imdecode(data, flags) if data.size else None

    def imwrite(path, image, params=None):
        path = os.fspath(path)
        ok, encoded = cv2.imencode(os.path.splitext(path)[1] or ".png", image, list(params or []))
        if not ok:
            return False
        try:
            encoded.tofile(path)
        except OSError:
            return False
        return True

    cv2.imread = imread
    cv2.imwrite = imwrite


if __name__ == "__main__":
    cli = os.path.abspath(sys.argv[1])
    sys.argv = [cli, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(cli)
    try:
        import cv2
        import numpy as np
    except ImportError:
        pass  # the runner reports a missing package itself
    else:
        use_python_file_access(cv2, np)
    runpy.run_path(cli, run_name="__main__")
'''

_ENV_LEAKS = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT")
_POLL_SECONDS = 0.5


def find_uv() -> Path | None:
    found = shutil.which("uv")
    if found:
        return Path(found)
    local = Path.home() / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    return local if local.is_file() else None


def runner_command(
    python: Path,
    cli: Path,
    in_dir: Path,
    out_dir: Path,
    model_dir: Path,
    resolution: int,
    *,
    launcher: Path | None = None,
    blocks_to_swap: int = 0,
) -> list[str]:
    """The CLI call that upscales every PNG in ``in_dir`` to ``resolution`` px on its short side.

    With ``launcher`` the CLI starts through it (LAUNCHER_SOURCE); ``blocks_to_swap`` turns on BlockSwap.
    """
    start = [str(python), str(launcher), str(cli)] if launcher else [str(python), str(cli)]
    cmd = [
        *start, str(in_dir),
        "--output", str(out_dir),
        "--output_format", "png",
        "--model_dir", str(model_dir),
        "--dit_model", DIT_MODEL,
        "--resolution", str(resolution),
        "--seed", str(SEED),
        "--color_correction", "lab",
        # Keep both models between tiles, parked in system RAM while the other one runs.
        # BlockSwap needs the DiT's offload device too.
        "--dit_offload_device", "cpu",
        "--vae_offload_device", "cpu",
        "--cache_dit",
        "--cache_vae",
        "--vae_encode_tiled",
        "--vae_decode_tiled",
    ]  # fmt: skip
    if blocks_to_swap > 0:
        cmd += ["--blocks_to_swap", str(blocks_to_swap)]
    return cmd


def runner_files_digest(zf: zipfile.ZipFile) -> str:
    """SHA-256 over every file in a runner archive: its path inside the top folder and its contents.

    The zip's own bytes don't count (compression, dates, the top folder's name), only the files.
    """
    entries = sorted(
        (info.filename.partition("/")[2], hashlib.sha256(zf.read(info)).hexdigest())
        for info in zf.infolist()
        if not info.is_dir()
    )
    return hashlib.sha256("".join(f"{digest}  {name}\n" for name, digest in entries).encode()).hexdigest()


class TorchSeedVR2Engine:
    name = "torch"

    def __init__(
        self,
        root: Path | None = None,
        *,
        torch_index: str | None = CUDA_TORCH_INDEX,
        tile: int = TILE,
        overlap: int = OVERLAP,
        vram_gb: float | None = None,
    ):
        self._root = root
        self.torch_index = torch_index
        self.tile = tile
        self.overlap = overlap
        self._vram_gb = vram_gb  # None: this computer's graphics card
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        if self._root is None:
            from upscaler.config import data_dir

            self._root = data_dir() / "seedvr2-torch"
        return self._root

    @property
    def runner_dir(self) -> Path:
        return self.root / "runner"

    @property
    def venv_python(self) -> Path:
        venv = self.root / ".venv"
        return venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"

    @property
    def cli_path(self) -> Path:
        return self.runner_dir / "inference_cli.py"

    @property
    def _marker(self) -> Path:
        return self.root / "installed.json"

    def runner_installed(self) -> bool:
        try:
            info = json.loads(self._marker.read_text())
        except (OSError, ValueError):
            return False
        return info.get("commit") == RUNNER_COMMIT and self.venv_python.exists() and self.cli_path.exists()

    def weights_present(self) -> bool:
        return self.runner_installed() and cached_weights_dir() is not None

    def download_weights(self, progress: ProgressFn, cancel: threading.Event) -> None:
        """First-time setup: the add-on (runner and its packages), then the 7.3 GB of weights."""
        check_disk_space([
            (self.root, 0 if self.runner_installed() else RUNNER_DISK_BYTES),
            (hub_cache_dir(), missing_download_bytes()),
        ])  # fmt: skip
        self.install_runner(_scaled(progress, 0.0, 0.25), cancel)
        download_seedvr2_weights(_scaled(progress, 0.25, 1.0), cancel)

    def release(self) -> None:
        """Nothing to free: the runner is a separate program that ends after every picture."""

    # --- installing the runner --------------------------------------------------------------

    def install_runner(self, progress: ProgressFn, cancel: threading.Event) -> None:
        if self.runner_installed():
            return
        uv = find_uv()
        if uv is None:
            raise SeedVR2Unavailable(
                "Photo-real needs the uv setup tool, which wasn't found. "
                "Start the app with its Start Re-Solution launcher."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._marker.unlink(missing_ok=True)

        progress(0.0, "Getting the Photo-real add-on…")
        self._fetch_runner_source(cancel)

        progress(0.1, "Setting up the Photo-real add-on…")
        venv = self.venv_python.parents[1]
        shutil.rmtree(venv, ignore_errors=True)  # a half-finished earlier attempt
        self._uv(uv, ["venv", "--python", RUNNER_PYTHON, str(venv)], cancel)

        progress(0.2, "Installing the Photo-real add-on (a few GB, one time)…")
        self._uv(uv, ["pip", "install", "--python", str(self.venv_python), *self._torch_requirements()], cancel)
        progress(0.7, "Installing the Photo-real add-on (almost done)…")
        requirements = self.runner_dir / "requirements.txt"
        packages = ["--exclude-newer", RUNNER_PACKAGES_CUTOFF, "-r", str(requirements)]
        self._uv(uv, ["pip", "install", "--python", str(self.venv_python), *packages], cancel)

        self._marker.write_text(json.dumps({"commit": RUNNER_COMMIT, "version": RUNNER_VERSION}))
        progress(1.0, "Photo-real add-on installed.")

    def _torch_requirements(self) -> list[str]:
        # Same versions as the app's own torch, so uv can reuse the wheels it already downloaded.
        packages = [_pinned("torch"), _pinned("torchvision")]
        return [*packages, "--index-url", self.torch_index] if self.torch_index else packages

    def _fetch_runner_source(self, cancel: threading.Event) -> None:
        from upscaler.config import Cancelled

        part = self.root / "runner.zip.part"
        try:
            with httpx.stream("GET", RUNNER_URL, follow_redirects=True, timeout=60) as response:
                response.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in response.iter_bytes():
                        if cancel.is_set():
                            raise Cancelled()
                        fh.write(chunk)
            self._unpack(part)
        except httpx.HTTPError as exc:
            raise SeedVR2Unavailable(
                "Couldn't download the Photo-real add-on. Check the internet connection and try again."
            ) from exc
        except OSError as exc:
            raise SeedVR2Unavailable("Couldn't save the Photo-real add-on. Is the disk full?") from exc
        finally:
            part.unlink(missing_ok=True)

    def _unpack(self, archive: Path) -> None:
        """Check the archive holds exactly the pinned runner's files, then put them in place."""
        damaged = "The Photo-real add-on download was damaged. Please try again."
        staging = self.root / "runner.unpacking"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                if runner_files_digest(zf) != RUNNER_FILES_SHA256:
                    raise SeedVR2Unavailable(damaged)
                # GitHub archives hold one folder named after the repo and commit, and nothing else.
                if len({name.partition("/")[0] for name in zf.namelist()}) != 1:
                    raise SeedVR2Unavailable(damaged)
                for member in zf.namelist():
                    if not (staging / member).resolve().is_relative_to(staging.resolve()):
                        raise SeedVR2Unavailable(damaged)
                zf.extractall(staging)
            (top,) = staging.iterdir()
            shutil.rmtree(self.runner_dir, ignore_errors=True)
            top.rename(self.runner_dir)
        except (zipfile.BadZipFile, zlib.error, EOFError) as exc:  # not a zip, or a file inside is damaged
            raise SeedVR2Unavailable(damaged) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _uv(self, uv: Path, args: list[str], cancel: threading.Event) -> None:
        log = self.root / "install.log"
        try:
            code = _run_process([str(uv), *args], log, cancel, append=True)
        except OSError as exc:
            raise SeedVR2Unavailable("The uv setup tool couldn't be started.") from exc
        if code != 0:
            raise SeedVR2Unavailable(f"Installing the Photo-real add-on failed. The details are in {log}.")

    # --- upscaling --------------------------------------------------------------------------

    def upscale(
        self, rgb: np.ndarray, scale: int, *, progress: ProgressFn, cancel: threading.Event, softness: float = 0.0
    ) -> np.ndarray:
        snapshot = cached_weights_dir()
        if not self.runner_installed() or snapshot is None:
            raise SeedVR2Unavailable("Photo-real isn't set up on this computer yet.")
        model_dir = self._linked_model_dir(snapshot)
        h, w = rgb.shape[:2]
        boxes = tile_grid(w, h, self.tile, self.overlap)
        tile_w, tile_h = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]  # every tile is this size

        with self._lock:  # one runner at a time: two would fight over the graphics card's memory
            work = self.root / "work"
            work.mkdir(parents=True, exist_ok=True)
            _remove_stale_work(work)
            launcher = self._write_launcher()
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
                in_dir.mkdir()
                names = [f"tile_{i:04d}.png" for i in range(len(boxes))]
                for name, (x0, y0, x1, y1) in zip(names, boxes):
                    tile = soften(np.ascontiguousarray(rgb[y0:y1, x0:x1]), scale, softness)  # the runner has no softness
                    Image.fromarray(tile).save(in_dir / name)

                progress(0.0, "Starting Photo-real (about a minute)…")
                cmd = runner_command(
                    self.venv_python, self.cli_path, in_dir, out_dir, model_dir, min(tile_w, tile_h) * scale,
                    launcher=launcher, blocks_to_swap=self._blocks_to_swap(),
                )  # fmt: skip
                _free_app_gpu_memory()
                self._run_runner(cmd, cwd=Path(tmp), cancel=cancel, on_tick=_tile_counter(out_dir, len(boxes), progress))

                tiles = [(box, self._read_tile(out_dir / name, tile_w * scale, tile_h * scale)) for name, box in zip(names, boxes)]
        progress(1.0, "Photo-real: putting the parts together")
        return blend_tiles(tiles, w * scale, h * scale, scale, ramp=self.overlap * scale)

    def _blocks_to_swap(self) -> int:
        vram_gb = self._vram_gb if self._vram_gb is not None else detect_hardware().memory_gb
        return BLOCKS_TO_SWAP if vram_gb < BLOCKSWAP_BELOW_GB else 0

    def _write_launcher(self) -> Path:
        """The launcher beside the runner (written on first use, so earlier installs get it too)."""
        launcher = self.runner_dir / LAUNCHER_NAME
        try:
            current = launcher.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != LAUNCHER_SOURCE:
            launcher.parent.mkdir(parents=True, exist_ok=True)
            launcher.write_text(LAUNCHER_SOURCE, encoding="utf-8")
        return launcher

    def _linked_model_dir(self, snapshot: Path) -> Path:
        """A folder of links to the cached weights, for the runner's ``--model_dir``.

        The runner writes a ``.validation_cache.json`` beside the weights and deletes any file that
        fails its hash check, so it gets links in its own folder rather than the Hugging Face cache.
        """
        models = self.root / "models"
        models.mkdir(parents=True, exist_ok=True)
        for file in SEEDVR2_FILES:
            source, link = (snapshot / file.name).resolve(), models / file.name
            if link.exists() and link.resolve() == source:
                continue
            link.unlink(missing_ok=True)
            try:
                link.symlink_to(source)
            except OSError:  # Windows without Developer Mode can't make symlinks
                try:
                    os.link(source, link)
                except OSError:
                    return snapshot
        return models

    def _run_runner(self, cmd: list[str], *, cwd: Path, cancel: threading.Event, on_tick: Callable[[], None]) -> None:
        log = self.root / "logs" / "runner.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            code = _run_process(cmd, log, cancel, cwd=cwd, on_tick=on_tick)
        except OSError as exc:
            raise SeedVR2Unavailable("The Photo-real add-on couldn't be started.") from exc
        if code != 0:
            raise SeedVR2Unavailable(_failure_reason(log))

    @staticmethod
    def _read_tile(path: Path, w: int, h: int) -> np.ndarray:
        try:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
        except OSError as exc:
            raise SeedVR2Unavailable("The Photo-real add-on didn't produce a result.") from exc
        return fit_tile(pixels, w, h)


def _pinned(package: str) -> str:
    try:
        return f"{package}=={metadata.version(package).split('+')[0]}"
    except metadata.PackageNotFoundError:
        return package


def _remove_stale_work(work: Path) -> None:
    """Delete tile folders left behind when the app was closed (or crashed) during Photo-real.

    Only folders untouched for an hour go, so a Photo-real run in another copy of the app (the
    command line, say) keeps its folder.
    """
    cutoff = time.time() - STALE_WORK_SECONDS
    try:
        folders = [path for path in work.iterdir() if path.is_dir()]
    except OSError:
        return
    for folder in folders:
        if _last_change(folder) < cutoff:
            shutil.rmtree(folder, ignore_errors=True)


def _last_change(folder: Path) -> float:
    """When something was last added to ``folder`` or to a folder directly inside it."""
    try:
        return max([folder.stat().st_mtime, *(path.stat().st_mtime for path in folder.iterdir() if path.is_dir())])
    except OSError:
        return time.time()  # can't tell, so treat it as in use


def _free_app_gpu_memory() -> None:
    """Hand back graphics memory this app holds but isn't using, so the runner can have it."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _scaled(progress: ProgressFn, start: float, end: float) -> ProgressFn:
    return lambda fraction, message: progress(start + (end - start) * fraction, message)


def _tile_counter(out_dir: Path, total: int, progress: ProgressFn) -> Callable[[], None]:
    """Reports progress whenever the runner has written another finished tile."""
    reported = 0

    def tick() -> None:
        nonlocal reported
        done = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
        if done > reported:
            reported = done
            progress(min(done / total, 0.99), f"Photo-real: part {min(done + 1, total)} of {total}")

    return tick


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ENV_LEAKS}
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"  # the runner prints emoji; a Windows code page can't encode them
    return env


def _run_process(
    cmd: list[str],
    log: Path,
    cancel: threading.Event,
    *,
    cwd: Path | None = None,
    append: bool = False,
    on_tick: Callable[[], None] | None = None,
) -> int:
    """Run ``cmd`` with its output in ``log``; stop it and raise Cancelled if ``cancel`` is set."""
    from upscaler.config import Cancelled

    with log.open("ab" if append else "wb") as out:
        process = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=_clean_env()
        )
        try:
            while True:
                try:
                    return process.wait(timeout=_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                if cancel.is_set():
                    raise Cancelled()
                if on_tick is not None:
                    on_tick()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _failure_reason(log: Path) -> str:
    try:
        tail = log.read_text(errors="replace")[-4000:].lower()
    except OSError:
        tail = ""
    if "out of memory" in tail:
        return "The graphics card ran out of memory."
    if "cuda is not available" in tail or "no cuda gpus" in tail:
        return "The graphics card couldn't be used."
    return f"The Photo-real add-on stopped with an error. The details are in {log}."
