import hashlib
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import huggingface_hub.constants  # loaded lazily otherwise: the attribute only exists once imported
import numpy as np
import pytest
from PIL import Image

from upscaler.config import Cancelled
from upscaler.hardware import Hardware
from upscaler.seedvr2 import SeedVR2Unavailable, base, get_seedvr2_engine
from upscaler.seedvr2.base import SEEDVR2_REVISION, WeightFile, download_seedvr2_weights, weights_cached
from upscaler.seedvr2.mlx_engine import MLXSeedVR2Engine


def _no_progress(fraction: float, message: str) -> None:
    pass


def _lanczos(rgb: np.ndarray, scale: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    return np.asarray(Image.fromarray(rgb).resize((w * scale, h * scale), Image.LANCZOS))


def _test_picture(w: int, h: int) -> np.ndarray:
    """A smooth scene with a few hard edges, so the model has something real to work on."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([xx / w * 200 + 30, yy / h * 160 + 40, np.full_like(xx, 120.0)], axis=-1)
    rgb[(xx - w * 0.35) ** 2 + (yy - h * 0.5) ** 2 < (min(w, h) * 0.2) ** 2] = (230, 220, 200)
    rgb[int(h * 0.7) :, int(w * 0.6) : int(w * 0.8)] = (40, 50, 60)
    return rgb.round().astype(np.uint8)


class FakeModelEngine(MLXSeedVR2Engine):
    """Runs the real tiling and blending, with Lanczos standing in for the diffusion model."""

    def __init__(
        self,
        *,
        tile: int = 64,
        overlap: int = 16,
        cancel_after: int | None = None,
        cancel=None,
        mid_tile: bool = False,
        idle_seconds: float = 60.0,
        seconds_per_tile: float = 0.0,
    ):
        super().__init__(tile=tile, overlap=overlap, idle_seconds=idle_seconds)
        self.tile_shapes: list[tuple[int, int]] = []
        self.softness: list[float] = []
        self.loads = 0
        self._cancel_after = cancel_after
        self._cancel = cancel
        self._mid_tile = mid_tile
        self._seconds_per_tile = seconds_per_tile

    def weights_present(self) -> bool:
        return True

    def _load_model(self):
        self.loads += 1
        return "fake-model"

    def _upscale_tile(self, model, crop: np.ndarray, scale: int, workdir, softness: float) -> np.ndarray:
        assert model == "fake-model"
        assert self._model == "fake-model"  # never freed while a picture is being worked on
        time.sleep(self._seconds_per_tile)
        self.tile_shapes.append(crop.shape[:2])
        self.softness.append(softness)
        if self._cancel_after is not None and len(self.tile_shapes) == self._cancel_after:
            self._cancel.set()
            if self._mid_tile:  # as mflux calls it between the model's phases
                self._cancel_check.call_after_loop(seed=42, prompt="", latents=None, config=None)
        return _lanczos(crop, scale)


def _wait_until(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_get_engine_selection():
    assert get_seedvr2_engine(Hardware("mps", "Apple M4", 16, "mac", True)).name == "mlx"
    assert get_seedvr2_engine(Hardware("cuda", "RTX 3060", 12, "windows", False)).name == "torch"
    assert get_seedvr2_engine(Hardware("cpu", "Intel Core i5", 16, "linux", False)) is None


def test_get_engine_returns_the_same_engine_so_the_model_stays_loaded():
    hw = Hardware("mps", "Apple M4", 16, "mac", True)
    assert get_seedvr2_engine(hw) is get_seedvr2_engine(hw)


def test_mlx_engine_tiles_and_blends_to_exact_size():
    rgb = _test_picture(150, 97)
    engine = FakeModelEngine(tile=64, overlap=16)
    messages: list[tuple[float, str]] = []

    out = engine.upscale(rgb, 4, progress=lambda f, m: messages.append((f, m)), cancel=threading.Event())

    assert out.shape == (97 * 4, 150 * 4, 3) and out.dtype == np.uint8
    assert len(engine.tile_shapes) > 1 and len(set(engine.tile_shapes)) == 1
    assert max(engine.tile_shapes[0]) <= 64
    diff = np.abs(out.astype(int) - _lanczos(rgb, 4).astype(int))
    assert diff.mean() < 0.5 and diff.max() <= 4  # only Lanczos edge effects at tile borders differ
    fractions = [f for f, _ in messages]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_mlx_engine_passes_softness_to_every_tile():
    engine = FakeModelEngine(tile=64, overlap=16)
    engine.upscale(_test_picture(150, 97), 2, progress=_no_progress, cancel=threading.Event())
    assert set(engine.softness) == {0.0}  # off unless asked for

    engine.softness.clear()
    engine.upscale(_test_picture(150, 97), 2, progress=_no_progress, cancel=threading.Event(), softness=0.3)
    assert len(engine.softness) > 1 and set(engine.softness) == {0.3}


def test_mlx_tile_hands_softness_to_mflux(tmp_path):
    pytest.importorskip("mlx.core")
    seen = {}

    class FakeMflux:
        def generate_image(self, **kwargs):
            seen.update(kwargs)
            with Image.open(kwargs["image_path"]) as tile:
                return type("Result", (), {"image": tile.resize((tile.width * 2, tile.height * 2))})()

    crop = _test_picture(40, 30)
    out = MLXSeedVR2Engine()._upscale_tile(FakeMflux(), crop, 2, tmp_path, 0.25)

    assert seen["softness"] == 0.25 and seen["resolution"] == 60 and seen["seed"] == 42
    assert out.shape == (60, 80, 3)


def test_mlx_tile_keeps_the_window_calm(tmp_path, capsys, caplog):
    # mflux warns "Width and height should be multiples of 16" and draws a progress bar for every tile.
    # The app shows its own progress, so neither reaches the Terminal window.
    pytest.importorskip("mlx.core")
    import logging

    from mflux.models.common.config import config as mflux_config

    original_bar = mflux_config.tqdm
    seen = {}

    class ChattyMflux:
        def generate_image(self, **kwargs):
            mflux_log = logging.getLogger(mflux_config.__name__)
            mflux_log.warning("Width and height should be multiples of 16. Rounding down.")
            bar = mflux_config.tqdm(range(1))  # what mflux's Config.time_steps makes
            for _ in bar:
                pass
            seen["elapsed"] = bar.format_dict["elapsed"]  # mflux reads this back for its own records
            with Image.open(kwargs["image_path"]) as tile:
                return type("Result", (), {"image": tile.resize((tile.width * 2, tile.height * 2))})()

    caplog.set_level(logging.WARNING)
    out = MLXSeedVR2Engine()._upscale_tile(ChattyMflux(), _test_picture(40, 30), 2, tmp_path, 0.0)

    assert out.shape == (60, 80, 3) and seen["elapsed"] >= 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert not [r for r in caplog.records if r.name.startswith("mflux")]
    # Only for the call: mflux's settings are back as they were afterwards.
    assert mflux_config.tqdm is original_bar
    assert logging.getLogger("mflux").level == logging.NOTSET
    logging.getLogger(mflux_config.__name__).warning("after")
    assert [r.getMessage() for r in caplog.records if r.name.startswith("mflux")] == ["after"]


def test_mlx_engine_cancel_between_tiles():
    cancel = threading.Event()
    engine = FakeModelEngine(cancel_after=1, cancel=cancel)

    with pytest.raises(Cancelled):
        engine.upscale(_test_picture(150, 97), 4, progress=_no_progress, cancel=cancel)
    assert len(engine.tile_shapes) == 1


def test_mlx_engine_cancel_inside_a_tile_stops_instead_of_using_faithful():
    cancel = threading.Event()
    engine = FakeModelEngine(cancel_after=1, cancel=cancel, mid_tile=True)

    with pytest.raises(Cancelled):  # not SeedVR2Unavailable, which would run Faithful instead
        engine.upscale(_test_picture(150, 97), 4, progress=_no_progress, cancel=cancel)
    assert len(engine.tile_shapes) == 1


def test_cancel_check_stops_mflux_between_phases():
    pytest.importorskip("mlx.core")
    from mflux.callbacks.callback_registry import CallbackRegistry

    from upscaler.seedvr2.mlx_engine import CancelCheck

    check = CancelCheck()
    registry = CallbackRegistry()
    registry.register(check)
    context = registry.start(seed=42, prompt="", config=None)
    context.before_loop(None)
    context.in_loop(0, None, time_steps="steps")
    context.after_loop(None)  # no job, or not cancelled: nothing happens

    check.event = threading.Event()
    check.event.set()
    phases = [lambda: context.before_loop(None), lambda: context.in_loop(0, None, "steps"), lambda: context.after_loop(None)]
    for phase in phases:
        with pytest.raises(Cancelled):
            phase()


def test_mlx_engine_frees_the_model_when_photoreal_is_idle():
    engine = FakeModelEngine(idle_seconds=0.05)

    engine.upscale(_test_picture(80, 80), 2, progress=_no_progress, cancel=threading.Event())

    assert _wait_until(lambda: engine._model is None)
    engine.upscale(_test_picture(80, 80), 2, progress=_no_progress, cancel=threading.Event())
    assert engine.loads == 2  # loaded again when needed


def test_mlx_engine_keeps_the_model_while_working_and_between_close_jobs():
    engine = FakeModelEngine(idle_seconds=0.3, seconds_per_tile=0.1)  # a picture takes longer than the wait

    for _ in range(3):
        engine.upscale(_test_picture(150, 97), 2, progress=_no_progress, cancel=threading.Event())

    assert engine.loads == 1 and engine._model == "fake-model"
    assert _wait_until(lambda: engine._model is None)


def test_mlx_engine_release_frees_the_model_now():
    engine = FakeModelEngine()
    engine.upscale(_test_picture(80, 80), 2, progress=_no_progress, cancel=threading.Event())

    engine.release()

    assert engine._model is None
    engine.release()  # harmless when nothing is loaded


def test_mlx_engine_waits_less_before_freeing_on_smaller_macs(monkeypatch):
    from upscaler.seedvr2 import mlx_engine

    monkeypatch.setattr(mlx_engine, "detect_hardware", lambda: Hardware("mps", "Apple M4", 16, "mac", True))
    small = MLXSeedVR2Engine()._idle_delay()
    monkeypatch.setattr(mlx_engine, "detect_hardware", lambda: Hardware("mps", "Apple M4 Max", 64, "mac", True))
    large = MLXSeedVR2Engine()._idle_delay()

    assert 0 < small < large <= 600


def test_mlx_engine_without_weights_is_unavailable(monkeypatch):
    engine = MLXSeedVR2Engine()
    monkeypatch.setattr(engine, "weights_present", lambda: False)

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        engine.upscale(_test_picture(64, 64), 4, progress=_no_progress, cancel=threading.Event())
    assert "download" in excinfo.value.reason.lower()


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Two tiny stand-in weight files and a fake Hugging Face download that serves ``payloads``.

    Downloads land like Hugging Face's own: the file in ``blobs/`` and a link to it in the snapshot
    folder, unless ``options["links"]`` is False (Windows without symlinks keeps the file itself there).
    """
    hub_cache = tmp_path / "hf-cache"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub_cache))
    payloads = {"dit.safetensors": b"dit weights", "vae.safetensors": b"vae"}
    files = tuple(WeightFile(n, len(b), hashlib.sha256(b).hexdigest()) for n, b in payloads.items())
    monkeypatch.setattr(base, "SEEDVR2_FILES", files)
    monkeypatch.setattr(base, "DOWNLOAD_BYTES", sum(f.size_bytes for f in files))
    cache: dict[str, object] = {}
    monkeypatch.setattr(base, "_cached_path", lambda f: cache.get(f.name))
    downloads: list[str] = []
    options = {"links": False}

    def fake_download(repo_id, filename, *, revision, tqdm_class):
        assert repo_id == "numz/SeedVR2_comfyUI" and revision == SEEDVR2_REVISION
        downloads.append(filename)
        data = payloads[filename]
        bar = tqdm_class(total=len(data))
        bar.update(len(data))
        bar.close()
        path = tmp_path / filename
        if options["links"]:
            blob = _blobs_dir(hub_cache) / hashlib.sha256(data).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            path.symlink_to(blob)
        else:
            path.write_bytes(data)
        cache[filename] = path
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    return payloads, cache, downloads, options


def _blobs_dir(hub_cache: Path) -> Path:
    return hub_cache / "models--numz--SeedVR2_comfyUI" / "blobs"


def test_download_weights_skips_finished_files_and_reports_progress(fake_hub, tmp_path):
    payloads, cache, downloads, _ = fake_hub
    cache["vae.safetensors"] = tmp_path / "vae.safetensors"
    messages: list[tuple[float, str]] = []

    folder = download_seedvr2_weights(lambda f, m: messages.append((f, m)), threading.Event())

    assert downloads == ["dit.safetensors"]
    assert folder == tmp_path
    assert messages[-1][0] == 1.0
    assert all(0 <= f <= 1 for f, _ in messages)


def test_download_progress_never_goes_backwards_while_checking(fake_hub):
    messages: list[tuple[float, str]] = []

    download_seedvr2_weights(lambda f, m: messages.append((f, m)), threading.Event())

    fractions = [f for f, _ in messages]
    assert fractions == sorted(fractions), messages  # the bar never jumps back while a file is checked
    checking = [f for f, m in messages if m.startswith("Checking")]
    assert len(checking) == 2 and checking[0] > 0.5  # the big file is checked where its download ended


def test_download_weights_removes_a_damaged_file(fake_hub, tmp_path):
    payloads, cache, downloads, _ = fake_hub
    payloads["dit.safetensors"] = b"dit wEights"  # same size, wrong bytes

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        download_seedvr2_weights(lambda f, m: None, threading.Event())
    assert "damaged" in excinfo.value.reason
    assert not (tmp_path / "dit.safetensors").exists()


def test_download_weights_cancel(fake_hub):
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(Cancelled):
        download_seedvr2_weights(lambda f, m: None, cancel)


def _cancel_when_checking(cancel: threading.Event):
    def progress(fraction: float, message: str) -> None:
        if message.startswith("Checking"):
            cancel.set()

    return progress


def test_cancel_while_checking_stops_and_leaves_the_file_to_be_checked_again(fake_hub, tmp_path):
    payloads, cache, downloads, options = fake_hub
    options["links"] = True
    cancel = threading.Event()

    with pytest.raises(Cancelled):
        download_seedvr2_weights(_cancel_when_checking(cancel), cancel)

    blob = _blobs_dir(tmp_path / "hf-cache") / hashlib.sha256(payloads["dit.safetensors"]).hexdigest()
    assert blob.read_bytes() == payloads["dit.safetensors"]  # the 7 GB download is kept...
    assert not (tmp_path / "dit.safetensors").is_symlink()  # ...but not counted as finished until checked


def test_cancel_while_checking_a_file_with_no_link_finishes_the_check_first(fake_hub, tmp_path):
    payloads, cache, downloads, options = fake_hub
    options["links"] = False  # Windows without symlinks: the snapshot holds the only copy
    payloads["dit.safetensors"] = b"dit wEights"  # damaged, so a finished check removes it
    cancel = threading.Event()

    with pytest.raises(SeedVR2Unavailable):
        download_seedvr2_weights(_cancel_when_checking(cancel), cancel)
    assert not (tmp_path / "dit.safetensors").exists()  # never left behind unchecked


def test_download_weights_removes_partial_files_left_by_a_closed_window(fake_hub, tmp_path):
    blobs = _blobs_dir(tmp_path / "hf-cache")
    blobs.mkdir(parents=True)
    left_behind, in_progress = blobs / "etag.1a2b3c4d.incomplete", blobs / "etag.5e6f7a8b.incomplete"
    finished = blobs / "abc123"
    for path in (left_behind, in_progress, finished):
        path.write_bytes(b"partial")
    an_hour_ago = time.time() - 3600
    os.utime(left_behind, (an_hour_ago, an_hour_ago))
    os.utime(finished, (an_hour_ago, an_hour_ago))

    download_seedvr2_weights(_no_progress, threading.Event())

    assert not left_behind.exists()
    assert in_progress.exists()  # still being written by another download
    assert finished.exists()


def test_download_weights_needs_room_on_the_disk_first(fake_hub, monkeypatch):
    payloads, cache, downloads, _ = fake_hub
    monkeypatch.setattr(base.shutil, "disk_usage", lambda path: SimpleNamespace(free=50))

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        download_seedvr2_weights(_no_progress, threading.Event())
    assert "disk space" in excinfo.value.reason and "GB" in excinfo.value.reason
    assert downloads == []  # nothing was started


def test_download_weights_with_everything_downloaded_needs_no_room(fake_hub, tmp_path, monkeypatch):
    payloads, cache, downloads, _ = fake_hub
    for name in payloads:
        cache[name] = tmp_path / name
    monkeypatch.setattr(base.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))

    assert download_seedvr2_weights(_no_progress, threading.Event()) == tmp_path


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not weights_cached(), reason="SeedVR2 weights are not downloaded")
def test_mlx_upscale_small():
    rgb = _test_picture(128, 128)

    out = MLXSeedVR2Engine().upscale(rgb, 4, progress=_no_progress, cancel=threading.Event())

    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    assert np.isfinite(out.astype(np.float32)).all()
    mean_diff = np.abs(out.reshape(-1, 3).mean(0) - _lanczos(rgb, 4).reshape(-1, 3).mean(0))
    assert (mean_diff <= 6).all(), mean_diff


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kernel, stride, padding, padding_causal",
    [
        (3, 1, 1, False),  # resnet convs
        ((3, 3, 3), (2, 2, 2), (1, 0, 0), False),  # temporal+spatial downsample
        (3, 1, 1, True),  # upsample conv
        (1, 1, 0, False),  # shortcuts and upscale_conv
    ],
)
def test_single_frame_convs_match_mflux_conv3d(kernel, stride, padding, padding_causal):
    mx = pytest.importorskip("mlx.core")
    from mflux.models.seedvr2.model.seedvr2_vae.common.conv3d import CausalConv3d
    from mlx import nn

    from upscaler.seedvr2.mlx_engine import use_single_frame_convs

    mx.random.seed(0)
    conv = CausalConv3d(8, 16, kernel_size=kernel, stride=stride, padding=padding, use_padding_causal=padding_causal)
    conv.update({"weight": mx.random.normal(conv.weight.shape), "bias": mx.random.normal(conv.bias.shape)})
    vae = nn.Sequential(conv)
    frame, video = mx.random.normal((1, 8, 1, 20, 18)), mx.random.normal((1, 8, 3, 20, 18))
    expected, expected_video = vae(frame), vae(video)

    use_single_frame_convs(vae)
    folded = vae(frame)

    assert type(conv).__name__ == "SingleFrameCausalConv3d"
    assert folded.shape == expected.shape
    assert mx.abs(folded - expected).max().item() < 1e-3
    assert mx.array_equal(vae(video), expected_video)  # real video still takes the 3D path
