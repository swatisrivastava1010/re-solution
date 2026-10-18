import threading
import tracemalloc

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from upscaler.config import Cancelled
from upscaler.models import LoadedModel
from upscaler.jobs import friendly_error
from upscaler.tiling import BROKEN_OUTPUT_MESSAGE, BrokenOutput, upscale_tiled, upscale_tiled_to


def fake_model(tile: int = 64) -> LoadedModel:
    """A fixed-weight 4x model: a 3x3 conv into PixelShuffle(4)."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 3 * 16, 3, padding=1), nn.PixelShuffle(4)).eval()
    return LoadedModel(net=net, scale=4, dtype=torch.float32, tile=tile, device=torch.device("cpu"))


def random_image(width: int, height: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def run_whole(model: LoadedModel, img: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.inference_mode():
        y = model.net(x)
    return (y.clamp(0, 1) * 255).round().byte().squeeze(0).permute(1, 2, 0).numpy()


class CountingNet(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.inner(x)


class NaNUnlessFloat32(nn.Module):
    """Mimics a model that overflows in half precision."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if x.dtype != torch.float32:
            return torch.full((1, 3, x.shape[2] * 4, x.shape[3] * 4), float("nan"), dtype=x.dtype)
        return self.inner(x)


def test_upscale_tiled_matches_untiled():
    model = fake_model(tile=64)
    img = random_image(300, 200)

    out = upscale_tiled(img, model, tile=64, pad=16)

    assert out.shape == (800, 1200, 3)
    assert out.dtype == np.uint8
    diff = np.abs(out.astype(int) - run_whole(model, img).astype(int))
    assert diff.max() <= 1


def test_upscale_tiled_cancel():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        upscale_tiled(random_image(100, 80), fake_model(), cancel=cancel)


def test_upscale_tiled_cancel_midway():
    cancel = threading.Event()
    seen = []

    def progress(fraction, message):
        seen.append(fraction)
        cancel.set()

    with pytest.raises(Cancelled):
        upscale_tiled(random_image(200, 200), fake_model(tile=64), progress=progress, cancel=cancel)
    assert len(seen) == 1


def test_upscale_tiled_reports_progress():
    reports = []
    upscale_tiled(random_image(200, 100), fake_model(tile=64), progress=lambda f, m: reports.append((f, m)))

    fractions = [f for f, _ in reports]
    assert len(reports) == 8  # 4 x 2 tiles
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert all(message for _, message in reports)


def test_tile_none_uses_model_tile():
    model = fake_model(tile=100)
    model.net = CountingNet(model.net)

    upscale_tiled(random_image(300, 200), model)

    assert model.net.calls == 6  # 3 x 2 tiles of 100 px


def test_upscale_tiled_retries_in_float32_on_non_finite_output():
    model = fake_model()
    reference = run_whole(model, random_image(90, 70))
    model.net = NaNUnlessFloat32(model.net)
    model.dtype = torch.bfloat16

    out = upscale_tiled(random_image(90, 70), model)

    assert model.dtype == torch.float32
    assert np.abs(out.astype(int) - reference.astype(int)).max() <= 1


class AlwaysNaN(nn.Module):
    def forward(self, x):
        return torch.full((1, 3, x.shape[2] * 4, x.shape[3] * 4), float("nan"), dtype=x.dtype)


def test_upscale_tiled_gives_up_when_float32_is_broken_too():
    model = fake_model()
    model.net = AlwaysNaN()
    model.dtype = torch.float16

    with pytest.raises(BrokenOutput) as raised:
        upscale_tiled(random_image(40, 40), model)

    # Worded for people, so the job card says to try a different mode instead of "Please try again".
    assert raised.value.user_message == BROKEN_OUTPUT_MESSAGE == str(raised.value)
    assert friendly_error(raised.value) == BROKEN_OUTPUT_MESSAGE


def test_upscale_tiled_to_gives_up_when_float32_is_broken_too():
    model = fake_model()
    model.net = AlwaysNaN()

    with pytest.raises(BrokenOutput):
        upscale_tiled_to(random_image(40, 40), model, (100, 100))


def test_upscale_tiled_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        upscale_tiled(np.zeros((10, 10), np.uint8), fake_model())
    with pytest.raises(ValueError):
        upscale_tiled(np.zeros((10, 10, 3), np.float32), fake_model())


# --- upscale_tiled_to (target-canvas mode) ---------------------------------------------


def lanczos(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


class RecordingNet(nn.Module):
    """Remembers the biggest tile it was given and returned."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.max_in = (0, 0)
        self.max_out = (0, 0)

    def forward(self, x):
        y = self.inner(x)
        self.max_in = max(self.max_in, tuple(x.shape[2:]))
        self.max_out = max(self.max_out, tuple(y.shape[2:]))
        return y


def test_upscale_tiled_to_matches_resize():
    model = fake_model(tile=64)
    img = random_image(300, 200)

    out = upscale_tiled_to(img, model, (600, 400))

    assert out.shape == (400, 600, 3) and out.dtype == np.uint8
    reference = lanczos(run_whole(model, img), (600, 400))
    diff = np.abs(out.astype(int) - reference.astype(int))
    assert diff.mean() <= 1.5
    assert diff.max() <= 2  # no seams between tiles


@pytest.mark.parametrize("size", [(768, 512), (1800, 1200), (301, 403), (310, 207)])
def test_upscale_tiled_to_any_size_has_no_seams(size):
    # 2.56x (800 px -> 8K), 6x (more than the model's 4x), a different stretch per side, and just
    # over 1x, where the model's 4x result is shrunk almost 4x (a one-pass job: 8000 px -> 8K).
    model = fake_model(tile=64)
    img = random_image(300, 200, seed=1)

    out = upscale_tiled_to(img, model, size)

    assert out.shape == (size[1], size[0], 3)
    diff = np.abs(out.astype(int) - lanczos(run_whole(model, img), size).astype(int))
    assert diff.mean() <= 1.5 and diff.max() <= 2


def test_upscale_tiled_to_memory_bounded():
    model = fake_model(tile=64)
    model.net = RecordingNet(model.net)
    img = random_image(300, 200)
    full_canvas_bytes = 1200 * 800 * 3  # what upscale_tiled would allocate

    tracemalloc.start()
    try:
        out = upscale_tiled_to(img, model, (600, 400), pad=16)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert out.shape == (400, 600, 3) and out.dtype == np.uint8
    assert model.net.max_in <= (64 + 2 * 16, 64 + 2 * 16)
    assert model.net.max_out <= (96 * 4, 96 * 4)  # one tile with its context, never the 800x1200 canvas
    assert peak < full_canvas_bytes


def test_upscale_tiled_to_cancel():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        upscale_tiled_to(random_image(100, 80), fake_model(), (200, 160), cancel=cancel)


def test_upscale_tiled_to_reports_progress():
    reports = []
    upscale_tiled_to(
        random_image(200, 100), fake_model(tile=64), (500, 250), progress=lambda f, m: reports.append((f, m))
    )

    fractions = [f for f, _ in reports]
    assert len(reports) == 8  # 4 x 2 tiles
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)
    assert all(message for _, message in reports)


def test_upscale_tiled_to_retries_in_float32_on_non_finite_output():
    model = fake_model()
    img = random_image(90, 70)
    reference = lanczos(run_whole(model, img), (180, 140))
    model.net = NaNUnlessFloat32(model.net)
    model.dtype = torch.bfloat16

    out = upscale_tiled_to(img, model, (180, 140))

    assert model.dtype == torch.float32
    assert np.abs(out.astype(int) - reference.astype(int)).max() <= 2


@pytest.mark.parametrize("size", [(0, 100), (100, -1)])
def test_upscale_tiled_to_rejects_bad_size(size):
    with pytest.raises(ValueError):
        upscale_tiled_to(random_image(40, 40), fake_model(), size)


@pytest.mark.gpu
def test_real_model_fast_to_target():
    from upscaler.hardware import detect_hardware
    from upscaler.models import MODELS, ensure_model, load_model

    if detect_hardware().device == "cpu":
        pytest.skip("needs an Apple or NVIDIA GPU")
    ensure_model(MODELS["fast"])
    model = load_model(MODELS["fast"])
    yy, xx = np.mgrid[0:96, 0:128]
    img = np.stack([xx * 2, yy * 2, 255 - xx - yy // 2], axis=-1).clip(0, 255).astype(np.uint8)

    out = upscale_tiled_to(img, model, (256, 192))

    assert out.shape == (192, 256, 3) and out.dtype == np.uint8
    assert np.abs(out.mean((0, 1)) - lanczos(img, (256, 192)).mean((0, 1))).max() < 6


@pytest.mark.gpu
def test_real_model_fast():
    from upscaler.hardware import detect_hardware
    from upscaler.models import MODELS, ensure_model, load_model

    if detect_hardware().device == "cpu":
        pytest.skip("needs an Apple or NVIDIA GPU")
    ensure_model(MODELS["fast"])
    model = load_model(MODELS["fast"])
    yy, xx = np.mgrid[0:128, 0:128]
    img = np.stack([xx * 2, yy * 2, 255 - xx - yy // 2], axis=-1).clip(0, 255).astype(np.uint8)

    out = upscale_tiled(img, model)

    assert out.shape == (512, 512, 3)
    assert out.dtype == np.uint8
    lanczos = np.asarray(Image.fromarray(img).resize((512, 512), Image.LANCZOS))
    assert np.abs(out.mean((0, 1)) - lanczos.mean((0, 1))).max() < 6
