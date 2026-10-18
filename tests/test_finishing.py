import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from upscaler import finishing
from upscaler.finishing import GRAIN_LEVELS, finish

ROOT = Path(__file__).resolve().parents[1]


def photo_like(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A smooth gradient with mild texture, kept in the mid range so nothing clips."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([60 + 80 * xx / width, 70 + 60 * yy / height, 120 - 40 * xx / width], axis=-1)
    return (base + rng.normal(0, 12, base.shape)).clip(40, 200).astype(np.uint8)


def lanczos(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


@pytest.fixture
def src():
    return photo_like(80, 60)


@pytest.fixture
def up(src):
    # Stands in for an AI result: Lanczos plus fake fine detail.
    noise = np.random.default_rng(1).normal(0, 10, (240, 320, 3))
    return (lanczos(src, (320, 240)) + noise).clip(0, 255).astype(np.uint8)


def test_grain_levels():
    assert GRAIN_LEVELS == {"off": 0.0, "subtle": 3.0, "medium": 6.0}


def test_finish_strength_zero_is_lanczos(src, up):
    out = finish(src, up, strength=0, tone_lock=False)
    assert out.shape == up.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - lanczos(src, (320, 240)).astype(int)).max() <= 1


def test_finish_full_strength_without_tone_lock_returns_ai_result(src, up):
    assert np.array_equal(finish(src, up, strength=1, tone_lock=False), up)


def test_finish_half_strength_blends(src, up):
    out = finish(src, up, strength=0.5, tone_lock=False)
    expected = (lanczos(src, (320, 240)).astype(float) + up) / 2
    assert np.abs(out - expected).max() <= 1.5


def test_finish_tone_lock_removes_color_cast(src):
    base = lanczos(src, (320, 240)).astype(int)
    up = (base + [20, 0, 0]).clip(0, 255).astype(np.uint8)

    out = finish(src, up, strength=1, tone_lock=True).astype(int)

    mean_diff = (out - base).mean(axis=(0, 1))
    assert abs(mean_diff[0]) < 2
    assert np.abs(mean_diff[1:]).max() < 1


def test_finish_tone_lock_keeps_fine_detail(src, up):
    out = finish(src, up, strength=1, tone_lock=True)
    detail_in = up.astype(float) - lanczos(src, (320, 240))
    detail_out = out.astype(float) - lanczos(src, (320, 240))
    assert np.corrcoef(detail_in.ravel(), detail_out.ravel())[0, 1] > 0.95


def test_finish_grain_mean_preserving(src, up):
    plain = finish(src, up, grain=0.0)
    grainy = finish(src, up, grain=3.0)
    assert abs(grainy.mean() - plain.mean()) < 0.5
    assert grainy.std() > plain.std()


def test_finish_grain_is_repeatable(src, up):
    assert np.array_equal(finish(src, up, grain=6.0, seed=3), finish(src, up, grain=6.0, seed=3))
    assert not np.array_equal(finish(src, up, grain=6.0, seed=3), finish(src, up, grain=6.0, seed=4))


def test_finish_strips_leave_no_seams(src, up, monkeypatch):
    whole = finish(src, up, strength=0.7, grain=3.0)
    monkeypatch.setattr(finishing, "_STRIP_PIXELS", 320 * 7)  # 7-row strips
    striped = finish(src, up, strength=0.7, grain=3.0)
    assert np.abs(whole.astype(int) - striped.astype(int)).max() <= 1


def test_finish_handles_tiny_images():
    src = photo_like(3, 2)
    up = lanczos(src, (12, 8))
    assert finish(src, up, grain=3.0).shape == (8, 12, 3)


def test_finish_rejects_bad_input(src, up):
    with pytest.raises(ValueError):
        finish(src, up, strength=1.5)
    with pytest.raises(ValueError):
        finish(src[..., 0], up)
    with pytest.raises(ValueError):
        finish(src, up.astype(np.float32))


def test_finish_in_place_writes_into_up(src, up):
    expected = finish(src, up, strength=0.5, tone_lock=True, grain=3.0)
    target = up.copy()

    out = finish(src, target, strength=0.5, tone_lock=True, grain=3.0, in_place=True)

    assert out is target
    assert np.array_equal(out, expected)


def test_finish_in_place_copies_a_read_only_picture(src, up):
    frozen = up.copy()
    frozen.flags.writeable = False

    out = finish(src, frozen, strength=0.5, in_place=True)

    assert out is not frozen and np.array_equal(frozen, up)
    assert np.array_equal(out, finish(src, up, strength=0.5))


def test_finish_without_in_place_leaves_up_alone(src, up):
    before = up.copy()
    finish(src, up, strength=0.5, grain=3.0)
    assert np.array_equal(up, before)


def _extra_memory(code: str) -> int:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip().splitlines()[-1])


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="uses the resource module")
def test_finish_16k_in_place_needs_no_second_canvas():
    # A 16K result is 0.6 GB; finishing it in place must not need another one.
    extra_bytes = _extra_memory(
        """
        import resource, numpy as np
        from upscaler.finishing import finish
        src = np.full((1536, 2048, 3), 120, np.uint8)
        up = np.full((12288, 16384, 3), 120, np.uint8)
        up[::2, ::3] = 140
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = finish(src, up, strength=0.8, tone_lock=True, grain=3.0, in_place=True)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if __import__("sys").platform == "darwin" else 1024
        print((after - before) * scale)
        """
    )
    # Measured on an M4: about 0.31 GB in place, 0.90 GB with a second canvas.
    assert extra_bytes < 450_000_000, f"finish() used {extra_bytes / 1e9:.2f} GB extra"


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="uses the resource module")
def test_finish_8k_peak_memory_under_1gb():
    code = textwrap.dedent(
        """
        import resource, numpy as np
        from upscaler.finishing import finish
        src = np.full((2048, 2048, 3), 120, np.uint8)
        up = np.full((8192, 8192, 3), 120, np.uint8)
        up[::2, ::3] = 140
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = finish(src, up, strength=0.8, tone_lock=True, grain=3.0)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if __import__("sys").platform == "darwin" else 1024
        print((after - before) * scale)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    extra_bytes = int(result.stdout.strip().splitlines()[-1])
    assert extra_bytes < 1_000_000_000, f"finish() used {extra_bytes / 1e9:.2f} GB extra"
