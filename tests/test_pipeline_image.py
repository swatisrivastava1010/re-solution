import io
import shutil
import subprocess
import threading
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from upscaler import pipeline_image
from upscaler.config import SPACE_RESERVE_BYTES, Cancelled
from upscaler.hardware import Hardware
from upscaler.imageio_utils import SUFFIXES, ImageFileError, LoadedImage, load_image, render_vector
from upscaler.pipeline_image import (
    UPSCALE_ROUTES,
    ImageOptions,
    parse_route,
    resolve_mode,
    upscale_array,
    upscale_image,
)
from upscaler.seedvr2 import SeedVR2Unavailable
from upscaler.sizing import BYTES_PER_PIXEL, PRESETS, TWO_PASS_NOTE, estimated_file_bytes, print_size

MAC_16GB = Hardware("mps", "Apple M4", 16, "mac", True)
MAC_8GB = Hardware("mps", "Apple M1", 8, "mac", True)
PC_NO_GPU = Hardware("cpu", "Intel", 32, "windows", False)


def nearest(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def nearest_to(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return rgb[(np.arange(h) * rgb.shape[0] // h)[:, None], np.arange(w) * rgb.shape[1] // w]


def gradient(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    return np.stack(np.broadcast_arrays(x + 0 * y, y + 0 * x, (x + y) / 2), axis=-1).astype(np.uint8)


def write_image(path: Path, width: int, height: int, **save_options) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gradient(width, height)).save(path, **save_options)
    return path


class FakeSeedVR2:
    name = "fake"

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.softness: list[float] = []

    def weights_present(self) -> bool:
        return True

    def download_weights(self, progress, cancel) -> None:
        pass

    def upscale(self, rgb, scale, *, progress, cancel, softness=0.0):
        self.calls.append((rgb.shape, scale))
        self.softness.append(softness)
        if self.error:
            raise self.error
        progress(1.0, "Photo-real")
        return nearest(rgb, scale)


class FakeProvider:
    id = "fal"
    name = "fal.ai"
    models: list = []

    def __init__(self, factor: int = 2):
        self.factor = factor  # how much bigger the pictures it sends back are
        self.calls = []
        self.images: list = []
        self.formats: list[str] = []
        self.icc: list[bytes | None] = []
        self.received: list[np.ndarray] = []

    def check_key(self, key):
        return True, ""

    def upscale(self, image, model_id, scale, key, *, progress, cancel):
        from upscaler.cloud.base import prepare_upload

        self.calls.append((image.name, model_id, scale, key))
        self.images.append(image)
        # What the real services send: cloud.base.prepare_upload's PNG or JPEG of the opened picture.
        upload = prepare_upload(image)
        with Image.open(io.BytesIO(upload.data)) as sent:
            self.formats.append(sent.format)
            self.icc.append(sent.info.get("icc_profile"))
            self.received.append(np.asarray(ImageOps.exif_transpose(sent).convert("RGB")))
        return nearest(self.received[-1], self.factor)


@pytest.fixture
def engines(monkeypatch):
    """Fast stand-ins for every engine: nearest-neighbour enlargement instead of AI."""
    state = SimpleNamespace(
        hardware=MAC_16GB,
        seedvr2=FakeSeedVR2(),
        tiled=[],
        tiled_to=[],
        loaded=[],
        cancel_during_engine=None,
        ensured=[],  # (model id, the cancel event it was given)
    )

    def fake_ensure_model(spec, progress=None, cancel=None):
        state.ensured.append((spec.id, cancel))
        return Path(spec.file)

    def fake_load_model(spec, device=None):
        state.loaded.append(spec.id)
        return SimpleNamespace(scale=4, tile=spec.tile, dtype=torch.float32)

    def fake_upscale_tiled(img, model, *, tile=None, pad=16, progress=None, cancel=None):
        state.tiled.append(img.shape)
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(1.0, "Upscaling")
        if state.cancel_during_engine is not None:
            state.cancel_during_engine.set()  # the user presses Cancel as the last part finishes
        return nearest(img, model.scale)

    def fake_upscale_tiled_to(img, model, out_size, *, tile=None, pad=16, progress=None, cancel=None):
        state.tiled_to.append((img.shape, tuple(out_size)))
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(1.0, "Upscaling to full size")
        if state.cancel_during_engine is not None:
            state.cancel_during_engine.set()
        return nearest_to(img, out_size)

    monkeypatch.setattr(pipeline_image, "detect_hardware", lambda: state.hardware)
    monkeypatch.setattr(pipeline_image, "get_seedvr2_engine", lambda hw: state.seedvr2)
    monkeypatch.setattr(pipeline_image, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(pipeline_image, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline_image, "upscale_tiled", fake_upscale_tiled)
    monkeypatch.setattr(pipeline_image, "upscale_tiled_to", fake_upscale_tiled_to)
    return state


def run(path: Path, out_dir: Path, cancel: threading.Event | None = None, **options):
    seen: list[tuple[float, str]] = []
    result = upscale_image(
        path,
        ImageOptions(**options),
        out_dir,
        progress=lambda fraction, message: seen.append((fraction, message)),
        cancel=cancel or threading.Event(),
    )
    return result, seen


def opened_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


# --- options and modes -------------------------------------------------------------


def test_photoreal_is_the_default_mode():
    assert ImageOptions().mode == "photoreal"
    assert ImageOptions().photoreal_level == "balanced"
    assert ImageOptions().size == "8k" and ImageOptions().output_format == "jpeg"
    assert ImageOptions().dpi == 300
    assert (ImageOptions().creativity, ImageOptions().resemblance, ImageOptions().strength) == (0.0, 1.0, 1.0)


def test_every_size_has_a_file_name_tag():
    assert pipeline_image.PRESET_TAGS.keys() == PRESETS.keys()


def test_every_format_has_a_size_estimate():
    assert BYTES_PER_PIXEL.keys() == SUFFIXES.keys()


def test_guest_key_is_not_shown_in_logs():
    assert "secret" not in repr(ImageOptions(guest_key="secret"))


def test_resolve_mode():
    assert resolve_mode("photoreal", MAC_16GB) == ("photoreal", [])
    assert resolve_mode("fast", MAC_8GB) == ("fast", [])
    assert resolve_mode("cloud:fal:topaz_precision", PC_NO_GPU) == ("cloud:fal:topaz_precision", [])

    mode, notes = resolve_mode("photoreal", MAC_8GB)
    assert mode == "faithful" and notes == ["Photo-real needs 16 GB of memory; used Faithful instead."]

    mode, notes = resolve_mode("photoreal", PC_NO_GPU)
    assert mode == "faithful" and "Faithful" in notes[0]


@pytest.mark.parametrize("mode", ["sharpest", "cloud:fal", "cloud:nobody:model", ""])
def test_resolve_mode_rejects_unknown(mode):
    with pytest.raises(ValueError):
        resolve_mode(mode, MAC_16GB)


# --- pipeline ----------------------------------------------------------------------


def test_pipeline_fast_4k_dims(engines, tmp_path):
    src = write_image(tmp_path / "in" / "table.png", 512, 385)

    started = time.monotonic()
    result, seen = run(src, tmp_path / "out", mode="fast", size="4k")

    assert time.monotonic() - started < 2
    assert (result.width, result.height) == (3840, 2888)
    assert result.output_path.parent == tmp_path / "out"
    assert result.output_path.name == "table-4K-fast.jpg"
    assert opened_size(result.output_path) == (3840, 2888)
    assert result.mode_used == "fast" and result.seconds > 0
    assert any("small" in note for note in result.notes)  # 7.5x needs two passes
    assert engines.tiled == [(385, 512, 3)]  # the chosen mode at 4x...
    assert engines.tiled_to == [((1540, 2048, 3), (3840, 2888))]  # ...then Fast straight to the exact size

    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[0] == 0 and fractions[-1] == 1
    for blob in (result.preview, result.compare_before, result.compare_after):
        with Image.open(io.BytesIO(blob)) as image:
            assert image.format == "JPEG"


def test_result_says_the_source_size(engines, tmp_path):
    result, _ = run(write_image(tmp_path / "room.png", 512, 385), tmp_path / "out", mode="fast", size="4k")
    assert (result.source_width, result.source_height) == (512, 385)
    big, _ = run(write_image(tmp_path / "big.png", 9000, 50), tmp_path / "out", mode="fast", size="8k")
    assert (big.source_width, big.source_height) == (9000, 50)


def test_photoreal_on_nvidia_unloads_the_cached_model_first(engines, tmp_path, monkeypatch):
    # SeedVR2 runs as its own program on NVIDIA: the Fast or Faithful model this app still holds on the
    # card would take graphics memory it needs.
    order = []
    monkeypatch.setattr(pipeline_image, "unload_model", lambda: order.append("unload"))
    real_upscale = engines.seedvr2.upscale
    monkeypatch.setattr(engines.seedvr2, "upscale", lambda *a, **k: order.append("seedvr2") or real_upscale(*a, **k))
    engines.hardware = Hardware("cuda", "RTX 4090", 24, "windows", False)
    run(write_image(tmp_path / "room.png", 64, 48), tmp_path / "out", mode="photoreal", size="4x")
    assert order == ["unload", "seedvr2"]

    order.clear()
    engines.hardware = MAC_16GB  # MLX keeps its own memory; nothing to unload on a Mac
    run(write_image(tmp_path / "room.png", 64, 48), tmp_path / "out", mode="photoreal", size="4x")
    assert order == ["seedvr2"]


def test_plain_words_problems_are_user_errors(engines, tmp_path, monkeypatch):
    from upscaler.config import UserError

    src = write_image(tmp_path / "room.png", 64, 48)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: None)
    with pytest.raises(UserError, match="connect fal.ai in Settings"):
        run(src, tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")
    with pytest.raises(UserError, match="smaller size"):
        pipeline_image._check_output_size((100_000, 100_000), "16k")
    with pytest.raises(UserError):
        run(src, tmp_path / "out", mode="fast", size="2x", strength=2.0)
    with pytest.raises(UserError):
        run(src, tmp_path / "out", mode="sharpest")
    with pytest.raises(UserError):
        parse_route("magic")


def test_pipeline_already_large(engines, tmp_path):
    engines.hardware = MAC_8GB  # no Photo-real here, but nothing needs to run anyway
    src = write_image(tmp_path / "wide.png", 9000, 60)

    result, _ = run(src, tmp_path / "out", size="8k")

    assert (result.width, result.height) == (9000, 60)
    assert opened_size(result.output_path) == (9000, 60)
    assert len(result.notes) == 1 and "already" in result.notes[0]
    assert result.mode_used == "original" and result.output_path.name == "wide-8K-original.jpg"
    assert engines.loaded == [] and engines.tiled == [] and engines.seedvr2.calls == []


def test_pipeline_photoreal_fallback(engines, tmp_path):
    engines.hardware = MAC_8GB
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="2x")

    assert result.mode_used == "faithful"
    assert "Photo-real needs 16 GB of memory; used Faithful instead." in result.notes
    assert result.output_path.name == "room-2x-faithful.jpg"
    assert engines.seedvr2.calls == [] and engines.loaded == ["faithful"]


def test_pipeline_photoreal_uses_seedvr2(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    result, seen = run(src, tmp_path / "out", size="4x")  # the default mode and level (Balanced)

    assert result.mode_used == "photoreal"
    assert result.output_path.name == "room-4x-photoreal.jpg"
    assert (result.width, result.height) == (256, 192)
    assert engines.seedvr2.calls == [((48, 64, 3), 2)]  # SeedVR2 2x...
    assert engines.loaded == ["fast"] and engines.tiled_to == [((96, 128, 3), (256, 192))]  # ...then Fast
    assert result.notes == []  # two steps are what Balanced is; nothing to warn about
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1


def test_pipeline_photoreal_rich_runs_4x(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", size="4x", photoreal_level="rich")

    assert result.mode_used == "photoreal" and (result.width, result.height) == (256, 192)
    assert engines.seedvr2.calls == [((48, 64, 3), 4)]
    assert engines.loaded == [] and engines.tiled_to == []


@pytest.mark.parametrize(
    ("level", "seedvr2_calls", "tiled_to", "loaded"),
    [
        ("balanced", [((12, 2048, 3), 2)], [((24, 4096, 3), (8192, 48))], ["fast"]),
        ("rich", [((12, 2048, 3), 4)], [], []),
        ("gentle", [((12, 2048, 3), 2)], [], []),  # then a plain Lanczos resize
    ],
)
def test_pipeline_photoreal_levels_2k_to_8k(engines, tmp_path, level, seedvr2_calls, tiled_to, loaded):
    src = write_image(tmp_path / "wide.png", 2048, 12)  # thin, so the test stays fast

    result, _ = run(src, tmp_path / "out", size="8k", photoreal_level=level)

    assert engines.seedvr2.calls == seedvr2_calls
    assert engines.tiled_to == tiled_to and engines.loaded == loaded
    assert (result.width, result.height) == (8192, 48) and opened_size(result.output_path) == (8192, 48)
    assert result.mode_used == "photoreal" and result.output_path.name == "wide-8K-photoreal.jpg"


@pytest.mark.parametrize("level", ["balanced", "rich", "gentle"])
def test_pipeline_photoreal_never_makes_a_bigger_picture_than_needed(engines, tmp_path, level):
    src = write_image(tmp_path / "wide.png", 5000, 12)  # a 5000 px photo going to 16K

    result, _ = run(src, tmp_path / "out", size="16k", photoreal_level=level)

    [(shape, scale)] = engines.seedvr2.calls
    assert shape[1] * scale <= 16384  # never the 20000 px picture
    assert (result.width, result.height) == (16384, 39)
    if level == "rich":
        assert (shape, scale) == ((10, 4096, 3), 4)  # 4x of a copy shrunk to land on 16K
    else:
        assert (shape, scale) == ((12, 5000, 3), 2)


def test_pipeline_photoreal_runs_2x_when_that_is_enough(engines, tmp_path):
    src = write_image(tmp_path / "phone.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="2x")

    assert engines.seedvr2.calls == [((48, 64, 3), 2)]  # a quarter of the work of 4x
    assert (result.width, result.height) == (128, 96)


@pytest.mark.parametrize(
    ("width", "seedvr2_input", "scale"),
    [
        (4032, (12, 4032), 2),
        (3300, (12, 3300), 2),
        (3200, (12, 3200), 2),
        (3000, (8, 2048), 4),
        (2048, (12, 2048), 4),
    ],
)
def test_pipeline_photoreal_rich_scale_for_8k(engines, tmp_path, width, seedvr2_input, scale):
    src = write_image(tmp_path / "wide.png", width, 12)  # thin, so the test stays fast

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="8k", photoreal_level="rich")

    # 2x within Lanczos reach of 8K, a little further when 4x would mean shrinking the picture
    # first; otherwise 4x on a copy shrunk so it lands on 8K, never past it.
    assert engines.seedvr2.calls == [((*seedvr2_input, 3), scale)]
    assert result.width == 8192 and engines.tiled_to == []


def test_pipeline_photoreal_two_passes_for_tiny_input(engines, tmp_path):
    src = write_image(tmp_path / "icon.png", 40, 30)

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="4k", photoreal_level="rich")

    assert (result.width, result.height) == (3840, 2880)
    assert engines.seedvr2.calls == [((30, 40, 3), 4)]  # Photo-real once...
    assert engines.tiled_to == [((120, 160, 3), (3840, 2880))] and engines.loaded == ["fast"]  # ...then Fast
    assert any("small" in note for note in result.notes)


@pytest.mark.parametrize("level", ["balanced", "gentle"])
def test_pipeline_photoreal_tiny_input_runs_4x_at_every_level(engines, tmp_path, level):
    src = write_image(tmp_path / "icon.png", 40, 30)

    result, _ = run(src, tmp_path / "out", size="4k", photoreal_level=level)

    # 96x in all: 2x, then the Fast model's 4x, would leave a 12x blurry stretch at the end.
    assert engines.seedvr2.calls == [((30, 40, 3), 4)]
    assert engines.tiled_to == [((120, 160, 3), (3840, 2880))] and engines.loaded == ["fast"]
    assert any("small" in note for note in result.notes)


@pytest.mark.parametrize("level", ["balanced", "rich", "gentle"])
def test_pipeline_photoreal_keeps_every_pixel_of_a_big_photo(engines, tmp_path, level):
    src = write_image(tmp_path / "scan.png", 3000, 12)  # a 3000 px scan going to 4K

    result, _ = run(src, tmp_path / "out", size="4k", photoreal_level=level)

    assert engines.seedvr2.calls == [((12, 3000, 3), 2)]  # never a smaller copy of the scan
    assert engines.tiled_to == [] and (result.width, result.height) == (3840, 15)


def test_pipeline_16k_fast_dims(engines, tmp_path):
    src = write_image(tmp_path / "in" / "table.png", 512, 385)

    result, seen = run(src, tmp_path / "out", mode="fast", size="16k")

    assert (result.width, result.height) == (16384, 12320)
    assert opened_size(result.output_path) == (16384, 12320)
    assert result.output_path.name == "table-16K-fast.jpg"
    assert engines.tiled == [(385, 512, 3)]
    assert engines.tiled_to == [((1540, 2048, 3), (16384, 12320))]  # never a 32K picture in between
    assert result.dpi == 300 and result.print == print_size(16384, 12320, 300)
    assert result.file_bytes == result.output_path.stat().st_size
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["table-16K-fast.jpg"]  # no temporary file


def test_pipeline_16k_photoreal_second_pass_is_fast(engines, tmp_path):
    src = write_image(tmp_path / "wide.png", 2048, 12)  # thin, so the test stays fast

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="16k", photoreal_level="rich")

    assert result.mode_used == "photoreal" and (result.width, result.height) == (16384, 96)
    assert engines.seedvr2.calls == [((12, 2048, 3), 4)]
    assert engines.loaded == ["fast"] and engines.tiled_to == [((48, 8192, 3), (16384, 96))]
    assert TWO_PASS_NOTE in result.notes and not any("small" in note for note in result.notes)


def test_pipeline_16k_photoreal_balanced(engines, tmp_path):
    src = write_image(tmp_path / "wide.png", 2048, 12)

    result, _ = run(src, tmp_path / "out", size="16k")

    assert (result.width, result.height) == (16384, 96)
    assert engines.seedvr2.calls == [((12, 2048, 3), 2)]
    assert engines.loaded == ["fast"] and engines.tiled == [(24, 4096, 3)]  # exactly 4x: no resize needed
    assert result.notes == []


# --- Photo-real creative controls ---------------------------------------------------


def test_creativity_maps_to_softness(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    run(src, tmp_path / "out", size="2x")
    run(src, tmp_path / "out", size="2x", creativity=0.6)
    run(src, tmp_path / "out", size="2x", creativity=1.0)

    assert engines.seedvr2.softness == [0.0, pytest.approx(0.3), 0.5]


def _lanczos(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, Image.LANCZOS)).astype(int)


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).astype(int)


def test_resemblance_zero_is_base(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", size="2x", resemblance=0.0, output_format="png")

    assert engines.seedvr2.calls  # Photo-real ran...
    diff = np.abs(_pixels(result.output_path) - _lanczos(src, (128, 96)))
    assert diff.max() <= 1  # ...but none of it was kept: the plain enlargement


def test_resemblance_replaces_strength_for_photoreal(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)
    base = _lanczos(src, (128, 96))

    kept, _ = run(src, tmp_path / "out", size="2x", strength=0.0, resemblance=1.0, output_format="png")
    dropped, _ = run(src, tmp_path / "out", size="2x", strength=1.0, resemblance=0.0, output_format="png")

    assert np.abs(_pixels(kept.output_path) - base).max() > 1  # strength is ignored for Photo-real
    assert np.abs(_pixels(dropped.output_path) - base).max() <= 1


def test_strength_still_works_for_other_modes(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(
        src, tmp_path / "out", mode="faithful", size="2x", strength=0.0, resemblance=1.0, output_format="png"
    )

    assert np.abs(_pixels(result.output_path) - _lanczos(src, (128, 96))).max() <= 1
    assert engines.seedvr2.calls == []


def test_photoreal_fallback_uses_detail_strength(engines, tmp_path):
    engines.hardware = MAC_8GB  # Faithful runs instead, so its own detail strength applies
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", size="2x", strength=0.0, resemblance=1.0, output_format="png")

    assert result.mode_used == "faithful"
    assert np.abs(_pixels(result.output_path) - _lanczos(src, (128, 96))).max() <= 1


@pytest.mark.parametrize(
    "options",
    [{"creativity": 1.5}, {"creativity": -0.1}, {"resemblance": 2.0}, {"photoreal_level": "extreme"}],
)
def test_pipeline_rejects_bad_creative_options(engines, tmp_path, options):
    src = write_image(tmp_path / "room.png", 64, 48)
    with pytest.raises(ValueError):
        run(src, tmp_path / "out", size="2x", **options)
    assert engines.seedvr2.calls == []


# --- upscale_array (for Preview styles) ---------------------------------------------------


def _array(rgb: np.ndarray, route: str, out_size: tuple[int, int], cancel: threading.Event | None = None):
    seen: list[float] = []
    out = upscale_array(rgb, route, out_size, progress=lambda f, m: seen.append(f), cancel=cancel or threading.Event())
    return out, seen


def test_upscale_routes():
    assert UPSCALE_ROUTES == (
        "photoreal:balanced",
        "photoreal:rich",
        "photoreal:gentle",
        "faithful",
        "faithful_sharp",
        "fast",
        "illustration",
    )


def test_parse_route():
    assert parse_route("photoreal:rich") == ("photoreal", "rich")
    assert parse_route("photoreal") == ("photoreal", "balanced")
    assert parse_route("faithful") == ("faithful", "balanced")
    for route in UPSCALE_ROUTES:
        assert parse_route(route)[0] in ("photoreal", *pipeline_image.LOCAL_MODES)


@pytest.mark.parametrize(
    ("route", "seedvr2_calls", "tiled", "tiled_to", "loaded"),
    [
        ("photoreal:balanced", [((24, 32, 3), 2)], [], [((48, 64, 3), (128, 96))], ["fast"]),
        ("photoreal:rich", [((24, 32, 3), 4)], [], [], []),
        ("photoreal:gentle", [((24, 32, 3), 2)], [], [], []),
        ("faithful", [], [(24, 32, 3)], [], ["faithful"]),
        ("faithful_sharp", [], [(24, 32, 3)], [], ["faithful_sharp"]),
        ("fast", [], [(24, 32, 3)], [], ["fast"]),
        ("illustration", [], [(24, 32, 3)], [], ["illustration"]),
    ],
)
def test_upscale_array_routes(engines, route, seedvr2_calls, tiled, tiled_to, loaded):
    rgb = gradient(32, 24)

    out, seen = _array(rgb, route, (128, 96))

    assert out.shape == (96, 128, 3) and out.dtype == np.uint8
    assert engines.seedvr2.calls == seedvr2_calls
    assert engines.tiled == tiled and engines.tiled_to == tiled_to and engines.loaded == loaded
    assert seen == sorted(seen)


def test_upscale_array_does_no_finishing(engines):
    rgb = gradient(32, 24)

    out, _ = _array(rgb, "faithful", (128, 96))

    assert np.array_equal(out, nearest(rgb, 4))  # the engine's own pixels: no tone lock, no blend


def test_upscale_array_resizes_to_the_exact_size(engines):
    out, _ = _array(gradient(32, 24), "photoreal:rich", (100, 75))
    assert out.shape == (75, 100, 3)
    assert engines.seedvr2.calls == [((24, 32, 3), 4)]  # the whole crop at 4x, then resized down to fit


@pytest.mark.parametrize("route", ["photoreal:extreme", "cloud:fal:seedvr2", "magic", ""])
def test_upscale_array_rejects_unknown_routes(engines, route):
    with pytest.raises(ValueError):
        _array(gradient(32, 24), route, (128, 96))


def test_upscale_array_creativity(engines):
    upscale_array(gradient(32, 24), "photoreal:rich", (128, 96), progress=lambda f, m: None,
                  cancel=threading.Event(), creativity=0.5)  # fmt: skip
    assert engines.seedvr2.softness == [0.25]
    with pytest.raises(ValueError):
        upscale_array(gradient(32, 24), "photoreal:rich", (128, 96), progress=lambda f, m: None,
                      cancel=threading.Event(), creativity=3)  # fmt: skip


def test_upscale_array_photoreal_unavailable_raises(engines):
    engines.hardware = MAC_8GB  # no silent fallback: a preview must show what really ran
    with pytest.raises(SeedVR2Unavailable):
        _array(gradient(32, 24), "photoreal:balanced", (128, 96))
    assert engines.seedvr2.calls == []


def test_upscale_array_cancel(engines):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        _array(gradient(32, 24), "fast", (128, 96), cancel=cancel)


def test_pipeline_second_pass_always_uses_fast(engines, tmp_path):
    src = write_image(tmp_path / "icon.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="faithful", size="4k")

    assert result.mode_used == "faithful" and (result.width, result.height) == (3840, 2880)
    assert engines.loaded == ["faithful", "fast"]
    assert engines.tiled == [(48, 64, 3)] and engines.tiled_to == [((192, 256, 3), (3840, 2880))]


@pytest.mark.parametrize(
    ("src_size", "size", "tiled", "tiled_to"),
    [
        ((64, 48), "4x", [(48, 64, 3)], []),  # exactly the model's own size: nothing to resize
        ((64, 48), "2x", [], [((48, 64, 3), (128, 96))]),  # smaller than the model's 4x picture
        ((800, 600), "4k", [], [((600, 800, 3), (3840, 2880))]),  # 4.8x: a little past the model's 4x
    ],
)
def test_pipeline_local_one_pass_goes_straight_to_the_size(engines, tmp_path, src_size, size, tiled, tiled_to):
    src = write_image(tmp_path / "room.png", *src_size)

    result, _ = run(src, tmp_path / "out", mode="faithful", size=size)

    assert engines.loaded == ["faithful"]
    assert engines.tiled == tiled and engines.tiled_to == tiled_to
    assert opened_size(result.output_path) == (result.width, result.height)


def test_pipeline_one_pass_16k_never_makes_the_4x_picture(engines, tmp_path):
    src = write_image(tmp_path / "scan.png", 8192, 12)  # thin, so the test stays fast

    result, _ = run(src, tmp_path / "out", mode="fast", size="16k")

    assert (result.width, result.height) == (16384, 24)
    assert engines.tiled == []  # never the 32768-wide picture...
    assert engines.tiled_to == [((12, 8192, 3), (16384, 24))]  # ...straight to 16K instead
    assert opened_size(result.output_path) == (16384, 24)


def test_pipeline_free_space_check_counts_transparency(engines, tmp_path, monkeypatch):
    needed = []
    monkeypatch.setattr(pipeline_image, "ensure_free_space", lambda folder, num_bytes, **_: needed.append(num_bytes))
    rgba = np.dstack([gradient(64, 48), np.full((48, 64), 255, np.uint8)])
    rgba[:, :16, 3] = 0
    src = tmp_path / "logo.png"
    Image.fromarray(rgba).save(src)

    run(src, tmp_path / "out", mode="fast", size="2x", output_format="tiff")

    assert needed == [round(128 * 96 * BYTES_PER_PIXEL["tiff"] * 4 / 3)]  # a fourth channel to store


def test_pipeline_photoreal_unavailable_falls_back(engines, tmp_path):
    engines.seedvr2 = FakeSeedVR2(SeedVR2Unavailable("Photo-real hasn't been downloaded yet."))
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="photoreal", size="2x")

    assert result.mode_used == "faithful"
    assert "Photo-real hasn't been downloaded yet. Used Faithful instead." in result.notes
    assert engines.loaded == ["faithful"]


@pytest.mark.parametrize("when", ["before", "during"])
def test_pipeline_cancel(engines, tmp_path, when):
    src = write_image(tmp_path / "in" / "table.png", 64, 48)
    out_dir = tmp_path / "out"
    cancel = threading.Event()
    if when == "before":
        cancel.set()
    else:
        engines.cancel_during_engine = cancel

    with pytest.raises(Cancelled):
        run(src, out_dir, cancel=cancel, mode="fast", size="2x")

    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_pipeline_cancel_between_passes(engines, tmp_path):
    src = write_image(tmp_path / "in" / "table.png", 512, 385)
    out_dir = tmp_path / "out"
    cancel = threading.Event()
    engines.cancel_during_engine = cancel  # pressed while the first pass finishes

    with pytest.raises(Cancelled):
        run(src, out_dir, cancel=cancel, mode="fast", size="16k")

    assert engines.tiled == [(385, 512, 3)] and engines.tiled_to == []  # the second pass never started
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_model_downloads_can_be_cancelled(engines, tmp_path):
    # A first-time download (140 MB for Faithful) stops when Cancel is pressed, in both passes.
    src = write_image(tmp_path / "in" / "table.png", 512, 385)
    cancel = threading.Event()

    run(src, tmp_path / "out", cancel=cancel, mode="faithful", size="16k")
    upscale_array(gradient(64, 48), "illustration", (256, 192), progress=lambda f, m: None, cancel=cancel)

    assert [model for model, _ in engines.ensured] == ["faithful", "fast", "illustration"]
    assert all(given is cancel for _, given in engines.ensured)


def test_pipeline_cloud_requires_key(engines, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: None)
    src = write_image(tmp_path / "room.png", 64, 48)

    with pytest.raises(ValueError, match="connect fal.ai in Settings"):
        run(src, tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")


def test_pipeline_cloud_two_passes(engines, tmp_path, monkeypatch):
    provider = FakeProvider()  # always gives back 2x
    monkeypatch.setattr(pipeline_image, "get_provider", lambda provider_id: provider)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: "host-key")
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="cloud:fal:topaz_precision", size="4k")

    assert provider.calls == [("room.png", "topaz_precision", 4.0, "host-key")]  # 4x, not the full 60x
    assert engines.tiled_to == [((96, 128, 3), (3840, 2880))] and engines.loaded == ["fast"]
    assert result.mode_used == "cloud-fal" and opened_size(result.output_path) == (3840, 2880)
    assert any("small" in note for note in result.notes)


def test_pipeline_cloud_uses_guest_key(engines, tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(pipeline_image, "get_provider", lambda provider_id: provider)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: "host-key")
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x", guest_key="guest-key")

    assert provider.calls == [("room.png", "topaz_precision", 2.0, "guest-key")]
    assert result.mode_used == "cloud-fal"
    assert result.output_path.name == "room-2x-cloud-fal.jpg"
    assert opened_size(result.output_path) == (128, 96)


def test_pipeline_cloud_result_too_small_gets_the_fast_pass(engines, tmp_path, monkeypatch):
    # A service that sends back less than asked (fal's Topaz stops at 4x, Magnific Creative at about
    # 25 MP) isn't stretched further than a plain resize can go: the Fast model finishes the job.
    provider = FakeProvider(factor=2)
    monkeypatch.setattr(pipeline_image, "get_provider", lambda provider_id: provider)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: "host-key")
    src = write_image(tmp_path / "room.png", 64, 48)

    result, seen = run(src, tmp_path / "out", mode="cloud:fal:topaz_precision", size="4x")

    assert provider.calls == [("room.png", "topaz_precision", 4.0, "host-key")]  # one paid call, never two
    assert engines.tiled_to == [((96, 128, 3), (256, 192))] and engines.loaded == ["fast"]
    assert opened_size(result.output_path) == (256, 192) and result.mode_used == "cloud-fal"
    assert any("smaller picture" in note and "Fast" in note for note in result.notes)
    fractions = [fraction for fraction, _ in seen]
    assert fractions == sorted(fractions)  # the bar never goes backwards


def test_pipeline_cloud_result_a_resize_can_finish_needs_no_fast_pass(engines, tmp_path, monkeypatch):
    provider = FakeProvider(factor=3)  # 1100 px to 4K is 3.49x: 3x, then a 1.16x resize
    monkeypatch.setattr(pipeline_image, "get_provider", lambda provider_id: provider)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: "host-key")
    src = write_image(tmp_path / "room.png", 1100, 825)
    result, _ = run(src, tmp_path / "out", mode="cloud:fal:seedvr2", size="4k")
    assert engines.tiled_to == [] and engines.loaded == []
    assert opened_size(result.output_path) == (3840, 2880)
    assert not any("smaller picture" in note for note in result.notes)


def test_pipeline_cloud_gets_the_opened_picture_not_a_copy(engines, cloud, tmp_path):
    # No temporary file, and a JPEG isn't compressed a second time on its way to the service.
    write_image(tmp_path / "room.jpg", 64, 48, quality=95)
    run(tmp_path / "room.jpg", tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")
    [image] = cloud.images
    assert isinstance(image, LoadedImage) and image.name == "room.jpg"
    np.testing.assert_array_equal(image.rgb, load_image(tmp_path / "room.jpg").rgb)


@pytest.fixture
def cloud(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(pipeline_image, "get_provider", lambda provider_id: provider)
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: "host-key")
    return provider


def test_pipeline_cloud_sends_what_only_ffmpeg_can_open(engines, cloud, tmp_path):
    # Pillow can't open Radiance HDR (nor EXR, JPEG XL or 16-bit PSD); the service gets the picture
    # the app decoded, so it never fails with "Something went wrong".
    write_image(tmp_path / "flat.png", 64, 48)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", tmp_path / "flat.png", tmp_path / "sky.hdr"], check=True)

    result, _ = run(tmp_path / "sky.hdr", tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")

    assert cloud.calls == [("sky.hdr", "topaz_precision", 2.0, "host-key")]
    assert cloud.formats == ["PNG"] and cloud.icc == [None]
    np.testing.assert_array_equal(cloud.received[0], load_image(tmp_path / "sky.hdr").rgb)
    assert opened_size(result.output_path) == (128, 96)


def test_pipeline_cloud_sends_the_developed_raw_not_its_preview(engines, cloud, tmp_path, monkeypatch):
    # Many RAW files start with a small preview, which is all Pillow would see: the service must get
    # the full picture LibRaw developed, or it upscales a thumbnail.
    developed = gradient(64, 48)
    Image.fromarray(gradient(16, 12)).save(tmp_path / "camera.dng", "TIFF")  # just the preview, for Pillow
    monkeypatch.setattr(
        pipeline_image,
        "load_image",
        lambda path: LoadedImage(developed, None, None, "camera.dng", "RAW (DNG)"),
    )

    run(tmp_path / "camera.dng", tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")

    np.testing.assert_array_equal(cloud.received[0], developed)
    assert cloud.formats == ["PNG"]


def test_pipeline_cloud_sends_lossy_sources_upright_as_jpeg(engines, cloud, tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # sideways, as phones save photos
    Image.fromarray(gradient(64, 48)).save(tmp_path / "phone.jpg", quality=95, exif=exif.tobytes())

    run(tmp_path / "phone.jpg", tmp_path / "out", mode="cloud:fal:topaz_precision", size="2x")

    assert cloud.calls[0][0] == "phone.jpg" and cloud.formats == ["JPEG"] and cloud.icc == [None]
    sent = cloud.received[0].astype(np.int16)
    assert sent.shape == (64, 48, 3)  # upright
    assert np.abs(sent - load_image(tmp_path / "phone.jpg").rgb).mean() < 2


def test_pipeline_transparent_png(engines, tmp_path):
    rgba = np.dstack([gradient(64, 48), np.full((48, 64), 255, np.uint8)])
    rgba[:, :16, 3] = 0
    src = tmp_path / "logo.png"
    Image.fromarray(rgba).save(src)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="jpeg")

    assert result.output_path.name == "logo-2x-fast.png"
    with Image.open(result.output_path) as image:
        assert image.mode == "RGBA" and image.size == (128, 96)
        alpha = np.asarray(image)[..., 3]
    assert alpha[:, :24].max() < 10 and alpha[:, 40:].min() > 245  # the edge moved with the picture
    assert any("PNG" in note for note in result.notes)


def test_pipeline_transparent_tiff_stays_tiff(engines, tmp_path):
    rgba = np.dstack([gradient(64, 48), np.full((48, 64), 255, np.uint8)])
    rgba[:, :16, 3] = 0
    src = tmp_path / "logo.png"
    Image.fromarray(rgba).save(src)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="tiff")

    assert result.output_path.name == "logo-2x-fast.tif"
    with Image.open(result.output_path) as image:
        assert image.mode == "RGBA" and image.size == (128, 96)
    assert not any("PNG" in note for note in result.notes)


def test_pipeline_tiff_with_dpi(engines, tmp_path):
    src = write_image(tmp_path / "poster.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="tiff", dpi=600)

    assert result.output_path.name == "poster-2x-fast.tif"
    with Image.open(result.output_path) as image:
        assert image.format == "TIFF" and image.info["dpi"] == (600, 600)
    assert result.dpi == 600
    assert result.print == {"inches": (0.2, 0.2), "cm": (0.5, 0.4), "dpi": 600}
    assert result.file_bytes == result.output_path.stat().st_size > 0


@pytest.mark.parametrize("dpi", [0, 100_000, 72.5])
def test_pipeline_rejects_bad_dpi(engines, tmp_path, dpi):
    src = write_image(tmp_path / "room.png", 64, 48)

    with pytest.raises(ValueError, match="DPI"):
        run(src, tmp_path / "out", mode="fast", size="2x", dpi=dpi)


def test_pipeline_not_enough_disk_space_refuses_before_starting(engines, tmp_path, monkeypatch):
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**12, 10**12, 2_000_000))  # 2 MB free
    src = write_image(tmp_path / "in" / "table.png", 512, 385)
    out_dir = tmp_path / "out"

    with pytest.raises(ImageFileError, match="free space"):
        run(src, out_dir, mode="fast", size="16k", output_format="tiff")  # about 360 MB

    assert engines.tiled == [] and engines.tiled_to == []  # no AI time wasted
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_pipeline_leaves_the_space_reserve_free(engines, tmp_path, monkeypatch):
    # Room for the picture itself isn't enough: a batch would otherwise fill the disk to the last
    # few megabytes, and the computer's other apps (and the job records) need room to save too.
    usage = namedtuple("Usage", "total used free")
    picture = estimated_file_bytes(1024, 770, "tiff")  # 512 x 385 at 2x
    free = {"bytes": picture + SPACE_RESERVE_BYTES - 1}
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**12, 10**12, free["bytes"]))
    src = write_image(tmp_path / "in" / "table.png", 512, 385)

    with pytest.raises(ImageFileError, match="always leaves 2 GB free") as refused:
        run(src, tmp_path / "out", mode="fast", size="2x", output_format="tiff")
    assert "the computer's other apps" in str(refused.value)
    assert engines.tiled == [] and engines.tiled_to == []

    free["bytes"] = picture + SPACE_RESERVE_BYTES + 10**6
    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="tiff")
    assert (result.width, result.height) == (1024, 770)


def test_pipeline_opaque_rgba_saves_jpeg(engines, tmp_path):
    rgba = np.dstack([gradient(64, 48), np.full((48, 64), 255, np.uint8)])
    src = tmp_path / "chatgpt.png"
    Image.fromarray(rgba).save(src)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x")

    assert result.output_path.suffix == ".jpg"
    assert not any("PNG" in note for note in result.notes)


def test_pipeline_exif_rotated_photo(engines, tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6
    src = write_image(tmp_path / "phone.jpg", 80, 40, exif=exif.tobytes())

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x")

    assert (result.width, result.height) == (80, 160)  # upright: portrait


def test_pipeline_batch_same_names_never_overwrite(engines, tmp_path):
    first = write_image(tmp_path / "in" / "0" / "image.png", 64, 48)
    second = tmp_path / "in" / "1" / "image.png"
    second.parent.mkdir(parents=True)
    Image.fromarray(255 - gradient(64, 48)).save(second)
    out_dir = tmp_path / "out"

    a, _ = run(first, out_dir, mode="fast", size="2x")
    b, _ = run(second, out_dir, mode="fast", size="2x")

    assert a.output_path.name == "image-2x-fast.jpg" and b.output_path.name == "image-2x-fast-2.jpg"
    assert a.output_path.read_bytes() != b.output_path.read_bytes()


def test_pipeline_emoji_and_spaces_in_name(engines, tmp_path):
    src = write_image(tmp_path / "Grandma's sofa 🛋️ v2.png", 64, 48)

    result, _ = run(src, tmp_path / "Pictures" / "Upscaled", mode="fast", size="2x", output_format="png")

    assert result.output_path.name == "Grandma's sofa 🛋️ v2-2x-fast.png"
    assert result.output_path.is_file()


def test_pipeline_refuses_gigantic_output(engines, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_image, "MAX_OUTPUT_PIXELS", 100_000)
    src = write_image(tmp_path / "huge.png", 100, 100)  # 4x would be 160,000 pixels

    with pytest.raises(ValueError, match="too big"):
        run(src, tmp_path / "out", mode="fast", size="4x")
    assert engines.tiled == [] and engines.tiled_to == []


# --- vectors ------------------------------------------------------------------------

VECTOR_NOTE = "Vector image: drawn directly at full size, no AI needed"
SVG_BADGE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48" viewBox="0 0 64 48">'
    '<rect width="64" height="48" fill="#3366cc"/><circle cx="32" cy="24" r="16" fill="#ffcc00"/></svg>'
)
SVG_SEE_THROUGH = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48">'
    '<circle cx="32" cy="24" r="16" fill="#e04040"/></svg>'
)


def no_ai_ran(engines) -> bool:
    return engines.tiled == [] and engines.tiled_to == [] and engines.loaded == [] and engines.seedvr2.calls == []


def test_pipeline_vector_drawn_directly(engines, tmp_path):
    src = tmp_path / "badge.svg"
    src.write_text(SVG_BADGE)

    result, seen = run(src, tmp_path / "out", mode="photoreal", size="4k")

    assert (result.width, result.height) == (3840, 2880)
    assert no_ai_ran(engines)
    assert result.mode_used == "vector"
    assert result.notes == [VECTOR_NOTE]
    assert result.output_path.name == "badge-4K-vector.jpg"
    drawn = render_vector(src, (3840, 2880))[..., :3].astype(np.int16)
    saved = _pixels(result.output_path)
    assert saved.shape == drawn.shape and np.abs(saved - drawn).mean() < 1.5  # JPEG
    assert seen[-1][0] == 1.0 and [f for f, _ in seen] == sorted(f for f, _ in seen)


def test_pipeline_vector_bigger_than_the_size_is_drawn_at_that_size(engines, tmp_path):
    # Pictures already bigger than the size are kept as they are, but a vector can be drawn at any
    # size, so "4K" gives 4K.
    src = tmp_path / "poster.svg"
    src.write_text(SVG_BADGE.replace('width="64" height="48"', 'width="5000" height="3750"'))

    result, _ = run(src, tmp_path / "out", mode="fast", size="4k")

    assert (result.width, result.height) == (3840, 2880) and opened_size(result.output_path) == (3840, 2880)
    assert result.mode_used == "vector" and result.notes == [VECTOR_NOTE]
    assert no_ai_ran(engines)


def test_pipeline_vector_needs_no_photoreal_or_cloud(engines, tmp_path, monkeypatch):
    engines.hardware = MAC_8GB  # Photo-real can't run here, but a vector never needs it
    monkeypatch.setattr(pipeline_image.keys, "get_key", lambda provider_id: None)
    src = tmp_path / "badge.svg"
    src.write_text(SVG_BADGE)

    for mode in ("photoreal", "cloud:fal:topaz_precision"):
        result, _ = run(src, tmp_path / "out", mode=mode, size="2x")
        assert result.notes == [VECTOR_NOTE] and result.mode_used == "vector"
    assert no_ai_ran(engines)


def test_pipeline_transparent_vector_keeps_alpha(engines, tmp_path):
    src = tmp_path / "dot.svg"
    src.write_text(SVG_SEE_THROUGH)

    result, _ = run(src, tmp_path / "out", mode="fast", size="3x", grain="medium")

    assert result.output_path.name == "dot-3x-vector.png"
    with Image.open(result.output_path) as image:
        assert image.mode == "RGBA" and image.size == (192, 144)
        alpha = np.asarray(image)[..., 3]
    assert alpha[0, 0] == 0 and alpha[72, 96] == 255
    assert result.notes[0] == VECTOR_NOTE and any("PNG" in note for note in result.notes)


# --- output formats -------------------------------------------------------------------

X_RESOLUTION = 0x011A


def test_pipeline_webp_output(engines, tmp_path):
    src = write_image(tmp_path / "room.png", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="webp", dpi=150)

    assert result.output_path.name == "room-2x-fast.webp"
    with Image.open(result.output_path) as image:
        assert image.format == "WEBP" and image.size == (128, 96)
        assert image.getexif()[X_RESOLUTION] == 150
    assert result.file_bytes == result.output_path.stat().st_size


def test_pipeline_webp_too_big_is_refused_before_upscaling(engines, tmp_path):
    src = write_image(tmp_path / "table.png", 512, 385)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="WebP can't hold pictures bigger than 16,383 pixels"):
        run(src, out_dir, mode="fast", size="16k", output_format="webp")

    assert engines.tiled == [] and engines.tiled_to == []  # no AI time wasted
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_pipeline_same_as_original_keeps_the_format(engines, tmp_path):
    jpeg = write_image(tmp_path / "phone.jpg", 64, 48, quality=90)
    webp = write_image(tmp_path / "web.webp", 64, 48)

    from_jpeg, _ = run(jpeg, tmp_path / "out", mode="fast", size="2x", output_format="original")
    from_webp, _ = run(webp, tmp_path / "out", mode="fast", size="2x", output_format="original")

    assert from_jpeg.output_path.name == "phone-2x-fast.jpg" and from_jpeg.notes == []
    assert from_webp.output_path.name == "web-2x-fast.webp" and from_webp.notes == []
    with Image.open(from_webp.output_path) as image:
        assert image.format == "WEBP"


def test_pipeline_same_as_original_unwritable_format_is_png(engines, tmp_path):
    src = write_image(tmp_path / "scan.bmp", 64, 48)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x", output_format="original")

    assert result.output_path.name == "scan-2x-fast.png"
    assert result.notes == ["Re:Solution can't save BMP files, so this one was saved as PNG."]


def test_pipeline_free_space_check_for_lossy_formats(engines, tmp_path, monkeypatch):
    needed = []
    monkeypatch.setattr(pipeline_image, "ensure_free_space", lambda folder, num_bytes, **_: needed.append(num_bytes))
    src = write_image(tmp_path / "room.webp", 64, 48)

    run(src, tmp_path / "out", mode="fast", size="2x", output_format="webp")
    run(src, tmp_path / "out", mode="fast", size="2x", output_format="original")

    assert needed == [round(128 * 96 * BYTES_PER_PIXEL["jpeg"])] * 2  # about the size of a JPEG


def test_pipeline_keeps_the_notes_from_opening(engines, tmp_path):
    frames = [Image.fromarray(gradient(64, 48)), Image.fromarray(255 - gradient(64, 48))]
    src = tmp_path / "wave.gif"
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=100, loop=0)

    result, _ = run(src, tmp_path / "out", mode="fast", size="2x")

    assert any("animated" in note and "first frame" in note for note in result.notes)


@pytest.mark.parametrize("fmt", ["gif", "heic", ""])
def test_pipeline_rejects_unknown_formats(engines, tmp_path, fmt):
    src = write_image(tmp_path / "room.png", 64, 48)
    with pytest.raises(ValueError, match="output format"):
        run(src, tmp_path / "out", mode="fast", size="2x", output_format=fmt)


# --- real model -------------------------------------------------------------------


def _has_gpu() -> bool:
    return torch.backends.mps.is_available() or torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="needs an Apple or NVIDIA GPU")
def test_real_faithful_crop(tmp_path):
    rng = np.random.default_rng(7)
    base = gradient(256, 256).astype(np.float32)
    texture = rng.normal(0, 12, (64, 64, 1)).repeat(4, 0).repeat(4, 1)
    crop = np.clip(base + texture, 0, 255).astype(np.uint8)
    src = tmp_path / "crop.png"
    Image.fromarray(crop).save(src)

    started = time.monotonic()
    result, _ = run(src, tmp_path / "out", mode="faithful", size="4x")
    elapsed = time.monotonic() - started

    assert (result.width, result.height) == (1024, 1024)
    assert opened_size(result.output_path) == (1024, 1024)
    assert result.mode_used == "faithful" and elapsed < 60
    with Image.open(result.output_path) as image:
        out = np.asarray(image).astype(np.float32)
    lanczos = np.asarray(Image.fromarray(crop).resize((1024, 1024), Image.LANCZOS)).astype(np.float32)
    assert np.abs(out.mean(axis=(0, 1)) - lanczos.mean(axis=(0, 1))).max() < 3  # tone lock held the colours


# --- The compare viewer's tiles (DESIGN "Deep-zoom compare viewer") ---------------------------------


def test_viewer_tiles_are_written_from_the_finished_picture(engines, tmp_path):
    from upscaler.imageio_utils import read_viewer_manifest

    src = write_image(tmp_path / "in" / "table.png", 512, 385)
    viewer = tmp_path / "job" / "results" / "0-viewer"
    seen = []
    result = upscale_image(
        src, ImageOptions(mode="fast", size="4k"), tmp_path / "out",
        progress=lambda f, m: seen.append(f), cancel=threading.Event(), viewer_dir=viewer,
    )  # fmt: skip
    assert result.viewer == viewer
    manifest = read_viewer_manifest(viewer)
    assert (manifest["width"], manifest["height"]) == (3840, 2888) == (result.width, result.height)
    assert (manifest["source_width"], manifest["source_height"]) == (512, 385)
    assert [lv["width"] for lv in manifest["levels"]] == [3840, 1920, 960]
    with Image.open(result.output_path) as saved:
        whole = np.asarray(saved.convert("RGB"))
    with Image.open(viewer / "0" / "3_2.jpg") as tile:
        assert tile.size == (512, 512)
        diff = np.abs(np.asarray(tile.convert("RGB")).astype(int) - whole[1024:1536, 1536:2048].astype(int)).mean()
    assert diff < 3  # the tile is the saved picture's own pixels
    assert seen == sorted(seen)


def test_no_viewer_folder_means_no_tiles(engines, tmp_path):
    result, _ = run(write_image(tmp_path / "room.png", 64, 48), tmp_path / "out", mode="fast", size="2x")
    assert result.viewer is None
    assert not list(tmp_path.rglob("viewer.json"))


def test_tiles_that_cant_be_written_never_fail_the_picture(engines, tmp_path, monkeypatch):
    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(pipeline_image, "write_tile_pyramid", full_disk)
    result = upscale_image(
        write_image(tmp_path / "room.png", 64, 48), ImageOptions(mode="fast", size="2x"), tmp_path / "out",
        progress=lambda f, m: None, cancel=threading.Event(), viewer_dir=tmp_path / "viewer",
    )  # fmt: skip
    assert result.viewer is None and result.output_path.is_file()


def test_transparent_results_are_tiled_on_white(engines, tmp_path):
    rgba = np.zeros((48, 64, 4), np.uint8)
    rgba[..., 0] = 255  # red, fully see-through
    rgba[:, 32:, 3] = 255  # except the right half
    src = tmp_path / "logo.png"
    Image.fromarray(rgba).save(src)
    viewer = tmp_path / "viewer"
    upscale_image(src, ImageOptions(mode="fast", size="2x"), tmp_path / "out",
                  progress=lambda f, m: None, cancel=threading.Event(), viewer_dir=viewer)  # fmt: skip
    with Image.open(viewer / "0" / "0_0.jpg") as tile:
        pixels = np.asarray(tile.convert("RGB")).astype(int)
    assert pixels[:, :50].min() > 235  # white where it's see-through
    assert pixels[:, 80:, 0].min() > 220 and pixels[:, 80:, 1].max() < 40  # red where it isn't


def test_cancelling_while_the_tiles_are_written_stops_the_picture(engines, tmp_path, monkeypatch):
    cancel = threading.Event()

    def cancelled_part_way(*args, **kwargs):
        cancel.set()
        raise Cancelled()

    monkeypatch.setattr(pipeline_image, "write_tile_pyramid", cancelled_part_way)
    with pytest.raises(Cancelled):
        upscale_image(write_image(tmp_path / "room.png", 64, 48), ImageOptions(mode="fast", size="2x"), tmp_path / "out",
                      progress=lambda f, m: None, cancel=cancel, viewer_dir=tmp_path / "viewer")  # fmt: skip
    assert not list((tmp_path / "out").glob("*")) if (tmp_path / "out").exists() else True
