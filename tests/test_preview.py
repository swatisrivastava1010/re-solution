import threading
import weakref

import cv2
import numpy as np
import pytest
from PIL import Image

from upscaler import pipeline_image, preview, sizing
from upscaler.config import Cancelled
from upscaler.hardware import Hardware
from upscaler.models import ModelDownloadError
from upscaler.preview import PREVIEW_ROUTES, pick_region, region_at, run_preview
from upscaler.seedvr2 import SeedVR2Unavailable
from upscaler.tiling import BROKEN_OUTPUT_MESSAGE, BrokenOutput

M4 = Hardware("mps", "Apple M4", 16, "mac", True)
SMALL_MAC = Hardware("mps", "Apple M1", 8, "mac", True)


def textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def smooth(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Soft texture, which JPEG keeps closely."""
    noise = np.random.default_rng(seed).normal(0, 1, (h, w, 3)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), 4)
    return np.clip(128 + 60 * blurred / blurred.std(), 0, 255).astype(np.uint8)


def save(tmp_path, rgb: np.ndarray, name: str = "photo.png"):
    path = tmp_path / name
    Image.fromarray(rgb).save(path)
    return path


class FakeEngines:
    """Stands in for pipeline_image.upscale_array: a plain enlargement, plus a record of each call."""

    def __init__(self, fail: dict[str, Exception] | None = None, cancel_after: threading.Event | None = None):
        self.calls: list[tuple[str, np.ndarray, tuple[int, int]]] = []
        self.fail = fail or {}
        self.cancel_after = cancel_after

    def __call__(self, rgb, route, out_size, *, progress, cancel, creativity=0.0):
        self.calls.append((route, rgb.copy(), tuple(out_size)))
        if route in self.fail:
            raise self.fail[route]
        progress(0.5, "Working")
        if self.cancel_after is not None:
            self.cancel_after.set()
        return np.asarray(Image.fromarray(rgb).resize(tuple(out_size), Image.LANCZOS))


@pytest.fixture
def engines(monkeypatch):
    fake = FakeEngines()
    monkeypatch.setattr(pipeline_image, "upscale_array", fake)
    monkeypatch.setattr(preview, "detect_hardware", lambda: M4)
    return fake


def run(path, tmp_path, routes=PREVIEW_ROUTES, region=None, **kwargs):
    reports = []
    items = run_preview(
        path,
        routes,
        region,
        progress=lambda fraction, message: reports.append((fraction, message)),
        cancel=threading.Event(),
        out_dir=tmp_path / "results",
        **kwargs,
    )
    return items, reports


# --- Choosing the crop -------------------------------------------------------------------


def test_pick_region_finds_detail():
    rgb = np.full((800, 1000, 3), 120, np.uint8)
    rgb[300:400, 600:700] = textured(100, 100)
    x, y, w, h = pick_region(rgb)
    assert (w, h) == (256, 256)
    assert x <= 600 and x + w >= 700 and y <= 300 and y + h >= 400


def test_pick_region_flat_picture_uses_the_middle():
    x, y, w, h = pick_region(np.full((600, 900, 3), 80, np.uint8))
    assert (x + w / 2, y + h / 2) == pytest.approx((450, 300), abs=2)


def test_pick_region_small_picture_is_whole():
    assert pick_region(textured(150, 200)) == (0, 0, 200, 150)


def test_region_at_keeps_box_inside():
    assert region_at(1000, 800, 500, 400) == (372, 272, 256, 256)
    assert region_at(1000, 800, 5, 790) == (0, 544, 256, 256)
    assert region_at(200, 150, 100, 70) == (0, 0, 200, 150)


# --- Running the styles ------------------------------------------------------------------


def test_preview_job_runs_routes(tmp_path, engines):
    rgb = textured(900, 1200)
    items, reports = run(save(tmp_path, rgb), tmp_path)

    assert [item["route"] for item in items] == ["original", *PREVIEW_ROUTES]
    assert [route for route, _, _ in engines.calls] == PREVIEW_ROUTES
    for _, crop, out_size in engines.calls:
        assert crop.shape == (256, 256, 3)
        assert out_size == (1024, 1024)
    for item in items:
        assert set(item) >= {"route", "label", "crop_path", "seconds", "est_full_seconds"}
        with Image.open(item["crop_path"]) as crop:
            assert (crop.size, crop.format) == ((1024, 1024), "JPEG")
        assert item["seconds"] >= 0
    assert items[0]["label"] == "Original (enlarged)"
    assert items[0]["est_full_seconds"] is None
    assert items[1]["label"] == "Photo-real · Balanced"
    assert reports[-1][0] == 1.0
    assert [f for f, _ in reports] == sorted(f for f, _ in reports)


def test_original_is_the_crop_enlarged(tmp_path, engines):
    rgb = smooth(600, 600, seed=3)
    items, _ = run(save(tmp_path, rgb), tmp_path, routes=["fast"], region=(100, 50, 256, 256))
    expected = Image.fromarray(rgb[50:306, 100:356]).resize((1024, 1024), Image.LANCZOS)
    with Image.open(items[0]["crop_path"]) as original:
        diff = np.abs(np.asarray(original, np.int16) - np.asarray(expected, np.int16))
    assert diff.mean() < 2  # JPEG
    assert np.array_equal(engines.calls[0][1], rgb[50:306, 100:356])


def test_full_image_estimates_use_the_size_and_level(tmp_path, engines):
    rgb = textured(1539, 2048)
    items, _ = run(save(tmp_path, rgb), tmp_path, size="8k")
    by_route = {item["route"]: item["est_full_seconds"] for item in items}
    for route in PREVIEW_ROUTES:
        mode, _, level = route.partition(":")
        expected = sizing.estimate(2048, 1539, "8k", mode, "jpeg", M4, level=level or "balanced")["seconds"]
        assert by_route[route] == expected
    assert by_route["photoreal:balanced"] < by_route["photoreal:rich"]
    assert by_route["fast"] < by_route["faithful"]


def test_preview_tiny_picture(tmp_path, engines):
    items, _ = run(save(tmp_path, textured(90, 120)), tmp_path, routes=["fast"])
    assert engines.calls[0][2] == (480, 360)
    with Image.open(items[1]["crop_path"]) as crop:
        assert crop.size == (480, 360)


def test_preview_region_is_clamped(tmp_path, engines):
    run(save(tmp_path, textured(400, 500)), tmp_path, routes=["fast"], region=(400, 300, 256, 256))
    crop = engines.calls[0][1]
    assert crop.shape == (256, 256, 3)


def test_preview_transparent_picture_shown_on_white(tmp_path, engines):
    rgba = np.zeros((300, 300, 4), np.uint8)
    rgba[..., 3] = 0  # fully see-through
    path = tmp_path / "clear.png"
    Image.fromarray(rgba, "RGBA").save(path)
    run(path, tmp_path, routes=["fast"])
    assert engines.calls[0][1].min() == 255


def test_photoreal_unavailable_marks_those_styles(tmp_path, engines):
    engines.fail = {route: SeedVR2Unavailable("Photo-real needs 16 GB of memory.") for route in PREVIEW_ROUTES[:3]}
    items, _ = run(save(tmp_path, textured(400, 400)), tmp_path)
    by_route = {item["route"]: item for item in items}
    for route in PREVIEW_ROUTES[:3]:
        assert by_route[route]["crop_path"] is None
        assert by_route[route]["error"] == "Photo-real needs 16 GB of memory."
        assert by_route[route]["est_full_seconds"] is None  # no time for a style that didn't run
    assert by_route["fast"]["crop_path"] and "error" not in by_route["fast"]
    assert by_route["fast"]["est_full_seconds"] > 0


def test_only_messages_written_for_people_reach_the_cards(tmp_path, engines):
    offline = "Couldn't download the Fast model. Check that this computer is connected to the internet, then try again."
    engines.fail = {
        "faithful": ValueError("4xBHI_dat2_real.safetensors is not an image upscaling model"),
        "fast": ModelDownloadError(offline),
        "illustration": BrokenOutput(),
    }
    items, _ = run(save(tmp_path, textured(400, 400)), tmp_path, routes=["faithful", "fast", "illustration"])
    by_route = {item["route"]: item for item in items}
    assert by_route["faithful"]["error"] == "This style couldn't be previewed. Please try again."
    assert by_route["fast"]["error"] == offline
    assert by_route["illustration"]["error"] == BROKEN_OUTPUT_MESSAGE  # not "try again": it would break again
    assert by_route["faithful"]["crop_path"] is None and by_route["fast"]["crop_path"] is None


def test_photoreal_asked_for_where_it_cant_run_has_no_estimate(tmp_path, engines, monkeypatch):
    # sizing.estimate would quietly time Faithful instead, which isn't what this style shows.
    monkeypatch.setattr(preview, "detect_hardware", lambda: SMALL_MAC)
    items, _ = run(save(tmp_path, textured(400, 400)), tmp_path, routes=["photoreal:balanced", "faithful"])
    by_route = {item["route"]: item["est_full_seconds"] for item in items}
    assert by_route["photoreal:balanced"] is None
    assert by_route["faithful"] > 0


def test_default_routes_skip_photoreal_where_it_cant_run(tmp_path, engines, monkeypatch):
    monkeypatch.setattr(preview, "detect_hardware", lambda: SMALL_MAC)
    items, _ = run(save(tmp_path, textured(400, 400)), tmp_path, routes=None)
    assert [item["route"] for item in items] == ["original", "faithful", "fast"]


def test_whole_picture_is_let_go_before_the_styles_run(tmp_path, engines, monkeypatch):
    # At 16K the decoded picture is about 600 MB; only the crop is needed once it's cut out.
    loaded = []

    def load_and_watch(path):
        image = preview_load_image(path)
        loaded.append(weakref.ref(image.rgb))
        return image

    preview_load_image = preview.load_image
    monkeypatch.setattr(preview, "load_image", load_and_watch)
    still_held = []
    fake_engine = pipeline_image.upscale_array

    def engine(rgb, route, out_size, **kwargs):
        still_held.append(loaded[0]() is not None)
        return fake_engine(rgb, route, out_size, **kwargs)

    monkeypatch.setattr(pipeline_image, "upscale_array", engine)
    run(save(tmp_path, textured(700, 900)), tmp_path, routes=["faithful", "fast"])
    assert still_held == [False, False]


def test_unknown_size_is_refused_before_any_work(tmp_path, engines):
    with pytest.raises(ValueError, match="size"):
        run(save(tmp_path, textured(300, 300)), tmp_path, size="99k")
    assert engines.calls == []
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("creativity", [-0.1, 1.5, float("nan")])
def test_creativity_out_of_range_is_refused_before_any_work(tmp_path, engines, creativity):
    with pytest.raises(ValueError, match="Creativity must be between 0 and 1"):
        run(save(tmp_path, textured(300, 300)), tmp_path, routes=["photoreal:rich"], creativity=creativity)
    assert engines.calls == []
    assert not (tmp_path / "results").exists()


def test_unknown_route_is_refused(tmp_path, engines):
    with pytest.raises(ValueError, match="cloud"):
        run(save(tmp_path, textured(300, 300)), tmp_path, routes=["cloud:fal:topaz"])
    assert engines.calls == []


def test_preview_cancel(tmp_path, monkeypatch):
    cancel = threading.Event()
    monkeypatch.setattr(pipeline_image, "upscale_array", FakeEngines(cancel_after=cancel))
    monkeypatch.setattr(preview, "detect_hardware", lambda: M4)
    with pytest.raises(Cancelled):
        run_preview(
            save(tmp_path, textured(300, 300)),
            PREVIEW_ROUTES,
            None,
            progress=lambda f, m: None,
            cancel=cancel,
            out_dir=tmp_path / "results",
        )


def test_creativity_is_passed_to_photoreal(tmp_path, monkeypatch):
    seen = []

    def engine(rgb, route, out_size, *, progress, cancel, creativity=0.0):
        seen.append((route, creativity))
        return np.zeros((out_size[1], out_size[0], 3), np.uint8)

    monkeypatch.setattr(pipeline_image, "upscale_array", engine)
    monkeypatch.setattr(preview, "detect_hardware", lambda: M4)
    run(save(tmp_path, textured(300, 300)), tmp_path, routes=["photoreal:rich", "fast"], creativity=0.4)
    assert seen == [("photoreal:rich", 0.4), ("fast", 0.0)]


def test_preview_problems_are_user_errors(tmp_path, engines):
    from upscaler.config import UserError

    picture = save(tmp_path, textured(300, 300))
    with pytest.raises(UserError, match="Click on the picture again"):
        run(picture, tmp_path, region=(0, 0, 0, 10))
    with pytest.raises(UserError, match="Click on the picture again"):
        run(picture, tmp_path, region=("a", 0, 10, 10))
    with pytest.raises(UserError, match="Creativity"):
        run(picture, tmp_path, creativity=2.0)
    with pytest.raises(UserError, match="can't be previewed"):
        run(picture, tmp_path, routes=["cloud:fal:topaz"])
