import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from upscaler import config, models, suggest
from upscaler.imageio_utils import ImageInfo
from upscaler.models import MODELS, ModelDownloadError
from upscaler.suggest import Signals, analyze

SAMPLES = Path(__file__).resolve().parents[1] / "upscaler" / "web" / "samples"
# Real photos that come with macOS; the tests that use them are skipped elsewhere.
TAHOE_PHOTO = Path("/System/Library/Wallpapers/.default/DefaultAerial.jpg")  # 3840 px, from a camera
ACCOUNT_PICTURES = Path("/Library/User Pictures")


# --- Pictures made for the tests ---------------------------------------------------------


def info_for(rgb: np.ndarray, *, is_vector: bool = False) -> ImageInfo:
    h, w = rgb.shape[:2]
    return ImageInfo(w, h, "PNG", False, 1, is_vector, [])


def natural_photo(w: int, h: int, seed: int = 0) -> np.ndarray:
    """A photo-like picture: random texture with the 1/f spectrum of real scenes, so detail at every scale."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    falloff = 1.0 / np.maximum(np.hypot(fx, fy), 1.0 / max(w, h)) ** 1.1
    channels = []
    base = np.fft.irfft2(falloff * np.exp(2j * np.pi * rng.random(falloff.shape)), s=(h, w))
    for _ in range(3):
        tint = np.fft.irfft2(falloff * np.exp(2j * np.pi * rng.random(falloff.shape)), s=(h, w))
        channels.append(base + 0.35 * tint)
    img = np.stack(channels, axis=-1)
    img = (img - img.mean()) / img.std()
    return np.clip(128 + 42 * img, 0, 255).astype(np.uint8)


def with_crosshatch(rgb: np.ndarray, period: float, amplitude: float = 3.0) -> np.ndarray:
    """Adds the faint diagonal crosshatch that AI image generators leave."""
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    hatch = np.cos(2 * np.pi * (x + y) / period) + np.cos(2 * np.pi * (x - y) / period)
    return np.clip(rgb + amplitude / 2 * hatch[..., None], 0, 255).astype(np.uint8)


def cartoon(w: int = 1200, h: int = 900, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (w, h), (250, 236, 200))
    draw = ImageDraw.Draw(image)
    palette = [(220, 60, 60), (60, 120, 220), (250, 200, 40), (90, 180, 90), (240, 150, 190)]
    for i in range(14):
        x, y, s = int(rng.integers(0, w - 200)), int(rng.integers(0, h - 200)), int(rng.integers(80, 300))
        shape = draw.ellipse if i % 2 else draw.rectangle
        shape((x, y, x + s, y + s * 0.8), fill=palette[i % len(palette)], outline=(20, 20, 20), width=6)
    return np.asarray(image)


def poster_with_text(w: int = 1600, h: int = 1200) -> np.ndarray:
    image = Image.fromarray(natural_photo(w, h, seed=3))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.rectangle((100, 100, 1300, 520), fill=(245, 245, 240))
    for i, line in enumerate(["Summer sale on all sofas", "Free delivery this week", "Visit our store today"]):
        draw.text((130, 130 + i * 95), line, fill=(20, 20, 30), font=font)
    return np.asarray(image)


def page_of_text(w: int = 2048, h: int = 1536, size: int = 16) -> np.ndarray:
    """A scanned letter or document: lines of small type from edge to edge."""
    words = "the quick brown fox jumps over a lazy dog while seven bakers quietly mix dough".split()
    rng = np.random.default_rng(size)
    image = Image.new("RGB", (w, h), (252, 252, 248))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=size)
    for y in range(10, h - size, round(size * 1.3)):
        draw.text((10, y), " ".join(rng.choice(words, 60)), fill=(20, 20, 20), font=font)
    return np.asarray(image)


def gradient(w: int = 2048, h: int = 1536, noise: float = 0.0) -> np.ndarray:
    """A smooth colour gradient, as in a studio backdrop or a designer's background."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([30 + 200 * x / w, 60 + 120 * y / h, 50 + 150 * (x + y) / (w + h)], axis=-1)
    rgb += np.random.default_rng(0).normal(0, noise, rgb.shape) if noise else 0
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def enlarged(rgb: np.ndarray, factor: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    return np.asarray(Image.fromarray(rgb).resize((w * factor, h * factor), Image.LANCZOS))


def as_jpeg(rgb: np.ndarray, quality: int, tmp_path) -> np.ndarray:
    path = tmp_path / "squashed.jpg"
    Image.fromarray(rgb).save(path, quality=quality)
    return np.asarray(Image.open(path).convert("RGB"))


def signals(**overrides) -> Signals:
    """Signals of an ordinary mid-size photo; tests change the ones they're about."""
    plain = Signals(
        faces=0,
        face_area=0.0,
        mesh_score=3.0,  # what natural detail scores
        illustration_score=0.0,
        text_score=0.0,
        sharpness=50.0,
        blockiness=1.0,
        long_edge=2048,
        is_vector=False,
    )
    return replace(plain, **overrides)


MESH = suggest.MESH_LIKELY + 5  # a clear AI mesh


def picture(long_edge: int = 2048, is_vector: bool = False) -> ImageInfo:
    return ImageInfo(long_edge, long_edge * 3 // 4, "SVG" if is_vector else "JPEG", False, 1, is_vector, [])


@pytest.fixture(autouse=True)
def no_face_model(monkeypatch):
    """Tests never download YuNet; the ones about faces fake the detector."""
    monkeypatch.setattr(suggest, "detect_faces", lambda rgb: [])


@pytest.fixture(scope="module")
def photo_2k():
    return natural_photo(2048, 1536)


# --- The rules, on measured signals ------------------------------------------------------


def test_suggest_illustration():
    rgb = cartoon()
    found = analyze(rgb)
    assert found.illustration_score >= suggest.ILLUSTRATION_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert (result.mode, result.photoreal_level) == ("illustration", None)
    assert result.signals is found
    assert any("drawing" in reason for reason in result.reasons)


def test_suggest_faces_faithful(monkeypatch, photo_2k):
    monkeypatch.setattr(suggest, "detect_faces", lambda rgb: [(0.30, 0.20, 0.35, 0.45)])  # 16% of the picture
    found = analyze(photo_2k)
    assert found.faces == 1
    assert found.face_area == pytest.approx(0.1575, abs=1e-3)
    result = suggest.suggest(info_for(photo_2k), found)
    assert (result.mode, result.photoreal_level) == ("faithful", None)
    assert any("Photo-real Balanced" in reason for reason in result.reasons)  # the tip from the design


@pytest.mark.parametrize("period", [2, 3, 4])
def test_suggest_mesh_photoreal(period):
    rgb = with_crosshatch(natural_photo(2048, 1536, seed=period), period)  # an AI picture at its own size
    found = analyze(rgb)
    assert found.mesh_score >= suggest.MESH_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert (result.mode, result.photoreal_level) == ("photoreal", "balanced")
    assert any("grid-like texture" in reason for reason in result.reasons)


@pytest.mark.parametrize("period", [2, 3, 4])
def test_enlarged_ai_picture_keeps_its_mesh(period):
    # Large and sharp, so it would go to Faithful for sharp photos, but the mesh is still there.
    rgb = enlarged(with_crosshatch(natural_photo(1600, 1200, seed=period), period), 2)
    found = analyze(rgb)
    assert (found.long_edge, found.sharpness >= suggest.SHARP) == (3200, True)
    assert found.mesh_score >= suggest.MESH_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert (result.mode, result.photoreal_level) == ("photoreal", "balanced")
    assert any("grid-like texture" in reason for reason in result.reasons)


@pytest.mark.skipif(not TAHOE_PHOTO.is_file(), reason="needs the macOS Tahoe wallpaper")
def test_large_camera_photo_is_not_mesh():
    # At its own 3840 px this photo has a fine pattern that scored like the mesh (26.8) and made
    # Re:Solution call a real photo AI-made; measured on a 2048 px copy, it scores about 5.
    rgb = np.asarray(Image.open(TAHOE_PHOTO).convert("RGB"))
    found = analyze(rgb)
    assert found.mesh_score < suggest.MESH_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert result.mode == "faithful_sharp"
    assert not any("grid-like texture" in reason for reason in result.reasons)


@pytest.mark.parametrize("noise", [0.0, 1.0])
def test_smooth_gradient_is_not_mesh(noise):
    # Its 8-bit steps repeat as regularly as the mesh does, but there's no texture to speak of.
    rgb = gradient(noise=noise)
    found = analyze(rgb)
    assert found.mesh_score == 0.0
    reasons = suggest.suggest(info_for(rgb), found).reasons
    assert not any("grid-like texture" in reason for reason in reasons)


def test_plain_photo_has_no_mesh(photo_2k):
    assert analyze(photo_2k).mesh_score < suggest.MESH_LIKELY


def test_jpeg_blocks_are_not_mesh(tmp_path, photo_2k):
    found = analyze(as_jpeg(photo_2k, 40, tmp_path))
    assert found.blockiness >= suggest.BLOCKY
    assert found.mesh_score < suggest.MESH_LIKELY


def test_suggest_sharp_large_photo():
    rgb = natural_photo(3200, 2400, seed=7)
    found = analyze(rgb)
    assert found.long_edge == 3200
    assert found.sharpness >= suggest.SHARP
    assert found.mesh_score < suggest.MESH_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert (result.mode, result.photoreal_level) == ("faithful_sharp", None)


def test_soft_large_photo_gets_photoreal():
    small = Image.fromarray(natural_photo(800, 600, seed=8))
    rgb = np.asarray(small.resize((3200, 2400), Image.BICUBIC))  # big, but soft at 100%
    found = analyze(rgb)
    assert found.sharpness < suggest.SHARP
    result = suggest.suggest(info_for(rgb), found)
    assert (result.mode, result.photoreal_level) == ("photoreal", "balanced")


def test_jpeg_cartoon_is_still_an_illustration(tmp_path):
    for quality in (60, 85):
        assert analyze(as_jpeg(cartoon(seed=3), quality, tmp_path)).illustration_score >= suggest.ILLUSTRATION_LIKELY


@pytest.mark.parametrize("name", ["Sports/Soccer.heic", "Sports/8ball.heic"])
def test_photo_of_an_object_on_a_plain_background_is_not_an_illustration(name):
    # Product photos cut out on a plain background are everyday work for designers.
    path = ACCOUNT_PICTURES / name
    if not path.is_file():
        pytest.skip("needs the macOS account pictures")
    rgb = np.asarray(Image.open(path).convert("RGB"))
    assert analyze(rgb).illustration_score < suggest.ILLUSTRATION_LIKELY


def test_suggest_text_faithful():
    rgb = poster_with_text()
    found = analyze(rgb)
    assert found.text_score >= suggest.TEXT_LIKELY
    result = suggest.suggest(info_for(rgb), found)
    assert result.mode == "faithful"
    assert any("text" in reason for reason in result.reasons)


def test_photo_has_no_text_or_illustration(photo_2k):
    found = analyze(photo_2k)
    assert found.text_score < suggest.TEXT_LIKELY
    assert found.illustration_score < suggest.ILLUSTRATION_LIKELY


def test_vector_is_drawn_directly():
    result = suggest.suggest(picture(is_vector=True), signals(illustration_score=1.0))
    assert result.mode == "fast"  # vectors skip AI whatever the mode; Fast keeps the estimate honest
    assert any("drawn directly" in reason for reason in result.reasons)


def test_rules_apply_in_order():
    def mode(long_edge=2048, **overrides):
        result = suggest.suggest(picture(long_edge), signals(**overrides))
        return result.mode, result.photoreal_level

    everything = dict(illustration_score=0.9, face_area=0.2, faces=1, text_score=0.05, mesh_score=MESH)
    assert mode(**everything) == ("illustration", None)
    assert mode(**{**everything, "illustration_score": 0.0}) == ("faithful", None)  # faces
    assert mode(text_score=0.05, mesh_score=MESH) == ("faithful", None)  # text beats mesh
    assert mode(mesh_score=MESH, sharpness=900.0, long_edge=4000) == ("photoreal", "balanced")
    assert mode(long_edge=900) == ("photoreal", "balanced")  # small picture
    assert mode(sharpness=900.0, long_edge=4000) == ("faithful_sharp", None)
    assert mode(sharpness=900.0, long_edge=2999) == ("photoreal", "balanced")  # not large
    assert mode(sharpness=900.0, long_edge=4000, blockiness=1.6) == ("photoreal", "balanced")  # not clean
    assert mode() == ("photoreal", "balanced")


def test_small_faces_dont_force_faithful():
    result = suggest.suggest(picture(), signals(faces=3, face_area=0.02))
    assert result.mode == "photoreal"


def test_reasons_are_plain_sentences():
    cases = [
        signals(illustration_score=0.9),
        signals(faces=1, face_area=0.2),
        signals(text_score=0.05),
        signals(mesh_score=MESH, long_edge=900),
        signals(sharpness=900.0, long_edge=4000),
        signals(blockiness=1.6),
        signals(),
    ]
    for found in cases:
        reasons = suggest.suggest(picture(found.long_edge), found).reasons
        assert reasons
        assert all(reason.endswith(".") and reason[0].isupper() for reason in reasons)


# --- Faces -----------------------------------------------------------------------------


def test_faces_skipped_when_model_cant_download(monkeypatch, tmp_path, photo_2k):
    monkeypatch.undo()  # the real detect_faces this time
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    attempts = []

    def offline(spec, progress=None):
        attempts.append(spec.id)
        raise ModelDownloadError("Couldn't download the face finder model.")

    monkeypatch.setattr(suggest, "ensure_model", offline)
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())

    started = time.monotonic()
    found = analyze(photo_2k)
    again = analyze(photo_2k)
    assert time.monotonic() - started < 5
    assert (found.faces, found.face_area, again.faces) == (0, 0.0, 0)
    assert attempts == ["face_yunet"]  # tried once, not on every picture
    reasons = suggest.suggest(info_for(photo_2k), found).reasons
    assert not any("face" in reason.lower() for reason in reasons)


def test_only_the_first_picture_waits_for_the_face_model(monkeypatch, tmp_path):
    # The download can queue behind a big model download; later pictures shouldn't wait for it too.
    monkeypatch.undo()
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    release, attempts = threading.Event(), []

    def slow(spec, progress=None):
        attempts.append(spec.id)
        release.wait(10)
        raise ModelDownloadError("Couldn't download the face finder model.")

    monkeypatch.setattr(suggest, "ensure_model", slow)
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())
    monkeypatch.setattr(suggest, "FACE_MODEL_WAIT_SECONDS", 0.3)
    blank = np.zeros((300, 400, 3), np.uint8)
    try:
        started = time.monotonic()
        assert suggest.detect_faces(blank) == []
        first = time.monotonic() - started
        started = time.monotonic()
        assert suggest.detect_faces(blank) == []
        second = time.monotonic() - started
    finally:
        release.set()
    assert first >= 0.3 and second < 0.1
    assert attempts == ["face_yunet"]


def test_face_model_download_doesnt_hold_up_models_already_here(monkeypatch, tmp_path):
    # Offline, the face model's tries take seconds (behind a silent firewall, minutes); a job
    # loading a model that's already on this computer mustn't wait for them.
    monkeypatch.undo()
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    started, release = threading.Event(), threading.Event()

    def hanging(spec, part, progress, cancel):
        started.set()
        release.wait(10)
        raise ModelDownloadError("Couldn't download the face finder model.")

    monkeypatch.setattr(models, "_download_with_retries", hanging)
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())
    monkeypatch.setattr(suggest, "FACE_MODEL_WAIT_SECONDS", 0.0)
    fast = MODELS["fast"]
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / fast.file).write_bytes(b"\0" * fast.size_bytes)
    try:
        assert suggest.detect_faces(np.zeros((300, 400, 3), np.uint8)) == []
        assert started.wait(5)  # the face model download is under way
        begun = time.monotonic()
        assert models.ensure_model(fast) == tmp_path / "models" / fast.file
        assert time.monotonic() - begun < 0.5
    finally:
        release.set()
        suggest._faces._download.join(5)


class FailingYuNet:
    """Stands in for cv2.FaceDetectorYN with a model OpenCV can't read."""

    def __init__(self):
        self.calls = []

    def create(self, *args):
        self.calls.append(args)
        raise cv2.error("unreadable model")


def complete_face_model(tmp_path) -> None:
    spec = MODELS["face_yunet"]
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / spec.file).write_bytes(b"\0" * spec.size_bytes)


def test_face_model_is_loaded_from_memory(monkeypatch, tmp_path):
    # OpenCV can't open files under a Windows profile folder such as C:\Users\Zoë, so the bytes are passed.
    monkeypatch.undo()
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    complete_face_model(tmp_path)
    factory = FailingYuNet()
    monkeypatch.setattr(suggest.cv2, "FaceDetectorYN", factory)
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())
    assert suggest._faces.detector() is None
    framework, model, config_bytes = factory.calls[0][:3]
    assert framework == "onnx"
    assert model.dtype == np.uint8 and model.size == MODELS["face_yunet"].size_bytes
    assert config_bytes.size == 0


def test_failed_face_model_load_is_retried_later(monkeypatch, tmp_path):
    monkeypatch.undo()
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    complete_face_model(tmp_path)
    factory = FailingYuNet()
    monkeypatch.setattr(suggest.cv2, "FaceDetectorYN", factory)
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())
    blank = np.zeros((300, 400, 3), np.uint8)
    assert suggest.detect_faces(blank) == suggest.detect_faces(blank) == []
    assert len(factory.calls) == 1  # not again for every picture
    suggest._faces._retry_at = time.monotonic() - 1  # ten minutes later
    assert suggest.detect_faces(blank) == []
    assert len(factory.calls) == 2


def test_detect_faces_merges_scales(monkeypatch):
    class FakeYuNet:
        """Finds one face covering the middle of whatever size it is given."""

        def setInputSize(self, size):
            self.size = size

        def detect(self, bgr):
            w, h = self.size
            assert bgr.shape[:2] == (h, w)
            face = [w * 0.25, h * 0.25, w * 0.5, h * 0.5] + [0.0] * 10 + [0.95]
            return 1, np.array([face], np.float32)

    monkeypatch.undo()
    monkeypatch.setattr(suggest._faces, "detector", lambda: FakeYuNet())
    boxes = suggest.detect_faces(np.zeros((900, 1200, 3), np.uint8))
    assert len(boxes) == 1  # the same face at both scales counts once
    x, y, w, h = boxes[0]
    assert (x, y, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5), abs=0.01)


@pytest.mark.network
def test_real_face_model_loads(monkeypatch, tmp_path):
    monkeypatch.undo()
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "Zoë"))  # a name OpenCV can't open on Windows
    monkeypatch.setattr(suggest, "_faces", suggest._FaceFinder())
    monkeypatch.setattr(suggest, "FACE_MODEL_WAIT_SECONDS", 60.0)
    rgb = np.asarray(Image.open(SAMPLES / "old-photo.jpg").convert("RGB"))
    boxes = suggest.detect_faces(rgb)
    assert suggest._faces.detector() is not None
    assert all(0 <= v <= 1 for box in boxes for v in box)


# --- Speed and odd inputs ----------------------------------------------------------------


def test_analyze_is_quick_on_a_2k_picture(photo_2k):
    analyze(photo_2k)  # warm up OpenCV
    started = time.perf_counter()
    analyze(photo_2k)
    assert time.perf_counter() - started < 1.5


def test_analyze_is_quick_on_a_page_of_text():
    # Scanned letters, menus and screenshots of spreadsheets are full of letters to group into lines.
    rgb = page_of_text()
    analyze(rgb[:300, :300])  # warm up OpenCV
    started = time.perf_counter()
    found = analyze(rgb)
    assert time.perf_counter() - started < 1.5
    assert found.text_score >= suggest.TEXT_LIKELY


def test_analyze_reports_no_faces_as_a_float(photo_2k):
    found = analyze(photo_2k[:256, :256])
    assert (found.faces, found.face_area) == (0, 0.0)
    assert isinstance(found.face_area, float)


@pytest.mark.parametrize("shape", [(5, 7, 3), (40, 30, 3), (120, 2000, 3)])
def test_analyze_tiny_or_thin_pictures(shape):
    found = analyze(np.full(shape, 128, np.uint8))
    assert found.long_edge == max(shape[:2])
    assert found.mesh_score == 0.0 and found.text_score == 0.0 and found.faces == 0


def test_analyze_ignores_alpha_channel(photo_2k):
    rgba = np.dstack([photo_2k[:512, :512], np.full((512, 512), 255, np.uint8)])
    assert analyze(rgba) == analyze(photo_2k[:512, :512])


def test_public_samples_get_sensible_suggestions():
    expected = {
        "illustration.jpg": "illustration",
        "interior.jpg": "photoreal",
        "food.jpg": "photoreal",
        "product.jpg": "photoreal",
        "old-photo.jpg": "photoreal",  # faces are faked away here; the real face finder may say Faithful
    }
    for name, mode in expected.items():
        rgb = np.asarray(Image.open(SAMPLES / name).convert("RGB"))
        result = suggest.suggest(info_for(rgb), analyze(rgb))
        assert result.mode == mode, (name, result)
