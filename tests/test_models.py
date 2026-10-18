import hashlib
import threading
import time
from collections import namedtuple
from dataclasses import replace

import httpx
import numpy as np
import pytest
import respx
import torch

from upscaler import config, models
from upscaler.config import Cancelled
from upscaler.models import (
    MODELS,
    ModelDownloadError,
    ModelSpec,
    choose_dtype,
    ensure_model,
    load_model,
)

URL = "https://models.example.com/tiny.safetensors"
CONTENT = np.random.default_rng(7).integers(0, 256, 300_000, dtype=np.uint8).tobytes()


def make_spec(content: bytes = CONTENT, **overrides) -> ModelSpec:
    spec = ModelSpec(
        id="tiny",
        label="Tiny",
        file="tiny.safetensors",
        url=URL,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        scale=4,
        tile=512,
        license="MIT",
        license_url="https://example.com/license",
        author="Test",
        source_url="https://example.com",
    )
    return replace(spec, **overrides)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(models, "_RETRY_DELAY_SECONDS", 0)
    return tmp_path / "models"


def leftovers(directory):
    return sorted(p.name for p in directory.iterdir())


# --- Registry ---------------------------------------------------------------


UPSCALERS = {"faithful", "faithful_sharp", "fast", "illustration", "video_general"}


def test_registry_has_every_local_model():
    assert set(MODELS) == UPSCALERS | {"face_yunet"}
    for key, spec in MODELS.items():
        assert spec.id == key
        assert spec.url.startswith("https://")
        assert spec.license_url.startswith("https://")
        assert spec.source_url.startswith("https://")
        assert len(spec.sha256) == 64
        assert spec.scale == (4 if key in UPSCALERS else 1)
        assert spec.author and spec.label and spec.license


def test_face_model_is_pinned_yunet():
    # OpenCV Zoo's YuNet (MIT), pinned to one commit and checked by SHA-256 like the upscalers.
    face = MODELS["face_yunet"]
    assert face.file == "face_detection_yunet_2023mar.onnx"
    assert "/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/" in face.url
    assert face.sha256 == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert face.size_bytes == 232589
    assert (face.scale, face.tile, face.license) == (1, 0, "MIT")


def test_every_download_is_pinned():
    # Each URL names one fixed upload (a release tag or a commit), never a branch that can move, so a
    # re-upload can't make every new install fail its SHA-256 check.
    for spec in MODELS.values():
        assert "/main/" not in spec.url and "/master/" not in spec.url, spec.id
    assert "/Kim2091/ClearRealityV1/resolve/56f9841c86d054ea19f820341c8f66f0fcaa16af/" in MODELS["fast"].url


def test_registry_values_match_research():
    fast = MODELS["fast"]
    assert fast.file == "4x-ClearRealityV1.safetensors"
    assert fast.sha256 == "83a9c8e279e3d07548e2b0d736dc2c73c1736a07072a748ed4fafbba99b44693"
    assert fast.size_bytes == 4492232
    assert fast.license == "Apache-2.0"
    # DAT2 models need small tiles to stay within memory.
    assert MODELS["faithful"].tile == 256
    assert MODELS["faithful_sharp"].tile == 256
    assert MODELS["fast"].tile == 512
    assert MODELS["faithful"].license == "CC-BY-4.0"


# --- Download ---------------------------------------------------------------


@respx.mock
def test_ensure_model_downloads_and_verifies(models_dir):
    respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT))
    reports = []

    path = ensure_model(make_spec(), progress=lambda f, msg: reports.append((f, msg)))

    assert path == models_dir / "tiny.safetensors"
    assert path.read_bytes() == CONTENT
    assert leftovers(models_dir) == ["tiny.safetensors"]
    fractions = [f for f, _ in reports]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert "Tiny" in reports[-1][1]


@respx.mock
def test_ensure_model_bad_hash(models_dir):
    wrong = bytes(len(CONTENT))  # right size, wrong bytes
    respx.get(URL).mock(return_value=httpx.Response(200, content=wrong))

    with pytest.raises(ModelDownloadError, match="try again"):
        ensure_model(make_spec())

    assert leftovers(models_dir) == []  # no file and no .part


@respx.mock
def test_ensure_model_resume_part(models_dir):
    models_dir.mkdir(parents=True)
    have = 100_000
    (models_dir / "tiny.safetensors.part").write_bytes(CONTENT[:have])
    route = respx.get(URL).mock(
        return_value=httpx.Response(
            206,
            content=CONTENT[have:],
            headers={"Content-Range": f"bytes {have}-{len(CONTENT) - 1}/{len(CONTENT)}"},
        )
    )

    path = ensure_model(make_spec())

    assert route.calls.last.request.headers["Range"] == f"bytes={have}-"
    assert path.read_bytes() == CONTENT
    assert leftovers(models_dir) == ["tiny.safetensors"]


@respx.mock
def test_ensure_model_restarts_when_server_ignores_range(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors.part").write_bytes(CONTENT[:50_000])
    respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT))

    assert ensure_model(make_spec()).read_bytes() == CONTENT


@respx.mock
def test_ensure_model_replaces_a_damaged_leftover_part(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors.part").write_bytes(bytes(len(CONTENT)))  # full size, bad bytes
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT))

    assert ensure_model(make_spec()).read_bytes() == CONTENT
    assert "Range" not in route.calls.last.request.headers
    assert leftovers(models_dir) == ["tiny.safetensors"]


@respx.mock
def test_ensure_model_replaces_a_wrong_size_cached_file(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors").write_bytes(b"old version")
    respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT))

    assert ensure_model(make_spec()).read_bytes() == CONTENT


@respx.mock(assert_all_mocked=True)
def test_ensure_model_skips_download_when_present(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors").write_bytes(CONTENT)

    assert ensure_model(make_spec()) == models_dir / "tiny.safetensors"  # no request made


@respx.mock
def test_ensure_model_network_error_keeps_part_for_later(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors.part").write_bytes(CONTENT[:1000])
    route = respx.get(URL).mock(side_effect=httpx.ConnectError("no route to host"))

    with pytest.raises(ModelDownloadError, match="internet"):
        ensure_model(make_spec())

    assert route.call_count == models._DOWNLOAD_ATTEMPTS
    assert leftovers(models_dir) == ["tiny.safetensors.part"]


@respx.mock
def test_ensure_model_recovers_from_a_dropped_connection(models_dir):
    respx.get(URL).mock(
        side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, content=CONTENT)]
    )
    assert ensure_model(make_spec()).read_bytes() == CONTENT


@respx.mock
def test_ensure_model_http_error_is_friendly(models_dir):
    respx.get(URL).mock(return_value=httpx.Response(404))

    with pytest.raises(ModelDownloadError, match="404"):
        ensure_model(make_spec())
    assert leftovers(models_dir) == []


@respx.mock
def test_ensure_model_server_outage_is_not_blamed_on_the_internet(models_dir):
    route = respx.get(URL).mock(return_value=httpx.Response(503))

    with pytest.raises(ModelDownloadError, match="503") as excinfo:
        ensure_model(make_spec())
    assert "internet" not in str(excinfo.value)
    assert route.call_count == models._DOWNLOAD_ATTEMPTS


@respx.mock
def test_ensure_model_rejects_oversized_download(models_dir):
    respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT + b"extra"))

    with pytest.raises(ModelDownloadError):
        ensure_model(make_spec())
    assert leftovers(models_dir) == []


@respx.mock
def test_ensure_model_stops_when_cancelled_and_keeps_the_part(models_dir, monkeypatch):
    # Cancel during a first-time download stops it at once, instead of the card sitting on
    # "Stopping…" until 140 MB have arrived; what came is kept, so the next run resumes.
    monkeypatch.setattr(models, "_CHUNK_BYTES", 50_000)
    respx.get(URL).mock(return_value=httpx.Response(200, content=CONTENT))
    cancel = threading.Event()
    reports = []

    def progress(fraction, message):
        reports.append(fraction)
        cancel.set()  # the person presses Cancel as the first megabytes arrive

    with pytest.raises(Cancelled):
        ensure_model(make_spec(), progress=progress, cancel=cancel)

    assert len(reports) == 1
    assert leftovers(models_dir) == ["tiny.safetensors.part"]
    assert (models_dir / "tiny.safetensors.part").read_bytes() == CONTENT[:50_000]


@respx.mock
def test_ensure_model_cancel_stops_the_wait_between_attempts(models_dir, monkeypatch):
    monkeypatch.setattr(models, "_RETRY_DELAY_SECONDS", 60)  # would wait a minute before trying again
    cancel = threading.Event()

    def drop(request):
        cancel.set()
        raise httpx.ConnectError("no route to host")

    route = respx.get(URL).mock(side_effect=drop)
    started = time.monotonic()

    with pytest.raises(Cancelled):
        ensure_model(make_spec(), cancel=cancel)

    assert route.call_count == 1 and time.monotonic() - started < 5


def test_ensure_model_cancel_stops_waiting_for_another_download(models_dir):
    cancel = threading.Event()
    cancel.set()
    with models._download_lock:  # another model is downloading
        with pytest.raises(Cancelled):
            ensure_model(make_spec(), cancel=cancel)


@respx.mock(assert_all_mocked=True)
def test_ensure_model_ignores_cancel_when_the_model_is_here(models_dir):
    models_dir.mkdir(parents=True)
    (models_dir / "tiny.safetensors").write_bytes(CONTENT)
    cancel = threading.Event()
    cancel.set()

    assert ensure_model(make_spec(), cancel=cancel) == models_dir / "tiny.safetensors"


def test_ensure_model_checks_free_space_first(models_dir, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _: usage(10**9, 10**9, 1000))

    with pytest.raises(ModelDownloadError, match="space"):
        ensure_model(make_spec())


# --- Precision and loading --------------------------------------------------


def test_choose_dtype():
    assert choose_dtype(True, True, "mps") == torch.float16
    assert choose_dtype(True, True, "cuda") == torch.float16
    assert choose_dtype(False, True, "mps") == torch.bfloat16  # DAT2
    assert choose_dtype(True, True, "cpu") == torch.float32
    assert choose_dtype(False, True, "cpu") == torch.float32
    assert choose_dtype(False, False, "cuda") == torch.float32


def write_tiny_compact_model(path):
    from safetensors.torch import save_file
    from spandrel.architectures.Compact import Compact

    torch.manual_seed(0)
    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=4)
    save_file(net.state_dict(), str(path))


def test_load_model_keeps_one_model_cached(models_dir, monkeypatch):
    models_dir.mkdir(parents=True)
    write_tiny_compact_model(models_dir / "a.safetensors")
    write_tiny_compact_model(models_dir / "b.safetensors")
    spec_a = make_spec(id="a", file="a.safetensors")
    spec_b = make_spec(id="b", file="b.safetensors")
    monkeypatch.setattr(models, "ensure_model", lambda spec, progress=None: models_dir / spec.file)
    monkeypatch.setattr(models, "_cached", None)

    first = load_model(spec_a, device="cpu")
    assert (first.scale, first.dtype, first.tile, first.device) == (4, torch.float32, 512, torch.device("cpu"))
    assert load_model(spec_a, device="cpu") is first
    second = load_model(spec_b, device="cpu")
    assert second is not first
    assert load_model(spec_a, device="cpu") is not first  # only one model stays cached

    x = torch.rand(1, 3, 10, 13)
    assert second.net(x).shape == (1, 3, 40, 52)


def test_unload_model_frees_the_cached_model(models_dir, monkeypatch):
    models_dir.mkdir(parents=True)
    write_tiny_compact_model(models_dir / "a.safetensors")
    spec = make_spec(id="a", file="a.safetensors")
    monkeypatch.setattr(models, "ensure_model", lambda spec, progress=None: models_dir / spec.file)
    monkeypatch.setattr(models, "_cached", None)
    freed = []
    monkeypatch.setattr(models, "_release_memory", freed.append)

    first = load_model(spec, device="cpu")
    models.unload_model()
    assert models._cached is None and freed == ["cpu"]  # the device's memory is given back
    assert load_model(spec, device="cpu") is not first  # loaded again when next needed
    models.unload_model()
    models.unload_model()  # nothing loaded: nothing to do
    assert freed == ["cpu", "cpu"]
