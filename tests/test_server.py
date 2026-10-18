import io
import json
import math
import os
import socket
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import keyring
import keyring.backend
import keyring.errors
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from upscaler import config, jobs, pipeline_image, preview, server, suggest
from upscaler.cloud import PROVIDERS
from upscaler.config import SPACE_RESERVE_BYTES
from upscaler.hardware import Hardware
from upscaler.imageio_utils import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
    render_vector,
    resolve_format,
    unique_path,
)
from upscaler.jobs import JobQueue, default_runners
from upscaler.pipeline_image import ImageResult
from upscaler.sizing import print_size

MAC_16GB = Hardware("mps", "Apple M4", 16, "mac", True)
MAC_8GB = Hardware("mps", "Apple M1", 8, "mac", True)
PC_NO_GPU = Hardware("cpu", "Intel", 32, "windows", False)
HOST_URL = "http://127.0.0.1:7860"
LAN_URL = "http://192.168.1.20:7860"


class MemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError(username)
        del self.store[(service, username)]


class FakeEngine:
    name = "mlx"

    def __init__(self):
        self.downloaded = False

    def weights_present(self):
        return self.downloaded

    def download_weights(self, progress, cancel):
        progress(0.5, "Downloading Photo-real: 3.6 of 7.3 GB")
        self.downloaded = True


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


# The real Wi-Fi listener, before the env fixture swaps in stand-ins (one test runs the real thing).
REAL_LISTEN_SOCKET = server.listen_socket
REAL_WIFI_SERVE = server.WifiSharing.__dict__["_serve"]  # the staticmethod itself


class FakeSocket:
    """Stands in for a listening socket, so tests never open real ports on the network."""

    def __init__(self, host: str, port: int):
        self.address = (host, port)
        self.closed = False

    def getsockname(self):
        return self.address

    def close(self):
        self.closed = True


def _serve_until_stopped(wifi_server, sock):
    """Stands in for uvicorn serving the Wi-Fi listener: "runs" until asked to stop."""
    while not wifi_server.should_exit:
        time.sleep(0.005)
    sock.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))  # Windows asks the system
    monkeypatch.setattr(server, "free_bytes", lambda path: 10**12)  # never this computer's real disk
    engine = FakeEngine()
    hardware = SimpleNamespace(value=MAC_16GB)
    monkeypatch.setattr(server, "detect_hardware", lambda: hardware.value)
    monkeypatch.setattr(server, "get_seedvr2_engine", lambda hw: engine)
    monkeypatch.setattr(jobs, "_photoreal_engine", lambda: (engine, ""))
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)
    monkeypatch.setattr(server, "lan_ipv4_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr(server, "LOGIN_FAILURE_DELAY_SECONDS", 0)
    monkeypatch.setattr(suggest, "detect_faces", lambda rgb: [])  # the face model would be downloaded
    monkeypatch.setattr(preview, "detect_hardware", lambda: hardware.value)
    wifi_sockets = []

    def fake_listen(host, port):
        wifi_sockets.append(FakeSocket(host, port))
        return wifi_sockets[-1]

    monkeypatch.setattr(server, "listen_socket", fake_listen)
    monkeypatch.setattr(server.WifiSharing, "_serve", staticmethod(_serve_until_stopped))
    previous = keyring.get_keyring()
    memory = MemoryKeyring()
    keyring.set_keyring(memory)

    upscale_calls = []

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        upscale_calls.append(SimpleNamespace(path=Path(path), opts=opts, out_dir=out_dir))
        progress(0.5, "Upscaling")
        output = unique_path(out_dir, f"{Path(path).stem}-4K-{opts.mode.replace(':', '-')}", ".jpg")
        output.write_bytes(b"upscaled " + Path(path).read_bytes())
        return ImageResult(
            output, 3840, 2880, opts.mode, ["A note."], 1.0, b"preview", b"before", b"after",
            dpi=opts.dpi, print=print_size(3840, 2880, opts.dpi), file_bytes=output.stat().st_size,
            source_width=960, source_height=720,
        )  # fmt: skip

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    yield SimpleNamespace(
        tmp=tmp_path,
        store=tmp_path / "data" / "jobs",
        engine=engine,
        hardware=hardware,
        keyring=memory,
        upscale_calls=upscale_calls,
        host_out=tmp_path / "home" / "Pictures" / "Upscaled",
        guest_out=tmp_path / "data" / "guest",
        wifi_sockets=wifi_sockets,
    )
    keyring.set_keyring(previous)


def make_app(env, runners=None, share=True):
    queue = JobQueue(env.store, runners or default_runners(env.store))
    return server.create_app(queue, share, port=7860), queue


def host_client(app) -> TestClient:
    return TestClient(app, base_url=HOST_URL, client=("127.0.0.1", 50000))


def guest_client(app, ip: str = "192.168.1.5") -> TestClient:
    return TestClient(app, base_url=LAN_URL, client=(ip, 50001))


def png_bytes(width: int = 8, height: int = 6) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buffer, "PNG")
    return buffer.getvalue()


def upload_images(client: TestClient, names: list[str], options: dict | None = None, data: bytes | None = None):
    files = [("files", (name, data or png_bytes(), "image/png")) for name in names]
    return client.post("/api/images", files=files, data={"options": json.dumps(options or {})})


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


def job_status(client: TestClient, job_id: str) -> str:
    return client.get(f"/api/jobs/{job_id}").json()["job"]["status"]


# --- Status -------------------------------------------------------------------------


def test_status_for_host(env):
    app, _ = make_app(env)
    with host_client(app) as host:  # the app starts sharing on Wi-Fi as it starts, like the Share launchers
        status = host.get("/api/status").json()
    assert status["is_host"] is True
    assert status["needs_login"] is False
    assert status["version"]
    assert status["app"] == {"name": "Re:Solution"}
    assert status["hardware"] == {
        "device": "mps",
        "device_name": "Apple M4",
        "memory_gb": 16,
        "cpu_only": False,
        "note": None,
    }
    assert status["photoreal"] == {"supported": True, "reason": "", "downloaded": False, "engine": "mlx"}
    assert [s["id"] for s in status["sizes"]] == ["4k", "6k", "8k", "8k_tv", "16k", "16k_tv", "2x", "3x", "4x"]
    assert status["formats"] == [
        {"id": "jpeg", "max_long_edge": 65535},
        {"id": "png"},
        {"id": "tiff"},
        {"id": "webp", "max_long_edge": 16383},
        {"id": "original"},
    ]
    assert status["dpi_choices"] == [72, 150, 300, 600]
    assert [m["id"] for m in status["video_modes"]] == ["fast", "general", "faithful"]
    assert {p["id"] for p in status["providers"]} == {"fal", "replicate", "magnific"}
    assert all(p["connected"] is False and p["models"] for p in status["providers"])
    assert shared_at(status["share"]) == {"enabled": True, "urls": ["http://192.168.1.20:7861"]}  # the Wi-Fi listener
    assert status["share"]["can_toggle"] is True
    assert status["settings"]["default_image_mode"] == "photoreal"
    assert status["settings"]["has_password"] is False
    assert "access_password_hash" not in status["settings"]


def test_photoreal_is_the_default_image_mode(env):
    app, _ = make_app(env)
    host = host_client(app)
    modes = host.get("/api/status").json()["image_modes"]
    assert modes[0]["id"] == "photoreal"
    assert modes[0]["label"] == "Photo-real"
    assert modes[0]["available"] is True
    for mode in modes:
        assert {"id", "label", "description", "eta_hint", "available", "needs_key"} <= mode.keys()
    cloud = [m for m in modes if m["id"].startswith("cloud:")]
    assert cloud and all(m["needs_key"] and not m["available"] for m in cloud)

    job = upload_images(host, ["table.png"]).json()["job"]
    assert job["options"]["mode"] == "photoreal"
    assert job["options"]["size"] == "8k"


def test_saved_default_mode_is_used_when_none_is_given(env):
    app, _ = make_app(env)
    host = host_client(app)
    assert (
        host.post("/api/settings", json={"default_image_mode": "faithful", "output_format": "png"}).status_code == 200
    )
    job = upload_images(host, ["table.png"]).json()["job"]
    assert job["options"]["mode"] == "faithful"
    assert job["options"]["output_format"] == "png"


def test_status_on_a_small_mac(env):
    env.hardware.value = MAC_8GB
    app, _ = make_app(env)
    status = host_client(app).get("/api/status").json()
    assert status["photoreal"]["supported"] is False
    assert "16 GB" in status["photoreal"]["reason"]
    photoreal = next(m for m in status["image_modes"] if m["id"] == "photoreal")
    assert photoreal["available"] is False
    assert host_client(app).post("/api/photoreal/download").status_code == 400


def test_status_has_presets_formats_samples(env):
    app, _ = make_app(env)
    host = host_client(app)
    status = guest_client(app).get("/api/status").json()

    presets = {p["id"]: p for p in status["presets"]}
    assert list(presets) == ["interiors", "products", "portraits", "ai_art", "print", "illustration", "old_photos"]
    for preset in presets.values():
        assert preset.keys() == {
            "id", "label", "icon", "mode", "photoreal_level", "size", "output_format", "dpi", "grain", "tip"
        }  # fmt: skip
        assert preset["label"] and preset["icon"] and preset["dpi"] == 300
        assert preset["photoreal_level"] == ("balanced" if preset["mode"] == "photoreal" else None)
    expected = {
        "interiors": ("photoreal", "8k", "jpeg"),
        "products": ("faithful", "8k", "jpeg"),
        "portraits": ("faithful", "8k", "jpeg"),
        "ai_art": ("photoreal", "8k", "jpeg"),
        "print": ("photoreal", "16k", "tiff"),
        "illustration": ("illustration", "8k", "jpeg"),
        "old_photos": ("photoreal", "4k", "jpeg"),
    }
    assert {i: (p["mode"], p["size"], p["output_format"]) for i, p in presets.items()} == expected
    assert "Photo-real" in presets["portraits"]["tip"] and "skin" in presets["portraits"]["tip"]
    assert "logos" in presets["products"]["tip"]

    samples = status["samples"]
    assert [s["id"] for s in samples] == ["interior", "product", "old-photo", "food", "illustration"]
    for sample in samples:
        assert sample.keys() == {
            "id", "title", "caption", "preset", "thumb_url", "url", "width", "height", "credit", "license", "source_url"
        }  # fmt: skip
        assert sample["preset"] in presets
        assert sample["url"].startswith("/samples/") and sample["thumb_url"].startswith("/samples/")
        for url in (sample["url"], sample["thumb_url"]):
            picture = host.get(url)
            assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    first = samples[0]
    with Image.open(io.BytesIO(host.get(first["url"]).content)) as image:
        assert image.size == (first["width"], first["height"])


@pytest.mark.parametrize("damage", ["missing", "not json", "not a list", "outside"])
def test_status_samples_survive_a_damaged_list(env, monkeypatch, tmp_path, damage):
    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "ok.jpg").write_bytes(b"jpeg")
    good = {
        "id": "ok",
        "title": "OK",
        "caption": "",
        "preset": "interiors",
        "file": "ok.jpg",
        "thumb": "ok.jpg",
        "width": 10,
        "height": 10,
        "credit": "Me",
        "license": "CC0",
        "source_url": "https://example.org",
    }
    index = folder / "samples.json"
    if damage == "not json":
        index.write_text("{oops")
    elif damage == "not a list":
        index.write_text(json.dumps({"id": "ok"}))
    elif damage == "outside":
        index.write_text(json.dumps([{**good, "id": "bad", "file": "../../secret.txt"}, good]))
    monkeypatch.setattr(server, "SAMPLES_DIR", folder)
    app, _ = make_app(env)

    samples = host_client(app).get("/api/status").json()["samples"]

    assert [s["id"] for s in samples] == (["ok"] if damage == "outside" else [])


def test_status_photoreal_levels(env):
    app, _ = make_app(env)
    modes = host_client(app).get("/api/status").json()["image_modes"]
    photoreal = next(m for m in modes if m["id"] == "photoreal")
    levels = {level["id"]: level for level in photoreal["levels"]}
    assert list(levels) == ["balanced", "rich", "gentle"]
    assert all(level["label"] and level["description"] for level in levels.values())
    assert "3–6 min" in levels["balanced"]["eta_hint"]
    assert "10–22 min" in levels["rich"]["eta_hint"]
    assert "3–5 min" in levels["gentle"]["eta_hint"]
    assert photoreal["eta_hint"] == levels["balanced"]["eta_hint"]  # Balanced is the default
    assert all("levels" not in m for m in modes if m["id"] != "photoreal")
    assert any("noise" in m["description"] for m in modes)  # "sharpens & cleans up noise", in plain words


def test_status_cpu_only(env):
    env.hardware.value = PC_NO_GPU
    app, _ = make_app(env)
    hardware = host_client(app).get("/api/status").json()["hardware"]
    assert hardware["cpu_only"] is True and hardware["note"] == "CPU only: slow"


def test_status_says_when_an_nvidia_card_is_too_old(env):
    from upscaler.hardware import OLD_NVIDIA_NOTE

    env.hardware.value = Hardware("cpu", "Intel64 Family 6", 32, "windows", False, OLD_NVIDIA_NOTE)
    app, _ = make_app(env)
    hardware = host_client(app).get("/api/status").json()["hardware"]
    assert hardware["cpu_only"] is True
    assert hardware["note"] == (
        "Your NVIDIA graphics card is too old for the graphics software Re:Solution uses, "
        "so it uses the processor instead (slow)"
    )


def test_status_lists_the_files_people_can_add(env):
    app, _ = make_app(env)
    accept = guest_client(app).get("/api/status").json()["accept"]
    assert accept["images"] == sorted(SUPPORTED_IMAGE_EXTS)
    assert accept["videos"] == sorted(SUPPORTED_VIDEO_EXTS)
    assert {".jpg", ".heic", ".png", ".tif", ".webp", ".avif", ".dng", ".svg", ".psd"} <= set(accept["images"])
    assert {".mp4", ".mov"} <= set(accept["videos"])


def test_webp_limit_in_status_matches_what_can_be_saved(env):
    app, _ = make_app(env)
    formats = {f["id"]: f for f in host_client(app).get("/api/status").json()["formats"]}
    limit = formats["webp"]["max_long_edge"]
    assert resolve_format("webp", None, (limit, 100)) == ("webp", None)
    with pytest.raises(ValueError, match="WebP"):
        resolve_format("webp", None, (limit + 1, 100))


def test_jpeg_limit_in_status_matches_what_can_be_saved(env):
    app, _ = make_app(env)
    formats = {f["id"]: f for f in host_client(app).get("/api/status").json()["formats"]}
    limit = formats["jpeg"]["max_long_edge"]
    assert resolve_format("jpeg", None, (limit, 100)) == ("jpeg", None)
    with pytest.raises(ValueError, match="JPEG"):
        resolve_format("jpeg", None, (limit + 1, 100))


def test_status_for_guest_has_no_settings(env):
    app, _ = make_app(env, share=False)
    status = guest_client(app).get("/api/status").json()
    assert status["is_host"] is False
    assert "settings" not in status
    assert shared_at(status["share"]) == {"enabled": False, "urls": []}
    assert status["share"]["can_toggle"] is False


# --- Host-only endpoints and the password -------------------------------------------


def test_api_host_only_endpoints_forbid_guest(env):
    app, _ = make_app(env)
    guest = guest_client(app)
    forbidden = [
        guest.post("/api/providers/fal/key", json={"key": "abc"}),
        guest.delete("/api/providers/fal/key"),
        guest.get("/api/settings"),
        guest.post("/api/settings", json={"output_format": "png"}),
        guest.post("/api/open-folder", json={}),
    ]
    assert [r.status_code for r in forbidden] == [403] * 5
    assert all(r.json()["detail"] for r in forbidden)
    assert config.load_settings().output_format == "jpeg"
    assert host_client(app).get("/api/settings").status_code == 200


def test_api_password_required_for_guest(env):
    app, _ = make_app(env)
    host, guest = host_client(app), guest_client(app)
    assert host.post("/api/settings", json={"access_password": "sesame"}).status_code == 200

    assert guest.get("/api/jobs").status_code == 401
    assert upload_images(guest, ["a.png"]).status_code == 401
    locked = guest.get("/api/status").json()
    assert locked["needs_login"] is True
    assert "image_modes" not in locked

    assert guest.post("/api/login", json={"password": "wrong"}).status_code == 401
    assert guest.post("/api/login", json={"password": "sesame"}).status_code == 200
    assert guest.get("/api/jobs").status_code == 200
    assert guest.get("/api/status").json()["needs_login"] is False

    assert host.get("/api/jobs").status_code == 200  # the host is never asked
    assert host.get("/api/status").json()["needs_login"] is False

    assert host.post("/api/settings", json={"access_password": "changed"}).status_code == 200
    assert guest.get("/api/jobs").status_code == 401  # a new password logs guests out


def test_password_is_stored_as_a_salted_hash(env):
    app, _ = make_app(env)
    host = host_client(app)
    host.post("/api/settings", json={"access_password": "sesame"})
    stored = config.load_settings().access_password_hash
    assert stored and "sesame" not in stored
    assert "sesame" not in (env.tmp / "data" / "settings.json").read_text()
    settings = host.get("/api/settings").json()["settings"]
    assert settings["has_password"] is True
    assert "access_password_hash" not in settings
    assert host.post("/api/settings", json={"access_password": "abc"}).status_code == 400  # too short
    assert host.post("/api/settings", json={"access_password": ""}).json()["settings"]["has_password"] is False


def test_allow_guests_cloud_requires_password(env):
    app, _ = make_app(env)
    host = host_client(app)
    response = host.post("/api/settings", json={"allow_guests_cloud": True})
    assert response.status_code == 400
    assert "password" in response.json()["detail"]
    assert config.load_settings().allow_guests_cloud is False

    assert host.post("/api/settings", json={"access_password": "sesame", "allow_guests_cloud": True}).status_code == 200
    assert config.load_settings().allow_guests_cloud is True
    cleared = host.post("/api/settings", json={"access_password": ""}).json()["settings"]
    assert cleared["allow_guests_cloud"] is False  # removing the password turns it off too


def test_settings_reject_unknown_values(env):
    app, _ = make_app(env)
    host = host_client(app)
    assert host.post("/api/settings", json={"default_image_mode": "magic"}).status_code == 400
    assert host.post("/api/settings", json={"output_format": "gif"}).status_code in (400, 422)


# --- Uploads and jobs -----------------------------------------------------------------


def test_api_upload_image_creates_job(env):
    app, queue = make_app(env)
    host = host_client(app)
    response = upload_images(
        host, ["image.png", "image.png", "my photo 🌅.png"], {"mode": "fast", "size": "4k", "strength": 0.8}
    )
    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "image"
    assert job["owner"] == "host"
    assert job["status"] == "queued"
    assert job["mine"] is True
    assert job["inputs"] == ["image.png", "image.png", "my photo 🌅.png"]
    assert job["options"] == {
        "mode": "fast",
        "size": "4k",
        "strength": 0.8,
        "grain": "off",
        "output_format": "jpeg",
        "dpi": 300,
        "photoreal_level": "balanced",
        "creativity": 0.0,
        "resemblance": 1.0,
    }

    saved = queue.get(job["id"]).inputs
    assert len(set(saved)) == 3  # same-named files never overwrite each other
    assert all(Path(p).is_file() and Path(p).is_relative_to(env.store / job["id"] / "in") for p in saved)
    assert Path(saved[2]).name == "my photo 🌅.png"


def test_image_options_for_designers_and_photo_real(env):
    app, queue = make_app(env)
    host = host_client(app)
    options = {
        "mode": "photoreal",
        "photoreal_level": "rich",
        "creativity": 0.4,
        "resemblance": 0.6,
        "dpi": 600,
        "output_format": "tiff",
        "size": "16k",
        "guest_key": "sneaky",  # never taken from the request, and never saved
        "something_new": True,  # unknown fields are ignored
    }
    job = upload_images(host, ["poster.png"], options).json()["job"]
    assert job["options"] == {
        "mode": "photoreal",
        "size": "16k",
        "strength": 1.0,
        "grain": "off",
        "output_format": "tiff",
        "dpi": 600,
        "photoreal_level": "rich",
        "creativity": 0.4,
        "resemblance": 0.6,
    }
    assert "sneaky" not in (env.store / job["id"] / "job.json").read_text()
    assert queue.get(job["id"]).secret is None


@pytest.mark.parametrize(
    "options",
    [
        {"dpi": 100},
        {"dpi": "300"},
        {"dpi": True},
        {"creativity": 2},
        {"resemblance": -0.1},
        {"photoreal_level": "extreme"},
        {"output_format": "gif"},
        {"size": ["8k"]},  # not even text: refused, not a server error
        {"size": {"a": 1}},
        {"grain": ["off"]},
        {"output_format": ["png"]},
        {"photoreal_level": ["rich"]},
    ],
)
def test_bad_designer_options_are_refused(env, options):
    app, queue = make_app(env)
    response = upload_images(host_client(app), ["a.png"], options)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str) and response.json()["detail"]
    assert queue.list() == []


def test_image_job_runs_and_results_download(env):
    app, _ = make_app(env)
    with host_client(app) as host:
        job = upload_images(host, ["image.png", "image.png"], {"mode": "fast", "size": "4k"}).json()["job"]
        wait_for(lambda: job_status(host, job["id"]) == "done")
        done = host.get(f"/api/jobs/{job['id']}").json()["job"]
        assert [r["filename"] for r in done["results"]] == ["image-4K-fast.jpg", "image-4K-fast-2.jpg"]
        assert done["results"][0]["width"] == 3840
        assert (done["results"][0]["source_width"], done["results"][0]["source_height"]) == (960, 720)
        assert done["results"][0]["notes"] == ["A note."]
        assert done["results"][0]["dpi"] == 300
        assert done["results"][0]["print"] == {"inches": [12.8, 9.6], "cm": [32.5, 24.4], "dpi": 300}
        assert done["results"][0]["file_bytes"] == (env.host_out / "image-4K-fast.jpg").stat().st_size
        assert "path" not in done["results"][0]  # no file system paths leave the server
        assert (env.host_out / "image-4K-fast.jpg").is_file()

        file = host.get(f"/api/jobs/{job['id']}/results/1/file")
        assert file.status_code == 200
        assert file.content.startswith(b"upscaled ")
        assert "image-4K-fast-2.jpg" in file.headers["content-disposition"]
        assert host.get(f"/api/jobs/{job['id']}/results/0/preview.jpg").content == b"preview"
        assert host.get(f"/api/jobs/{job['id']}/results/0/before.jpg").content == b"before"
        assert host.get(f"/api/jobs/{job['id']}/results/0/after.jpg").content == b"after"
        assert host.get(f"/api/jobs/{job['id']}/results/5/file").status_code == 404

        archive = host.get(f"/api/jobs/{job['id']}/zip")
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
            assert sorted(zf.namelist()) == ["image-4K-fast-2.jpg", "image-4K-fast.jpg"]
            assert zf.read("image-4K-fast.jpg").startswith(b"upscaled ")
    assert env.upscale_calls[0].opts.mode == "fast"


def test_upload_names_are_sanitised(env):
    app, queue = make_app(env)
    host = host_client(app)
    job = upload_images(host, ["../../evil.png", "..\\..\\sneaky.png", "CON.png", ".hidden.png"]).json()["job"]
    saved = [Path(p) for p in queue.get(job["id"]).inputs]
    inbox = env.store / job["id"] / "in"
    assert all(p.resolve().is_relative_to(inbox.resolve()) for p in saved)
    assert [p.name for p in saved] == ["evil.png", "sneaky.png", "_CON.png", "hidden.png"]
    assert not (env.store.parent / "evil.png").exists()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", "photo.jpg"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\pic.png", "pic.png"),
        ("my photo 🌅.png", "my photo 🌅.png"),
        ('a<b>c:d"e|f?g*h.png', "a_b_c_d_e_f_g_h.png"),
        ("nul.txt", "_nul.txt"),
        ("   ", "upload"),
        ("...", "upload"),
        ("bad\x00name\x1f.png", "badname.png"),
        ("x" * 300 + ".png", "x" * 196 + ".png"),
    ],
)
def test_safe_filename(name, expected):
    assert server.safe_filename(name) == expected


def test_image_too_big_is_refused(env, monkeypatch):
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 100)
    app, queue = make_app(env)
    response = upload_images(host_client(app), ["huge.png"], data=b"x" * 1000)
    assert response.status_code == 413
    assert "huge.png" in response.json()["detail"]
    assert queue.list() == []
    assert list(env.store.iterdir()) == []  # nothing left behind


def test_bad_image_options_are_refused(env):
    app, queue = make_app(env)
    host = host_client(app)
    for options in ({"size": "10k"}, {"strength": 2}, {"mode": "magic"}, {"grain": "lots"}, {"output_format": "gif"}):
        response = upload_images(host, ["a.png"], options)
        assert response.status_code == 400, options
        assert isinstance(response.json()["detail"], str)
    assert (
        host.post(
            "/api/images", files=[("files", ("a.png", png_bytes(), "image/png"))], data={"options": "{oops"}
        ).status_code
        == 400
    )
    assert host.post("/api/images", data={"options": "{}"}).status_code == 400  # no pictures
    assert queue.list() == []
    assert list(env.store.iterdir()) == []


def test_video_upload_creates_job(env, monkeypatch):
    monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=50, duration=2.0))
    app, queue = make_app(env)
    host = host_client(app)
    files = {"file": ("clip.mov", b"fake video bytes", "video/quicktime")}
    response = host.post(
        "/api/videos", files=files, data={"options": json.dumps({"mode": "general", "preview_seconds": 3})}
    )
    assert response.status_code == 200
    job = response.json()["job"]
    assert job["kind"] == "video"
    assert job["inputs"] == ["clip.mov"]
    assert job["options"] == {"mode": "general", "preview_seconds": 3.0, "strength": 1.0}
    assert Path(queue.get(job["id"]).inputs[0]).read_bytes() == b"fake video bytes"

    assert host.post("/api/videos", files=files, data={"options": json.dumps({"mode": "photoreal"})}).status_code == 400
    assert host.post("/api/videos", files=files, data={"options": json.dumps({"mode": ["fast"]})}).status_code == 400
    assert (
        host.post("/api/videos", files=files, data={"options": json.dumps({"preview_seconds": -1})}).status_code == 400
    )


def test_video_that_is_not_a_video_is_refused(env, monkeypatch):
    from upscaler.video import VideoError

    def refuse(path):
        raise VideoError("This file doesn't look like a video.")

    monkeypatch.setattr(server, "probe_video", refuse)
    app, queue = make_app(env)
    response = host_client(app).post(
        "/api/videos", files={"file": ("cat.jpg", png_bytes(), "image/jpeg")}, data={"options": "{}"}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "This file doesn't look like a video."
    assert queue.list() == []
    assert list(env.store.iterdir()) == []


def test_single_picture_is_not_a_video(env, monkeypatch):
    monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=1, duration=0.04))
    app, queue = make_app(env)
    response = host_client(app).post(
        "/api/videos", files={"file": ("table.png", png_bytes(), "image/png")}, data={"options": "{}"}
    )
    assert response.status_code == 400
    assert "Photos page" in response.json()["detail"]
    assert queue.list() == []


def test_video_needs_free_disk_space(env, monkeypatch):
    monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=50, duration=2.0))
    monkeypatch.setattr(server, "free_bytes", lambda path: 10)
    app, queue = make_app(env)
    response = host_client(app).post(
        "/api/videos", files={"file": ("clip.mov", b"x" * 1000, "video/quicktime")}, data={"options": "{}"}
    )
    assert response.status_code == 507
    assert "space" in response.json()["detail"]
    assert queue.list() == []


def test_video_space_check_counts_the_length_of_the_video(env, monkeypatch):
    # A 4K result is about 60 Mbit/s whatever the source, so a long video from a small file still needs
    # many GB for its segments and the joined result (2.2x that), on top of the space kept free.
    gb = 10**9
    monkeypatch.setattr(server, "free_bytes", lambda path: 5 * gb)
    app, queue = make_app(env)
    host = host_client(app)

    def send(duration, options=None):
        monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=50, duration=duration))
        files = {"file": ("clip.mov", b"x" * 1000, "video/quicktime")}
        return host.post("/api/videos", files=files, data={"options": json.dumps(options or {})})

    long_video = send(600.0)  # 10 minutes: 600 s x 7.5 MB/s x 2.2 = 9.9 GB
    assert long_video.status_code == 507
    assert "12.0 GB" in long_video.json()["detail"]  # 9.9 GB plus the 2 GiB always kept free
    assert queue.list() == []
    assert send(60.0).status_code == 200  # a minute needs about 1 GB
    assert send(600.0, {"preview_seconds": 3}).status_code == 200  # a preview only makes its few seconds
    assert server.video_space_needed(10.0, 0) == math.ceil(10 * 60e6 / 8 * 2.2)
    assert server.video_space_needed(10.0, 5000) == server.video_space_needed(10.0, 0) + 5000


def test_video_upload_bigger_than_the_free_space_is_refused_before_it_arrives(env, monkeypatch):
    monkeypatch.setattr(server, "probe_video", lambda path: pytest.fail("never got that far"))
    monkeypatch.setattr(server, "free_bytes", lambda path: server.SPACE_RESERVE_BYTES + 500)
    app, queue = make_app(env)
    files = {"file": ("clip.mov", b"x" * 1000, "video/quicktime")}
    response = host_client(app).post("/api/videos", files=files, data={"options": "{}"})
    assert response.status_code == 507
    assert queue.list() == []


def test_a_video_that_ran_out_of_space_can_be_retried(env, monkeypatch):
    from upscaler.video import NotEnoughDiskSpace

    monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=50, duration=2.0))
    attempts = []

    def video(job, progress, cancel):
        attempts.append(job.id)
        if len(attempts) == 1:
            raise NotEnoughDiskSpace("There isn't enough free disk space to finish this video. Free up about 2 GB.")
        return []

    app, queue = make_app(env, runners={"video": video, "image": lambda job, progress, cancel: []})
    files = {"file": ("clip.mov", b"fake video bytes", "video/quicktime")}
    with host_client(app) as host:
        guest, other = guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
        job = guest.post("/api/videos", files=files, data={"options": "{}"}).json()["job"]
        assert job["retryable"] is False
        wait_for(lambda: job_status(host, job["id"]) == "failed")
        failed = guest.get(f"/api/jobs/{job['id']}").json()["job"]
        assert failed["retryable"] is True and "press Retry" in failed["error"]
        assert other.post(f"/api/jobs/{job['id']}/retry").status_code == 403
        retried = guest.post(f"/api/jobs/{job['id']}/retry")
        assert retried.status_code == 200 and retried.json()["job"]["status"] in ("queued", "running")
        wait_for(lambda: job_status(host, job["id"]) == "done")
        assert guest.post(f"/api/jobs/{job['id']}/retry").status_code == 409  # nothing left to carry on
        assert host.post("/api/jobs/0123456789ab/retry").status_code == 404
    assert attempts == [job["id"], job["id"]]


def test_concurrent_guest_uploads_order(env):
    runs = []
    lock = threading.Lock()

    def runner(job, progress, cancel):
        with lock:
            runs.append(("start", job.id))
        time.sleep(0.1)
        with lock:
            runs.append(("end", job.id))
        return []

    app, queue = make_app(env, runners={"image": runner})
    guests = [guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")]
    responses = [None, None]
    barrier = threading.Barrier(2)

    def send(n):
        barrier.wait()
        responses[n] = upload_images(guests[n], [f"guest{n}.png"], {"mode": "fast"})

    threads = [threading.Thread(target=send, args=(n,)) for n in range(2)]
    with host_client(app):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert [r.status_code for r in responses] == [200, 200]
        ids = [r.json()["job"]["id"] for r in responses]
        wait_for(lambda: all(queue.get(i).status == "done" for i in ids))

    in_order = [job.id for job in sorted(queue.list(), key=lambda j: j.created)]
    assert runs == [("start", in_order[0]), ("end", in_order[0]), ("start", in_order[1]), ("end", in_order[1])]
    owners = {queue.get(i).owner for i in ids}
    assert len(owners) == 2 and all(owner.startswith("guest:") for owner in owners)


def test_guest_closing_tab_does_not_stop_job(env):
    app, _ = make_app(env)
    with host_client(app):
        guest = guest_client(app)
        job = upload_images(guest, ["a.png"], {"mode": "fast"}).json()["job"]
        guest.close()  # the guest goes away; the job is the server's now
        wait_for(lambda: job_status(host_client(app), job["id"]) == "done")


def test_guests_only_see_their_own_results(env):
    app, _ = make_app(env)
    with host_client(app) as host:
        alice, bob = guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
        job = upload_images(alice, ["alice.png"], {"mode": "fast"}).json()["job"]
        wait_for(lambda: job_status(alice, job["id"]) == "done")

        mine = alice.get(f"/api/jobs/{job['id']}").json()["job"]
        assert mine["mine"] is True and mine["results"] and mine["inputs"] == ["alice.png"]
        assert alice.get(f"/api/jobs/{job['id']}/results/0/file").status_code == 200

        [theirs] = [j for j in bob.get("/api/jobs").json()["jobs"] if j["id"] == job["id"]]
        assert theirs["mine"] is False
        assert theirs["status"] == "done"
        assert theirs["inputs"] == [] and theirs["results"] == []
        for path in ("results/0/file", "results/0/preview.jpg", "zip"):
            assert bob.get(f"/api/jobs/{job['id']}/{path}").status_code == 403

        assert host.get(f"/api/jobs/{job['id']}/results/0/file").status_code == 200  # the host sees all
    assert (env.guest_out / job["id"] / "alice-4K-fast.jpg").is_file()
    assert not (env.host_out / "alice-4K-fast.jpg").exists()  # never in the host's Pictures


def test_job_json_never_contains_a_guest_session(env):
    app, queue = make_app(env)
    with host_client(app) as host:
        alice, bob = guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
        job = upload_images(alice, ["alice.png"], {"mode": "fast"}).json()["job"]
        cookie = alice.cookies.get(server.SESSION_COOKIE)
        assert cookie and len(cookie) >= 32
        wait_for(lambda: job_status(alice, job["id"]) == "done")

        views = {
            "alice": alice.get(f"/api/jobs/{job['id']}").json()["job"],
            "bob": next(j for j in bob.get("/api/jobs").json()["jobs"] if j["id"] == job["id"]),
            "host": host.get(f"/api/jobs/{job['id']}").json()["job"],
        }
        for view in views.values():
            assert cookie not in json.dumps(view)
        assert (views["alice"]["mine"], views["bob"]["mine"], views["host"]["mine"]) == (True, False, False)
        assert views["alice"]["owner"].startswith("guest:") and views["alice"]["owner"] == views["host"]["owner"]
        assert views["bob"]["owner"] == "guest"  # other guests can't even tell whose job it is

        live = queue.get(job["id"])
        for viewer in (server.Viewer(True, "host", None, True), server.Viewer(False, "guest:other", "x" * 40, True)):
            assert cookie not in server._server_sent_event({"event": "job", "job": live}, viewer)
    assert cookie not in (env.store / job["id"] / "job.json").read_text()


def test_result_files_outside_the_app_folders_are_refused(env):
    job_id = "0123456789ab"
    record = env.store / job_id / "job.json"
    record.parent.mkdir(parents=True)
    secret = env.tmp / "secret.txt"
    secret.write_text("private")
    record.write_text(
        json.dumps(
            {
                "id": job_id,
                "kind": "image",
                "owner": "host",
                "status": "done",
                "created": time.time(),
                "finished": time.time(),
                "results": [{"path": str(secret), "filename": "secret.txt", "preview": str(secret)}],
            }
        )
    )
    app, _ = make_app(env)
    host = host_client(app)
    assert host.get(f"/api/jobs/{job_id}/results/0/file").status_code == 404
    assert host.get(f"/api/jobs/{job_id}/results/0/preview.jpg").status_code == 404
    assert host.get("/api/jobs/..%2F..%2Fetc/results/0/file").status_code == 404


def test_a_cancelled_batch_lists_the_pictures_finished_before(env, monkeypatch):
    reached, release = threading.Event(), threading.Event()
    finished_one = env.upscale_calls  # filled by the stand-in engine

    real_fake = pipeline_image.upscale_image

    def slow_second(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        if len(finished_one) == 1:
            reached.set()
            release.wait(5)
            if cancel.is_set():
                raise jobs.Cancelled()
        return real_fake(path, opts, out_dir, progress=progress, cancel=cancel)

    monkeypatch.setattr(pipeline_image, "upscale_image", slow_second)
    app, _ = make_app(env)
    with host_client(app) as host:
        job = upload_images(host, ["a.png", "b.png", "c.png"], {"mode": "fast"}).json()["job"]
        assert reached.wait(5)
        assert host.post(f"/api/jobs/{job['id']}/cancel").json()["ok"] is True
        release.set()
        wait_for(lambda: job_status(host, job["id"]) == "cancelled")
        cancelled = host.get(f"/api/jobs/{job['id']}").json()["job"]
        assert [r["filename"] for r in cancelled["results"]] == ["a-4K-fast.jpg"]
        assert host.get(f"/api/jobs/{job['id']}/results/0/file").status_code == 200
        assert host.get(f"/api/jobs/{job['id']}/zip").status_code == 200


def test_cancel_permissions_api(env):
    app, _ = make_app(env)  # not started, so jobs stay queued
    host, guest, other = host_client(app), guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
    host_job = upload_images(host, ["h.png"], {"mode": "fast"}).json()["job"]
    guest_job = upload_images(guest, ["g.png"], {"mode": "fast"}).json()["job"]

    assert guest.post(f"/api/jobs/{host_job['id']}/cancel").status_code == 403
    assert other.post(f"/api/jobs/{guest_job['id']}/cancel").status_code == 403
    assert host.post(f"/api/jobs/{guest_job['id']}/cancel").status_code == 200
    assert job_status(host, guest_job["id"]) == "cancelled"
    assert job_status(host, host_job["id"]) == "queued"
    assert host.post("/api/jobs/0123456789ab/cancel").status_code == 404


# --- Live updates -------------------------------------------------------------------


def _parse_sse(lines):
    event = {}
    for line in lines:
        if line.startswith("event:"):
            event["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            event["data"] = json.loads(line.split(":", 1)[1])
        elif not line and event:
            yield event
            event = {}


def test_events_stream_emits_job_updates(env):
    app, _ = make_app(env)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for(lambda: uv.started, timeout=10)
        with (
            httpx.Client(base_url=base, timeout=10) as listener,
            httpx.Client(base_url=base, timeout=10) as sender,
            listener.stream("GET", "/api/events") as stream,
        ):
            assert stream.headers["content-type"].startswith("text/event-stream")
            lines = stream.iter_lines()
            assert next(lines).startswith("retry:")  # subscribed

            files = [("files", ("a.png", png_bytes(), "image/png"))]
            job_id = sender.post("/api/images", files=files, data={"options": '{"mode": "fast"}'}).json()["job"]["id"]

            statuses = []
            for event in _parse_sse(lines):
                if event.get("event") == "job" and event["data"]["id"] == job_id:
                    statuses.append(event["data"]["status"])
                    if event["data"]["status"] == "done":
                        assert event["data"]["results"][0]["filename"] == "a-4K-fast.jpg"
                        break
            assert statuses[0] == "queued"
            assert "running" in statuses

            sender.post("/api/settings", json={"grain": "subtle"})
            for event in _parse_sse(lines):
                if event.get("event") == "status":
                    break
    finally:
        uv.should_exit = True
        thread.join(10)
    assert not thread.is_alive()


# --- Estimate --------------------------------------------------------------------------


@pytest.fixture
def roomy_disk(monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**12, 0, 10**12))


def test_estimate_endpoint(env, roomy_disk):
    app, _ = make_app(env)
    body = {"width": 2048, "height": 1539, "size": "16k", "mode": "photoreal", "output_format": "tiff", "dpi": 300}

    rich = guest_client(app).post("/api/estimate", json={**body, "photoreal_level": "rich"})
    assert rich.status_code == 200
    rich = rich.json()
    assert rich.keys() == {"out", "megapixels", "seconds", "file_mb", "print", "passes", "warnings", "room_mb"}
    assert rich["room_mb"] == round((10**12 - SPACE_RESERVE_BYTES) / 1e6, 1)  # for a warning about a whole batch
    assert rich["out"] == [16384, 12312] and rich["megapixels"] == 201.7 and rich["passes"] == 2
    assert rich["print"] == {"inches": [54.6, 41.0], "cm": [138.7, 104.2], "dpi": 300}
    assert any("16K files are very large" in w for w in rich["warnings"])

    balanced = host_client(app).post("/api/estimate", json=body).json()  # the default level
    assert balanced["seconds"] < rich["seconds"] / 2

    print_150 = host_client(app).post("/api/estimate", json={**body, "dpi": 150}).json()
    assert print_150["print"]["inches"] == [109.2, 82.1]


def test_estimate_cost_for_cloud_modes(env, roomy_disk):
    app, _ = make_app(env)
    host = host_client(app)

    def cost(**change):
        body = {"width": 2048, "height": 1536, "size": "8k", "mode": "cloud:fal:topaz_precision", **change}
        response = host.post("/api/estimate", json=body)
        assert response.status_code == 200
        return response.json()["cost_usd"]

    topaz = next(m for m in PROVIDERS["fal"].models if m.id == "topaz_precision")
    assert cost() == pytest.approx(topaz.est_cost_8k_usd)  # 8192 x 6144 is the 8K the prices are for
    assert cost(size="4k") == pytest.approx(0.08)  # 3840 x 2880 is 11 MP, but Topaz charges per started 24 MP
    # 16K: the service makes 4x (8192 x 6144) and the Fast model on this computer does the rest, for free.
    assert cost(size="16k") == pytest.approx(topaz.est_cost_8k_usd)
    assert cost(width=9000, height=6000) == 0  # already big enough: nothing is sent
    assert cost(is_vector=True) == 0  # drawn here at full size, never sent
    assert cost(mode="cloud:magnific:precision_v2") is None  # comes out of the plan's credits; no price
    local = host.post("/api/estimate", json={"width": 2048, "height": 1536, "size": "8k", "mode": "fast"})
    assert "cost_usd" not in local.json()


# What each service charges for a 2048 x 1536 picture, worked out from its published prices (see the
# notes at the top of cloud/fal.py, replicate.py and magnific.py), not from a price per 8K picture.
@pytest.mark.parametrize(
    ("mode", "size", "price"),
    [
        ("cloud:fal:topaz_precision", "4k", 0.08),  # 3840 x 2880, 11 MP: one started 24 MP
        ("cloud:fal:topaz_precision", "6k", 0.16),  # 6144 x 4608, 28 MP: two
        ("cloud:fal:topaz_precision", "8k", 0.24),  # 8192 x 6144, 50 MP: three
        ("cloud:fal:topaz_generative", "4k", 0.16),  # Wonder: per started 8 MP
        ("cloud:fal:topaz_generative", "6k", 0.32),
        ("cloud:fal:topaz_generative", "8k", 0.56),
        ("cloud:fal:seedvr2", "4k", 0.011),  # $0.001 per megapixel made
        ("cloud:fal:seedvr2", "6k", 0.028),
        ("cloud:fal:seedvr2", "8k", 0.050),
        ("cloud:fal:crystal", "4k", 0.177),  # $0.016 per megapixel made
        ("cloud:fal:crystal", "8k", 0.805),
        ("cloud:replicate:topaz_hf", "4k", 0.05),  # Replicate makes 2x: 4096 x 3072, 12.6 MP
        ("cloud:replicate:topaz_hf", "6k", 0.15),  # 4x: 8192 x 6144, 50.3 MP, priced as its 60 MP row
        ("cloud:replicate:topaz_hf", "8k", 0.15),
        ("cloud:magnific:creative", "4k", 0.40),  # 2x: 12.6 MP, about €0.35
        ("cloud:magnific:creative", "6k", 0.70),  # it stops at 25.3 MP, about €0.61
        ("cloud:magnific:creative", "8k", 0.70),
    ],
)
def test_estimate_cost_follows_each_services_own_pricing(env, roomy_disk, mode, size, price):
    app, _ = make_app(env)
    body = {"width": 2048, "height": 1536, "size": size, "mode": mode}
    assert host_client(app).post("/api/estimate", json=body).json()["cost_usd"] == pytest.approx(price, abs=0.006)


def test_estimate_cost_of_a_small_picture_is_the_services_least_charge(env, roomy_disk):
    app, _ = make_app(env)
    body = {"width": 1200, "height": 900, "size": "2x"}  # 2400 x 1800 is only 4.3 MP
    for mode, least in (("cloud:fal:topaz_precision", 0.08), ("cloud:replicate:topaz_hf", 0.05)):
        assert host_client(app).post("/api/estimate", json={**body, "mode": mode}).json()["cost_usd"] == least


def test_estimate_for_a_vector_picture(env, roomy_disk):
    app, _ = make_app(env)
    host = host_client(app)
    body = {"width": 10000, "height": 5000, "size": "8k", "mode": "photoreal", "is_vector": True}
    est = host.post("/api/estimate", json=body).json()
    assert est["out"] == [8192, 4096]  # drawn at exactly the size asked for, even smaller than declared
    assert est["passes"] == 0 and est["seconds"] < 10
    env.hardware.value = MAC_8GB
    est = host.post("/api/estimate", json={**body, "width": 500, "height": 250}).json()
    assert not any("Faithful" in w for w in est["warnings"])  # no AI, so no fallback to mention
    assert host.post("/api/estimate", json={**body, "is_vector": "yes"}).status_code == 400


def test_estimate_uses_saved_defaults(env, roomy_disk):
    app, _ = make_app(env)
    host = host_client(app)
    host.post("/api/settings", json={"output_format": "png"})
    est = host.post("/api/estimate", json={"width": 1000, "height": 500, "size": "2x", "mode": "fast"}).json()
    assert est["out"] == [2000, 1000] and est["file_mb"] == pytest.approx(3.2)


def test_estimate_warns_when_too_big(env, roomy_disk):
    app, _ = make_app(env)
    est = host_client(app).post("/api/estimate", json={"width": 12000, "height": 9000, "size": "4x", "mode": "fast"})
    assert est.status_code == 200
    assert any("too big" in w for w in est.json()["warnings"])


@pytest.mark.parametrize(
    "change",
    [
        {"size": "10k"},
        {"mode": "magic"},
        {"output_format": "gif"},
        {"dpi": 100},
        {"photoreal_level": "extreme"},
        {"width": 0},
        {"height": -5},
        {"width": "wide"},
        {"width": 10**7},
        {"size": ["8k"]},
        {"size": {"a": 1}},
        {"photoreal_level": ["rich"]},
    ],
)
def test_estimate_refuses_bad_input(env, roomy_disk, change):
    app, _ = make_app(env)
    body = {"width": 2048, "height": 1539, "size": "8k", "mode": "fast", **change}
    response = host_client(app).post("/api/estimate", json=body)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str) and response.json()["detail"]


def test_estimate_needs_login_for_guests(env, roomy_disk):
    app, _ = make_app(env)
    host_client(app).post("/api/settings", json={"access_password": "sesame"})
    response = guest_client(app).post("/api/estimate", json={"width": 100, "height": 100})
    assert response.status_code == 401


# --- Sample pictures ---------------------------------------------------------------------


def test_sample_job_creates_from_bundle(env):
    app, queue = make_app(env)
    with host_client(app) as host:
        body = {"sample_id": "interior", "options": {"mode": "fast", "size": "4k"}}
        response = host.post("/api/images/sample", json=body)
        assert response.status_code == 200
        job = response.json()["job"]
        assert job["kind"] == "image" and job["inputs"] == ["interior.jpg"] and job["mine"] is True
        assert job["options"]["mode"] == "fast" and job["options"]["size"] == "4k"

        [copied] = map(Path, queue.get(job["id"]).inputs)
        assert copied.is_relative_to(env.store / job["id"] / "in")
        assert copied.read_bytes() == (server.SAMPLES_DIR / "interior.jpg").read_bytes()  # a copy, not the bundle
        wait_for(lambda: job_status(host, job["id"]) == "done")
    assert (server.SAMPLES_DIR / "interior.jpg").is_file()  # the bundled sample is never removed
    assert env.upscale_calls[-1].path.name == "interior.jpg"


def test_sample_job_for_guests_and_bad_requests(env):
    app, queue = make_app(env)
    guest = guest_client(app)
    job = guest.post("/api/images/sample", json={"sample_id": "food", "options": {}}).json()["job"]
    assert queue.get(job["id"]).owner.startswith("guest:")
    assert job["options"]["mode"] == "photoreal"  # the usual defaults

    unknown = guest.post("/api/images/sample", json={"sample_id": "../../etc/passwd", "options": {}})
    assert unknown.status_code == 404 and unknown.json()["detail"]
    bad = guest.post("/api/images/sample", json={"sample_id": "food", "options": {"size": "10k"}})
    assert bad.status_code == 400
    not_text = guest.post("/api/images/sample", json={"sample_id": "food", "options": {"size": ["8k"]}})
    assert not_text.status_code == 400 and not_text.json()["detail"]
    cloud = guest.post("/api/images/sample", json={"sample_id": "food", "options": {"mode": "cloud:fal:seedvr2"}})
    assert cloud.status_code == 403
    assert len(queue.list()) == 1
    assert [p.name for p in env.store.iterdir()] == [job["id"]]  # nothing left behind by the refusals


# --- Checking a picture on drop, and staged pictures ---------------------------------------


def photo_bytes(width: int = 320, height: int = 240, fmt: str = "PNG", seed: int = 0, alpha: bool = False) -> bytes:
    """A small photo-like picture: soft gradients plus a little noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(40, 215, width, dtype=np.float32)
    y = np.linspace(30, 200, height, dtype=np.float32)[:, None]
    base = np.stack(np.broadcast_arrays(x + 0 * y, y + 0 * x, (x + y) / 2), axis=-1)
    rgb = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    if alpha:
        mask = np.full((height, width), 255, np.uint8)
        mask[:, : width // 4] = 0
        image = Image.fromarray(np.dstack([rgb, mask]))
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


SVG_LOGO = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">'
    b'<rect width="200" height="100" fill="#3366cc"/><circle cx="100" cy="50" r="30" fill="#ffcc00"/></svg>'
)


def docx_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as docx:
        docx.writestr("word/document.xml", "<w:document/>")
    return buffer.getvalue()


def inspect(client: TestClient, name: str, data: bytes, content_type: str = "application/octet-stream"):
    return client.post("/api/inspect", files={"file": (name, data, content_type)})


def staging(env) -> Path:
    return env.store / "staging"


def job_folders(env) -> list[str]:
    return sorted(p.name for p in env.store.iterdir() if p.name != "staging") if env.store.exists() else []


def test_inspect_endpoint_returns_thumb(env):
    app, queue = make_app(env)
    host = host_client(app)
    data = photo_bytes(640, 480)

    response = inspect(host, "living room.png", data, "image/png")

    assert response.status_code == 200
    info = response.json()
    assert info.keys() == {
        "token", "name", "width", "height", "format", "has_alpha", "frames", "is_vector", "warnings",
        "thumb_url", "suggestion",
    }  # fmt: skip
    assert (info["name"], info["width"], info["height"], info["format"]) == ("living room.png", 640, 480, "PNG")
    assert (info["has_alpha"], info["frames"], info["is_vector"], info["warnings"]) == (False, 1, False, [])
    assert server._STAGING_TOKEN.fullmatch(info["token"])
    assert info["thumb_url"] == f"/api/staging/{info['token']}/thumb.jpg"

    # The suggestion is the auto-suggest rules applied to this very picture.
    expected = suggest.suggest(
        server.inspect_image(staging(env) / info["token"] / "in" / "0" / "living room.png"),
        suggest.analyze(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))),
    )
    assert info["suggestion"] == {
        "mode": expected.mode,
        "photoreal_level": expected.photoreal_level,
        "reasons": expected.reasons,
        "faces": expected.signals.faces,
    }
    assert info["suggestion"]["reasons"]

    thumb = host.get(info["thumb_url"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    with Image.open(io.BytesIO(thumb.content)) as picture:
        assert picture.format == "JPEG" and picture.size == (480, 360)
    assert (staging(env) / info["token"]).is_dir()
    assert queue.list() == []  # looking at a picture makes no job


def test_inspect_transparent_and_vector_pictures(env):
    app, _ = make_app(env)
    host = host_client(app)

    logo = inspect(host, "logo.png", photo_bytes(alpha=True)).json()
    assert logo["has_alpha"] is True

    vector = inspect(host, "badge.svg", SVG_LOGO, "image/svg+xml")
    assert vector.status_code == 200
    vector = vector.json()
    assert (vector["is_vector"], vector["format"], vector["width"], vector["height"]) == (True, "SVG", 200, 100)
    assert vector["suggestion"]["mode"] == "fast"
    assert "vector" in vector["suggestion"]["reasons"][0]
    with Image.open(io.BytesIO(host.get(vector["thumb_url"]).content)) as thumb:
        assert thumb.size == (480, 240)


def test_inspect_still_answers_when_the_suggestion_breaks(env, monkeypatch):
    def broken(rgb):
        raise RuntimeError("boom")

    monkeypatch.setattr(suggest, "analyze", broken)
    app, _ = make_app(env)
    response = inspect(host_client(app), "photo.jpg", photo_bytes(fmt="JPEG"))
    assert response.status_code == 200
    assert response.json()["suggestion"] is None and response.json()["width"] == 320


@pytest.mark.parametrize(
    ("name", "data", "words"),
    [
        ("letter.docx", docx_bytes(), "This file type isn't an image Re:Solution can read"),
        ("menu.pdf", b"%PDF-1.7\n" + b"\x00" * 64, "Export the page as PNG or TIFF"),
        ("holiday.mov", b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64, "Videos tab"),
        ("broken.png", b"not really a png", "couldn't be opened as a picture"),
    ],
    ids=["document", "pdf", "video", "damaged"],
)
def test_inspect_refuses_files_it_cant_read(env, name, data, words):
    app, queue = make_app(env)
    response = inspect(guest_client(app), name, data)
    assert response.status_code == 400
    assert words in response.json()["detail"] and name in response.json()["detail"]
    assert queue.list() == []
    assert not staging(env).exists() or list(staging(env).iterdir()) == []  # nothing kept


def test_unsupported_file_friendly_message(env):
    app, queue = make_app(env)
    host = host_client(app)
    message = "This file type isn't an image Re:Solution can read (letter.docx)."

    on_drop = inspect(host, "letter.docx", docx_bytes())
    assert on_drop.status_code == 400 and on_drop.json()["detail"].startswith(message)

    sent = host.post(
        "/api/images",
        files=[("files", ("photo.png", photo_bytes(), "image/png")), ("files", ("letter.docx", docx_bytes(), "x"))],
        data={"options": "{}"},
    )
    assert sent.status_code == 400 and sent.json()["detail"].startswith(message)
    assert queue.list() == []
    assert job_folders(env) == []


def test_upload_without_a_known_extension_is_checked_by_its_content(env):
    app, queue = make_app(env)
    host = host_client(app)
    response = upload_images(host, ["pasted image"], data=photo_bytes())
    assert response.status_code == 200 and response.json()["job"]["inputs"] == ["pasted image"]
    refused = upload_images(host, ["notes.txt"], data=b"shopping list")
    assert refused.status_code == 400 and "isn't an image" in refused.json()["detail"]
    assert len(queue.list()) == 1


def test_inspect_too_big_is_refused(env, monkeypatch):
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 100)
    app, _ = make_app(env)
    response = inspect(host_client(app), "huge.png", photo_bytes())
    assert response.status_code == 413 and "huge.png" in response.json()["detail"]
    assert not staging(env).exists() or list(staging(env).iterdir()) == []


def test_images_from_tokens(env):
    app, queue = make_app(env)
    with host_client(app) as host:
        first = inspect(host, "image.png", photo_bytes(seed=1)).json()
        second = inspect(host, "image.png", photo_bytes(seed=2)).json()
        body = {"tokens": [first["token"], second["token"]], "options": {"mode": "fast", "size": "4k"}}

        response = host.post("/api/images", json=body)

        assert response.status_code == 200
        job = response.json()["job"]
        assert (job["kind"], job["inputs"], job["mine"]) == ("image", ["image.png", "image.png"], True)
        assert job["options"]["mode"] == "fast" and job["options"]["size"] == "4k"
        saved = [Path(p) for p in queue.get(job["id"]).inputs]
        assert len(set(saved)) == 2 and all(p.is_relative_to(env.store / job["id"] / "in") for p in saved)
        wait_for(lambda: job_status(host, job["id"]) == "done")
        done = host.get(f"/api/jobs/{job['id']}").json()["job"]
        assert [r["filename"] for r in done["results"]] == ["image-4K-fast.jpg", "image-4K-fast-2.jpg"]
        # Each job worked on its own picture (the fake engine copies its input into the result).
        assert (env.host_out / "image-4K-fast.jpg").read_bytes() == b"upscaled " + photo_bytes(seed=1)
        assert (env.host_out / "image-4K-fast-2.jpg").read_bytes() == b"upscaled " + photo_bytes(seed=2)

        # The staged pictures stay, for a preview or another go with other settings.
        assert host.get(first["thumb_url"]).status_code == 200
        again = host.post("/api/images", json={"tokens": [first["token"]], "options": {"mode": "faithful"}})
        assert again.status_code == 200
        wait_for(lambda: job_status(host, again.json()["job"]["id"]) == "done")
    assert [call.opts.mode for call in env.upscale_calls] == ["fast", "fast", "faithful"]


@pytest.mark.parametrize(
    "body",
    [
        {"tokens": [], "options": {}},
        {"tokens": "abc", "options": {}},
        {"tokens": [7], "options": {}},
        {"tokens": ["../../etc/passwd"], "options": {}},
        {"tokens": ["A" * 32], "options": {}},  # well formed, but nothing staged under it
        {"options": {}},
        ["not", "an", "object"],
    ],
)
def test_images_from_tokens_bad_requests(env, body):
    app, queue = make_app(env)
    response = host_client(app).post("/api/images", json=body)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str) and response.json()["detail"]
    assert queue.list() == [] and job_folders(env) == []


def test_images_from_tokens_checks_options_and_batch_size(env, monkeypatch):
    app, queue = make_app(env)
    host = host_client(app)
    token = inspect(host, "a.png", photo_bytes()).json()["token"]
    bad = host.post("/api/images", json={"tokens": [token], "options": {"size": "10k"}})
    assert bad.status_code == 400
    monkeypatch.setattr(server, "MAX_IMAGES_PER_BATCH", 2)
    too_many = host.post("/api/images", json={"tokens": [token] * 3, "options": {}})
    assert too_many.status_code == 400 and "at most 2" in too_many.json()["detail"]
    assert queue.list() == [] and job_folders(env) == []


def test_staged_pictures_are_private(env):
    app, queue = make_app(env)
    alice, bob = guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
    info = inspect(alice, "alice.png", photo_bytes()).json()

    assert alice.get(info["thumb_url"]).status_code == 200
    assert bob.get(info["thumb_url"]).status_code == 404
    assert bob.post("/api/images", json={"tokens": [info["token"]], "options": {}}).status_code == 400
    assert bob.post("/api/preview", json={"token": info["token"]}).status_code == 400
    assert alice.get("/api/staging/..%2F..%2Fsecret/thumb.jpg").status_code == 404
    assert alice.get(f"/api/staging/{'B' * 32}/thumb.jpg").status_code == 404
    assert queue.list() == []
    record = (staging(env) / info["token"] / "info.json").read_text()
    assert alice.cookies.get(server.SESSION_COOKIE) not in record  # an opaque owner id, never the cookie


def test_staging_is_tidied_after_24_hours(env):
    app, _ = make_app(env)
    host = host_client(app)
    old = inspect(host, "old.png", photo_bytes()).json()["token"]
    new = inspect(host, "new.png", photo_bytes()).json()["token"]
    record = staging(env) / old / "info.json"
    data = json.loads(record.read_text())
    data["created"] -= 25 * 3600
    record.write_text(json.dumps(data))
    stray = staging(env) / ("C" * 32)  # a staged folder an interrupted upload left without its record
    stray.mkdir()
    long_ago = time.time() - 25 * 3600
    os.utime(stray, (long_ago, long_ago))

    with host_client(app):  # tidied when the app starts, then every hour
        wait_for(lambda: not (staging(env) / old).exists() and not stray.exists())
    assert (staging(env) / new).is_dir()
    assert host.post("/api/images", json={"tokens": [old], "options": {}}).status_code == 400


# --- Formats ---------------------------------------------------------------------------------


def test_webp_and_same_as_original_formats(env, roomy_disk):
    app, _ = make_app(env)
    host = host_client(app)
    for fmt in ("webp", "original"):
        job = upload_images(host, ["a.png"], {"mode": "fast", "output_format": fmt}).json()["job"]
        assert job["options"]["output_format"] == fmt
    assert host.post("/api/settings", json={"output_format": "original"}).status_code == 200
    assert host.get("/api/settings").json()["settings"]["output_format"] == "original"

    body = {"width": 2048, "height": 1539, "mode": "fast"}
    big = host.post("/api/estimate", json={**body, "size": "16k", "output_format": "webp"})
    assert big.status_code == 200
    assert any(w.startswith("WebP can't hold pictures bigger than 16,383 pixels") for w in big.json()["warnings"])
    small = host.post("/api/estimate", json={**body, "size": "8k", "output_format": "webp"}).json()
    jpeg = host.post("/api/estimate", json={**body, "size": "8k", "output_format": "jpeg"}).json()
    assert not any("WebP" in w for w in small["warnings"])
    assert small["file_mb"] == jpeg["file_mb"]  # a lossy file, about the size of a JPEG
    original = host.post("/api/estimate", json={**body, "size": "8k", "output_format": "original"})
    assert original.status_code == 200 and original.json()["out"] == [8192, 6156]


# --- Preview styles ----------------------------------------------------------------------------


@pytest.fixture
def fake_styles(monkeypatch):
    """Stands in for the engines behind the preview crops: a plain enlargement, plus a record of each call."""
    calls = []

    def upscale_array(rgb, route, out_size, *, progress, cancel, creativity=0.0):
        calls.append(SimpleNamespace(route=route, shape=rgb.shape, out_size=tuple(out_size), creativity=creativity))
        progress(0.5, "Working")
        return np.asarray(Image.fromarray(rgb).resize(tuple(out_size), Image.LANCZOS))

    monkeypatch.setattr(pipeline_image, "upscale_array", upscale_array)
    return calls


def test_preview_job_runs_routes(env, fake_styles):
    app, queue = make_app(env)
    with host_client(app) as host:
        token = inspect(host, "room.png", photo_bytes(900, 600)).json()["token"]
        routes = ["fast", "faithful", "photoreal:balanced"]
        request = {"token": token, "routes": routes, "size": "16k", "creativity": 0.3}

        response = host.post("/api/preview", json=request)

        assert response.status_code == 200
        job = response.json()["job"]
        assert (job["kind"], job["inputs"], job["mine"]) == ("preview", ["room.png"], True)
        assert job["options"] == {
            "routes": ["fast", "faithful", "photoreal:balanced"],
            "region": None,
            "size": "16k",
            "creativity": 0.3,
        }
        wait_for(lambda: job_status(host, job["id"]) == "done")
        results = host.get(f"/api/jobs/{job['id']}").json()["job"]["results"]

        assert [r["route"] for r in results] == ["original", "fast", "faithful", "photoreal:balanced"]
        assert [r["label"] for r in results] == [
            "Original (enlarged)",
            "Fast",
            pipeline_image.MODELS["faithful"].label,
            "Photo-real · Balanced",
        ]
        for n, result in enumerate(results):
            assert result["url"] == f"/api/jobs/{job['id']}/results/{n}/file"
            assert {"route", "label", "url", "seconds", "est_full_seconds"} <= result.keys()
            assert not {"path", "crop_path"} & result.keys()  # no file system paths leave the server
            crop = host.get(result["url"])
            assert crop.status_code == 200
            with Image.open(io.BytesIO(crop.content)) as picture:
                assert (picture.format, picture.size) == ("JPEG", (1024, 1024))
        assert results[0]["est_full_seconds"] is None
        fast, faithful, balanced = (r["est_full_seconds"] for r in results[1:])
        assert 0 < fast < faithful and 0 < balanced
    assert [(c.route, c.shape, c.out_size) for c in fake_styles] == [
        (route, (256, 256, 3), (1024, 1024)) for route in ("fast", "faithful", "photoreal:balanced")
    ]
    assert [c.creativity for c in fake_styles] == [0.0, 0.0, 0.3]  # creativity is for Photo-real only
    assert not (env.store / job["id"] / "in").exists()  # the copy of the picture goes once the job is over
    assert (staging(env) / token).is_dir()  # the staged picture stays for the real job


def test_preview_default_routes_region_and_samples(env, fake_styles):
    app, queue = make_app(env)
    with host_client(app) as host:
        job = host.post("/api/preview", json={"sample_id": "interior", "region": [10, 20, 256, 256]}).json()["job"]
        assert job["inputs"] == ["interior.jpg"] and job["options"]["routes"] is None
        assert job["options"]["region"] == [10, 20, 256, 256] and job["options"]["size"] == "8k"
        wait_for(lambda: job_status(host, job["id"]) == "done")
        results = host.get(f"/api/jobs/{job['id']}").json()["job"]["results"]
        assert [r["route"] for r in results] == ["original", *preview.PREVIEW_ROUTES]

        env.hardware.value = MAC_8GB  # no Photo-real here: those styles are left out
        small = host.post("/api/preview", json={"sample_id": "interior"}).json()["job"]
        wait_for(lambda: job_status(host, small["id"]) == "done")
        results = host.get(f"/api/jobs/{small['id']}").json()["job"]["results"]
        assert [r["route"] for r in results] == ["original", "faithful", "fast"]
    assert (server.SAMPLES_DIR / "interior.jpg").is_file()


def test_preview_style_that_cannot_run_has_no_url(env, monkeypatch):
    from upscaler.seedvr2 import SeedVR2Unavailable

    def upscale_array(rgb, route, out_size, *, progress, cancel, creativity=0.0):
        if route.startswith("photoreal"):
            raise SeedVR2Unavailable("Photo-real hasn't been downloaded yet.")
        return np.asarray(Image.fromarray(rgb).resize(tuple(out_size)))

    monkeypatch.setattr(pipeline_image, "upscale_array", upscale_array)
    app, _ = make_app(env)
    with host_client(app) as host:
        token = inspect(host, "room.png", photo_bytes()).json()["token"]
        job = host.post("/api/preview", json={"token": token, "routes": ["photoreal:rich", "fast"]}).json()["job"]
        wait_for(lambda: job_status(host, job["id"]) == "done")
        original, rich, fast = host.get(f"/api/jobs/{job['id']}").json()["job"]["results"]
        assert rich["error"] == "Photo-real hasn't been downloaded yet." and rich["url"] is None
        assert host.get(f"/api/jobs/{job['id']}/results/1/file").status_code == 404
        assert fast["url"] and "error" not in fast and original["url"]


def test_preview_jumps_queue(env):
    events = []
    release = threading.Event()

    def image_runner(job, progress, cancel):
        name = Path(job.inputs[0]).name
        events.append(("image", name))
        if name == "first.png":
            release.wait(5)
        return []

    def preview_runner(job, progress, cancel):
        events.append(("preview", Path(job.inputs[0]).name))
        return []

    app, queue = make_app(env, runners={"image": image_runner, "preview": preview_runner})
    with host_client(app) as host:
        first = upload_images(host, ["first.png"], {"mode": "fast"}).json()["job"]
        wait_for(lambda: job_status(host, first["id"]) == "running")
        second = upload_images(host, ["second.png"], {"mode": "fast"}).json()["job"]
        token = inspect(host, "look.png", photo_bytes()).json()["token"]
        look = host.post("/api/preview", json={"token": token, "routes": ["fast"]}).json()["job"]
        time.sleep(0.1)
        assert job_status(host, first["id"]) == "running"  # never interrupted
        assert job_status(host, second["id"]) == job_status(host, look["id"]) == "queued"
        release.set()
        wait_for(lambda: all(job_status(host, j["id"]) == "done" for j in (first, second, look)))
    assert events == [("image", "first.png"), ("preview", "look.png"), ("image", "second.png")]


@pytest.mark.parametrize(
    "change",
    [
        {"token": None},  # neither a picture nor a sample
        {"sample_id": "interior"},  # both
        {"routes": ["magic"]},
        {"routes": ["original"]},
        {"routes": "fast"},
        {"routes": []},
        {"routes": [["fast"]]},
        {"region": [1, 2, 3]},
        {"region": ["a", 0, 10, 10]},
        {"region": [0, 0, 0, 10]},
        {"region": [0, 0, True, 10]},
        {"size": "10k"},
        {"size": ["8k"]},
        {"creativity": 2},
        {"token": "A" * 32},  # nothing staged under it
        {"token": "../../etc"},
    ],
)
def test_preview_refuses_bad_requests(env, change):
    app, queue = make_app(env)
    host = host_client(app)
    token = inspect(host, "room.png", photo_bytes()).json()["token"]
    body = {key: value for key, value in {"token": token, **change}.items() if value is not None}
    response = host.post("/api/preview", json=body)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str) and response.json()["detail"]
    assert queue.list() == [] and job_folders(env) == []


def test_preview_unknown_sample_and_login(env):
    app, queue = make_app(env)
    host = host_client(app)
    assert host.post("/api/preview", json={"sample_id": "nope"}).status_code == 404
    host.post("/api/settings", json={"access_password": "sesame"})
    guest = guest_client(app)
    assert guest.post("/api/preview", json={"sample_id": "interior"}).status_code == 401
    assert inspect(guest, "a.png", photo_bytes()).status_code == 401
    assert queue.list() == []


# --- Photo-real download ----------------------------------------------------------------


def test_photoreal_download_job(env):
    app, _ = make_app(env)
    with host_client(app) as host:
        first = host.post("/api/photoreal/download").json()["job"]
        assert first["kind"] == "download"
        wait_for(lambda: job_status(host, first["id"]) == "done")
        assert host.get("/api/status").json()["photoreal"]["downloaded"] is True
    assert env.engine.downloaded


def test_photoreal_download_is_started_only_once(env):
    app, _ = make_app(env)  # not started: the first download stays queued
    host = host_client(app)
    first = host.post("/api/photoreal/download").json()["job"]
    again = host.post("/api/photoreal/download").json()["job"]
    assert again["id"] == first["id"]


def test_only_the_host_can_start_the_photoreal_download(env):
    # 7 GB of the host's disk and internet: the host's decision, like installing anything else.
    app, queue = make_app(env)
    refused = guest_client(app).post("/api/photoreal/download")
    assert refused.status_code == 403 and refused.json()["detail"] == server.PHOTOREAL_HOST_ONLY
    assert "Ask the person at the main computer" in server.PHOTOREAL_HOST_ONLY
    assert queue.list() == []


# --- Cloud keys -------------------------------------------------------------------------


def test_provider_key_is_checked_then_stored(env, monkeypatch):
    monkeypatch.setattr(
        PROVIDERS["fal"], "check_key", lambda key: (key == "good:key", "fal.ai didn't accept this key.")
    )
    app, _ = make_app(env)
    host = host_client(app)

    refused = host.post("/api/providers/fal/key", json={"key": "bad"})
    assert refused.status_code == 400
    assert refused.json()["detail"] == "fal.ai didn't accept this key."
    assert env.keyring.store == {}

    assert host.post("/api/providers/fal/key", json={"key": "good:key"}).json()["ok"] is True
    assert env.keyring.store == {("Upscaler", "fal"): "good:key"}
    status = host.get("/api/status").json()
    assert next(p for p in status["providers"] if p["id"] == "fal")["connected"] is True
    assert next(m for m in status["image_modes"] if m["id"] == "cloud:fal:seedvr2")["available"] is True
    assert "good:key" not in json.dumps(status)

    assert host.delete("/api/providers/fal/key").status_code == 200
    assert env.keyring.store == {}
    status = host.get("/api/status").json()
    assert next(p for p in status["providers"] if p["id"] == "fal")["connected"] is False
    assert host.post("/api/providers/nope/key", json={"key": "x"}).status_code == 404


def test_key_check_that_breaks_is_a_friendly_400_and_nothing_is_stored(env, monkeypatch):
    def broken(key):
        raise RuntimeError("TLS handshake failed")

    monkeypatch.setattr(PROVIDERS["fal"], "check_key", broken)
    app, _ = make_app(env)
    host, guest = host_client(app), guest_client(app)

    for client, path in ((host, "/api/providers/fal/key"), (guest, "/api/providers/fal/session-key")):
        response = client.post(path, json={"key": "abc:def"})
        assert response.status_code == 400
        assert "fal.ai" in response.json()["detail"] and "TLS" not in response.json()["detail"]
    assert env.keyring.store == {}
    for client in (host, guest):
        fal = next(p for p in client.get("/api/status").json()["providers"] if p["id"] == "fal")
        assert fal["connected"] is False


def test_rejected_session_key_is_not_kept(env, monkeypatch):
    monkeypatch.setattr(PROVIDERS["fal"], "check_key", lambda key: (False, "fal.ai didn't accept this key."))
    app, _ = make_app(env)
    guest = guest_client(app)
    response = guest.post("/api/providers/fal/session-key", json={"key": "nope"})
    assert response.status_code == 400 and response.json()["detail"] == "fal.ai didn't accept this key."
    assert app.state.upscaler.session_keys == {}


def test_host_key_counts_for_guests_only_when_allowed(env):
    app, _ = make_app(env)
    host, guest = host_client(app), guest_client(app)
    env.keyring.store[("Upscaler", "fal")] = "host:key"

    def fal_connected(client):
        return next(p for p in client.get("/api/status").json()["providers"] if p["id"] == "fal")["connected"]

    assert fal_connected(host) is True
    assert fal_connected(guest) is False  # the host's key isn't the guests' to use...
    host.post("/api/settings", json={"access_password": "sesame", "allow_guests_cloud": True})
    guest.post("/api/login", json={"password": "sesame"})
    assert fal_connected(guest) is True  # ...until the host allows it


def test_guest_session_key_stays_in_memory(env, monkeypatch):
    monkeypatch.setattr(
        PROVIDERS["fal"], "check_key", lambda key: (key == "guest:key", "fal.ai didn't accept this key.")
    )
    app, queue = make_app(env)
    guest, other = guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")

    assert upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).status_code == 403
    assert guest.post("/api/providers/fal/session-key", json={"key": "nope"}).status_code == 400
    assert guest.post("/api/providers/fal/session-key", json={"key": "guest:key"}).json()["ok"] is True
    assert env.keyring.store == {}  # never saved

    def fal_connected(client):
        return next(p for p in client.get("/api/status").json()["providers"] if p["id"] == "fal")["connected"]

    assert fal_connected(guest) is True
    assert fal_connected(other) is False
    assert fal_connected(host_client(app)) is False

    job = upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).json()["job"]
    assert queue.get(job["id"]).secret == "guest:key"
    assert "guest:key" not in (env.store / job["id"] / "job.json").read_text()
    with host_client(app):
        wait_for(lambda: queue.get(job["id"]).status == "done")
    assert env.upscale_calls[-1].opts.guest_key == "guest:key"


def test_providers_say_whether_the_key_is_the_viewers_own(env, fal_accepts):
    app, _ = make_app(env)
    host, guest, other = host_client(app), guest_client(app, "192.168.1.5"), guest_client(app, "192.168.1.6")
    env.keyring.store[("Upscaler", "replicate")] = "host:key"
    host.post("/api/settings", json={"access_password": "sesame", "allow_guests_cloud": True})
    for client in (guest, other):
        client.post("/api/login", json={"password": "sesame"})
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})

    def own(client):
        return {p["id"]: (p["connected"], p["own_key"]) for p in client.get("/api/status").json()["providers"]}

    assert own(guest) == {"fal": (True, True), "replicate": (True, False), "magnific": (False, False)}
    assert own(other) == {"fal": (False, False), "replicate": (True, False), "magnific": (False, False)}
    assert own(host) == {"fal": (False, False), "replicate": (True, False), "magnific": (False, False)}
    guest.delete("/api/providers/fal/session-key")  # "Forget my key"
    assert own(guest)["fal"] == (False, False)


def test_guests_may_use_host_cloud_when_allowed(env):
    app, queue = make_app(env)
    host, guest = host_client(app), guest_client(app)
    env.keyring.store[("Upscaler", "fal")] = "host:key"
    host.post("/api/settings", json={"access_password": "sesame", "allow_guests_cloud": True})
    guest.post("/api/login", json={"password": "sesame"})
    fal = next(p for p in guest.get("/api/status").json()["providers"] if p["id"] == "fal")
    assert fal["connected"] is True
    job = upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).json()["job"]
    assert queue.get(job["id"]).secret is None  # the pipeline uses the host's key


def test_guide_endpoint(env):
    app, _ = make_app(env)
    guide = guest_client(app).get("/api/providers/magnific/guide").json()
    assert guide["provider_id"] == "magnific"
    assert len(guide["steps"]) >= 4
    assert {"title", "body", "link_label", "link_url"} <= guide["steps"][0].keys()
    assert guest_client(app).get("/api/providers/nope/guide").status_code == 404


# --- Open folder, pages and safety ------------------------------------------------------


def test_open_folder(env, monkeypatch):
    opened = []
    monkeypatch.setattr(server, "reveal_in_file_manager", lambda path, select: opened.append((path, select)))
    app, _ = make_app(env)
    with host_client(app) as host:
        assert host.post("/api/open-folder", json={}).json() == {"ok": True}
        assert opened[-1] == (env.host_out, False)
        job = upload_images(host, ["a.png"], {"mode": "fast"}).json()["job"]
        wait_for(lambda: job_status(host, job["id"]) == "done")
        host.post("/api/open-folder", json={"job_id": job["id"]})
        assert opened[-1] == (env.host_out / "a-4K-fast.jpg", True)


def test_index_and_static_files_are_served(env):
    app, _ = make_app(env)
    guest = guest_client(app)
    index = guest.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "app.js" in index.text
    assert guest.get("/app.js").status_code == 200
    assert guest.get("/icons.svg").status_code == 200
    assert guest.get("/api/nothing-here").status_code == 404


def test_foreign_host_names_are_refused(env):
    app, _ = make_app(env)
    rebinding = TestClient(app, base_url="http://evil.example", client=("127.0.0.1", 50000))
    assert rebinding.get("/api/status").status_code == 400
    assert (
        TestClient(app, base_url="http://localhost:7860", client=("127.0.0.1", 50000)).get("/api/status").status_code
        == 200
    )


def test_cross_site_posts_are_refused(env):
    app, _ = make_app(env)
    host = host_client(app)
    attack = host.post("/api/settings", json={"output_format": "png"}, headers={"Origin": "http://evil.example"})
    assert attack.status_code == 403
    assert config.load_settings().output_format == "jpeg"
    same_site = host.post("/api/settings", json={"output_format": "png"}, headers={"Origin": HOST_URL})
    assert same_site.status_code == 200


def test_guest_outputs_cleanup(env):
    old = env.guest_out / "oldjob" / "old-8K-fast.jpg"
    new = env.guest_out / "newjob" / "new-8K-fast.jpg"
    for path in (old, new):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    long_ago = time.time() - 25 * 3600
    os.utime(old, (long_ago, long_ago))
    app, _ = make_app(env)
    with host_client(app):
        wait_for(lambda: not old.exists())
    assert new.exists()


# --- Sharing on Wi-Fi ----------------------------------------------------------------------
#
# The app listens only on 127.0.0.1. Sharing adds a second listener for the same app on every
# network at port + 1, switched on and off while the app runs; whoever comes in through it is a guest.

WIFI_URL = "http://192.168.1.20:7861"


def shared_at(block: dict) -> dict:
    """Whether the app is shared, and at which addresses (tests/test_share.py checks the rest)."""
    return {"enabled": block["enabled"], "urls": block["urls"]}


def share_status(client: TestClient) -> dict:
    return shared_at(client.get("/api/status").json()["share"])


def status_events(subscription) -> list[dict]:
    events = []
    while True:
        try:
            events.append(subscription.get_nowait())
        except Exception:
            return events


def test_share_starts_with_the_app_and_stops_with_it(env):
    app, _ = make_app(env)  # share=True, as the "Share on Wi-Fi" launchers start it
    sharing = app.state.upscaler.sharing
    assert not sharing.enabled  # nothing listens before the app starts
    with host_client(app) as host:
        assert share_status(host) == {"enabled": True, "urls": [WIFI_URL]}
        assert [s.address for s in env.wifi_sockets] == [("0.0.0.0", 7861)]
    assert not sharing.enabled
    assert env.wifi_sockets[0].closed


def test_share_switch_turns_wifi_on_and_off_without_a_restart(env):
    app, queue = make_app(env, share=False)
    events = queue.subscribe()
    with host_client(app) as host:
        assert share_status(host) == {"enabled": False, "urls": []}
        assert env.wifi_sockets == []  # only this computer can reach the app

        on = host.post("/api/share", json={"enabled": True})
        assert on.status_code == 200
        assert shared_at(on.json()) == {"enabled": True, "urls": [WIFI_URL]}
        assert share_status(host) == shared_at(on.json())
        assert {"event": "status"} in status_events(events)  # every open page shows "Sharing on"
        assert host.post("/api/share", json={"enabled": True}).json()["enabled"] is True  # twice is fine
        assert len(env.wifi_sockets) == 1

        off = host.post("/api/share", json={"enabled": False})
        assert shared_at(off.json()) == {"enabled": False, "urls": []}
        assert share_status(host) == {"enabled": False, "urls": []}
        assert env.wifi_sockets[0].closed
        assert {"event": "status"} in status_events(events)

        host.post("/api/share", json={"enabled": True})  # and on again
        assert share_status(host)["enabled"] is True and len(env.wifi_sockets) == 2


def test_requests_through_wifi_are_guests_even_from_this_computer(env):
    app, _ = make_app(env, share=False)
    wifi = app.state.upscaler.sharing.app  # what the Wi-Fi listener serves
    via_wifi = TestClient(wifi, base_url="http://127.0.0.1:7861", client=("127.0.0.1", 50002))

    status = via_wifi.get("/api/status").json()
    assert status["is_host"] is False and "settings" not in status
    assert via_wifi.get("/api/settings").status_code == 403
    assert via_wifi.post("/api/share", json={"enabled": True}).status_code == 403
    assert via_wifi.post("/api/open-folder", json={}).status_code == 403
    assert guest_client(app).post("/api/share", json={"enabled": True}).status_code == 403
    assert env.wifi_sockets == []
    assert host_client(app).get("/api/status").json()["is_host"] is True  # the usual address is still the host


def test_share_port_taken_is_friendly(env, monkeypatch):
    def taken(host, port):
        raise OSError(48, "Address already in use")

    monkeypatch.setattr(server, "listen_socket", taken)
    app, _ = make_app(env, share=False)
    host = host_client(app)
    response = host.post("/api/share", json={"enabled": True})
    assert response.status_code == 409
    assert "7861" in response.json()["detail"] and "Re:Solution" in response.json()["detail"]
    assert share_status(host) == {"enabled": False, "urls": []}

    started = make_app(env)[0]  # a launcher asking for sharing: the app still starts, just not shared
    with host_client(started) as host:
        assert share_status(host) == {"enabled": False, "urls": []}
        assert host.get("/api/status").json()["is_host"] is True


def test_share_bad_requests(env):
    app, _ = make_app(env, share=False)
    host = host_client(app)
    for body in ({}, {"enabled": "maybe"}, {"enabled": None}):
        assert host.post("/api/share", json=body).status_code == 400
    assert share_status(host)["enabled"] is False


def test_share_qr_code(env, tmp_path):
    app, _ = make_app(env, share=False)
    with host_client(app) as host:
        url = {"url": WIFI_URL}
        assert host.get("/api/share/qr.svg", params=url).status_code == 404  # not sharing yet
        host.post("/api/share", json={"enabled": True})

        response = host.get("/api/share/qr.svg", params=url)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")
        assert response.headers["cache-control"] == "no-store"
        assert response.content.startswith(b"<svg") and b"viewBox" in response.content  # scales to any size
        assert b"<script" not in response.content

        # It really is that address: draw it and read it back, as a phone's camera would.
        (tmp_path / "qr.svg").write_bytes(response.content)
        rgba = render_vector(tmp_path / "qr.svg", (330, 330))
        grey = np.ascontiguousarray(rgba[..., :3].mean(axis=2).astype(np.uint8))
        assert cv2.QRCodeDetector().detectAndDecode(grey)[0] == WIFI_URL

        assert host.get("/api/share/qr.svg", params={"url": "http://evil.example/"}).status_code == 404
        assert host.get("/api/share/qr.svg").status_code == 400
        assert guest_client(app).get("/api/share/qr.svg", params=url).status_code == 403


def test_wifi_listener_really_serves_the_app(env, monkeypatch):
    """The real thing: a second uvicorn server in its own thread, serving the same app."""
    monkeypatch.setattr(server.WifiSharing, "_serve", REAL_WIFI_SERVE)
    app, queue = make_app(env, share=False)
    sharing = app.state.upscaler.sharing
    sock = REAL_LISTEN_SOCKET("127.0.0.1", 0)  # this computer only, so no firewall question in tests
    port = sock.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    sharing.start(sock)
    try:
        assert sharing.enabled and sharing.port == port

        def answers() -> bool:
            try:
                return httpx.get(f"{address}/api/status", timeout=2, trust_env=False).status_code == 200
            except httpx.TransportError:
                return False

        wait_for(answers)
        status = httpx.get(f"{address}/api/status", timeout=5, trust_env=False).json()
        assert status["is_host"] is False  # through the Wi-Fi listener: a guest, even from this computer
        assert shared_at(status["share"]) == {"enabled": True, "urls": [f"http://192.168.1.20:{port}"]}

        # A guest's live updates end at once when sharing stops, so switching off is quick.
        streaming = threading.Event()

        def follow_events():
            with httpx.stream("GET", f"{address}/api/events", timeout=10, trust_env=False) as response:
                streaming.set()
                for _ in response.iter_lines():
                    pass

        follower = threading.Thread(target=follow_events, daemon=True)
        follower.start()
        assert streaming.wait(5)
    finally:
        started = time.monotonic()
        sharing.stop()
        stopping_took = time.monotonic() - started
    assert not sharing.enabled
    assert stopping_took < server.WIFI_GRACE_SECONDS
    follower.join(5)
    assert not follower.is_alive()
    # Refused at once on Macs and Linux; Windows retries a refused local connection for about 2 s, so it may time out.
    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        httpx.get(f"{address}/api/status", timeout=2, trust_env=False)
    assert queue.list() == []


def test_lan_addresses_skip_loopback_and_link_local(monkeypatch):
    monkeypatch.setattr(
        server,
        "_interface_addresses",
        lambda: ["127.0.0.1", "169.254.10.2", "192.168.1.20", "10.0.0.5", "192.168.1.20"],
    )
    addresses = server.lan_ipv4_addresses()
    assert "192.168.1.20" in addresses and "10.0.0.5" in addresses
    assert "127.0.0.1" not in addresses and "169.254.10.2" not in addresses
    assert len(addresses) == len(set(addresses))
    assert server.lan_urls(7860)[0].startswith("http://") and server.lan_urls(7860)[0].endswith(":7860")


def test_mac_ifconfig_is_understood():
    sample = "lo0: flags=8049\n\tinet 127.0.0.1 netmask 0xff000000\nen0: flags=8863\n\tinet 192.168.1.20 netmask 0xffffff00 broadcast 192.168.1.255\n"
    assert server._INET.findall(sample) == ["127.0.0.1", "192.168.1.20"]
    windows = "   IPv4 Address. . . . . . . . . . . : 192.168.1.30\r\n   Subnet Mask . . . . . . . . . . . : 255.255.255.0\r\n"
    assert server._IPCONFIG.findall(windows) == ["192.168.1.30"]


# --- Guessing the password, big requests, disk space and guests' keys ------------------------


def test_password_needs_six_characters(env):
    app, _ = make_app(env)
    host = host_client(app)
    short = host.post("/api/settings", json={"access_password": "abcde"})
    assert short.status_code == 400 and short.json()["detail"] == "Use at least 6 characters."
    assert host.post("/api/settings", json={"access_password": "abcdef"}).status_code == 200


def _waits(response) -> bool:
    return response.status_code == 429 and "Too many wrong passwords" in response.json()["detail"]


def test_wrong_passwords_make_that_device_wait_longer_each_time(env):
    app, _ = make_app(env)
    host_client(app).post("/api/settings", json={"access_password": "sesame"})
    clock = [1000.0]
    app.state.upscaler.clock = lambda: clock[0]
    guesser, family = guest_client(app, "192.168.1.66"), guest_client(app, "192.168.1.5")
    for _ in range(server.LOGIN_FREE_TRIES):
        assert guesser.post("/api/login", json={"password": "wrong!"}).status_code == 401
    refused = guesser.post("/api/login", json={"password": "sesame"})
    assert _waits(refused) and "30 seconds" in refused.json()["detail"]  # not even the right one is checked
    assert family.post("/api/login", json={"password": "sesame"}).status_code == 200  # other devices are fine

    clock[0] += 31
    assert guesser.post("/api/login", json={"password": "wrong!"}).status_code == 401
    refused = guesser.post("/api/login", json={"password": "sesame"})
    assert _waits(refused) and "1 minute" in refused.json()["detail"]
    clock[0] += 61
    assert guesser.post("/api/login", json={"password": "sesame"}).status_code == 200
    assert guesser.post("/api/login", json={"password": "wrong!"}).status_code == 401  # a fresh start


def test_guesses_sent_all_at_once_are_still_checked_one_at_a_time(env, monkeypatch):
    app, _ = make_app(env)
    host_client(app).post("/api/settings", json={"access_password": "sesame"})
    checked = []
    real_verify = server.verify_password
    monkeypatch.setattr(
        server, "verify_password", lambda guess, stored: checked.append(guess) or real_verify(guess, stored)
    )
    guesser = guest_client(app, "192.168.1.66")

    def guess(n: int) -> int:
        return guesser.post("/api/login", json={"password": f"guess{n:03d}"}).status_code

    with ThreadPoolExecutor(16) as pool:
        codes = list(pool.map(guess, range(48)))
    assert len(checked) == codes.count(401) == server.LOGIN_FREE_TRIES
    assert codes.count(429) == 48 - server.LOGIN_FREE_TRIES


def test_guesses_from_many_devices_make_everyone_wait(env):
    app, _ = make_app(env)
    host = host_client(app)
    host.post("/api/settings", json={"access_password": "sesame"})
    for n in range(server.LOGIN_FREE_TRIES_EVERYONE):
        guest = guest_client(app, f"192.168.1.{100 + n}")
        assert guest.post("/api/login", json={"password": "wrong!"}).status_code == 401
    assert _waits(guest_client(app, "192.168.1.5").post("/api/login", json={"password": "sesame"}))
    assert host.get("/api/status").json()["needs_login"] is False  # the host is never asked


def test_big_request_bodies_are_refused_before_they_are_read(env):
    app, _ = make_app(env)
    host = host_client(app)
    host.post("/api/settings", json={"access_password": "sesame"})
    padding = "x" * (server.MAX_FIELD_BYTES + 1)
    login = guest_client(app).post("/api/login", json={"password": padding})
    assert login.status_code == 413 and login.json()["detail"] == server.REQUEST_TOO_BIG
    change = host.post("/api/settings", json={"grain": "subtle", "padding": padding})
    assert change.status_code == 413
    assert config.load_settings().grain == "off"
    assert host.post("/api/estimate", json={"width": 10, "height": 10, "mode": padding}).status_code == 413


def test_big_bodies_sent_without_a_length_are_cut_off(env):
    app, _ = make_app(env)

    def chunks():
        yield b'{"password": "'
        for _ in range(100):
            yield b"x" * 1024
        yield b'"}'

    response = guest_client(app).post("/api/login", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413 and response.json()["detail"] == server.REQUEST_TOO_BIG


def test_pictures_may_still_be_big(env):
    app, _ = make_app(env)
    host = host_client(app)
    picture = photo_bytes()
    assert len(picture) > server.MAX_FIELD_BYTES
    assert upload_images(host, ["a.png"], data=picture).status_code == 200
    assert inspect(host, "a.png", picture).status_code == 200


def test_uploads_only_take_the_fields_the_app_sends(env):
    app, _ = make_app(env)
    host = host_client(app)
    files = [("files", ("a.png", png_bytes(), "image/png"))]
    assert host.post("/api/images", files=files, data={"options": "{}", "note": "hello"}).status_code == 400
    # Many text fields can't add up to more than one field may hold.
    stacked = {"options": ["x" * 40_000, "y" * 40_000, "{}"]}
    assert host.post("/api/images", files=files, data=stacked).status_code == 400
    picture = [("file", ("a.png", png_bytes(), "image/png"))]
    assert host.post("/api/inspect", files=picture, data={"options": "{}"}).status_code == 400
    assert not [path for path in env.store.iterdir() if path.name != "staging"]  # nothing left behind


def test_two_settings_changes_at_once_both_count(env, monkeypatch):
    app, _ = make_app(env)
    host = host_client(app)
    host.post("/api/settings", json={"access_password": "sesame"})
    real_save = config.save_settings
    first = threading.Event()

    def slow_save(settings):
        if not first.is_set():
            first.set()
            time.sleep(0.3)  # the second change arrives meanwhile
        real_save(settings)

    monkeypatch.setattr(config, "save_settings", slow_save)
    monkeypatch.setattr(server, "save_settings", slow_save, raising=False)
    with ThreadPoolExecutor(2) as pool:
        png = pool.submit(host.post, "/api/settings", json={"output_format": "png"})
        first.wait(5)
        grain = pool.submit(host.post, "/api/settings", json={"grain": "subtle"})
        assert png.result().status_code == grain.result().status_code == 200
    saved = config.load_settings()
    assert (saved.output_format, saved.grain) == ("png", "subtle")
    assert saved.access_password_hash is not None


@pytest.fixture
def fal_accepts(monkeypatch):
    monkeypatch.setattr(PROVIDERS["fal"], "check_key", lambda key: (True, "fal.ai accepted this key."))


def fal_connected(client: TestClient, **kwargs) -> bool:
    providers = client.get("/api/status", **kwargs).json()["providers"]
    return next(p for p in providers if p["id"] == "fal")["connected"]


def test_a_guests_key_is_forgotten_after_a_few_idle_hours(env, fal_accepts):
    app, _ = make_app(env)
    clock = [1000.0]
    app.state.upscaler.clock = lambda: clock[0]
    guest = guest_client(app)
    assert guest.post("/api/providers/fal/session-key", json={"key": "guest:key"}).status_code == 200
    assert fal_connected(guest)
    clock[0] += server.SESSION_KEY_IDLE_SECONDS - 60
    assert upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).status_code == 200  # using it keeps it
    clock[0] += server.SESSION_KEY_IDLE_SECONDS - 60
    assert fal_connected(guest)  # just looking doesn't
    clock[0] += 120
    assert not fal_connected(guest)
    assert upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).status_code == 403
    assert app.state.upscaler.session_keys == {}


def test_a_guests_key_lasts_a_day_at_most(env, fal_accepts):
    app, _ = make_app(env)
    clock = [1000.0]
    app.state.upscaler.clock = lambda: clock[0]
    guest = guest_client(app)
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})
    while clock[0] + 3600 < 1000.0 + server.SESSION_KEY_MAX_SECONDS:
        clock[0] += 3600
        assert upload_images(guest, ["a.png"], {"mode": "cloud:fal:seedvr2"}).status_code == 200
    clock[0] += 3601
    assert not fal_connected(guest)


def test_a_guests_key_goes_with_the_browser_session(env, fal_accepts):
    app, queue = make_app(env)
    guest = guest_client(app)
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})
    job = upload_images(guest, ["a.png"]).json()["job"]
    assert fal_connected(guest)
    # The browser was closed and opened again: its 30-day cookie is still there, the visit's isn't.
    cookie = {"cookie": f"{server.SESSION_COOKIE}={guest.cookies[server.SESSION_COOKIE]}"}
    reopened = guest_client(app)
    assert not fal_connected(reopened, headers=cookie)
    assert reopened.get(f"/api/jobs/{job['id']}", headers=cookie).json()["job"]["mine"] is True  # still theirs


def test_a_guest_can_forget_their_key(env, fal_accepts):
    app, _ = make_app(env)
    guest, other = guest_client(app), guest_client(app, "192.168.1.6")
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})
    other.post("/api/providers/fal/session-key", json={"key": "other:key"})
    assert guest.delete("/api/providers/fal/session-key").json() == {"ok": True, "stopped": 0}
    assert not fal_connected(guest)
    assert fal_connected(other)
    assert guest.delete("/api/providers/fal/session-key").json() == {"ok": True, "stopped": 0}  # twice is fine
    assert guest.delete("/api/providers/nope/session-key").status_code == 404


def test_forgetting_a_key_stops_the_jobs_still_waiting_to_use_it(env, fal_accepts):
    app, queue = make_app(env)  # not started: the jobs wait in the queue
    guest, other = guest_client(app), guest_client(app, "192.168.1.6")
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})
    other.post("/api/providers/fal/session-key", json={"key": "other:key"})
    waiting = upload_images(guest, ["a.png", "b.png"], {"mode": "cloud:fal:seedvr2"}).json()["job"]
    local = upload_images(guest, ["c.png"], {"mode": "fast"}).json()["job"]
    theirs = upload_images(other, ["d.png"], {"mode": "cloud:fal:seedvr2"}).json()["job"]

    forgot = guest.delete("/api/providers/fal/session-key")
    assert forgot.json() == {"ok": True, "stopped": 1}
    stopped = queue.get(waiting["id"])
    assert stopped.status == "cancelled" and stopped.secret is None
    assert stopped.error == jobs.KEY_FORGOTTEN_BY_PERSON
    assert queue.get(local["id"]).status == "queued"  # it never needed the key
    assert queue.get(theirs["id"]).status == "queued" and queue.get(theirs["id"]).secret == "other:key"

    with host_client(app):  # the queue runs: nothing is sent with the forgotten key
        wait_for(lambda: queue.get(theirs["id"]).status == "done")
    assert queue.get(waiting["id"]).status == "cancelled"
    assert [call.path.name for call in env.upscale_calls] == ["c.png", "d.png"]
    assert [call.opts.guest_key for call in env.upscale_calls] == [None, "other:key"]


def test_housekeeping_forgets_old_jobs_and_keys_while_the_app_runs(env, fal_accepts):
    app, queue = make_app(env)
    ctx = app.state.upscaler
    clock = [1000.0]
    ctx.clock = lambda: clock[0]
    guest = guest_client(app)
    guest.post("/api/providers/fal/session-key", json={"key": "guest:key"})
    job_id = upload_images(guest, ["a.png"]).json()["job"]["id"]
    with host_client(app):
        wait_for(lambda: queue.get(job_id).status == "done")
    assert guest.get(f"/api/jobs/{job_id}/results/0/preview.jpg").status_code == 200
    with queue._lock:  # a day and a bit later
        queue._jobs[job_id].finished -= 25 * 3600
    clock[0] += server.SESSION_KEY_MAX_SECONDS + 1

    server._tidy_up(ctx)  # what runs every hour
    assert queue.get(job_id) is None and not (env.store / job_id).exists()  # the guest's previews too
    assert guest.get(f"/api/jobs/{job_id}/results/0/preview.jpg").status_code == 404
    assert ctx.session_keys == {}


def test_picture_uploads_need_free_disk_space(env, monkeypatch):
    app, queue = make_app(env)
    host = host_client(app)
    monkeypatch.setattr(server, "free_bytes", lambda path: server.SPACE_RESERVE_BYTES - 1)
    response = upload_images(host, ["a.png"])
    assert response.status_code == 507 and "disk space" in response.json()["detail"]
    checked = inspect(host, "a.png", photo_bytes())
    assert checked.status_code == 507 and "disk space" in checked.json()["detail"]
    assert queue.list() == []
    assert not [path for path in env.store.iterdir() if path.name != "staging"]
    assert not (env.store / "staging").exists() or not any((env.store / "staging").iterdir())


def test_an_upload_stops_when_the_disk_fills_up_meanwhile(env, monkeypatch):
    app, queue = make_app(env)
    answers = iter([10**12])  # plenty at the start...
    monkeypatch.setattr(server, "free_bytes", lambda path: next(answers, 10))  # ...then the disk fills up
    assert upload_images(host_client(app), ["a.png"]).status_code == 507
    assert queue.list() == []


def test_each_guest_can_keep_only_so_many_pictures_waiting(env, monkeypatch):
    app, _ = make_app(env)
    picture = photo_bytes()
    monkeypatch.setattr(server, "GUEST_UPLOAD_QUOTA_BYTES", len(picture) * 2)
    guest, other, host = guest_client(app), guest_client(app, "192.168.1.6"), host_client(app)
    assert inspect(guest, "a.png", picture).status_code == 200
    assert inspect(guest, "b.png", picture).status_code == 200
    full = inspect(guest, "c.png", picture)
    assert full.status_code == 507 and full.json()["detail"] == server.STAGING_FULL
    assert inspect(other, "c.png", picture).status_code == 200  # each guest has their own share
    for n in range(3):
        assert inspect(host, f"h{n}.png", picture).status_code == 200  # the host's own disk: no share


def test_a_guests_jobs_count_towards_their_share_of_the_disk(env, monkeypatch):
    # Pictures sent straight to a job, and videos, are kept with the job for a day after it ends, so
    # they count along with the pictures waiting: one guest can't fill the main computer's disk.
    monkeypatch.setattr(server, "probe_video", lambda path: SimpleNamespace(frames=50, duration=2.0))
    app, queue = make_app(env)  # not started: the jobs and their uploads stay
    picture = photo_bytes()
    monkeypatch.setattr(server, "GUEST_UPLOAD_QUOTA_BYTES", len(picture) * 3)
    guest, host = guest_client(app), host_client(app)
    assert upload_images(guest, ["a.png", "b.png"], data=picture).status_code == 200
    full = upload_images(guest, ["c.png", "d.png"], data=picture)
    assert full.status_code == 507 and full.json()["detail"] == server.STAGING_FULL
    assert inspect(guest, "c.png", picture).status_code == 200  # exactly the share
    assert inspect(guest, "d.png", picture).status_code == 507
    video = {"file": ("clip.mov", b"x" * 100, "video/quicktime")}
    assert guest.post("/api/videos", files=video, data={"options": "{}"}).status_code == 507
    assert len(queue.list()) == 1
    assert host.post("/api/videos", files=video, data={"options": "{}"}).status_code == 200  # the host's own disk
    assert upload_images(host, ["e.png", "f.png", "g.png"], data=picture).status_code == 200


def test_a_picture_and_its_jobs_copy_count_once(env, monkeypatch):
    app, _ = make_app(env)
    picture = photo_bytes()
    monkeypatch.setattr(server, "GUEST_UPLOAD_QUOTA_BYTES", len(picture) * 3)
    guest = guest_client(app)
    tokens = [inspect(guest, name, picture).json()["token"] for name in ("a.png", "b.png")]
    assert guest.post("/api/images", json={"tokens": tokens, "options": {"mode": "fast"}}).status_code == 200
    assert inspect(guest, "c.png", picture).status_code == 200  # the job's copies are the same files


def test_a_batch_whose_results_wont_fit_is_refused_before_it_starts(env, monkeypatch):
    # 16K TIFFs are about 360 MB each: three won't fit in 1 GB, with the 2 GB always kept free on top.
    monkeypatch.setattr(server, "free_bytes", lambda path: server.SPACE_RESERVE_BYTES + 10**9)
    app, queue = make_app(env)
    host = host_client(app)
    options = {"mode": "fast", "size": "16k", "output_format": "tiff"}
    tokens = [inspect(host, f"{name}.png", photo_bytes(seed=n)).json()["token"] for n, name in enumerate("abc")]

    refused = host.post("/api/images", json={"tokens": tokens, "options": options})
    assert refused.status_code == 507
    detail = refused.json()["detail"]
    assert detail.startswith("These 3 pictures will take about 1.1 GB once finished")
    assert "always leaves 2 GB free" in detail and "choose JPEG or a smaller size" in detail
    assert upload_images(host, ["a.png", "b.png", "c.png"], options).status_code == 507  # sent with the job too
    assert queue.list() == []

    assert host.post("/api/images", json={"tokens": tokens[:2], "options": options}).status_code == 200
    jpeg = host.post("/api/images", json={"tokens": tokens, "options": {**options, "output_format": "jpeg"}})
    assert jpeg.status_code == 200  # about 40 MB each


def test_a_single_picture_too_big_for_the_disk_says_so(env, monkeypatch):
    monkeypatch.setattr(server, "free_bytes", lambda path: server.SPACE_RESERVE_BYTES + 10**7)
    app, _ = make_app(env)
    refused = upload_images(guest_client(app), ["poster.png"], {"mode": "fast", "size": "16k"})
    assert refused.status_code == 507
    assert refused.json()["detail"].startswith("This picture will take about 40 MB once finished")
    assert "choose a smaller size." in refused.json()["detail"]  # already JPEG
