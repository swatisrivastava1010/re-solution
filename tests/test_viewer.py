"""The compare viewer's backend: tile pyramids, the viewer endpoints, and keeping the originals.

(DESIGN "Deep-zoom compare viewer".) The viewer shows the whole picture at Fit and every pixel at
100% and 200% by loading 512 px tiles of a pyramid written when the job finishes; the Before side's
tiles are made on request from the job's original, simply enlarged.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageCms

from upscaler import config, jobs, server
from upscaler.config import Cancelled
from upscaler.imageio_utils import (
    VIEWER_MANIFEST,
    before_tile,
    pyramid_levels,
    read_viewer_manifest,
    tile_bounds,
    write_tile_pyramid,
)
from upscaler.jobs import Job, JobQueue

HOST_URL = "http://127.0.0.1:7860"
LAN_URL = "http://192.168.1.20:7860"


def gradient(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A picture with smooth colour and some fine detail, so tiles can be checked by eye and by number."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    detail = np.random.default_rng(seed).integers(0, 24, (height, width), dtype=np.uint8).astype(np.float32)
    rgb = np.stack(np.broadcast_arrays(x + 0 * y, y + 0 * x, (x + y) / 2 + detail), axis=-1)
    return rgb.clip(0, 255).astype(np.uint8)


def jpeg_pixels(data: bytes | Path) -> np.ndarray:
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with Image.open(source) as image:
        assert image.format == "JPEG"
        return np.asarray(image.convert("RGB"))


def mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# --- The pyramid -----------------------------------------------------------------------------


def test_levels_halve_until_the_whole_picture_fits_1024():
    assert [(lv["width"], lv["height"]) for lv in pyramid_levels(3840, 2160)] == [
        (3840, 2160),
        (1920, 1080),
        (960, 540),
    ]
    assert [lv["width"] for lv in pyramid_levels(8192, 6144)] == [8192, 4096, 2048, 1024]
    assert [lv["width"] for lv in pyramid_levels(16384, 12312)] == [16384, 8192, 4096, 2048, 1024]
    assert pyramid_levels(800, 600) == [{"level": 0, "width": 800, "height": 600, "cols": 2, "rows": 2}]
    # Odd sides round up, so no pixel is lost; a tall picture fits by its height.
    assert [(lv["width"], lv["height"]) for lv in pyramid_levels(1025, 3001)] == [(1025, 3001), (513, 1501), (257, 751)]
    first = pyramid_levels(3840, 2981)[0]
    assert (first["cols"], first["rows"]) == (8, 6)  # 7.5 and 5.8 tiles: edge tiles are smaller
    with pytest.raises(ValueError):
        pyramid_levels(0, 10)


def test_tile_bounds_cover_each_level_exactly():
    level = pyramid_levels(1300, 700)[0]
    assert tile_bounds(level, 0, 0) == (0, 0, 512, 512)
    assert tile_bounds(level, 2, 1) == (1024, 512, 1300, 700)  # the corner tile is 276 x 188
    assert tile_bounds(level, 3, 0) is None and tile_bounds(level, 0, 2) is None and tile_bounds(level, -1, 0) is None


def test_pyramid_writes_every_tile_and_its_viewer_json(tmp_path):
    rgb = gradient(1300, 2100)
    folder = tmp_path / "0-viewer"
    manifest = write_tile_pyramid(rgb, folder, source_size=(325, 525))
    assert manifest == read_viewer_manifest(folder) == json.loads((folder / VIEWER_MANIFEST).read_text())
    assert (manifest["width"], manifest["height"], manifest["tile_size"]) == (1300, 2100, 512)
    assert (manifest["source_width"], manifest["source_height"]) == (325, 525)
    assert manifest["levels"] == pyramid_levels(1300, 2100)
    assert not (tmp_path / "0-viewer.part").exists()  # written next to it, then moved into place
    for level in manifest["levels"]:
        names = sorted(p.name for p in (folder / str(level["level"])).iterdir())
        assert names == sorted(f"{c}_{r}.jpg" for c in range(level["cols"]) for r in range(level["rows"]))
        for c in range(level["cols"]):
            for r in range(level["rows"]):
                x0, y0, x1, y1 = tile_bounds(level, c, r)
                with Image.open(folder / str(level["level"]) / f"{c}_{r}.jpg") as tile:
                    assert tile.size == (x1 - x0, y1 - y0)
    # Level 0 is the picture itself, cut into tiles (JPEG at quality 88 stays very close).
    assert mean_difference(jpeg_pixels(folder / "0" / "1_2.jpg"), rgb[1024:1536, 512:1024]) < 3
    assert mean_difference(jpeg_pixels(folder / "0" / "2_4.jpg"), rgb[2048:2100, 1024:1300]) < 3


def test_smaller_levels_are_the_one_before_averaged(tmp_path):
    rgb = gradient(1500, 900)
    manifest = write_tile_pyramid(rgb, tmp_path / "v")
    assert [lv["width"] for lv in manifest["levels"]] == [1500, 750]
    expected = rgb.reshape(450, 2, 750, 2, 3).astype(np.float32).mean(axis=(1, 3))
    level1 = jpeg_pixels(tmp_path / "v" / "1" / "0_0.jpg")
    assert level1.shape == (450, 512, 3)  # level 1 is 750 x 450
    assert mean_difference(level1, expected[:, :512].round().astype(np.uint8)) < 3


def test_tiles_keep_the_pictures_colour_profile(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    write_tile_pyramid(gradient(600, 400), tmp_path / "v", icc=icc)
    with Image.open(tmp_path / "v" / "0" / "0_0.jpg") as tile:
        assert tile.info.get("icc_profile") == icc


def test_a_cancelled_pyramid_leaves_nothing_behind(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        write_tile_pyramid(gradient(1200, 1200), tmp_path / "v", cancel=cancel)
    assert not (tmp_path / "v").exists() and not (tmp_path / "v.part").exists()


def test_a_new_pyramid_replaces_an_old_one(tmp_path):
    write_tile_pyramid(gradient(2000, 1000), tmp_path / "v")
    write_tile_pyramid(gradient(300, 200), tmp_path / "v")
    assert sorted(p.name for p in (tmp_path / "v").iterdir()) == ["0", VIEWER_MANIFEST]
    assert read_viewer_manifest(tmp_path / "v")["width"] == 300


def test_damaged_or_missing_viewer_json_reads_as_no_pyramid(tmp_path):
    assert read_viewer_manifest(tmp_path) is None
    (tmp_path / VIEWER_MANIFEST).write_text("{not json")
    assert read_viewer_manifest(tmp_path) is None
    (tmp_path / VIEWER_MANIFEST).write_text(json.dumps({"levels": []}))
    assert read_viewer_manifest(tmp_path) is None


def test_before_tiles_are_the_original_simply_enlarged():
    source = Image.fromarray(gradient(250, 160, seed=3))
    levels = pyramid_levels(1000, 640, tile_size=256)
    whole = np.asarray(source.resize((1000, 640), Image.LANCZOS))
    for col, row in ((0, 0), (1, 1), (3, 2)):
        x0, y0, x1, y1 = tile_bounds(levels[0], col, row, 256)
        tile = np.asarray(before_tile(source, levels[0], col, row, 256))
        assert tile.shape == (y1 - y0, x1 - x0, 3)
        # The same pixels as cutting the tile out of the whole enlarged picture.
        assert mean_difference(tile, whole[y0:y1, x0:x1]) < 1.0
    with pytest.raises(ValueError):
        before_tile(source, levels[0], 4, 0, 256)


def test_pyramid_timings_for_8k_and_16k(tmp_path):
    """Writing the tiles must not slow a job down noticeably (they're made while the picture is in memory)."""
    timings = {}
    for name, (w, h) in {"8K": (8192, 6144), "16K": (16384, 12312)}.items():
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[::7] = 200  # some content, so the JPEGs aren't trivially empty
        started = time.perf_counter()
        manifest = write_tile_pyramid(rgb, tmp_path / name)
        timings[name] = time.perf_counter() - started
        assert len(manifest["levels"]) == (4 if name == "8K" else 5)
        del rgb
    # Measured on an M4: about 0.1 s for 8K and 0.3 s for 16K. Generous limits for slower computers.
    assert timings["8K"] < 10 and timings["16K"] < 30, timings


# --- The viewer endpoints ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))
    monkeypatch.setattr(server, "lan_ipv4_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)
    monkeypatch.setattr(server, "_before_sources", server._SourceCache())
    monkeypatch.setattr(server, "_manifests", type(server._manifests)())
    monkeypatch.setattr(server, "_pyramid_failed", {})
    return SimpleNamespace(
        tmp=tmp_path, store=tmp_path / "data" / "jobs", out=tmp_path / "home" / "Pictures" / "Upscaled"
    )


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def host_client(app) -> TestClient:
    return TestClient(app, base_url=HOST_URL, client=("127.0.0.1", 50000))


def guest_client(app, ip: str = "192.168.1.5") -> TestClient:
    return TestClient(app, base_url=LAN_URL, client=(ip, 50001))


def finished_image_job(
    env,
    queue: JobQueue,
    *,
    owner: str = "host",
    source=(400, 300),
    scale: int = 4,
    pyramid: bool = True,
    name="room.png",
) -> tuple[Job, np.ndarray, np.ndarray]:
    """A finished picture job as run_image_job leaves it: the original in in/, the result in the
    output folder, the previews and (unless `pyramid` is False, as for older results) the tiles."""
    job_id = queue.new_job_id()
    job_dir = queue.job_dir(job_id)
    original = job_dir / "in" / "0" / name
    original.parent.mkdir(parents=True)
    src = gradient(*source, seed=7)
    Image.fromarray(src).save(original)
    out = np.asarray(Image.fromarray(src).resize((source[0] * scale, source[1] * scale), Image.LANCZOS))
    env.out.mkdir(parents=True, exist_ok=True)
    output = env.out / f"{Path(name).stem}-4K-fast.jpg"
    Image.fromarray(out).save(output, quality=95)
    result = {
        "path": str(output),
        "filename": output.name,
        "width": out.shape[1],
        "height": out.shape[0],
        "source_width": source[0],
        "source_height": source[1],
        "mode_used": "fast",
        "notes": [],
        "input": 0,
    }
    if pyramid:
        result["viewer"] = str(job_dir / "results" / "0-viewer")
        write_tile_pyramid(out, Path(result["viewer"]), source_size=source)
    job = Job(id=job_id, kind="image", owner=owner, status="done", progress=1.0, inputs=[str(original)],
              results=[result], finished=time.time())  # fmt: skip
    with queue._lock:
        queue._jobs[job.id] = job
        queue._save(job)
    return job, src, out


def make_app(env):
    queue = JobQueue(env.store, {"image": lambda job, progress, cancel: [], "video": lambda job, progress, cancel: []})
    return server.create_app(queue, False, port=7860), queue


def test_viewer_json_describes_the_pyramid_and_the_before_side(env):
    app, queue = make_app(env)
    job, _, out = finished_image_job(env, queue, source=(960, 745))
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/viewer.json")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-cache"
    body = response.json()
    assert (body["width"], body["height"]) == (3840, 2980) == (out.shape[1], out.shape[0])
    assert (body["source_width"], body["source_height"]) == (960, 745)
    assert body["tile_size"] == 512 and [lv["width"] for lv in body["levels"]] == [3840, 1920, 960]
    assert body["before"] is True
    assert "path" not in json.dumps(body) and str(env.tmp) not in json.dumps(body)  # no file system paths


def test_tiles_are_served_with_a_long_cache(env):
    app, queue = make_app(env)
    job, _, out = finished_image_job(env, queue)
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/tiles/0/1_2.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert "immutable" in response.headers["cache-control"]
        assert mean_difference(jpeg_pixels(response.content), out[1024:1200, 512:1024]) < 3
        top = host.get(f"/api/jobs/{job.id}/results/0/tiles/1/0_0.jpg")
        assert top.status_code == 200 and jpeg_pixels(top.content).shape == (512, 512, 3)
        for missing in ("tiles/0/4_0.jpg", "tiles/0/0_3.jpg", "tiles/2/0_0.jpg", "tiles/-1/0_0.jpg", "tiles/0/0_0.png"):
            assert host.get(f"/api/jobs/{job.id}/results/0/{missing}").status_code == 404, missing
        assert host.get(f"/api/jobs/{job.id}/results/1/tiles/0/0_0.jpg").status_code == 404


def test_before_tiles_come_from_the_original_simply_enlarged(env):
    app, queue = make_app(env)
    job, src, _ = finished_image_job(env, queue)
    whole = np.asarray(Image.fromarray(src).resize((1600, 1200), Image.LANCZOS))
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/before/0/2_1.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg" and "immutable" in response.headers["cache-control"]
        tile = jpeg_pixels(response.content)
        assert tile.shape == (512, 512, 3)
        assert mean_difference(tile, whole[512:1024, 1024:1536]) < 3
        # The last level: the whole original, at the size the viewer shows when it fits the picture.
        small = jpeg_pixels(host.get(f"/api/jobs/{job.id}/results/0/before/1/1_1.jpg").content)
        assert small.shape == (88, 288, 3)  # level 1 is 800 x 600: the corner tile is 288 x 88
        assert host.get(f"/api/jobs/{job.id}/results/0/before/0/9_9.jpg").status_code == 404


def test_the_original_is_opened_once_for_a_screenful_of_tiles(env, monkeypatch):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue)
    opened = []
    real = server.load_image
    monkeypatch.setattr(server, "load_image", lambda path, **kw: opened.append(Path(path).name) or real(path, **kw))
    with host_client(app) as host:
        for col in range(4):
            for row in range(3):
                assert host.get(f"/api/jobs/{job.id}/results/0/before/0/{col}_{row}.jpg").status_code == 200
    assert opened == ["room.png"]


def test_source_cache_keeps_the_newest_few(tmp_path):
    cache = server._SourceCache(max_items=2, max_pixels=10**9)
    paths = []
    for n in range(3):
        paths.append(tmp_path / f"{n}.png")
        Image.fromarray(gradient(40, 30, seed=n)).save(paths[-1])
    first = cache.get(paths[0])
    assert cache.get(paths[0]) is first
    cache.get(paths[1])
    cache.get(paths[2])
    assert cache.get(paths[0]) is not first  # pushed out by the two newer ones, then opened again
    tight = server._SourceCache(max_items=5, max_pixels=40 * 30)
    tight.get(paths[0])
    kept = tight.get(paths[1])
    assert len(tight._items) == 1 and tight.get(paths[1]) is kept  # one picture is always kept


def test_see_through_originals_are_shown_on_white(env):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue, source=(200, 150), name="logo.png")
    rgba = np.zeros((150, 200, 4), np.uint8)
    rgba[..., 0] = 255  # red, but fully see-through
    Image.fromarray(rgba).save(job.inputs[0])
    with host_client(app) as host:
        tile = jpeg_pixels(host.get(f"/api/jobs/{job.id}/results/0/before/0/0_0.jpg").content)
    assert tile.min() > 240  # white, not the junk colour under the see-through pixels


def test_guests_only_see_their_own_viewer(env):
    app, queue = make_app(env)
    hosts, _, _ = finished_image_job(env, queue)
    with guest_client(app) as guest:
        for path in ("viewer.json", "tiles/0/0_0.jpg", "before/0/0_0.jpg"):
            response = guest.get(f"/api/jobs/{hosts.id}/results/0/{path}")
            assert response.status_code == 403, path
            assert response.json()["detail"] == server.NOT_YOURS
        guest.get("/api/status")  # gets the guest a session cookie
        owner = server.GUEST_PREFIX + app.state.upscaler.guest_id(guest.cookies[server.SESSION_COOKIE])
        mine, _, _ = finished_image_job(env, queue, owner=owner, name="mine.png")
        assert guest.get(f"/api/jobs/{mine.id}/results/0/viewer.json").json()["before"] is True
        assert guest.get(f"/api/jobs/{mine.id}/results/0/before/0/0_0.jpg").status_code == 200
        assert guest.get(f"/api/jobs/{mine.id}/results/0/tiles/0/0_0.jpg").status_code == 200


def test_a_guest_needs_the_password_for_tiles(env, monkeypatch):
    app, queue = make_app(env)
    settings = config.Settings(access_password_hash=server.hash_password("secret1"))
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    job, _, _ = finished_image_job(env, queue)
    with guest_client(app) as guest:
        assert guest.get(f"/api/jobs/{job.id}/results/0/tiles/0/0_0.jpg").status_code == 401


def test_older_results_get_their_tiles_made_once_from_the_saved_picture(env, monkeypatch):
    app, queue = make_app(env)
    job, _, out = finished_image_job(env, queue, pyramid=False)
    built = []
    real = server.write_tile_pyramid
    monkeypatch.setattr(server, "write_tile_pyramid", lambda *a, **k: built.append(a[1]) or real(*a, **k))
    with host_client(app) as host:
        bodies = [host.get(f"/api/jobs/{job.id}/results/0/viewer.json").json() for _ in range(3)]
        tile = host.get(f"/api/jobs/{job.id}/results/0/tiles/0/0_0.jpg")
    assert len(built) == 1 and built[0] == (env.store / job.id / "results" / "0-viewer").resolve()
    assert bodies[0] == bodies[2] and (bodies[0]["width"], bodies[0]["height"]) == (1600, 1200)
    assert (bodies[0]["source_width"], bodies[0]["source_height"]) == (400, 300)
    assert tile.status_code == 200 and mean_difference(jpeg_pixels(tile.content), out[:512, :512]) < 4


def test_tiles_are_made_once_however_many_ask_at_the_same_time(env, monkeypatch):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue, pyramid=False)
    built = []
    real = server.write_tile_pyramid

    def slow_build(*args, **kwargs):
        built.append(args[1])
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(server, "write_tile_pyramid", slow_build)
    with host_client(app) as host:
        url = f"/api/jobs/{job.id}/results/0/viewer.json"
        threads = [threading.Thread(target=host.get, args=(url,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert len(built) == 1


def test_a_result_whose_picture_is_gone_says_unavailable(env):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue, pyramid=False)
    Path(job.results[0]["path"]).unlink()
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/viewer.json")
        assert response.status_code == 200 and response.json() == {"unavailable": True}
        assert host.get(f"/api/jobs/{job.id}/results/0/tiles/0/0_0.jpg").status_code == 404
        # The old close-ups still work, for the page to fall back on.
        assert host.get(f"/api/jobs/{job.id}/results/0/before/0/0_0.jpg").status_code == 404


def test_without_the_original_only_the_upscaled_side_is_offered(env):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue)
    Path(job.inputs[0]).unlink()  # e.g. a result made before originals were kept
    with host_client(app) as host:
        assert host.get(f"/api/jobs/{job.id}/results/0/viewer.json").json()["before"] is False
        response = host.get(f"/api/jobs/{job.id}/results/0/before/0/0_0.jpg")
        assert response.status_code == 404 and response.json()["detail"] == server.ORIGINAL_NOT_KEPT
        assert host.get(f"/api/jobs/{job.id}/results/0/tiles/0/0_0.jpg").status_code == 200


def test_viewer_never_reads_outside_the_jobs_own_folder(env, tmp_path):
    app, queue = make_app(env)
    job, _, _ = finished_image_job(env, queue)
    elsewhere = tmp_path / "elsewhere"
    write_tile_pyramid(gradient(64, 64), elsewhere / "viewer")
    secret = tmp_path / "secret.png"
    Image.fromarray(gradient(64, 64)).save(secret)
    with queue._lock:
        live = queue._jobs[job.id]
        live.results[0]["viewer"] = str(elsewhere / "viewer")  # a damaged or tampered record
        live.inputs[0] = str(secret)
    with host_client(app) as host:
        body = host.get(f"/api/jobs/{job.id}/results/0/viewer.json").json()
        assert body["width"] == 1600  # the job's own tiles (made again in its folder), not the other ones
        assert body["before"] is False  # and the original must be in the job's folder too
        assert host.get(f"/api/jobs/{job.id}/results/0/tiles/0/..%2F..%2Fsecret.jpg").status_code == 404
        assert host.get(f"/api/jobs/{job.id}/results/0/before/0/0_0.jpg").status_code == 404


def test_only_picture_jobs_have_a_viewer(env):
    app, queue = make_app(env)
    job = Job(id=queue.new_job_id(), kind="video", owner="host", status="done", results=[{"path": "x.mp4"}])
    with queue._lock:
        queue._jobs[job.id] = job
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/viewer.json")
        assert response.status_code == 404 and response.json()["detail"] == server.NOT_ZOOMABLE
        assert host.get("/api/jobs/000000000000/results/0/viewer.json").status_code == 404


# --- The job leaves what the viewer needs ----------------------------------------------------


def test_image_jobs_write_the_viewer_tiles_and_record_them(env, monkeypatch):
    from upscaler import pipeline_image
    from upscaler.pipeline_image import ImageResult

    seen = []

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        seen.append(viewer_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{Path(path).stem}-4K-fast.jpg"
        output.write_bytes(b"jpeg")
        write_tile_pyramid(gradient(64, 48), viewer_dir)
        return ImageResult(output, 64, 48, "fast", [], 1.0, b"p", b"b", b"a", viewer=viewer_dir)

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    queue = JobQueue(env.store, jobs.default_runners(env.store))
    job_id = queue.new_job_id()
    inputs = []
    for n, name in enumerate(("a.png", "b.png")):
        path = queue.job_dir(job_id) / "in" / str(n) / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        inputs.append(str(path))
    queue.submit("image", "host", inputs, {"mode": "fast", "size": "4k"}, job_id=job_id)
    queue.start()
    try:
        deadline = time.monotonic() + 10
        while queue.get(job_id).status != "done":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        queue.stop()
    folder = env.store / job_id / "results"
    assert seen == [folder / "0-viewer", folder / "1-viewer"]
    results = queue.get(job_id).results
    assert [r["viewer"] for r in results] == [str(folder / "0-viewer"), str(folder / "1-viewer")]
    assert [r["input"] for r in results] == [0, 1]
    # The originals stay for the Before side (they used to be deleted when the job finished).
    assert all(Path(path).is_file() for path in inputs)


def test_originals_stay_until_the_job_is_forgotten(env):
    def runner(job, progress, cancel):
        return [{"filename": "a.jpg", "input": 0}] if job.kind != "image" or "keep" in job.inputs[0] else []

    queue = JobQueue(env.store, {"image": runner, "video": runner, "preview": runner})
    made = {}
    for kind, name in (("image", "keep.png"), ("video", "clip.mov"), ("preview", "crop.png"), ("image", "nothing.png")):
        job_id = queue.new_job_id()
        path = queue.job_dir(job_id) / "in" / "0" / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        (queue.job_dir(job_id) / "work").mkdir()
        made[name] = queue.submit(kind, "guest:x", [str(path)], {}, job_id=job_id)
    queue.start()
    try:
        deadline = time.monotonic() + 10
        while any(queue.get(job.id).active for job in made.values()):
            assert time.monotonic() < deadline
            time.sleep(0.02)
    finally:
        queue.stop()
    folder = lambda name: env.store / made[name].id  # noqa: E731
    assert (folder("keep.png") / "in" / "0" / "keep.png").is_file()  # a picture's before side
    assert (folder("clip.mov") / "in" / "0" / "clip.mov").is_file()  # the side-by-side video comparison
    assert not (folder("crop.png") / "in").exists()  # Preview styles only shows crops
    assert not (folder("nothing.png") / "in").exists()  # nothing was made to compare with
    assert not any((folder(name) / "work").exists() for name in made)  # half-made work always goes

    with queue._lock:  # a day and a bit later, the guest's jobs (and their originals) are forgotten
        for job in made.values():
            queue._jobs[job.id].finished = time.time() - jobs.GUEST_RESULTS_SECONDS - 60
    assert queue.prune_expired() == 4
    assert not folder("keep.png").exists() and not folder("clip.mov").exists()


def test_keeps_originals_rule():
    def job(kind, results):
        return Job(id="0" * 12, kind=kind, owner="host", results=results)

    assert jobs.keeps_originals(job("image", [{}])) and jobs.keeps_originals(job("video", [{}]))
    assert not jobs.keeps_originals(job("image", [])) and not jobs.keeps_originals(job("preview", [{}]))
    assert not jobs.keeps_originals(job("download", []))


def test_open_in_new_tab_shows_the_whole_picture(env):
    app, queue = make_app(env)
    job, _, out = finished_image_job(env, queue)
    previews = env.store / job.id / "results"
    previews.mkdir(parents=True, exist_ok=True)
    (previews / "0-preview.jpg").write_bytes(b"preview jpeg")
    with queue._lock:
        queue._jobs[job.id].results[0]["preview"] = str(previews / "0-preview.jpg")
    with host_client(app) as host:
        response = host.get(f"/api/jobs/{job.id}/results/0/view")
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert response.headers["content-disposition"].startswith("inline")  # shown in the tab, not downloaded
        assert jpeg_pixels(response.content).shape == out.shape  # at full size
        # A format browsers can't show in a tab (TIFF, HEIC): the screen-sized preview instead.
        tiff = Path(job.results[0]["path"]).with_suffix(".tif")
        Image.fromarray(out).save(tiff)
        with queue._lock:
            queue._jobs[job.id].results[0]["path"] = str(tiff)
        assert host.get(f"/api/jobs/{job.id}/results/0/view").content == b"preview jpeg"
    with guest_client(app) as guest:
        assert guest.get(f"/api/jobs/{job.id}/results/0/view").status_code == 403
