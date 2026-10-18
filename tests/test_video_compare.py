"""Side-by-side video comparison (DESIGN "Side-by-side video compare"): browser-friendly H.264 copies
of a video job's original and its result, frame for frame, made on request and kept with the job."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import av
import imageio_ffmpeg
import pytest
from fastapi.testclient import TestClient

from upscaler import config, jobs, server
from upscaler.config import Cancelled
from upscaler.jobs import Job, JobQueue
from upscaler.video import compare
from upscaler.video.probe import VideoError

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
H264 = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")
# Each frame shows its own number: the left half is grey level n % 16, the right half n // 16.
NUMBERED_FRAMES = (
    "color=c=gray:s={w}x{h}:r=25:d={seconds},format=yuv420p,"
    r"geq=lum='if(lt(X\,{half})\,mod(N\,16)\,trunc(N/16))*14+23':cb=128:cr=128"
)


def run_ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def numbered_clip(
    path: Path, *, size=(64, 36), seconds: float = 2.0, audio: bool = True, rotation=None, video=H264
) -> Path:
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["-f", "lavfi", "-i", NUMBERED_FRAMES.format(w=w, h=h, seconds=seconds, half=w // 2)]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    target = path if rotation is None else path.with_name("unrotated-" + path.name)
    run_ffmpeg(*inputs, *video, *(["-c:a", "aac"] if audio else []), str(target))
    if rotation is not None:
        run_ffmpeg("-display_rotation:v:0", str(rotation), "-i", str(target), "-c", "copy", str(path))
    return path


def fake_result(original: Path, path: Path, *, scale: int = 2, frames: int | None = None,
                video=H264, audio=("-c:a", "aac")) -> Path:  # fmt: skip
    """What a video job leaves: the original upright and bigger, at the same rate, with its sound."""
    limit = ["-frames:v", str(frames)] if frames else []
    run_ffmpeg("-i", str(original), "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor", *limit, *video,
               *audio, "-shortest", str(path))  # fmt: skip
    return path


def facts(path: Path) -> dict:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        cc = video.codec_context
        return {
            "codec": cc.name,
            "pix_fmt": cc.pix_fmt,
            "size": (cc.width, cc.height),
            "rate": video.average_rate,
            "frames": video.frames,
            "audio": [s.codec_context.name for s in container.streams.audio],
            "first_pts": float(next(container.decode(video)).time),
        }


def frame_numbers(path: Path) -> list[int]:
    """The number each frame shows (see NUMBERED_FRAMES), read from its grey levels."""
    numbers = []
    step = 14 * 255 / 219  # one digit, in full-range RGB
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            h, w = rgb.shape[:2]
            low, high = (rgb[h // 4 : 3 * h // 4, w * e // 8 : w * (e + 2) // 8].mean() for e in (1, 5))
            numbers.append(int(low // step) + 16 * int(high // step))
    return numbers


def prepare(folder: Path, original: Path | None, upscaled: Path, cancel: threading.Event | None = None):
    seen: list[float] = []
    info = compare.prepare(
        folder, original, upscaled, progress=lambda f, m: seen.append(f), cancel=cancel or threading.Event()
    )
    return info, seen


# --- The copies ---------------------------------------------------------------------------------


def test_copies_are_browser_friendly_h264_frame_for_frame(tmp_path):
    original = numbered_clip(tmp_path / "in" / "clip.mov")
    result = fake_result(original, tmp_path / "clip-4K-fast.mp4")
    folder = tmp_path / "job" / "compare" / "0"
    info, seen = prepare(folder, original, result)

    first, second = facts(compare.copy_path(folder, "original")), facts(compare.copy_path(folder, "upscaled"))
    for copy in (first, second):
        assert copy["codec"] == "h264" and copy["pix_fmt"] == "yuv420p"
        assert copy["rate"] == 25 and copy["frames"] == 50 and copy["first_pts"] == 0
    assert first["size"] == (64, 36) and second["size"] == (128, 72)
    assert first["audio"] == [] and second["audio"] == ["aac"]  # sound only from the upscaled side
    assert frame_numbers(compare.copy_path(folder, "original")) == list(range(50))
    assert frame_numbers(compare.copy_path(folder, "upscaled")) == list(range(50))
    assert info == {
        "fps": "25/1", "frames": 50, "duration": 2.0, "has_audio": True,
        "original": {"width": 64, "height": 36}, "upscaled": {"width": 128, "height": 72},
    }  # fmt: skip
    assert compare.ready_info(folder) == info
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert sorted(p.name for p in folder.iterdir()) == [
        "original.json",
        "original.mp4",
        "upscaled.json",
        "upscaled.mp4",
    ]


def test_a_rotated_phone_video_is_compared_upright(tmp_path):
    original = numbered_clip(tmp_path / "phone.mp4", size=(320, 180), seconds=1, rotation=-90)
    result = tmp_path / "phone-4K-fast.mp4"
    run_ffmpeg(
        "-i", str(original), "-vf", "scale=iw*2:ih*2", *H264, "-c:a", "aac", str(result)
    )  # upright, like the job's
    info, _ = prepare(tmp_path / "compare", original, result)
    assert info["original"] == {"width": 180, "height": 320} and info["upscaled"] == {"width": 360, "height": 640}
    assert facts(compare.copy_path(tmp_path / "compare", "original"))["size"] == (180, 320)


def test_a_preview_is_compared_with_the_same_first_seconds(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=3)
    result = fake_result(original, tmp_path / "clip-4K-fast-preview1s.mp4", frames=25)
    info, _ = prepare(tmp_path / "compare", original, result)
    assert info["frames"] == 25 and info["duration"] == 1.0
    assert frame_numbers(compare.copy_path(tmp_path / "compare", "original")) == list(range(25))


def test_odd_sizes_lose_a_row_but_say_the_real_size(tmp_path):
    odd = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p")  # 4:2:0 can't hold odd sides
    original = numbered_clip(tmp_path / "clip.mp4", size=(66, 37), seconds=1, audio=False, video=odd)
    result = fake_result(original, tmp_path / "out.mp4", audio=())
    info, _ = prepare(tmp_path / "compare", original, result)
    assert info["original"] == {"width": 66, "height": 37} and info["has_audio"] is False
    assert facts(compare.copy_path(tmp_path / "compare", "original"))["size"] == (66, 36)


def test_a_ten_bit_hevc_result_becomes_eight_bit_h264_with_aac(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=1)
    result = fake_result(
        original, tmp_path / "out.mp4",
        video=("-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params", "log-level=error"),
        audio=("-c:a", "alac"),
    )  # fmt: skip
    assert facts(result)["pix_fmt"] == "yuv420p10le" and facts(result)["audio"] == ["alac"]
    prepare(tmp_path / "compare", original, result)
    copy = facts(compare.copy_path(tmp_path / "compare", "upscaled"))
    assert (copy["codec"], copy["pix_fmt"], copy["audio"]) == ("h264", "yuv420p", ["aac"])


def test_stopping_part_way_leaves_no_half_made_copy(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=2)
    result = fake_result(original, tmp_path / "out.mp4")
    cancel = threading.Event()
    with pytest.raises(Cancelled):
        compare.prepare(tmp_path / "compare", original, result, progress=lambda f, m: cancel.set(), cancel=cancel)
    assert not list((tmp_path / "compare").glob("*.mp4")) and compare.ready_info(tmp_path / "compare") is None


def test_without_the_original_there_is_nothing_to_compare(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=1)
    result = fake_result(original, tmp_path / "out.mp4")
    with pytest.raises(VideoError) as caught:
        prepare(tmp_path / "compare", None, result)
    assert str(caught.value) == compare.ORIGINAL_GONE
    with pytest.raises(VideoError) as caught:
        prepare(tmp_path / "compare", original, tmp_path / "gone.mp4")
    assert str(caught.value) == compare.RESULT_GONE


def test_a_hardware_encoder_that_fails_hands_over_to_x264(tmp_path, monkeypatch):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=1)
    result = fake_result(original, tmp_path / "out.mp4")
    broken = ("-c:v", "h264_not_a_real_encoder")
    monkeypatch.setitem(compare._H264_ENCODERS, "broken", broken)
    monkeypatch.setattr(compare, "h264_encoder", lambda: "broken")
    info, _ = prepare(tmp_path / "compare", original, result)
    assert info["frames"] == 25 and facts(compare.copy_path(tmp_path / "compare", "upscaled"))["codec"] == "h264"


def test_hardware_h264_is_preferred_where_it_works(monkeypatch):
    compare.h264_encoder.cache_clear()
    try:
        monkeypatch.setattr(compare, "_encoder_works", lambda args: args[1] == "h264_videotoolbox")
        monkeypatch.setattr(compare.sys, "platform", "darwin")
        assert compare.h264_encoder() == "h264_videotoolbox"
        compare.h264_encoder.cache_clear()
        monkeypatch.setattr(compare.sys, "platform", "win32")  # no NVIDIA card here
        assert compare.h264_encoder() == "libx264"
    finally:
        compare.h264_encoder.cache_clear()


# --- In the background ---------------------------------------------------------------------


def wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_comparisons_run_in_the_background_and_asking_twice_is_fine(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=2)
    result = fake_result(original, tmp_path / "out.mp4")
    comparisons = compare.Comparisons()
    folder = tmp_path / "compare"
    assert comparisons.status(folder) == {"status": "idle", "progress": 0.0, "message": ""}
    first = comparisons.start(folder, original, result)
    again = comparisons.start(folder, original, result)
    assert first["status"] in ("waiting", "preparing", "ready") and again["status"] in ("waiting", "preparing", "ready")
    wait_until(lambda: comparisons.status(folder)["status"] == "ready")
    ready = comparisons.status(folder)
    assert ready["progress"] == 1.0 and ready["frames"] == 50 and ready["upscaled"] == {"width": 128, "height": 72}
    assert comparisons.start(folder, original, result)["status"] == "ready"  # made once, kept with the job
    assert compare.Comparisons().status(folder)["status"] == "ready"  # and after a restart


def test_copies_that_went_missing_are_made_again(tmp_path):
    import shutil

    original = numbered_clip(tmp_path / "clip.mp4", seconds=1)
    result = fake_result(original, tmp_path / "out.mp4")
    comparisons = compare.Comparisons()
    folder = tmp_path / "compare"
    comparisons.start(folder, original, result)
    wait_until(lambda: comparisons.status(folder)["status"] == "ready")
    shutil.rmtree(folder)  # e.g. tidied away with its job while the app kept running
    assert comparisons.status(folder)["status"] == "idle"  # not "ready" with nothing to play
    comparisons.start(folder, original, result)
    wait_until(lambda: comparisons.status(folder)["status"] == "ready")
    assert compare.copy_path(folder, "upscaled").is_file()


def test_one_comparison_at_a_time_and_each_can_be_stopped(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", size=(320, 180), seconds=6)
    result = fake_result(original, tmp_path / "out.mp4", scale=4)
    comparisons = compare.Comparisons()
    first, second = tmp_path / "a", tmp_path / "b"
    comparisons.start(first, original, result)
    assert comparisons.start(second, original, result)["status"] == "waiting"
    assert comparisons.cancel(second) is True
    wait_until(lambda: comparisons.status(second)["status"] == "cancelled")
    assert comparisons.status(second)["message"] == compare.STOPPED
    wait_until(lambda: comparisons.status(first)["status"] in ("preparing", "ready"))
    comparisons.cancel(first)
    wait_until(lambda: comparisons.status(first)["status"] in ("cancelled", "ready"))
    assert comparisons.cancel(tmp_path / "never") is False
    if comparisons.status(first)["status"] == "cancelled":
        assert not list(first.glob("*.part.mp4"))
        # Try again carries on: the finished copy (if any) is kept, the rest is made.
        comparisons.start(first, original, result)
        wait_until(lambda: comparisons.status(first)["status"] == "ready", timeout=60)


def test_a_missing_original_fails_straight_away_in_plain_words(tmp_path):
    original = numbered_clip(tmp_path / "clip.mp4", seconds=1)
    result = fake_result(original, tmp_path / "out.mp4")
    state = compare.Comparisons().start(tmp_path / "compare", None, result)
    assert state["status"] == "failed" and state["error"] == compare.ORIGINAL_GONE


# --- Very large originals -------------------------------------------------------------------


def test_huge_originals_are_swapped_for_their_small_copy(tmp_path, monkeypatch):
    original = tmp_path / "in" / "0" / "camcorder.avi"
    original.parent.mkdir(parents=True)
    run_ffmpeg(
        "-f",
        "lavfi",
        "-i",
        NUMBERED_FRAMES.format(w=320, h=180, seconds=2, half=160),
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        str(original),
    )  # fmt: skip  (uncompressed, like old capture files)
    result = fake_result(original, tmp_path / "out.mp4", audio=())
    monkeypatch.setattr(compare, "COMPACT_OVER_BYTES", 1000)
    assert compare.worth_compacting(original) is True
    folder = compare.compare_folder(tmp_path, 0)
    assert compare.keep_compact_original(original, folder, result) is True
    assert not original.exists() and compare.copy_path(folder, "original").is_file()
    info, _ = prepare(folder, None, result)  # the comparison uses the small copy
    assert info["original"] == {"width": 320, "height": 180} and info["frames"] == 50
    assert frame_numbers(compare.copy_path(folder, "original")) == list(range(50))


def test_ordinary_phone_videos_are_kept_as_they_are(tmp_path, monkeypatch):
    original = numbered_clip(tmp_path / "phone.mp4", size=(320, 180), seconds=1)
    assert compare.worth_compacting(original) is False  # small
    monkeypatch.setattr(compare, "COMPACT_OVER_BYTES", 10)
    assert compare.worth_compacting(original) is False  # big enough, but already compact
    assert compare.worth_compacting(tmp_path / "missing.mp4") is False


def test_a_finished_video_job_compacts_a_huge_original(tmp_path, monkeypatch):
    from upscaler.video import VideoResult, pipeline

    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))
    monkeypatch.setattr(jobs, "release_photoreal_on_small_mac", lambda: None)
    output = tmp_path / "out.mp4"
    monkeypatch.setattr(
        pipeline,
        "upscale_video",
        lambda src, out_dir, job_dir, opts, *, progress, cancel: VideoResult(output, 128, 72, 1.0, []),
    )
    calls = []
    monkeypatch.setattr(compare, "worth_compacting", lambda source: True)
    monkeypatch.setattr(compare, "keep_compact_original", lambda *a, **k: calls.append(a) or True)
    job = Job(id="a" * 12, kind="video", owner="host", inputs=[str(tmp_path / "in" / "0" / "clip.avi")], options={})
    seen = []
    [result] = jobs.run_video_job(job, tmp_path / "job", lambda f, m: seen.append(m), threading.Event())
    assert result["path"] == str(output)
    assert calls == [(Path(job.inputs[0]), tmp_path / "job" / "compare" / "0", output)]
    assert "Keeping a small copy of the original for the comparison" in seen

    def broken(*args, **kwargs):
        raise VideoError("nope")

    monkeypatch.setattr(compare, "keep_compact_original", broken)
    assert jobs.run_video_job(job, tmp_path / "job", lambda f, m: None, threading.Event())  # still finishes


# --- The endpoints ----------------------------------------------------------------------------


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))
    monkeypatch.setattr(server, "lan_ipv4_addresses", lambda: ["192.168.1.20"])
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)
    monkeypatch.setattr(compare, "COMPARISONS", compare.Comparisons())
    queue = JobQueue(tmp_path / "data" / "jobs", {"video": lambda job, progress, cancel: [], "image": lambda *a: []})
    app = server.create_app(queue, False, port=7860)
    return SimpleNamespace(tmp=tmp_path, queue=queue, app=app, out=tmp_path / "home" / "Pictures" / "Upscaled")


def finished_video_job(env, *, owner="host", seconds=2.0) -> Job:
    job_id = env.queue.new_job_id()
    original = numbered_clip(env.queue.job_dir(job_id) / "in" / "0" / "clip.mov", seconds=seconds)
    env.out.mkdir(parents=True, exist_ok=True)
    result = fake_result(original, env.out / "clip-4K-fast.mp4")
    job = Job(id=job_id, kind="video", owner=owner, status="done", progress=1.0, inputs=[str(original)],
              results=[{"path": str(result), "filename": result.name, "width": 128, "height": 72}])  # fmt: skip
    with env.queue._lock:
        env.queue._jobs[job.id] = job
        env.queue._save(job)
    return job


def test_compare_endpoints_prepare_and_serve_the_copies(env):
    job = finished_video_job(env)
    base = f"/api/jobs/{job.id}/results/0/compare"
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        assert host.get(f"{base}/status").json()["status"] == "idle"
        assert host.get(f"{base}/original.mp4").status_code == 404  # not made yet
        started = host.post(base).json()
        assert started["status"] in ("waiting", "preparing", "ready")
        wait_until(lambda: host.get(f"{base}/status").json()["status"] == "ready")
        ready = host.post(base).json()  # asking again just says it's ready
        assert ready["status"] == "ready" and ready["frames"] == 50 and ready["fps"] == "25/1"
        assert ready["original_url"] == f"{base}/original.mp4" and ready["upscaled_url"] == f"{base}/upscaled.mp4"
        assert ready["original"] == {"width": 64, "height": 36} and ready["has_audio"] is True
        whole = host.get(ready["upscaled_url"])
        assert whole.status_code == 200 and whole.headers["content-type"] == "video/mp4"
        part = host.get(ready["original_url"], headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.content) == 100  # browsers seek by asking for parts
        assert host.get(f"{base}/other.mp4").status_code == 404
        assert host.post(f"{base}/cancel").json()["ok"] is False  # nothing to stop
    assert str(env.tmp) not in str(ready)  # no file system paths


def test_only_the_owner_can_compare_a_video(env):
    job = finished_video_job(env)
    with TestClient(env.app, base_url="http://192.168.1.20:7860", client=("192.168.1.5", 50001)) as guest:
        for method, path in (("post", ""), ("get", "/status"), ("post", "/cancel"), ("get", "/original.mp4")):
            response = getattr(guest, method)(f"/api/jobs/{job.id}/results/0/compare{path}")
            assert response.status_code == 403, path


def test_pictures_and_missing_results_cant_be_compared(env):
    picture = Job(id=env.queue.new_job_id(), kind="image", owner="host", status="done", results=[{"path": "x"}])
    with env.queue._lock:
        env.queue._jobs[picture.id] = picture
    job = finished_video_job(env, seconds=1)
    Path(job.results[0]["path"]).unlink()
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        response = host.post(f"/api/jobs/{picture.id}/results/0/compare")
        assert response.status_code == 404 and response.json()["detail"] == server.NOT_A_VIDEO
        assert host.post(f"/api/jobs/{job.id}/results/1/compare").status_code == 404
        response = host.post(f"/api/jobs/{job.id}/results/0/compare")
        assert response.status_code == 404 and response.json()["detail"] == compare.RESULT_GONE


def test_a_video_whose_original_is_gone_says_so(env):
    job = finished_video_job(env, seconds=1)
    Path(job.inputs[0]).unlink()
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        state = host.post(f"/api/jobs/{job.id}/results/0/compare").json()
    assert state["status"] == "failed" and state["error"] == compare.ORIGINAL_GONE


def test_stopping_a_comparison_from_the_page(env, monkeypatch):
    job = finished_video_job(env, seconds=1)
    release = threading.Event()
    real = compare.prepare

    def slow_prepare(folder, original, upscaled, *, progress, cancel):
        progress(0.42, "Making a copy of the original your browser can play")
        release.wait(5)
        if cancel.is_set():
            raise Cancelled()
        return real(folder, original, upscaled, progress=progress, cancel=cancel)

    monkeypatch.setattr(compare, "prepare", slow_prepare)
    base = f"/api/jobs/{job.id}/results/0/compare"
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        host.post(base)
        wait_until(lambda: host.get(f"{base}/status").json()["progress"] == 0.42)
        assert host.get(f"{base}/status").json()["status"] == "preparing"
        assert host.post(f"{base}/cancel").json()["ok"] is True
        release.set()
        wait_until(lambda: host.get(f"{base}/status").json()["status"] == "cancelled")


# --- The still on a finished video's card -------------------------------------------------------


def test_a_still_is_made_once_from_the_result(tmp_path):
    from PIL import Image

    original = numbered_clip(tmp_path / "clip.mov", size=(64, 36), seconds=2.0)
    hevc = ("-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
            "-x265-params", "log-level=error")  # fmt: skip
    result = fake_result(original, tmp_path / "clip-4K-fast.mp4", scale=40, video=hevc)  # 2560 x 1440, 10-bit HEVC
    folder = tmp_path / "compare" / "0"
    still = compare.poster(folder, result)
    assert still == folder / compare.POSTER
    with Image.open(still) as picture:
        assert picture.format == "JPEG" and picture.size == (960, 540)  # the longer side at most POSTER_EDGE
    made = still.stat().st_mtime_ns
    assert compare.poster(folder, result).stat().st_mtime_ns == made  # kept, not made again
    assert not list(folder.glob("*.part.*"))
    assert compare.poster_time(5) == 1.0 and compare.poster_time(2) == 0.5 and compare.poster_time(0) == 0.0
    result.unlink()
    with pytest.raises(VideoError, match="isn't here any more"):
        compare.poster(folder, result)


def test_a_still_of_a_very_short_clip_falls_back_to_its_first_frame(tmp_path, monkeypatch):
    original = numbered_clip(tmp_path / "clip.mov", seconds=0.2, audio=False)
    result = fake_result(original, tmp_path / "short.mp4", audio=())
    # The clip claims to be longer than it is: the still is taken from the start instead.
    monkeypatch.setattr(compare, "poster_time", lambda duration: 30.0)
    assert compare.poster(tmp_path / "compare", result).stat().st_size > 0


def test_the_card_still_endpoint(env):
    job = finished_video_job(env, seconds=1)
    picture = Job(id=env.queue.new_job_id(), kind="image", owner="host", status="done", results=[{"path": "x"}])
    with env.queue._lock:
        env.queue._jobs[picture.id] = picture
    url = f"/api/jobs/{job.id}/results/0/poster.jpg"
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        response = host.get(url)
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert response.content[:3] == b"\xff\xd8\xff" and "max-age" in response.headers["cache-control"]
        assert host.get(f"/api/jobs/{picture.id}/results/0/poster.jpg").json()["detail"] == server.NOT_A_VIDEO
        assert host.get(f"/api/jobs/{job.id}/results/3/poster.jpg").status_code == 404
    with TestClient(env.app, base_url="http://192.168.1.20:7860", client=("192.168.1.5", 50001)) as guest:
        assert guest.get(url).status_code == 403  # someone else's video
    Path(job.results[0]["path"]).unlink()
    shutil.rmtree(compare.compare_folder(env.queue.job_dir(job.id), 0), ignore_errors=True)
    with TestClient(env.app, base_url="http://127.0.0.1:7860", client=("127.0.0.1", 50000)) as host:
        gone = host.get(url)
        assert gone.status_code == 404 and gone.json()["detail"] == compare.RESULT_GONE
