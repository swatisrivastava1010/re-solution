import errno
import json
import os
import threading
import time
from pathlib import Path

import pytest

from upscaler import config, jobs, keepawake
from upscaler.config import Cancelled
from upscaler.jobs import EtaTracker, Job, JobQueue, default_runners, delete_old_files, friendly_error
from upscaler.keepawake import KeepAwake


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.01)


def finished(queue: JobQueue, job_id: str) -> bool:
    return queue.get(job_id).status in ("done", "failed", "cancelled")


@pytest.fixture
def store(tmp_path) -> Path:
    return tmp_path / "jobs"


# --- Order and permissions --------------------------------------------------------


def test_queue_runs_in_order(store):
    events = []
    release_first = threading.Event()

    def runner(job, progress, cancel):
        events.append(("start", job.inputs[0]))
        if job.inputs[0] == "a":
            release_first.wait(5)
        events.append(("end", job.inputs[0]))
        return [{"filename": job.inputs[0]}]

    queue = JobQueue(store, {"image": runner})
    queue.start()
    try:
        first = queue.submit("image", "host", ["a"], {})
        second = queue.submit("image", "guest:x", ["b"], {})
        wait_for(lambda: queue.get(first.id).status == "running")
        time.sleep(0.2)
        assert queue.get(second.id).status == "queued"
        release_first.set()
        wait_for(lambda: queue.get(second.id).status == "done")
    finally:
        queue.stop()
    assert events == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]
    assert queue.get(first.id).results == [{"filename": "a"}]
    assert queue.get(first.id).progress == 1.0


def test_preview_jobs_jump_ahead_of_queued_jobs(store):
    events = []
    release_first = threading.Event()

    def runner(job, progress, cancel):
        events.append(job.inputs[0])
        if job.inputs[0] == "running":
            release_first.wait(5)
        return []

    queue = JobQueue(store, {"image": runner, "preview": runner})
    queue.start()
    try:
        running = queue.submit("image", "host", ["running"], {})
        wait_for(lambda: queue.get(running.id).status == "running")
        waiting = queue.submit("image", "host", ["waiting"], {})
        preview = queue.submit("preview", "guest:x", ["preview"], {})
        time.sleep(0.1)
        assert queue.get(running.id).status == "running"  # never interrupted
        release_first.set()
        wait_for(lambda: finished(queue, waiting.id) and finished(queue, preview.id))
    finally:
        queue.stop()
    assert events == ["running", "preview", "waiting"]


def test_queue_cancel_permissions(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})  # never started: jobs stay queued
    host_job = queue.submit("image", "host", ["a"], {})
    guest_job = queue.submit("image", "guest:x", ["b"], {})
    own_job = queue.submit("image", "guest:x", ["c"], {})

    assert queue.cancel(host_job.id, "guest:x") is False
    assert queue.get(host_job.id).status == "queued"
    assert queue.cancel(own_job.id, "guest:other") is False
    assert queue.cancel(own_job.id, "guest:x") is True
    assert queue.cancel(guest_job.id, "host") is True
    assert queue.get(guest_job.id).status == "cancelled"
    assert queue.get(own_job.id).status == "cancelled"
    assert queue.cancel("no-such-job", "host") is False


def test_cancel_running_job(store):
    def runner(job, progress, cancel):
        while not cancel.wait(0.01):
            progress(0.5, "Working")
        raise Cancelled()

    queue = JobQueue(store, {"video": runner})
    queue.start()
    try:
        job = queue.submit("video", "host", ["v.mp4"], {})
        wait_for(lambda: queue.get(job.id).status == "running")
        assert queue.cancel(job.id, "host") is True
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    assert queue.get(job.id).status == "cancelled"


def test_finished_job_cannot_be_cancelled(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    queue.start()
    try:
        job = queue.submit("image", "host", ["a"], {})
        wait_for(lambda: finished(queue, job.id))
        assert queue.cancel(job.id, "host") is False
    finally:
        queue.stop()
    assert queue.get(job.id).status == "done"


def test_unknown_kind_is_refused(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    with pytest.raises(ValueError):
        queue.submit("sculpture", "host", [], {})


# --- Persistence ------------------------------------------------------------------


def test_queue_survives_restart(store):
    ran = []
    queue = JobQueue(store, {"video": lambda job, progress, cancel: ran.append(job.id) or []})
    job = queue.submit("video", "host", ["v.mp4"], {"mode": "fast"})

    # Simulate a crash in the middle of the job: its saved record still says "running".
    record = store / job.id / "job.json"
    saved = json.loads(record.read_text())
    saved.update(status="running", progress=0.4, message="Upscaling frame 40 of 100")
    record.write_text(json.dumps(saved))

    restarted = JobQueue(store, {"video": lambda job, progress, cancel: ran.append(job.id) or []})
    again = restarted.get(job.id)
    assert again.status == "queued"
    assert again.progress == 0.0
    assert again.options == {"mode": "fast"}
    restarted.start()
    try:
        wait_for(lambda: finished(restarted, job.id))
    finally:
        restarted.stop()
    assert ran == [job.id]
    assert restarted.get(job.id).status == "done"


def test_stopping_the_app_requeues_the_running_job(store):
    def runner(job, progress, cancel):
        cancel.wait(5)
        raise Cancelled()

    queue = JobQueue(store, {"video": runner})
    queue.start()
    job = queue.submit("video", "host", ["v.mp4"], {})
    wait_for(lambda: queue.get(job.id).status == "running")
    queue.stop()

    assert queue.get(job.id).status == "queued"
    assert JobQueue(store, {"video": runner}).get(job.id).status == "queued"


def _crash(store: Path, job_id: str) -> None:
    """What a crash leaves behind: the job's saved record still says "running"."""
    record = store / job_id / "job.json"
    saved = json.loads(record.read_text())
    saved.update(status="running", progress=0.4, message="Working")
    record.write_text(json.dumps(saved))


def test_a_job_that_keeps_crashing_the_app_is_set_aside(store):
    runners = {"image": lambda job, progress, cancel: []}
    job = JobQueue(store, runners).submit("image", "host", ["huge.png"], {})
    for crashes in range(1, jobs.MAX_CRASHES):
        _crash(store, job.id)
        again = JobQueue(store, runners).get(job.id)
        assert (again.status, again.crashes) == ("queued", crashes)  # a crash or two: it carries on
    _crash(store, job.id)
    restarted = JobQueue(store, runners)
    set_aside = restarted.get(job.id)
    assert set_aside.status == "failed" and set_aside.error == jobs.CRASHED_TOO_OFTEN
    assert JobQueue(store, runners).get(job.id).status == "failed"  # and it stays that way
    restarted.start()
    try:
        time.sleep(0.1)
    finally:
        restarted.stop()
    assert restarted.get(job.id).status == "failed"  # never run again


def test_stopping_the_app_saves_the_running_job_as_waiting_at_once(store, monkeypatch):
    # Closing the app may not wait for a slow job to wind down; it must still run again next time,
    # without counting as a crash.
    monkeypatch.setattr(jobs, "_STOP_WAIT_SECONDS", 0.05)
    release = threading.Event()

    def stubborn(job, progress, cancel):
        release.wait(5)  # ignores the cancel for a while, like a long Photo-real tile
        raise Cancelled()

    queue = JobQueue(store, {"image": stubborn})
    queue.start()
    job = queue.submit("image", "host", ["a.png"], {})
    wait_for(lambda: queue.get(job.id).status == "running")
    queue.stop()
    try:
        saved = json.loads((store / job.id / "job.json").read_text())
        assert saved["status"] == "queued" and saved.get("crashes", 0) == 0
        again = JobQueue(store, {"image": stubborn}).get(job.id)
        assert (again.status, again.crashes) == ("queued", 0)
    finally:
        release.set()
        wait_for(lambda: queue.get(job.id).status == "queued")


def test_a_cancel_is_remembered_even_if_the_app_dies_before_the_job_stops(store):
    release = threading.Event()

    def slow(job, progress, cancel):
        release.wait(5)
        raise Cancelled()

    queue = JobQueue(store, {"image": slow})
    queue.start()
    job = queue.submit("image", "guest:x", ["a.png"], {})
    wait_for(lambda: queue.get(job.id).status == "running")
    try:
        assert queue.cancel(job.id, "guest:x") is True
        # The app is killed now, before the job notices: next time it must not run again.
        assert JobQueue(store, {"image": slow}).get(job.id).status == "cancelled"
    finally:
        release.set()
        wait_for(lambda: finished(queue, job.id))
        queue.stop()
    assert queue.get(job.id).status == "cancelled"


# --- Cloud jobs are never sent twice -------------------------------------------------

CLOUD = {"mode": "cloud:fal:topaz_precision", "size": "8k"}


def test_cloud_prefix_matches_the_pipeline():
    from upscaler import pipeline_image
    from upscaler.seedvr2 import mlx_engine

    assert jobs.CLOUD_PREFIX == pipeline_image.CLOUD_PREFIX
    assert jobs.PHOTOREAL == pipeline_image.PHOTOREAL
    assert jobs.SMALL_MAC_GB == mlx_engine.SMALL_MAC_GB


def test_a_cloud_job_running_when_the_app_crashed_is_not_sent_again(store):
    ran = []
    runners = {"image": lambda job, progress, cancel: ran.append(job.id) or []}
    job = JobQueue(store, runners).submit("image", "host", ["a.png"], CLOUD)
    _crash(store, job.id)

    restarted = JobQueue(store, runners)
    again = restarted.get(job.id)
    assert again.status == "cancelled"
    assert again.message == jobs.CLOUD_INTERRUPTED and again.error == jobs.CLOUD_INTERRUPTED
    assert "won't be charged twice" in jobs.CLOUD_INTERRUPTED
    restarted.start()
    try:
        time.sleep(0.1)
    finally:
        restarted.stop()
    assert ran == [] and restarted.get(job.id).status == "cancelled"
    assert JobQueue(store, runners).get(job.id).status == "cancelled"  # and it stays that way


def test_stopping_the_app_ends_a_running_cloud_job_instead_of_requeueing_it(store):
    def runner(job, progress, cancel):
        cancel.wait(5)
        raise Cancelled()

    queue = JobQueue(store, {"image": runner})
    queue.start()
    job = queue.submit("image", "host", ["a.png"], CLOUD)
    wait_for(lambda: queue.get(job.id).status == "running")
    queue.stop()

    stopped = queue.get(job.id)
    assert (stopped.status, stopped.message) == ("cancelled", jobs.CLOUD_INTERRUPTED)
    again = JobQueue(store, {"image": runner}).get(job.id)
    assert (again.status, again.message) == ("cancelled", jobs.CLOUD_INTERRUPTED)


def test_a_cloud_job_is_saved_as_ended_the_moment_the_app_starts_stopping(store, monkeypatch):
    # The app may be gone before the job winds down: what's on disk must already say "don't send again".
    monkeypatch.setattr(jobs, "_STOP_WAIT_SECONDS", 0.05)
    release = threading.Event()

    def stubborn(job, progress, cancel):
        release.wait(5)
        raise Cancelled()

    queue = JobQueue(store, {"image": stubborn})
    queue.start()
    job = queue.submit("image", "host", ["a.png"], CLOUD)
    wait_for(lambda: queue.get(job.id).status == "running")
    queue.stop()
    try:
        saved = json.loads((store / job.id / "job.json").read_text())
        assert saved["status"] == "cancelled" and saved["message"] == jobs.CLOUD_INTERRUPTED
        assert JobQueue(store, {"image": stubborn}).get(job.id).status == "cancelled"
    finally:
        release.set()
        wait_for(lambda: queue.get(job.id).status == "cancelled")


def test_a_cloud_job_saved_as_waiting_to_continue_is_not_sent_again(store):
    # E.g. a record written by an older copy of the app, which put stopped cloud jobs back in the queue.
    runners = {"image": lambda job, progress, cancel: pytest.fail("must not be sent again")}
    job = JobQueue(store, runners).submit("image", "host", ["a.png"], CLOUD)
    record = store / job.id / "job.json"
    saved = json.loads(record.read_text())
    saved.update(status="queued", message=jobs.RESUME_MESSAGE)
    record.write_text(json.dumps(saved))
    restarted = JobQueue(store, runners)
    assert restarted.get(job.id).status == "cancelled"
    assert restarted.get(job.id).message == jobs.CLOUD_INTERRUPTED


def test_an_interrupted_cloud_batch_keeps_the_pictures_already_made(store):
    runners = {"image": lambda job, progress, cancel: []}
    job = JobQueue(store, runners).submit("image", "host", ["a.png", "b.png"], CLOUD)
    record = store / job.id / "job.json"
    saved = json.loads(record.read_text())
    saved.update(status="running", results=[{"filename": "a-8K-cloud-fal.jpg", "input": 0}])
    record.write_text(json.dumps(saved))
    again = JobQueue(store, runners).get(job.id)
    assert again.status == "cancelled"
    assert again.results == [{"filename": "a-8K-cloud-fal.jpg", "input": 0}]


def test_a_local_job_still_carries_on_after_a_restart(store):
    runners = {"image": lambda job, progress, cancel: []}
    job = JobQueue(store, runners).submit("image", "host", ["a.png"], {"mode": "fast"})
    _crash(store, job.id)
    assert JobQueue(store, runners).get(job.id).status == "queued"


def test_a_cloud_job_that_never_started_still_runs_after_a_restart(store):
    # Nothing was sent, so nothing can be charged twice: it runs with the host's saved key as asked.
    ran = []
    runners = {"image": lambda job, progress, cancel: ran.append(job.id) or []}
    job = JobQueue(store, runners).submit("image", "host", ["a.png"], CLOUD)
    restarted = JobQueue(store, runners)
    assert restarted.get(job.id).status == "queued"
    restarted.start()
    try:
        wait_for(lambda: finished(restarted, job.id))
    finally:
        restarted.stop()
    assert ran == [job.id]


def test_a_guests_own_key_job_is_not_switched_to_the_hosts_key_after_a_restart(store):
    # The guest's key lived only in memory. Without it the job must not quietly use the host's account.
    runners = {"image": lambda job, progress, cancel: pytest.fail("must not run")}
    job = JobQueue(store, runners).submit("image", "guest:x", ["a.png"], CLOUD, secret="guest-key")
    assert "guest-key" not in (store / job.id / "job.json").read_text()
    restarted = JobQueue(store, runners)
    again = restarted.get(job.id)
    assert again.status == "cancelled" and again.message == jobs.OWN_KEY_FORGOTTEN
    assert again.secret is None


def test_image_runner_never_uses_the_hosts_key_for_a_job_that_had_its_own(store, monkeypatch):
    from upscaler import pipeline_image

    monkeypatch.setattr(pipeline_image, "upscale_image", lambda *a, **k: pytest.fail("must not run"))
    with config.changing_settings() as settings:
        settings.access_password_hash, settings.allow_guests_cloud = "salt:hash", True
    job = _image_job(store, "guest:x", ["a.png"], CLOUD)
    job.own_key = True  # submitted with the guest's own key, which has since been forgotten
    with pytest.raises(jobs.JobError) as caught:
        default_runners(store)["image"](job, lambda f, m: None, threading.Event())
    assert str(caught.value) == jobs.OWN_KEY_FORGOTTEN


# --- A video that ran out of disk space carries on with Retry ---------------------------------


def _video_with_work(queue: JobQueue, owner: str = "host") -> Job:
    job_id = queue.new_job_id()
    src = queue.job_dir(job_id) / "in" / "0" / "clip.mov"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"video")
    return queue.submit("video", owner, [str(src)], {"mode": "fast"}, job_id=job_id)


def test_a_video_that_ran_out_of_disk_space_keeps_its_work_for_retry(store):
    from upscaler.video import NotEnoughDiskSpace

    runs = []

    def runner(job, progress, cancel):
        runs.append(job.id)
        segments = store / job.id / "work" / "segments"
        if len(runs) == 1:
            segments.mkdir(parents=True)
            (segments / "segment-00000.mp4").write_bytes(b"done already")
            raise NotEnoughDiskSpace("There isn't enough free disk space to finish this video. Free up about 3 GB.")
        assert (segments / "segment-00000.mp4").read_bytes() == b"done already"  # carried on from it
        return [{"filename": "clip-4K-fast.mp4"}]

    queue = JobQueue(store, {"video": runner})
    queue.start()
    try:
        job = _video_with_work(queue)
        wait_for(lambda: finished(queue, job.id))
        failed = queue.get(job.id)
        assert failed.status == "failed" and failed.retryable is True
        assert failed.error.startswith("There isn't enough free disk space to finish this video. Free up about 3 GB.")
        assert failed.error.endswith(jobs.RETRY_HINT) and "press Retry" in jobs.RETRY_HINT
        assert (store / job.id / "in" / "0" / "clip.mov").is_file()  # the video and the finished parts stay
        assert (store / job.id / "work" / "segments" / "segment-00000.mp4").is_file()

        assert JobQueue(store, {"video": runner}).get(job.id).retryable is True  # remembered across a restart
        assert queue.retry(job.id, "guest:someone") is False  # not theirs
        assert queue.retry(job.id, "host") is True
        assert queue.get(job.id).status in ("queued", "running")
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    done = queue.get(job.id)
    assert (done.status, done.retryable, done.error) == ("done", False, None)
    assert runs == [job.id, job.id]
    assert not (store / job.id / "work").exists()  # tidied once done
    assert (store / job.id / "in" / "0" / "clip.mov").is_file()  # the original stays, for the side-by-side comparison


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        # Every segment finished, then joining them failed (the disk filled up meanwhile, say).
        ("join", "Couldn't put the finished video together. Check that the computer has free disk space."),
        # A plain "disk full" from anywhere along the way (the list of finished segments, a log file).
        ("disk full", "The disk is full. Free up some space."),
    ],
)
def test_a_video_whose_join_failed_or_whose_disk_filled_up_keeps_its_work(store, failure, error):
    from upscaler.video.pipeline import JoinFailed

    runs = []

    def runner(job, progress, cancel):
        runs.append(job.id)
        segments = store / job.id / "work" / "segments"
        if len(runs) == 1:
            segments.mkdir(parents=True)
            (segments / "segment-00000.mp4").write_bytes(b"hours of work")
            if failure == "join":
                raise JoinFailed(error)
            raise OSError(errno.ENOSPC, "No space left on device")
        assert (segments / "segment-00000.mp4").read_bytes() == b"hours of work"
        return [{"filename": "clip-4K-fast.mp4"}]

    queue = JobQueue(store, {"video": runner})
    queue.start()
    try:
        job = _video_with_work(queue)
        wait_for(lambda: finished(queue, job.id))
        failed = queue.get(job.id)
        assert failed.status == "failed" and failed.retryable is True
        assert failed.error == f"{error} {jobs.RETRY_HINT}"
        assert (store / job.id / "work" / "segments" / "segment-00000.mp4").is_file()
        assert (store / job.id / "in" / "0" / "clip.mov").is_file()  # the upload too
        assert queue.retry(job.id, "host") is True
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    assert queue.get(job.id).status == "done"


def test_a_picture_job_that_fills_the_disk_isnt_kept_for_retry(store):
    # Pictures already made are saved with the job as they're made; there's nothing half-made to keep.
    def runner(job, progress, cancel):
        raise OSError(errno.ENOSPC, "No space left on device")

    queue = JobQueue(store, {"image": runner})
    queue.start()
    try:
        job = queue.submit("image", "host", ["a.png"], {})
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    failed = queue.get(job.id)
    assert (failed.retryable, failed.error) == (False, jobs.DISK_FULL)


def test_forgetting_a_key_stops_a_running_job_that_uses_it(store):
    started, stopped = threading.Event(), threading.Event()

    def runner(job, progress, cancel):
        started.set()
        cancel.wait(5)
        stopped.set()
        raise Cancelled()

    queue = JobQueue(store, {"image": runner})
    queue.start()
    try:
        running = queue.submit("image", "guest:x", ["a.png"], {"mode": "cloud:fal:seedvr2"}, secret="key")
        waiting = queue.submit("image", "guest:x", ["b.png"], {"mode": "cloud:fal:seedvr2"}, secret="key")
        other_service = queue.submit("image", "guest:x", ["c.png"], {"mode": "cloud:replicate:topaz_hf"}, secret="r8")
        hosts = queue.submit("image", "host", ["d.png"], {"mode": "cloud:fal:seedvr2"})
        assert started.wait(5)
        assert queue.stop_own_key_jobs("guest:x", "fal") == 2
        assert stopped.wait(5)
        wait_for(lambda: finished(queue, running.id))
        assert queue.get(running.id).status == "cancelled"
        assert queue.get(waiting.id).status == "cancelled"
        assert queue.get(waiting.id).error == jobs.KEY_FORGOTTEN_BY_PERSON
        assert queue.get(other_service.id).secret == "r8"  # another service's key stays
        assert queue.get(hosts.id).status in ("queued", "running")  # the host's account, not the guest's key
        assert queue.stop_own_key_jobs("guest:y", "replicate") == 0  # only their own
    finally:
        queue.stop()


def test_only_jobs_that_kept_their_work_can_be_retried(store):
    def runner(job, progress, cancel):
        raise RuntimeError("something else")

    queue = JobQueue(store, {"video": runner})
    queue.start()
    try:
        job = _video_with_work(queue, owner="guest:x")
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    assert queue.get(job.id).retryable is False
    assert queue.retry(job.id, "guest:x") is False
    assert queue.retry("000000000000", "host") is False
    assert not (store / job.id / "in").exists()


def test_a_guest_can_retry_their_own_video(store):
    from upscaler.video import NotEnoughDiskSpace

    attempts = []

    def runner(job, progress, cancel):
        attempts.append(job.id)
        if len(attempts) == 1:
            raise NotEnoughDiskSpace("There isn't enough free disk space to finish this video. Free up about 1 GB.")
        return []

    queue = JobQueue(store, {"video": runner})
    queue.start()
    try:
        job = _video_with_work(queue, owner="guest:x")
        wait_for(lambda: finished(queue, job.id))
        assert queue.retry(job.id, "guest:x") is True
        wait_for(lambda: queue.get(job.id).status == "done")
    finally:
        queue.stop()


def test_finished_jobs_are_forgotten_while_the_app_runs(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: [], "preview": lambda job, progress, cancel: []})
    guest_job = queue.submit("image", "guest:x", ["a"], {})
    host_job = queue.submit("image", "host", ["b"], {})
    preview_job = queue.submit("preview", "host", ["c"], {})
    recent_job = queue.submit("image", "guest:x", ["d"], {})
    waiting_job = queue.submit("image", "guest:x", ["e"], {})
    for job in (guest_job, host_job, preview_job, recent_job):
        (store / job.id / "results").mkdir()
        (store / job.id / "results" / "0-preview.jpg").write_bytes(b"preview")
    queue.start()
    try:
        wait_for(lambda: all(finished(queue, j.id) for j in (guest_job, host_job, preview_job, recent_job)))
    finally:
        queue.stop()
    with queue._lock:  # a day and a bit later
        for job in (guest_job, host_job, preview_job):
            queue._jobs[job.id].finished = time.time() - 25 * 3600
        queue._jobs[waiting_job.id].status = "queued"
    events = queue.subscribe()

    assert queue.prune_expired() == 2
    assert queue.get(guest_job.id) is None and not (store / guest_job.id).exists()  # its preview goes too
    assert queue.get(preview_job.id) is None and not (store / preview_job.id).exists()
    assert queue.get(host_job.id) is not None  # the host's list entries last a week
    assert queue.get(recent_job.id) is not None and queue.get(waiting_job.id) is not None
    assert events.get_nowait() == {"event": "status"}
    assert queue.prune_expired() == 0


def test_secret_is_never_saved_or_shown(store):
    seen = []
    queue = JobQueue(store, {"image": lambda job, progress, cancel: seen.append(job.secret) or []})
    job = queue.submit("image", "guest:x", ["a.png"], {"mode": "cloud:fal:seedvr2"}, secret="key-123")
    assert "key-123" not in (store / job.id / "job.json").read_text()
    assert "key-123" not in json.dumps(job.to_dict())
    assert "key-123" not in repr(job)
    queue.start()
    try:
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    assert seen == ["key-123"]


def test_submit_with_reserved_id_uses_its_folder(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    job_id = queue.new_job_id()
    upload = queue.job_dir(job_id) / "in" / "0" / "a.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"x")
    job = queue.submit("image", "host", [str(upload)], {}, job_id=job_id)
    assert job.id == job_id
    assert (store / job_id / "job.json").is_file()


def test_inputs_are_removed_once_finished(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    job_id = queue.new_job_id()
    upload = queue.job_dir(job_id) / "in" / "0" / "a.png"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"x")
    queue.submit("image", "host", [str(upload)], {}, job_id=job_id)
    queue.start()
    try:
        wait_for(lambda: finished(queue, job_id))
    finally:
        queue.stop()
    assert not (store / job_id / "in").exists()
    assert (store / job_id / "job.json").is_file()


def test_old_finished_jobs_are_forgotten_on_start(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    guest_job = queue.submit("image", "guest:x", ["a"], {})
    host_job = queue.submit("image", "host", ["b"], {})
    recent_guest_job = queue.submit("image", "guest:x", ["c"], {})
    waiting_job = queue.submit("image", "guest:x", ["d"], {})
    day = 24 * 3600
    for job, age in ((guest_job, 1.5 * day), (host_job, 1.5 * day), (recent_guest_job, 0.5 * day)):
        record = store / job.id / "job.json"
        saved = json.loads(record.read_text())
        saved.update(status="done", finished=time.time() - age)
        record.write_text(json.dumps(saved))
    stale_upload = store / queue.new_job_id()  # an upload interrupted by a crash
    stale_upload.mkdir()

    restarted = JobQueue(store, {"image": lambda job, progress, cancel: []})
    assert restarted.get(guest_job.id) is None
    assert not (store / guest_job.id).exists()
    assert restarted.get(host_job.id) is not None
    assert restarted.get(recent_guest_job.id) is not None
    assert restarted.get(waiting_job.id).status == "queued"
    assert not stale_upload.exists()


def test_old_preview_jobs_are_forgotten_after_a_day_even_the_hosts(store):
    runners = {"preview": lambda job, progress, cancel: []}
    queue = JobQueue(store, runners)
    old, recent = queue.submit("preview", "host", ["a"], {}), queue.submit("preview", "host", ["b"], {})
    for job, age in ((old, 1.5 * 24 * 3600), (recent, 3600)):
        record = store / job.id / "job.json"
        saved = json.loads(record.read_text())
        saved.update(status="done", finished=time.time() - age)
        record.write_text(json.dumps(saved))

    restarted = JobQueue(store, runners)
    assert restarted.get(old.id) is None and not (store / old.id).exists()
    assert restarted.get(recent.id) is not None


def test_damaged_record_is_skipped(store):
    queue = JobQueue(store, {"image": lambda job, progress, cancel: []})
    good = queue.submit("image", "host", ["a"], {})
    damaged = store / queue.new_job_id()
    damaged.mkdir()
    (damaged / "job.json").write_text("{not json")
    restarted = JobQueue(store, {"image": lambda job, progress, cancel: []})
    assert [job.id for job in restarted.list()] == [good.id]


# --- Progress, errors and events --------------------------------------------------


def test_failed_job_gets_a_friendly_error(store):
    def runner(job, progress, cancel):
        if job.inputs[0] == "friendly":
            raise config.UserError("To use this mode, connect fal.ai in Settings first.")
        raise RuntimeError("MPS backend internal assertion at 0x7ff")

    queue = JobQueue(store, {"image": runner})
    queue.start()
    try:
        friendly = queue.submit("image", "host", ["friendly"], {})
        technical = queue.submit("image", "host", ["technical"], {})
        wait_for(lambda: finished(queue, technical.id))
    finally:
        queue.stop()
    assert queue.get(friendly.id).status == "failed"
    assert queue.get(friendly.id).error == "To use this mode, connect fal.ai in Settings first."
    assert "0x7ff" not in queue.get(technical.id).error
    assert queue.get(technical.id).error == jobs.GENERIC_ERROR


def test_friendly_error_messages():
    from upscaler.cloud.base import CloudError
    from upscaler.imageio_utils import ImageFileError
    from upscaler.seedvr2 import SeedVR2Unavailable

    assert (
        friendly_error(CloudError("fal.ai says your account has no credit."))
        == "fal.ai says your account has no credit."
    )
    assert friendly_error(SeedVR2Unavailable("Photo-real isn't installed.")) == "Photo-real isn't installed."
    assert friendly_error(ImageFileError("cat.png couldn't be opened.")) == "cat.png couldn't be opened."
    assert "memory" in friendly_error(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert "disk" in friendly_error(OSError(28, "No space left on device"))
    assert friendly_error(KeyError("x")) == jobs.GENERIC_ERROR
    assert friendly_error(config.UserError("Choose JPEG, PNG or TIFF instead.")) == "Choose JPEG, PNG or TIFF instead."


def _raised(work):
    try:
        work()
    except ValueError as exc:
        return exc
    raise AssertionError("nothing was raised")


def test_technical_value_errors_are_never_shown():
    import numpy as np
    from PIL import Image

    # Libraries raise plain ValueErrors in their own words; people get the general message.
    broadcast = _raised(lambda: np.zeros((4, 4, 3)) + np.zeros((4, 4, 4)))
    resize = _raised(lambda: Image.new("RGB", (4, 4)).resize((0, 0)))
    number = _raised(lambda: int("abc"))
    for exc in (broadcast, resize, number):
        assert friendly_error(exc) == jobs.GENERIC_ERROR
    assert friendly_error(ValueError("To use this mode, connect fal.ai in Settings first.")) == jobs.GENERIC_ERROR


def test_the_apps_own_plain_words_still_reach_people():
    from upscaler import pipeline_image

    # The app's own checks raise UserError, written for people.
    too_big = _raised(lambda: pipeline_image._check_output_size((100_000, 100_000), "16k"))
    assert isinstance(too_big, config.UserError)
    assert friendly_error(too_big) == str(too_big) and "smaller size" in str(too_big)


def test_only_user_errors_are_shown_as_they_are():
    from upscaler.seedvr2 import blend

    # An internal check's ValueError reads like a sentence but means nothing to people.
    internal = _raised(lambda: blend.blend_tiles([], 8, 8, 4, 4))
    assert str(internal) == "The tiles don't cover the whole image."
    assert friendly_error(internal) == jobs.GENERIC_ERROR
    assert not hasattr(jobs, "_apps_own_words")


def test_image_results_say_the_source_size(store, monkeypatch):
    from upscaler import pipeline_image

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        result = _fake_image_result(Path(path), out_dir)
        result.source_width, result.source_height = 960, 720
        return result

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "host", ["table.png"], {"mode": "fast", "size": "4k"})
    [result] = default_runners(store)["image"](job, lambda f, m: None, threading.Event())
    assert (result["source_width"], result["source_height"]) == (960, 720)


def test_subscribers_get_job_updates(store):
    def runner(job, progress, cancel):
        progress(0.5, "Halfway there")
        return []

    queue = JobQueue(store, {"image": runner})
    events = queue.subscribe()
    queue.start()
    try:
        job = queue.submit("image", "host", ["a"], {})
        wait_for(lambda: finished(queue, job.id))
    finally:
        queue.stop()
    seen = []
    while not events.empty():
        event = events.get_nowait()
        assert event["event"] == "job"
        seen.append((event["job"].status, event["job"].message))
    assert seen[0][0] == "queued"
    assert ("running", "Halfway there") in seen
    assert seen[-1][0] == "done"

    queue.unsubscribe(events)
    queue.publish_status()
    assert events.empty()


def test_publish_status_reaches_subscribers(store):
    queue = JobQueue(store, {})
    events = queue.subscribe()
    queue.publish_status()
    assert events.get_nowait() == {"event": "status"}


def test_eta_from_recent_progress():
    eta = EtaTracker()
    assert eta.update(0.0, 0.0) is None
    assert eta.update(2.0, 0.1) is None  # too early to tell
    assert eta.update(10.0, 0.5) == pytest.approx(10.0)
    assert eta.update(20.0, 1.0) == 0.0


# --- Guest outputs ----------------------------------------------------------------


def test_delete_old_files(tmp_path):
    guest = tmp_path / "guest"
    old = guest / "job1" / "old-8K-fast.jpg"
    new = guest / "job2" / "new-8K-fast.jpg"
    for path in (old, new):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    day_and_a_bit = time.time() - 25 * 3600
    os.utime(old, (day_and_a_bit, day_and_a_bit))

    assert delete_old_files(guest, 24 * 3600) == 1
    assert not old.exists()
    assert not old.parent.exists()  # empty folders go too
    assert new.exists()
    assert delete_old_files(tmp_path / "missing", 24 * 3600) == 0


# --- Runners ----------------------------------------------------------------------


def _fake_image_result(path: Path, out_dir: Path, mode: str = "fast"):
    from upscaler.pipeline_image import ImageResult

    output = out_dir / f"{path.stem}-4K-{mode}.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"jpeg")
    return ImageResult(
        output_path=output,
        width=3840,
        height=2880,
        mode_used=mode,
        notes=["A note."],
        seconds=1.5,
        preview=b"preview",
        compare_before=b"before",
        compare_after=b"after",
        dpi=300,
        print={"inches": (12.8, 9.6), "cm": (32.5, 24.4), "dpi": 300},
        file_bytes=4,
    )


def _image_job(store: Path, owner: str, names: list[str], options: dict, job_id: str = "abc123def456") -> Job:
    inputs = []
    for n, name in enumerate(names):
        path = store / job_id / "in" / str(n) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        inputs.append(str(path))
    return Job(id=job_id, kind="image", owner=owner, inputs=inputs, options=options)


def test_image_runner_saves_results_and_previews(store, monkeypatch, tmp_path):
    from upscaler import pipeline_image

    calls = []

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        calls.append((Path(path).name, opts, out_dir))
        progress(1.0, "Done")
        return _fake_image_result(Path(path), out_dir)

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "host", ["table.png"], {"mode": "fast", "size": "4k"})
    results = default_runners(store)["image"](job, lambda f, m: None, threading.Event())

    host_out = tmp_path / "home" / "Pictures" / "Upscaled"
    assert calls[0][0] == "table.png"
    assert calls[0][1].mode == "fast" and calls[0][1].size == "4k"
    assert calls[0][2] == host_out
    [result] = results
    assert result["filename"] == "table-4K-fast.jpg"
    assert Path(result["path"]).parent == host_out
    assert (result["width"], result["height"], result["mode_used"]) == (3840, 2880, "fast")
    assert result["notes"] == ["A note."]
    assert Path(result["preview"]).read_bytes() == b"preview"
    assert Path(result["before"]).read_bytes() == b"before"
    assert Path(result["after"]).read_bytes() == b"after"
    assert Path(result["preview"]).is_relative_to(store / job.id)
    assert result["dpi"] == 300 and result["file_bytes"] == 4
    assert result["print"] == {"inches": [12.8, 9.6], "cm": [32.5, 24.4], "dpi": 300}
    json.dumps(result)  # saved in job.json as it is


def test_image_runner_passes_the_new_options(store, monkeypatch):
    from upscaler import pipeline_image

    seen = []
    monkeypatch.setattr(
        pipeline_image,
        "upscale_image",
        lambda path, opts, out_dir, *, progress, cancel, viewer_dir=None: seen.append(opts) or _fake_image_result(Path(path), out_dir),
    )
    options = {"mode": "photoreal", "photoreal_level": "rich", "creativity": 0.4, "resemblance": 0.7, "dpi": 600}
    default_runners(store)["image"](_image_job(store, "host", ["a.png"], options), lambda f, m: None, threading.Event())
    [opts] = seen
    assert (opts.photoreal_level, opts.creativity, opts.resemblance, opts.dpi) == ("rich", 0.4, 0.7, 600)


def test_image_runner_guest_outputs_and_key(store, monkeypatch, tmp_path):
    from upscaler import pipeline_image

    seen = {}

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        seen["key"], seen["out"] = opts.guest_key, out_dir
        return _fake_image_result(Path(path), out_dir, "cloud-fal")

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "guest:x", ["a.png"], {"mode": "cloud:fal:seedvr2"})
    job.secret = "guest-key"
    default_runners(store)["image"](job, lambda f, m: None, threading.Event())
    assert seen["key"] == "guest-key"
    assert seen["out"] == tmp_path / "data" / "guest" / job.id


def test_image_runner_refuses_host_cloud_for_guests_without_permission(store, monkeypatch):
    from upscaler import pipeline_image

    monkeypatch.setattr(pipeline_image, "upscale_image", lambda *a, **k: pytest.fail("must not run"))
    job = _image_job(store, "guest:x", ["a.png"], {"mode": "cloud:fal:seedvr2"})  # no key of their own
    with pytest.raises(jobs.JobError, match="cloud"):
        default_runners(store)["image"](job, lambda f, m: None, threading.Event())


def test_image_runner_batch_keeps_going_after_a_bad_picture(store, monkeypatch):
    from upscaler import pipeline_image
    from upscaler.imageio_utils import ImageFileError

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        if Path(path).name == "broken.png":
            raise ImageFileError("broken.png couldn't be opened as a picture.")
        return _fake_image_result(Path(path), out_dir)

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "host", ["good.png", "broken.png", "fine.png"], {"mode": "fast"})
    progress = []
    results = default_runners(store)["image"](job, lambda f, m: progress.append((f, m)), threading.Event())
    assert [r["filename"] for r in results] == ["good-4K-fast.jpg", "fine-4K-fast.jpg"]
    assert any("broken.png couldn't be opened" in note for note in results[0]["notes"])
    assert progress[-1][0] == pytest.approx(1.0)


def test_image_runner_fails_when_every_picture_fails(store, monkeypatch):
    from upscaler import pipeline_image
    from upscaler.imageio_utils import ImageFileError

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        raise ImageFileError(f"{Path(path).name} couldn't be opened as a picture.")

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "host", ["a.png", "b.png"], {"mode": "fast"})
    with pytest.raises(jobs.JobError, match="a.png couldn't be opened"):
        default_runners(store)["image"](job, lambda f, m: None, threading.Event())


def test_image_runner_lets_cancel_through(store, monkeypatch):
    from upscaler import pipeline_image

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        raise Cancelled()

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    job = _image_job(store, "host", ["a.png", "b.png"], {"mode": "fast"})
    with pytest.raises(Cancelled):
        default_runners(store)["image"](job, lambda f, m: None, threading.Event())


def _batch(store: Path, monkeypatch, names: list[str], *, stop_at: str | None = None):
    """A queue with the real image runner and a stand-in engine. Returns (queue, job, calls, at_stop):
    the engine waits at the picture named `stop_at` until `at_stop` is set, then lets the cancel through."""
    from upscaler import pipeline_image

    calls, reached, at_stop = [], threading.Event(), threading.Event()

    def fake_upscale(path, opts, out_dir, *, progress, cancel, viewer_dir=None):
        calls.append(Path(path).name)
        if Path(path).name == stop_at:
            reached.set()
            at_stop.wait(5)
            if cancel.is_set():
                raise Cancelled()
        return _fake_image_result(Path(path), out_dir)

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    queue = JobQueue(store, default_runners(store))
    job_id = queue.new_job_id()
    inputs = []
    for n, name in enumerate(names):
        path = queue.job_dir(job_id) / "in" / str(n) / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        inputs.append(str(path))
    job = queue.submit("image", "guest:x", inputs, {"mode": "fast"}, job_id=job_id)
    return queue, job, calls, reached, at_stop


def test_a_batch_saves_each_picture_as_soon_as_it_is_made(store, monkeypatch):
    queue, job, calls, reached, at_stop = _batch(store, monkeypatch, ["a.png", "b.png", "c.png"], stop_at="c.png")
    queue.start()
    try:
        reached.wait(5)
        saved = json.loads((store / job.id / "job.json").read_text())
        assert saved["status"] == "running"
        assert [r["filename"] for r in saved["results"]] == ["a-4K-fast.jpg", "b-4K-fast.jpg"]
        at_stop.set()
        wait_for(lambda: finished(queue, job.id))
    finally:
        at_stop.set()
        queue.stop()
    done = queue.get(job.id)
    assert done.status == "done"
    assert [r["filename"] for r in done.results] == ["a-4K-fast.jpg", "b-4K-fast.jpg", "c-4K-fast.jpg"]
    assert [Path(r["preview"]).name for r in done.results] == ["0-preview.jpg", "1-preview.jpg", "2-preview.jpg"]


def test_a_cancelled_batch_keeps_the_pictures_already_made(store, monkeypatch):
    names = ["a.png", "b.png", "c.png", "d.png"]
    queue, job, calls, reached, at_stop = _batch(store, monkeypatch, names, stop_at="d.png")
    queue.start()
    try:
        reached.wait(5)
        assert queue.cancel(job.id, "guest:x") is True
        at_stop.set()
        wait_for(lambda: finished(queue, job.id))
    finally:
        at_stop.set()
        queue.stop()
    cancelled = queue.get(job.id)
    assert cancelled.status == "cancelled"
    assert [r["filename"] for r in cancelled.results] == ["a-4K-fast.jpg", "b-4K-fast.jpg", "c-4K-fast.jpg"]
    assert cancelled.message == "Cancelled after 3 of 4 pictures"
    assert JobQueue(store, default_runners(store)).get(job.id).results == cancelled.results  # saved


def test_a_batch_carries_on_after_a_restart_without_redoing_pictures(store, monkeypatch):
    queue, job, calls, reached, at_stop = _batch(store, monkeypatch, ["a.png", "b.png", "c.png"], stop_at="c.png")
    queue.start()
    reached.wait(5)
    queue.stop()  # the app is closed during the third picture
    at_stop.set()
    wait_for(lambda: queue.get(job.id).status == "queued")
    assert calls == ["a.png", "b.png", "c.png"]

    calls.clear()
    restarted = JobQueue(store, default_runners(store))
    assert [r["filename"] for r in restarted.get(job.id).results] == ["a-4K-fast.jpg", "b-4K-fast.jpg"]
    restarted.start()
    try:
        wait_for(lambda: finished(restarted, job.id))
    finally:
        restarted.stop()
    assert calls == ["c.png"]  # only the picture that wasn't finished
    done = restarted.get(job.id)
    assert done.status == "done"
    assert [r["filename"] for r in done.results] == ["a-4K-fast.jpg", "b-4K-fast.jpg", "c-4K-fast.jpg"]
    assert sorted(p.name for p in (store.parent / "data" / "guest" / job.id).iterdir()) == [
        "a-4K-fast.jpg", "b-4K-fast.jpg", "c-4K-fast.jpg",
    ]  # fmt: skip  # no "-2" copies


def test_video_runner_passes_job_dir_for_resume(store, monkeypatch, tmp_path):
    from upscaler.video import VideoResult, pipeline

    seen = {}

    def fake_upscale_video(src, out_dir, job_dir, opts, *, progress, cancel):
        seen.update(src=src, out_dir=out_dir, job_dir=job_dir, opts=opts)
        output = out_dir / "clip-4K-fast.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        return VideoResult(output, 3840, 2160, 12.0, ["A note."])

    monkeypatch.setattr(pipeline, "upscale_video", fake_upscale_video)
    src = store / "vid000000001" / "in" / "0" / "clip.mov"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    job = Job(
        id="vid000000001", kind="video", owner="host", inputs=[str(src)], options={"mode": "fast", "preview_seconds": 3}
    )
    [result] = default_runners(store)["video"](job, lambda f, m: None, threading.Event())
    assert seen["job_dir"] == store / job.id / "work"
    assert seen["opts"].preview_seconds == 3
    assert seen["out_dir"] == tmp_path / "home" / "Pictures" / "Upscaled"
    assert result["filename"] == "clip-4K-fast.mp4"
    assert (result["width"], result["height"], result["mode_used"]) == (3840, 2160, "fast")


def test_preview_runner_passes_the_options_and_keeps_the_crops(store, monkeypatch):
    from upscaler import preview

    seen = {}

    def fake_run_preview(path, routes, region, *, progress, cancel, out_dir, size, creativity):
        seen.update(path=path, routes=routes, region=region, out_dir=out_dir, size=size, creativity=creativity)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "0-original.jpg").write_bytes(b"original")
        (out_dir / "1-fast.jpg").write_bytes(b"fast")
        return [
            {"route": "original", "label": "Original (enlarged)", "crop_path": str(out_dir / "0-original.jpg"),
             "seconds": 0.0, "est_full_seconds": None},
            {"route": "fast", "label": "Fast", "crop_path": str(out_dir / "1-fast.jpg"),
             "seconds": 0.4, "est_full_seconds": 3.2},
            {"route": "photoreal:rich", "label": "Photo-real · Rich", "crop_path": None,
             "seconds": 0.0, "est_full_seconds": None, "error": "Photo-real hasn't been downloaded yet."},
        ]  # fmt: skip

    monkeypatch.setattr(preview, "run_preview", fake_run_preview)
    src = store / "pre000000001" / "in" / "0" / "my room.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    options = {"routes": ["fast", "photoreal:rich"], "region": [10, 20, 256, 256], "size": "16k", "creativity": 0.2}
    job = Job(id="pre000000001", kind="preview", owner="guest:abc", inputs=[str(src)], options=options)

    original, fast, rich = default_runners(store)["preview"](job, lambda f, m: None, threading.Event())

    assert seen == {
        "path": src,
        "routes": ["fast", "photoreal:rich"],
        "region": (10, 20, 256, 256),
        "out_dir": store / job.id / "results",
        "size": "16k",
        "creativity": 0.2,
    }
    assert original == {
        "route": "original",
        "label": "Original (enlarged)",
        "path": str(store / job.id / "results" / "0-original.jpg"),
        "filename": "my room-preview-original.jpg",
        "seconds": 0.0,
        "est_full_seconds": None,
    }
    assert fast["filename"] == "my room-preview-fast.jpg" and fast["est_full_seconds"] == 3.2
    assert rich["path"] is None and rich["error"] == "Photo-real hasn't been downloaded yet."
    assert "filename" not in rich
    json.dumps([original, fast, rich])  # saved in job.json as it is


def test_preview_runner_default_routes_and_region(store, monkeypatch):
    from upscaler import preview

    seen = {}
    monkeypatch.setattr(
        preview, "run_preview", lambda path, routes, region, **kwargs: seen.update(routes=routes, region=region) or []
    )
    job = Job(id="pre000000002", kind="preview", owner="host", inputs=[str(store / "x.png")], options={})
    assert default_runners(store)["preview"](job, lambda f, m: None, threading.Event()) == []
    assert seen == {"routes": None, "region": None}


class _ReleasableEngine:
    name = "mlx"

    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


@pytest.fixture
def photoreal_engine(monkeypatch):
    """A stand-in Photo-real engine on a 16 GB Mac. Returns (engine, set_hardware)."""
    from upscaler import hardware, pipeline_image, preview, seedvr2
    from upscaler.hardware import Hardware
    from upscaler.video import VideoResult, pipeline

    engine = _ReleasableEngine()
    current = {"hw": Hardware("mps", "Apple M4", 16, "mac", True)}
    monkeypatch.setattr(hardware, "detect_hardware", lambda: current["hw"])
    monkeypatch.setattr(seedvr2, "get_seedvr2_engine", lambda hw: engine)
    monkeypatch.setattr(
        pipeline_image,
        "upscale_image",
        lambda path, opts, out_dir, *, progress, cancel, viewer_dir=None: _fake_image_result(Path(path), out_dir, opts.mode),
    )
    monkeypatch.setattr(preview, "run_preview", lambda path, routes, region, **kwargs: [])

    def fake_upscale_video(src, out_dir, job_dir, opts, *, progress, cancel):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "clip-4K-fast.mp4").write_bytes(b"mp4")
        return VideoResult(out_dir / "clip-4K-fast.mp4", 3840, 2160, 1.0, [])

    monkeypatch.setattr(pipeline, "upscale_video", fake_upscale_video)
    return engine, lambda hw: current.update(hw=hw)


def _run_kind(store: Path, kind: str, options: dict) -> None:
    job = _image_job(store, "host", ["a.png"], options)
    job.kind = kind
    default_runners(store)[kind](job, lambda f, m: None, threading.Event())


def test_photoreal_lets_go_of_its_memory_when_other_work_starts_on_a_small_mac(store, photoreal_engine):
    # On a Mac under 24 GB, Photo-real's model (about 4.8 GB) goes as soon as other work starts,
    # rather than a minute later, so Faithful, a video (and Photoshop or Figma) get that memory.
    engine, _ = photoreal_engine
    _run_kind(store, "image", {"mode": "photoreal"})
    assert engine.released == 0  # it's about to be used again
    _run_kind(store, "image", {"mode": "faithful"})
    assert engine.released == 1
    _run_kind(store, "video", {"mode": "fast"})
    assert engine.released == 2
    _run_kind(store, "preview", {"routes": ["fast", "faithful"]})
    assert engine.released == 3
    _run_kind(store, "preview", {})  # every style, Photo-real's too
    _run_kind(store, "preview", {"routes": ["photoreal:gentle"]})
    assert engine.released == 3


def test_photoreal_keeps_its_memory_on_bigger_macs_and_pcs(store, photoreal_engine):
    from upscaler.hardware import Hardware

    engine, set_hardware = photoreal_engine
    set_hardware(Hardware("mps", "Apple M4 Max", 36, "mac", True))
    _run_kind(store, "image", {"mode": "fast"})
    set_hardware(Hardware("cuda", "RTX 4090", 24, "windows", False))  # its runner frees everything itself
    _run_kind(store, "image", {"mode": "fast"})
    assert engine.released == 0


def test_download_runner_fetches_photoreal(store, monkeypatch):
    class FakeEngine:
        name = "fake"
        downloaded = False

        def weights_present(self):
            return self.downloaded

        def download_weights(self, progress, cancel):
            progress(0.5, "Downloading Photo-real")
            self.downloaded = True

    engine = FakeEngine()
    monkeypatch.setattr(jobs, "_photoreal_engine", lambda: (engine, ""))
    job = Job(id="dl0000000001", kind="download", owner="host")
    assert default_runners(store)["download"](job, lambda f, m: None, threading.Event()) == []
    assert engine.downloaded

    monkeypatch.setattr(jobs, "_photoreal_engine", lambda: (None, "Photo-real needs 16 GB of memory"))
    with pytest.raises(jobs.JobError, match="16 GB"):
        default_runners(store)["download"](job, lambda f, m: None, threading.Event())


# --- Keeping the computer awake ---------------------------------------------------


def test_keep_awake_mac_runs_caffeinate(monkeypatch):
    started = []

    class FakeProcess:
        def __init__(self, args, **kwargs):
            started.append(args)
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(keepawake.sys, "platform", "darwin")
    monkeypatch.setattr(keepawake.subprocess, "Popen", FakeProcess)
    with KeepAwake() as awake:
        assert started == [["caffeinate", "-i", "-w", str(os.getpid())]]
        process = awake._process
    assert process.terminated


def test_keep_awake_windows_sets_execution_state(monkeypatch):
    calls = []

    class Kernel32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 1

    class WinDLL:
        kernel32 = Kernel32()

    monkeypatch.setattr(keepawake.sys, "platform", "win32")
    monkeypatch.setattr(keepawake, "_windll", lambda: WinDLL())
    with KeepAwake():
        assert calls == [0x80000000 | 0x00000001]
    assert calls[-1] == 0x80000000


def test_keep_awake_never_breaks_a_job(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("caffeinate not found")

    monkeypatch.setattr(keepawake.sys, "platform", "darwin")
    monkeypatch.setattr(keepawake.subprocess, "Popen", broken)
    with KeepAwake():
        pass  # no exception
