"""The job queue: one background worker runs jobs one at a time, in the order they arrived.
"Preview styles" jobs are the exception: they go ahead of waiting jobs (never a running one),
because someone is looking at the screen waiting for them.

Every job is saved to `<store>/<job id>/job.json` whenever its state changes, so jobs survive a
closed browser tab, a crash or a restart: unfinished jobs run again, picture batches skip the
pictures already made, and videos pick up from their last finished segment. A job that was running
each time the app crashed is set aside after MAX_CRASHES, so it can't crash the app on every start.
Cloud jobs are the exception: one that was interrupted (a crash, a restart, the app closing) is ended
instead of run again, because the service may already have charged for it. A video that fails for
lack of disk space, or while its finished segments are being joined, keeps its work: Retry carries on
from it (video_work_worth_keeping). Uploads live in
`<store>/<job id>/in/`. A finished picture or video job keeps them (the compare viewer's "Before"
side and the video comparison are made from them) until the job itself is forgotten (_expired);
other jobs, and jobs that made nothing, let them go when they end.

The runners for each kind of job ("image", "video", "preview", "download") are at the bottom. They import
the heavy engine modules only when a job actually runs.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import queue as queue_module
import re
import shutil
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from upscaler import config
from upscaler.config import Cancelled, ProgressFn, UserError
from upscaler.keepawake import KeepAwake

log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)
HOST = "host"
GUEST_PREFIX = "guest:"
PRIORITY_KINDS = ("preview",)  # quick jobs someone is waiting for: they go first
CLOUD_PREFIX = "cloud:"  # pipeline_image.CLOUD_PREFIX, without loading the engines
PHOTOREAL = "photoreal"  # pipeline_image.PHOTOREAL
# On Macs with less memory than this, Photo-real's model (about 4.8 GB) is let go as soon as other
# work starts (seedvr2.mlx_engine frees it after a minute's rest anyway).
SMALL_MAC_GB = 24

GUEST_RESULTS_SECONDS = 24 * 3600  # guests' results (and their job records) are kept this long
PREVIEW_RECORD_SECONDS = 24 * 3600  # "Preview styles" crops are only for choosing, so they go after a day
HOST_JOB_RECORD_SECONDS = 7 * 24 * 3600  # the host's results stay in Pictures; the list entry goes
MAX_CRASHES = 3  # a job running when the app crashed this many times is set aside, not run again
ORIGINALS_KEPT_FOR = ("image", "video")  # job kinds whose results are compared with the originals

GENERIC_ERROR = "Something went wrong while working on this. Please try again."
OUT_OF_MEMORY = (
    "The computer ran out of memory. Close other apps and try again, or pick a smaller size or the Fast mode."
)
DISK_FULL = "The disk is full. Free up some space and try again."
DISK_FULL_KEPT = "The disk is full. Free up some space."  # followed by RETRY_HINT
CRASHED_TOO_OFTEN = (
    "Re:Solution stopped unexpectedly several times while working on this, so it was set aside. "
    "Try again, perhaps with a smaller size or the Fast mode."
)
RESUME_MESSAGE = "Waiting to continue after a restart"
RETRY_MESSAGE = "Waiting to carry on"
CLOUD_INTERRUPTED = (
    "This cloud job was interrupted, so Re:Solution didn't send it again (you won't be charged twice). "
    "Start it again if you still want it."
)
OWN_KEY_FORGOTTEN = (
    "Your own cloud key is kept only while Re:Solution is running, so this job wasn't sent after the restart. "
    "Add your key again, then start the job again if you still want it."
)
KEY_FORGOTTEN_BY_PERSON = (
    "You asked Re:Solution to forget your own cloud key, so this job was stopped and not sent. "
    "Add your key again, then start the job again if you still want it."
)
RETRY_HINT = "Then press Retry to carry on from where it stopped: the parts already done are kept."
GUEST_CLOUD_NOT_ALLOWED = (
    "This cloud mode needs a key. Use “Use my own key for this visit”, "
    "or ask the person who runs Re:Solution to let guests use their cloud account."
)

_RECORD = "job.json"
_JOB_ID = re.compile(r"[0-9a-f]{12}")
_BROADCAST_EVERY_SECONDS = 0.5
_STOP_WAIT_SECONDS = 10.0
_EVENTS_PER_SUBSCRIBER = 1000

Runner = Callable[["Job", ProgressFn, threading.Event], list[dict]]


class JobError(Exception):
    """A job can't go on. The message is written for the person using the app."""


@dataclass
class Job:
    id: str
    kind: str  # "image" | "video" | "download" | "preview"
    owner: str  # "host" | "guest:<id>", where <id> is derived from the guest's cookie but never is it
    status: str = QUEUED
    progress: float = 0.0
    message: str = ""
    eta_seconds: float | None = None
    created: float = field(default_factory=time.time)
    inputs: list[str] = field(default_factory=list)  # paths of the uploaded files
    options: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    error: str | None = None
    finished: float | None = None
    crashes: int = 0  # times the app stopped unexpectedly (crashed, was killed) while this job ran
    own_key: bool = False  # sent with the person's own cloud key ("for this visit"), held in `secret`
    retryable: bool = False  # failed, but its work was kept: retry() carries on from where it stopped
    secret: str | None = field(default=None, repr=False, compare=False)  # a guest's cloud key; memory only

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    @property
    def cloud(self) -> bool:
        """Sent to a paid cloud service, which may charge again if the job is sent again."""
        mode = self.options.get("mode") if isinstance(self.options, dict) else None
        return isinstance(mode, str) and mode.startswith(CLOUD_PREFIX)

    def to_dict(self) -> dict:
        data = asdict(self)
        del data["secret"]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        known = {f.name for f in fields(cls)} - {"secret"}
        return cls(**{name: value for name, value in data.items() if name in known})

    def copy(self) -> Job:
        """A snapshot that callers can read while the worker keeps changing the original."""
        snapshot = Job.from_dict(json.loads(json.dumps(self.to_dict())))
        snapshot.secret = self.secret
        return snapshot


class EtaTracker:
    """Time left, from how fast progress moved over the last couple of minutes."""

    def __init__(self, window_seconds: float = 120.0, min_seconds: float = 5.0, min_progress: float = 0.01):
        self._samples: deque[tuple[float, float]] = deque()
        self._window = window_seconds
        self._min_seconds = min_seconds
        self._min_progress = min_progress

    def update(self, now: float, fraction: float) -> float | None:
        if fraction >= 1.0:
            return 0.0
        self._samples.append((now, fraction))
        while len(self._samples) > 2 and now - self._samples[1][0] >= self._window:
            self._samples.popleft()
        first_time, first_fraction = self._samples[0]
        elapsed, done = now - first_time, fraction - first_fraction
        if elapsed < self._min_seconds or done < self._min_progress:
            return None
        return (1.0 - fraction) * elapsed / done


class JobQueue:
    def __init__(self, store: Path, runners: dict[str, Runner]):
        self._store = Path(store)
        self._runners = dict(runners)
        self._jobs: dict[str, Job] = {}  # in the order they arrived
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._cancels: dict[str, threading.Event] = {}  # running job -> its cancel signal
        self._cancelled_by_person: set[str] = set()  # as opposed to the app shutting down
        self._subscribers: list[queue_module.Queue] = []
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._store.mkdir(parents=True, exist_ok=True)
        self._load()

    # --- Lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._work, name="upscaler-jobs", daemon=True)
            self._thread.start()

    def stop(self, wait: float | None = None) -> None:
        """Stop the worker. A running job is interrupted and stays queued, so it runs again next time
        (a cloud job is ended instead, see _save). It is saved that way straight away: the app may be
        gone before the job winds down. Waits up to `wait` seconds (default _STOP_WAIT_SECONDS) for it."""
        with self._lock:
            self._stopping = True
            for job_id, cancel in self._cancels.items():
                cancel.set()
                self._save(self._jobs[job_id])
            self._wake.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(_STOP_WAIT_SECONDS if wait is None else wait)

    @property
    def stopping(self) -> bool:
        return self._stopping

    # --- Jobs ------------------------------------------------------------------------

    @staticmethod
    def new_job_id() -> str:
        return uuid.uuid4().hex[:12]

    def job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError(f"Not a job id: {job_id!r}")
        return self._store / job_id

    def submit(
        self,
        kind: str,
        owner: str,
        inputs: list[str],
        options: dict,
        *,
        job_id: str | None = None,
        secret: str | None = None,
    ) -> Job:
        """Add a job to the end of the queue. `job_id` is one reserved with `new_job_id()`, for
        uploads that were saved into `job_dir(job_id)` before the job existed."""
        if kind not in self._runners:
            raise ValueError(f"Unknown job kind {kind!r}")
        job = Job(id=job_id or self.new_job_id(), kind=kind, owner=owner, inputs=list(inputs), options=dict(options))
        job.secret = secret
        job.own_key = secret is not None
        self.job_dir(job.id)  # validates the id
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"Job {job.id} already exists")
            self._jobs[job.id] = job
            self._save(job)
            snapshot = job.copy()
            self._wake.notify_all()
        self._broadcast_job(snapshot)
        return snapshot

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.copy() if job else None

    def list(self) -> list[Job]:
        with self._lock:
            return [job.copy() for job in self._jobs.values()]

    def cancel(self, job_id: str, requester: str) -> bool:
        """The host may cancel any job; a guest only their own. False if not allowed or already over."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (requester != HOST and requester != job.owner) or not job.active:
                return False
            self._stop(job, "Cancelled")
            snapshot = job.copy()
        self._broadcast_job(snapshot)
        return True

    def _stop(self, job: Job, message: str, error: str | None = None) -> None:
        """Stop an active job for a person (with the lock held): a waiting one ends now, with `message`
        and `error`; a running one is told to stop, and ends as cancelled (pictures already made stay)."""
        if job.status == QUEUED:
            self._finish(job, CANCELLED, message=message, error=error)
        else:
            self._cancelled_by_person.add(job.id)
            self._cancels[job.id].set()
            job.message = "Stopping…"
            job.eta_seconds = None
            self._save(job)  # saved as cancelled now, in case the app is gone before the job stops

    def retry(self, job_id: str, requester: str) -> bool:
        """Put a failed job whose work was kept (Job.retryable: a video that ran out of disk space, or
        whose finished segments couldn't be joined) back in the queue, to carry on from where it
        stopped. The host may retry any job; a guest only their own. False if not allowed, or the job
        can't be retried."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (requester != HOST and requester != job.owner):
                return False
            if job.status != FAILED or not job.retryable:
                return False
            job.status, job.progress, job.message, job.eta_seconds = QUEUED, 0.0, RETRY_MESSAGE, None
            job.error, job.finished, job.retryable, job.crashes = None, None, False, 0
            self._save(job)
            snapshot = job.copy()
            self._wake.notify_all()
        self._broadcast_job(snapshot)
        return True

    def stop_own_key_jobs(self, owner: str, provider_id: str) -> int:
        """`owner` asked to forget their own key for `provider_id`, so none of their jobs may send
        another picture with it: a waiting one ends at once, a running one stops as if they had pressed
        Cancel (pictures already made stay), and the key goes from every job's memory. Returns how many
        jobs were stopped."""
        mode_prefix = f"{CLOUD_PREFIX}{provider_id}:"
        snapshots = []
        with self._lock:
            for job in self._jobs.values():
                mode = job.options.get("mode") if isinstance(job.options, dict) else None
                if job.owner != owner or not job.own_key or not job.active:
                    continue
                if not (isinstance(mode, str) and mode.startswith(mode_prefix)):
                    continue
                job.secret = None
                self._stop(job, KEY_FORGOTTEN_BY_PERSON, error=KEY_FORGOTTEN_BY_PERSON)
                snapshots.append(job.copy())
        for snapshot in snapshots:
            self._broadcast_job(snapshot)
        return len(snapshots)

    # --- Events ----------------------------------------------------------------------

    def subscribe(self) -> queue_module.Queue[dict]:
        """Events: {"event": "job", "job": Job} on every change, and {"event": "status"} when
        something everyone's status page shows has changed."""
        events: queue_module.Queue[dict] = queue_module.Queue(maxsize=_EVENTS_PER_SUBSCRIBER)
        with self._lock:
            self._subscribers.append(events)
        return events

    def unsubscribe(self, events: queue_module.Queue) -> None:
        with self._lock:
            if events in self._subscribers:
                self._subscribers.remove(events)

    def publish_status(self) -> None:
        self._publish({"event": "status"})

    def _broadcast_job(self, job: Job) -> None:
        self._publish({"event": "job", "job": job})

    def _publish(self, event: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for events in subscribers:
            try:
                events.put_nowait(event)
            except queue_module.Full:
                pass  # a stalled viewer; it reloads the job list when it reconnects

    # --- The worker ------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            with self._wake:
                job = self._next_queued()
                while job is None and not self._stopping:
                    self._wake.wait()
                    job = self._next_queued()
                if self._stopping:
                    return
                cancel = threading.Event()
                self._cancels[job.id] = cancel
                job.status, job.progress, job.message, job.eta_seconds = RUNNING, 0.0, "Starting", None
                self._save(job)
                snapshot = job.copy()
            self._broadcast_job(snapshot)
            self._run(snapshot, cancel)

    def _next_queued(self) -> Job | None:
        queued = [job for job in self._jobs.values() if job.status == QUEUED]
        return next((job for job in queued if job.kind in PRIORITY_KINDS), queued[0] if queued else None)

    def _run(self, job: Job, cancel: threading.Event) -> None:
        runner = self._runners[job.kind]
        try:
            with KeepAwake():
                results = runner(job, _Reporter(self, job.id, cancel), cancel)
        except Cancelled:
            outcome: tuple[str, dict[str, Any]] = (CANCELLED, {})
        except Exception as exc:
            log.exception("Job %s (%s) failed", job.id, job.kind)
            if job.kind == "video" and video_work_worth_keeping(exc):  # the work so far is kept, for Retry
                error = DISK_FULL_KEPT if isinstance(exc, OSError) else friendly_error(exc)
                failed = {"error": f"{error} {RETRY_HINT}", "message": "Didn't finish", "keep_work": True}
            else:
                failed = {"error": friendly_error(exc), "message": "Didn't finish"}
            outcome = (FAILED, failed)
        else:
            outcome = (DONE, {"results": list(results or []), "message": "Done"})

        with self._lock:
            live = self._jobs[job.id]
            del self._cancels[job.id]
            by_person = job.id in self._cancelled_by_person
            self._cancelled_by_person.discard(job.id)
            if outcome[0] == CANCELLED and self._stopping and not by_person and live.cloud:
                self._finish(live, CANCELLED, message=CLOUD_INTERRUPTED, error=CLOUD_INTERRUPTED)
            elif outcome[0] == CANCELLED and self._stopping and not by_person:
                self._requeue(live, RESUME_MESSAGE)
            elif outcome[0] == CANCELLED:
                self._finish(live, CANCELLED, message=_cancelled_message(live))  # pictures already made stay
            else:
                status, changes = outcome
                self._finish(live, status, **changes)
            snapshot = live.copy()
        self._broadcast_job(snapshot)

    def _keep_result(self, job_id: str, result: dict) -> None:
        """A finished part of a running batch, saved with the job straight away."""
        with self._lock:
            job = self._jobs[job_id]
            if job.status != RUNNING:
                return
            job.results.append(result)
            self._save(job)
            snapshot = job.copy()
        self._broadcast_job(snapshot)

    def _finish(
        self,
        job: Job,
        status: str,
        *,
        message: str,
        results: list[dict] | None = None,
        error: str | None = None,
        keep_work: bool = False,
    ) -> None:
        """End the job. `keep_work` keeps its uploads and half-made work for retry()."""
        job.status, job.message, job.error, job.eta_seconds = status, message, error, None
        job.finished = time.time()
        job.retryable = keep_work
        job.secret = None  # a guest's key is kept no longer than their job needs it
        if results is not None:
            job.results = results
        if status == DONE:
            job.progress = 1.0
        self._save(job)
        if not keep_work:
            self._remove_work_files(job)

    def _requeue(self, job: Job, message: str) -> None:
        job.status, job.progress, job.message, job.eta_seconds = QUEUED, 0.0, message, None
        self._save(job)

    def prune_expired(self) -> int:
        """Forget finished jobs that are past their time, with their folders (and so guests' preview
        pictures): the same rule as on start, applied every hour while the app runs. Returns how many."""
        with self._lock:
            old = [job for job in self._jobs.values() if self._expired(job)]
            for job in old:
                del self._jobs[job.id]
        for job in old:
            shutil.rmtree(self._store / job.id, ignore_errors=True)
        if old:
            self.publish_status()
        return len(old)

    # --- Storage ---------------------------------------------------------------------

    def _save(self, job: Job) -> None:
        """Save the job's record. A running job is saved as what should happen if the app were gone
        the next moment: cancelled once someone has cancelled it, waiting again while the app stops
        (a cloud job ended instead, as sending it again could charge twice). Called with the lock held."""
        if job.status == RUNNING and job.id in self._cancelled_by_person:
            message = _cancelled_message(job)
            job = _record_as(job, status=CANCELLED, message=message, eta_seconds=None, finished=time.time())
        elif job.status == RUNNING and self._stopping and job.cloud:
            job = _record_as(
                job,
                status=CANCELLED,
                message=CLOUD_INTERRUPTED,
                error=CLOUD_INTERRUPTED,
                eta_seconds=None,
                finished=time.time(),
            )
        elif job.status == RUNNING and self._stopping:
            job = _record_as(job, status=QUEUED, progress=0.0, message=RESUME_MESSAGE, eta_seconds=None)
        folder = self._store / job.id
        record = folder / _RECORD
        temporary = folder / (_RECORD + ".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(job.to_dict(), indent=1), encoding="utf-8")
            os.replace(temporary, record)  # atomic, so a crash never leaves half a record
        except OSError:
            log.exception("Couldn't save job %s", job.id)

    def _remove_work_files(self, job: Job) -> None:
        """Half-made video segments aren't needed once a job is over, and nor are the uploads,
        unless the job made something to compare them with (keeps_originals)."""
        folder = self._store / job.id
        shutil.rmtree(folder / "work", ignore_errors=True)
        if not keeps_originals(job):
            shutil.rmtree(folder / "in", ignore_errors=True)

    def _load(self) -> None:
        loaded = []
        for folder in self._store.iterdir():
            if not folder.is_dir() or not _JOB_ID.fullmatch(folder.name):
                continue
            job = self._read_record(folder)
            if job is None:
                shutil.rmtree(folder, ignore_errors=True)  # an upload a crash interrupted, or a damaged record
            elif self._expired(job):
                shutil.rmtree(folder, ignore_errors=True)
            else:
                loaded.append(job)
        for job in sorted(loaded, key=lambda j: (j.created, j.id)):
            interrupted = job.status == RUNNING or (job.status == QUEUED and job.message == RESUME_MESSAGE)
            if job.active and job.kind not in self._runners:
                self._finish(job, FAILED, message="Didn't finish", error=GENERIC_ERROR)
            elif interrupted and job.cloud:  # the service may have charged already: never send it twice
                self._finish(job, CANCELLED, message=CLOUD_INTERRUPTED, error=CLOUD_INTERRUPTED)
            elif job.active and job.cloud and job.own_key:  # the key lived in memory and is gone
                self._finish(job, CANCELLED, message=OWN_KEY_FORGOTTEN, error=OWN_KEY_FORGOTTEN)
            elif job.status == RUNNING:  # the app didn't stop calmly: it crashed or was killed
                job.crashes += 1
                if job.crashes >= MAX_CRASHES:
                    log.error("Job %s was running the last %d times the app crashed; set aside", job.id, job.crashes)
                    self._finish(job, FAILED, message="Didn't finish", error=CRASHED_TOO_OFTEN)
                else:
                    self._requeue(job, RESUME_MESSAGE)
            self._jobs[job.id] = job

    @staticmethod
    def _read_record(folder: Path) -> Job | None:
        try:
            job = Job.from_dict(json.loads((folder / _RECORD).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        return job if job.id == folder.name else None

    @staticmethod
    def _expired(job: Job) -> bool:
        if job.active:
            return False
        if job.kind == "preview":
            keep = PREVIEW_RECORD_SECONDS
        else:
            keep = HOST_JOB_RECORD_SECONDS if job.owner == HOST else GUEST_RESULTS_SECONDS
        return time.time() - (job.finished or job.created) > keep


class _Reporter:
    """What a runner tells the queue as it works. Call it with progress (a ProgressFn). A batch also
    hands over each finished part with `keep`, which saves it with the job at once, so a cancel or a
    restart never loses it."""

    def __init__(self, queue: JobQueue, job_id: str, cancel: threading.Event):
        self._queue = queue
        self._job_id = job_id
        self._cancel = cancel
        self._eta = EtaTracker()
        self._last_sent = 0.0

    def __call__(self, fraction: float, message: str) -> None:
        queue = self._queue
        now = time.monotonic()
        fraction = min(max(float(fraction), 0.0), 1.0)
        with queue._lock:
            job = queue._jobs[self._job_id]
            if job.status != RUNNING:
                return  # a late report from a job that has already ended
            job.progress = fraction
            if not self._cancel.is_set():
                job.message = message
                job.eta_seconds = self._eta.update(now, fraction)
            if now - self._last_sent < _BROADCAST_EVERY_SECONDS:
                return
            self._last_sent = now
            snapshot = job.copy()
        queue._broadcast_job(snapshot)

    def keep(self, result: dict) -> None:
        self._queue._keep_result(self._job_id, result)


def keeps_originals(job: Job) -> bool:
    """A finished picture or video job keeps its originals (the uploads) for the compare viewer's
    "Before" side and the side-by-side video comparison, until the job is forgotten with its folder."""
    return job.kind in ORIGINALS_KEPT_FOR and bool(job.results)


def _record_as(job: Job, **changes: Any) -> Job:
    """A copy of the job with these changes, to save without changing the job itself."""
    record = job.copy()
    for name, value in changes.items():
        setattr(record, name, value)
    return record


def video_work_worth_keeping(exc: BaseException) -> bool:
    """Whether a video that failed with `exc` should keep its work folder (finished segments and all),
    to carry on from with Retry, or the same command again on the command line: it ran out of disk
    space (NotEnoughDiskSpace, or a disk-full error from anywhere), or every segment was finished and
    only joining them failed (JoinFailed). Segments that can't be reused are thrown away when it
    carries on (the video pipeline's manifest checks them)."""
    from upscaler.video import NotEnoughDiskSpace
    from upscaler.video.pipeline import JoinFailed

    if isinstance(exc, (NotEnoughDiskSpace, JoinFailed)):
        return True
    return isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC))


def _cancelled_message(job: Job) -> str:
    if job.kind == "image" and job.results and len(job.inputs) > 1:
        return f"Cancelled after {len(job.results)} of {len(job.inputs)} pictures"
    return "Cancelled"


def delete_old_files(folder: Path, max_age_seconds: float, now: float | None = None) -> int:
    """Delete files older than `max_age_seconds` under `folder`, then any folders left empty.
    Returns how many files were deleted."""
    if not folder.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_seconds
    deleted = 0
    for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_dir() and not path.is_symlink():
                if not any(path.iterdir()):
                    path.rmdir()
            elif path.lstat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            log.warning("Couldn't tidy up %s", path, exc_info=True)
    return deleted


def friendly_error(exc: BaseException) -> str:
    """A message for the person using the app. Technical details go to the log, not the screen."""
    from upscaler.imageio_utils import ImageFileError
    from upscaler.models import ModelDownloadError
    from upscaler.video import VideoError

    if isinstance(exc, (JobError, UserError, ImageFileError, VideoError, ModelDownloadError)):
        return str(exc) or GENERIC_ERROR
    for attribute in ("user_message", "reason"):  # CloudError, SeedVR2Unavailable
        message = getattr(exc, attribute, None)
        if isinstance(message, str) and message:
            return message
    if isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and _mentions_memory(str(exc))):
        return OUT_OF_MEMORY
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return DISK_FULL
    return GENERIC_ERROR


def _mentions_memory(text: str) -> bool:
    text = text.lower()
    return "out of memory" in text or "metal::malloc" in text


# --- Runners --------------------------------------------------------------------------


def default_runners(store: Path) -> dict[str, Runner]:
    """The real work for each kind of job. `store` is the JobQueue's store (for job folders)."""
    store = Path(store)

    def image(job: Job, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
        return run_image_job(job, store / job.id, progress, cancel)

    def video(job: Job, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
        return run_video_job(job, store / job.id, progress, cancel)

    def preview(job: Job, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
        return run_preview_job(job, store / job.id, progress, cancel)

    return {"image": image, "video": video, "preview": preview, "download": run_download_job}


def output_dir_for(job: Job) -> Path:
    """The host's results go to Pictures/Upscaled; each guest job gets its own temporary folder."""
    if job.owner == HOST:
        return config.host_outputs_dir()
    return config.guest_outputs_dir() / job.id


def guests_may_use_host_cloud(settings: config.Settings) -> bool:
    return bool(settings.allow_guests_cloud and settings.access_password_hash)


def release_photoreal_on_small_mac() -> None:
    """Let Photo-real's model go now, on a Mac under SMALL_MAC_GB, for work that doesn't use it: that
    work, and apps such as Photoshop or Figma alongside, need the memory more. It loads again, in a few
    seconds, when Photo-real next runs. NVIDIA's runner is a program of its own that frees everything."""
    from upscaler.hardware import detect_hardware
    from upscaler.seedvr2 import get_seedvr2_engine

    hardware = detect_hardware()
    if not hardware.apple_silicon or (hardware.memory_gb or 0) >= SMALL_MAC_GB:
        return
    engine = get_seedvr2_engine(hardware)
    if engine is None:
        return
    try:
        engine.release()
    except Exception:  # never a reason for the job itself to fail
        log.warning("Couldn't free Photo-real's memory", exc_info=True)


def run_image_job(job: Job, job_dir: Path, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
    from upscaler import pipeline_image

    options = pipeline_image.ImageOptions(**job.options, guest_key=job.secret)
    if job.own_key and not job.secret and options.mode.startswith(pipeline_image.CLOUD_PREFIX):
        raise JobError(OWN_KEY_FORGOTTEN)  # never quietly switched to the host's account
    if (
        job.owner != HOST
        and options.mode.startswith(pipeline_image.CLOUD_PREFIX)
        and not job.secret
        and not guests_may_use_host_cloud(config.load_settings())
    ):
        raise JobError(GUEST_CLOUD_NOT_ALLOWED)

    if options.mode != pipeline_image.PHOTOREAL:
        release_photoreal_on_small_mac()
    out_dir = output_dir_for(job)
    total = len(job.inputs)
    # Pictures made before a restart are kept, not made again (each result names its input).
    results = [r for r in job.results if isinstance(r.get("input"), int) and 0 <= r["input"] < total]
    made = {r["input"] for r in results}
    keep = getattr(progress, "keep", None)  # the queue's reporter saves each picture as soon as it's made
    problems: list[str] = []
    for n, path in enumerate(map(Path, job.inputs)):
        if n in made:
            continue
        report = _part_of(progress, n, total)
        viewer_dir = job_dir / "results" / f"{len(results)}-viewer"  # the compare viewer's tiles
        try:
            result = pipeline_image.upscale_image(
                path, options, out_dir, progress=report, cancel=cancel, viewer_dir=viewer_dir
            )
        except Cancelled:
            raise
        except Exception as exc:  # one bad picture shouldn't stop the rest of the batch
            if total == 1:
                raise
            log.warning("Picture %s in job %s failed", path.name, job.id, exc_info=True)
            problems.append(f"{path.name} wasn't upscaled: {friendly_error(exc)}")
            continue
        entry = {**_image_result(result, job_dir, len(results)), "input": n}
        results.append(entry)
        if keep is not None:
            keep(entry)
    progress(1.0, "Done")
    if not results:
        raise JobError(" ".join(problems))
    if problems:
        results[0]["notes"] = [*results[0]["notes"], *problems]
    return results


def _part_of(progress: ProgressFn, n: int, total: int) -> ProgressFn:
    """Progress for picture n of a batch, as its share of the whole job."""
    prefix = f"Picture {n + 1} of {total}: " if total > 1 else ""

    def report(fraction: float, message: str) -> None:
        progress((n + min(max(fraction, 0.0), 1.0)) / total, prefix + message)

    return report


def _image_result(result: Any, job_dir: Path, n: int) -> dict:
    previews = job_dir / "results"
    previews.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in (("preview", result.preview), ("before", result.compare_before), ("after", result.compare_after)):
        files[name] = previews / f"{n}-{name}.jpg"
        files[name].write_bytes(data)
    viewer = getattr(result, "viewer", None)  # the compare viewer's tile pyramid, if it was written
    if viewer is not None:
        files["viewer"] = Path(viewer)
    return {
        "path": str(result.output_path),
        "filename": result.output_path.name,
        "width": result.width,
        "height": result.height,
        "mode_used": result.mode_used,
        "notes": list(result.notes),
        "seconds": round(result.seconds, 1),
        "dpi": result.dpi,
        "print": {key: list(value) if isinstance(value, tuple) else value for key, value in result.print.items()},
        "file_bytes": result.file_bytes,
        "source_width": result.source_width,
        "source_height": result.source_height,
        **{name: str(path) for name, path in files.items()},
    }


def run_video_job(job: Job, job_dir: Path, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
    from upscaler.video import pipeline

    options = pipeline.VideoOptions(**job.options)
    release_photoreal_on_small_mac()  # videos never use Photo-real
    # A fixed work folder per job lets an interrupted video resume from its last segment.
    result = pipeline.upscale_video(
        Path(job.inputs[0]), output_dir_for(job), job_dir / "work", options, progress=progress, cancel=cancel
    )
    _keep_compact_original(Path(job.inputs[0]), job_dir, result.output_path, progress, cancel)
    return [
        {
            "path": str(result.output_path),
            "filename": result.output_path.name,
            "width": result.width,
            "height": result.height,
            "mode_used": options.mode,
            "notes": list(result.notes),
            "seconds": round(result.seconds, 1),
        }
    ]


def _keep_compact_original(
    source: Path, job_dir: Path, output: Path, progress: ProgressFn, cancel: threading.Event
) -> None:
    """The original is kept for the side-by-side comparison. A very large one (a camcorder or
    ProRes file, many GB) is swapped for the comparison's own small copy of it instead. Never a
    reason for the finished job to fail: if the copy can't be made, the original stays."""
    from upscaler.video import compare

    try:
        if not compare.worth_compacting(source):
            return
        progress(1.0, "Keeping a small copy of the original for the comparison")
        compare.keep_compact_original(source, compare.compare_folder(job_dir, 0), output, cancel=cancel)
    except Exception:
        log.warning("Couldn't make a small copy of %s; keeping the original", source.name, exc_info=True)


_PREVIEW_NAME_CHARACTERS = 100  # of the picture's name, in a crop's download name


def run_preview_job(job: Job, job_dir: Path, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
    """Run "Preview styles": one crop of the picture through each style (preview.run_preview).

    Options: routes (None = every style this computer can run), region ([x, y, width, height] in the
    picture, or None for its most detailed spot), size (for the whole-picture time estimates) and
    creativity (Photo-real). Results: "Original (enlarged)" first, then one per style, each with its
    crop in the job's folder; a style that couldn't run has no file, and an `error` for people.
    """
    from upscaler import preview

    options = job.options
    region = options.get("region")
    source = Path(job.inputs[0])
    routes = options.get("routes")
    if routes is not None and not any(route.split(":")[0] == PHOTOREAL for route in routes):
        release_photoreal_on_small_mac()
    items = preview.run_preview(
        source,
        routes,
        None if region is None else tuple(region),
        progress=progress,
        cancel=cancel,
        out_dir=job_dir / "results",
        size=options.get("size", "8k"),
        creativity=options.get("creativity", 0.0),
    )
    stem = source.stem[:_PREVIEW_NAME_CHARACTERS] or "picture"
    return [_preview_result(item, stem) for item in items]


def _preview_result(item: dict, stem: str) -> dict:
    result = {
        "route": item["route"],
        "label": item["label"],
        "path": item.get("crop_path"),
        "seconds": item["seconds"],
        "est_full_seconds": item.get("est_full_seconds"),
    }
    if result["path"]:
        result["filename"] = f"{stem}-preview-{item['route'].replace(':', '-')}.jpg"
    if item.get("error"):
        result["error"] = item["error"]
    return result


def _photoreal_engine() -> tuple[Any, str]:
    """This computer's Photo-real engine, or None and the reason it can't run here."""
    from upscaler.hardware import detect_hardware, photoreal_support
    from upscaler.seedvr2 import get_seedvr2_engine

    hardware = detect_hardware()
    supported, reason = photoreal_support(hardware)
    engine = get_seedvr2_engine(hardware) if supported else None
    return engine, reason or "Photo-real can't run on this computer"


def run_download_job(job: Job, progress: ProgressFn, cancel: threading.Event) -> list[dict]:
    engine, reason = _photoreal_engine()
    if engine is None:
        raise JobError(f"{reason}, so there's nothing to download.")
    engine.download_weights(progress, cancel)
    return []
