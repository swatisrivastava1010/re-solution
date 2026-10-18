"""The web server: the API the browser app talks to, and the app's own pages from upscaler/web.

Who is asking matters:
- The host is whoever sits at the computer running the app (requests from 127.0.0.1 or ::1).
  They are never asked for the password, and only they may change settings and cloud keys.
- Everyone else is a guest on the home Wi-Fi. Each browser gets a session cookie that ties it to
  its own jobs. With an access password set, guests log in first.

The app itself listens only on 127.0.0.1, so the computer's firewall never asks about it. "Share
on Wi-Fi" (the host's switch, or the Share launchers' --share) adds a second listener for the same
app on every network at port + 1 (or the next free port), without a restart (WifiSharing). Whoever
comes in through that listener is a guest, even from this computer, whatever its headers claim.
Sharing always starts off unless the app was launched with --share; it isn't remembered.

Pictures are checked as soon as they're dropped (POST /api/inspect): the file is staged under
`jobs/staging/<token>/` with a thumbnail and a suggested mode, and a job or a "Preview styles" run
then names it by its token instead of sending it again. Staged pictures are deleted after 24 hours.

The compare viewer zooms into finished pictures through their tile pyramids (viewer.json and
512 px tiles, written by the job; built from the saved file the first time for older results).
The "Before" tiles are made on request from the job's original, simply enlarged. Finished videos
are compared side by side through browser-friendly copies made on request (video.compare).
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import math
import os
import queue as queue_module
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import zipfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, StrictBool
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import ClientDisconnect

from upscaler import __version__, config
from upscaler.cloud import PROVIDERS, CloudError, CloudProvider, keys
from upscaler.cloud.base import clean_key
from upscaler.cloud.guides import GUIDES
from upscaler.config import (
    DEFAULT_PORT,
    GRAIN_CHOICES,
    OUTPUT_FORMATS,
    PRODUCT_NAME,
    SPACE_RESERVE_BYTES,
    SPACE_RESERVE_TEXT,
    Settings,
    load_settings,
)
from upscaler.hardware import Hardware, detect_hardware, photoreal_support
from upscaler.imageio_utils import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
    ImageFileError,
    ImageInfo,
    before_tile,
    inspect_image,
    load_image,
    make_thumbnail,
    on_white,
    read_viewer_manifest,
    resolve_format,
    tile_bounds,
    tile_jpeg,
    write_tile_pyramid,
)
from upscaler.jobs import (
    GUEST_CLOUD_NOT_ALLOWED,
    GUEST_PREFIX,
    GUEST_RESULTS_SECONDS,
    HOST,
    Job,
    JobQueue,
    delete_old_files,
    guests_may_use_host_cloud,
)
from upscaler.models import MODELS
from upscaler.pipeline_image import (
    CLOUD_PREFIX,
    DEFAULT_MODE,
    PHOTOREAL,
    UPSCALE_ROUTES,
    cloud_scale,
    estimate_format,
    output_size_problem,
)
from upscaler.seedvr2 import get_seedvr2_engine
from upscaler.sizing import (
    DEFAULT_DPI,
    DEFAULT_PHOTOREAL_LEVEL,
    DPI_CHOICES,
    PHOTOREAL_LEVELS,
    PRESETS,
    describe_size,
    estimate,
    estimated_file_bytes,
    plan_passes,
    target_size,
)
from upscaler.video import VIDEO_MODES, VideoError, probe_video
from upscaler.video import compare as video_compare
from upscaler.video.pipeline import OUTPUT_BITS_PER_SECOND

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
SAMPLES_DIR = WEB_DIR / "samples"  # public-domain / CC0 pictures to try; served at /samples/<file>
HOST_ADDRESSES = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})

SESSION_COOKIE = "upscaler_session"
AUTH_COOKIE = "upscaler_auth"
VISIT_COOKIE = "upscaler_visit"  # no expiry date: browsers forget it when they close
COOKIE_MAX_AGE = 30 * 24 * 3600
MIN_PASSWORD_LENGTH = 6
LOGIN_FAILURE_DELAY_SECONDS = 1.0  # each wrong password is answered a little later
# Guessing the password: after this many wrong ones in a row, a device has to wait before its next
# try is even checked (and after LOGIN_FREE_TRIES_EVERYONE, every guest does), twice as long each time.
LOGIN_FREE_TRIES = 5
LOGIN_FREE_TRIES_EVERYONE = 20
LOGIN_WAIT_SECONDS = 30
LOGIN_MAX_WAIT_SECONDS = 15 * 60
LOGIN_FORGET_SECONDS = 3600  # wrong passwords this long ago no longer count
SESSION_KEY_IDLE_SECONDS = 4 * 3600  # a guest's own cloud key is forgotten when unused this long...
SESSION_KEY_MAX_SECONDS = 24 * 3600  # ...and after a day in any case

MAX_IMAGE_BYTES = 300 * 1024 * 1024
MAX_IMAGES_PER_BATCH = 100
MAX_FIELD_BYTES = 64 * 1024  # a text field in an upload, and any request that isn't an upload
MAX_PART_HEADER_BYTES = 16 * 1024
BIG_BODY_PATHS = frozenset({"/api/images", "/api/videos", "/api/inspect"})  # uploads, streamed to disk
# A video job's 4K segments and the joined result: up to about 60 Mbit/s each (HEVC at 4K, whatever the
# source), so 2.2x that with a margin, for as many seconds as it makes.
VIDEO_OUTPUT_BITS_PER_SECOND = OUTPUT_BITS_PER_SECOND
VIDEO_WORK_FACTOR = 2.2
PICTURE_SPACE_FACTOR = 2  # the pictures, and room for what they become
# Uploads (like jobs) always leave config.SPACE_RESERVE_BYTES free, for the jobs already running and
# the computer's other apps; a picture job also has to fit its finished pictures (_check_room_for_results).
SPACE_CHECK_BYTES = 64 * 1024 * 1024  # free space is checked again each time this much has arrived
# What one guest's uploads can take up on the main computer: pictures waiting to be used (24 hours) and
# the pictures and videos their jobs keep (until the job is forgotten, 24 hours after it ends).
GUEST_UPLOAD_QUOTA_BYTES = 10 * 1024**3
MAX_PREVIEW_SECONDS = 600
MAX_NAME_BYTES = 200
MAX_ESTIMATE_SIDE = 1_000_000  # pixels; anything bigger isn't a real picture

IMAGE_FIELDS = frozenset({"files", "files[]"})
VIDEO_FIELDS = frozenset({"file"})
INSPECT_FIELDS = frozenset({"file"})
OPTIONS_FIELDS = frozenset({"options"})  # the only text field the page sends with an upload
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

STAGING_SECONDS = 24 * 3600  # pictures checked on drop are kept this long for a job or a preview
MAX_INSPECTIONS_AT_ONCE = 2  # each opens a whole picture; more at once only competes for memory
FORMAT_MAX_LONG_EDGE = {"jpeg": 65_535, "webp": 16_383}  # imageio_utils refuses to save bigger ones

SHARE_PORT_OFFSET = 1  # the Wi-Fi listener's port is the app's port + 1 (7861 by default)
SHARE_PORT_TRIES = 20  # when another program has that port, the next free one of this many is used
WIFI_HOST = "0.0.0.0"  # every network this computer is on
WIFI_GRACE_SECONDS = 3  # requests still running when sharing stops get this long to finish
WIFI_STOP_SECONDS = 10
VIA_WIFI = "upscaler.via_wifi"  # marks requests that came in through the Wi-Fi listener

HOUSEKEEPING_SECONDS = 3600
LAN_CACHE_SECONDS = 60
EVENT_POLL_SECONDS = 0.2
KEEPALIVE_SECONDS = 15

HOST_ONLY = "Only the person on the main computer can change this."
NEEDS_PASSWORD = f"{PRODUCT_NAME} needs the password."
WRONG_PASSWORD = f"That password didn't work. Check with the person who runs {PRODUCT_NAME}."
NOT_YOURS = "That's someone else's job."
FILE_GONE = "This file isn't available any more. Guests' results are kept for 24 hours."
OTHER_WEBSITE = "Requests from other websites aren't allowed."
BAD_REQUEST = "That request wasn't quite right. Reload the page and try again."
NO_SPACE = "The main computer is out of disk space. Free up some space and try again."
REQUEST_TOO_BIG = "That request was too big. Reload the page and try again."
STAGING_FULL = (
    "You've added as much as the main computer keeps for one person: pictures and videos are kept there for "
    "24 hours. Cancel jobs you no longer need, or try again later."
)
PHOTOREAL_HOST_ONLY = (
    "Ask the person at the main computer to download Photo-real. Until then, Faithful keeps your pictures "
    "true to the original."
)
CPU_ONLY_NOTE = "CPU only: slow"
STAGED_GONE = (
    "That picture isn't here any more. Pictures are kept for 24 hours after you add them, so please add it again."
)
NO_PICTURE = "Add at least one picture first."
NOT_SHARED = "That address isn't being shared right now."
SHARE_PORT_TAKEN = (
    "Sharing couldn't start because other programs are using port {port} and the ones after it. Quit them "
    f"and try again, or start {PRODUCT_NAME} on another port."
)
NO_NETWORK = "You don't seem to be connected to Wi-Fi or a network."
NO_NETWORK_TO_SHARE = f"{NO_NETWORK} Connect to your Wi-Fi, then try again."
NO_NETWORK_WHILE_SHARING = f"{NO_NETWORK} Connect to your Wi-Fi and the address will show up here."
NO_PASSWORD_WARNING = f"Anyone on your Wi-Fi can use {PRODUCT_NAME}. You can set a password in Settings."
FIREWALL_NOTES = {  # the question the computer's firewall asks the first time sharing starts
    "darwin": "If your Mac asks whether to allow incoming connections, click Allow.",
    "win32": "If Windows asks about the firewall, click Allow access.",
}

# --- What the app offers ---------------------------------------------------------------

# Photo-real detail levels, Balanced (the default) first.
PHOTOREAL_LEVEL_TEXT: dict[str, tuple[str, str, str]] = {  # id -> (label, description, time for 2K to 8K, M4)
    "balanced": (
        "Balanced",
        "Natural detail with crisp edges. Right for most pictures.",
        "About 3–6 minutes per picture",
    ),
    "rich": (
        "Rich",
        "The most new detail. Can make food and other natural surfaces look over-textured.",
        "About 10–22 minutes per picture",
    ),
    "gentle": ("Gentle", "The softest and most true to the original.", "About 3–5 minutes per picture"),
}

# Listed in this order; Photo-real (SeedVR2) first, as it is the default for pictures.
IMAGE_MODES: dict[str, tuple[str, str]] = {  # id -> (description, time for a 2K picture to 8K on an M4)
    PHOTOREAL: (
        (
            "Sharpens, cleans up noise and compression, and adds natural new detail, so pictures look like "
            "genuine high-resolution photos (SeedVR2)."
        ),
        PHOTOREAL_LEVEL_TEXT[DEFAULT_PHOTOREAL_LEVEL][2],
    ),
    "faithful": (
        "Sharpens and cleans up noise and compression, keeping faces, logos, text and shapes exactly as they are.",
        "About 3 minutes per picture",
    ),
    "fast": ("Sharpens and cleans up in seconds. Good for trying things out.", "A few seconds per picture"),
    "illustration": ("Sharpens drawings, cartoons and anime, keeping lines clean.", "About 15 seconds per picture"),
    "faithful_sharp": ("For camera photos that are already clean and sharp.", "About 3 minutes per picture"),
}
PHOTOREAL_LABEL = "Photo-real"
CLOUD_ETA_HINT = "About a minute per picture"


def _use_case(
    preset_id: str,
    label: str,
    icon: str,
    mode: str,
    size: str,
    tip: str,
    *,
    output_format: str = "jpeg",
    grain: str = "off",
) -> dict:
    return {
        "id": preset_id,
        "label": label,
        "icon": icon,  # a symbol name for the page's icon set
        "mode": mode,
        "photoreal_level": DEFAULT_PHOTOREAL_LEVEL if mode == PHOTOREAL else None,
        "size": size,
        "output_format": output_format,
        "dpi": DEFAULT_DPI,
        "grain": grain,
        "tip": tip,
    }


# One click sets the mode, size, format and grain; everything stays editable (docs/DESIGN.md,
# "Designer features"). The sample pictures name these ids too.
USE_CASE_PRESETS: tuple[dict, ...] = (
    _use_case(
        "interiors", "Interiors & furniture", "sofa", PHOTOREAL, "8k", "Adds believable fabric, wood and stone detail."
    ),
    _use_case("products", "Products & e-commerce", "tag", "faithful", "8k", "Keeps logos and text exact."),
    _use_case("portraits", "People & portraits", "user", "faithful", "8k", "Try Photo-real for more skin detail."),
    _use_case(
        "ai_art", "AI art", "sparkle", PHOTOREAL, "8k", "Cleans up the faint mesh texture AI pictures often have."
    ),
    _use_case(
        "print",
        "Print & posters",
        "printer",
        PHOTOREAL,
        "16k",
        "A 16K TIFF at 300 DPI, for big prints.",
        output_format="tiff",
    ),
    _use_case(
        "illustration", "Illustration & anime", "brush", "illustration", "8k", "Keeps lines clean and colours flat."
    ),
    _use_case(
        "old_photos", "Old photos", "photo", PHOTOREAL, "4k", "Brings back natural detail in old prints and scans."
    ),
)

VIDEO_MODE_TEXT: dict[str, tuple[str, str, str]] = {  # id -> (label, description, time hint)
    "fast": (
        "Fast",
        "Natural-looking and the quickest. Right for most videos.",
        "About 4 minutes for a 10-second clip",
    ),
    "general": (
        "Balanced",
        "A little more clean-up for noisy or soft footage.",
        "About 7 minutes for a 10-second clip",
    ),
    "faithful": ("Best", "The most detail, but very slow.", "Many hours, even for a short clip"),
}


# --- Who is asking -------------------------------------------------------------------


@dataclass(frozen=True)
class Viewer:
    is_host: bool
    owner: str  # "host" or "guest:<id>": who a job belongs to
    token: str | None  # the guest's session cookie (secret); None for the host
    logged_in: bool
    visit: str | None = None  # this browser session's cookie (secret), which a guest's own cloud key needs


@dataclass
class _SessionKey:
    """A cloud key someone pasted "for this visit": in memory only, for one browser session."""

    key: str
    visit: str  # the browser session it belongs to
    saved: float
    used: float


class _Context:
    """Everything the endpoints share for one running app."""

    def __init__(self, queue: JobQueue, sharing: WifiSharing):
        self.queue = queue
        self.sharing = sharing
        self.secret = _server_secret()
        self.lock = threading.Lock()
        self.clock = time.monotonic  # for guests' keys and password tries; tests move it on
        self.logins = LoginThrottle(lambda: self.clock())
        self.session_keys: dict[str, dict[str, _SessionKey]] = {}  # owner -> provider id -> key; memory only
        self._host_keys: dict[str, bool] = {}  # provider id -> has a saved key (the keychain is slow)
        self._lan: tuple[float, list[str]] | None = None  # when this computer's addresses were looked up, and them
        self._qr: tuple[str, str] | None = None  # the last address drawn as a QR code, and its SVG

    def guest_id(self, token: str) -> str:
        """A public id for a guest's jobs. Derived from the secret cookie, so it survives restarts
        but can't be turned back into the cookie."""
        return hmac.new(self.secret, b"owner:" + token.encode(), hashlib.sha256).hexdigest()[:16]

    def auth_value(self, token: str, password_hash: str) -> str:
        """Proof of login; a new password makes every old one invalid."""
        message = f"auth:{token}:{password_hash}".encode()
        return hmac.new(self.secret, message, hashlib.sha256).hexdigest()

    def host_has_key(self, provider_id: str) -> bool:
        with self.lock:
            if provider_id not in self._host_keys:
                self._host_keys[provider_id] = keys.get_key(provider_id) is not None
            return self._host_keys[provider_id]

    def remember_host_key(self, provider_id: str, present: bool) -> None:
        with self.lock:
            self._host_keys[provider_id] = present

    def session_key(self, viewer: Viewer, provider_id: str, *, use: bool = False) -> str | None:
        """The key this viewer pasted for this visit, if it's still kept. `use` counts as using it,
        which keeps it for another SESSION_KEY_IDLE_SECONDS."""
        with self.lock:
            entry = self.session_keys.get(viewer.owner, {}).get(provider_id)
            if entry is None:
                return None
            now = self.clock()
            if self._key_expired(entry, now):
                self._drop_key(viewer.owner, provider_id)
                return None
            if viewer.visit is None or not hmac.compare_digest(entry.visit, viewer.visit):
                return None  # another browser session: the key went when the browser closed
            if use:
                entry.used = now
            return entry.key

    def keep_session_key(self, owner: str, provider_id: str, key: str, visit: str) -> None:
        now = self.clock()
        with self.lock:
            self.session_keys.setdefault(owner, {})[provider_id] = _SessionKey(key, visit, now, now)

    def forget_session_key(self, owner: str, provider_id: str) -> None:
        with self.lock:
            self._drop_key(owner, provider_id)

    def prune_session_keys(self) -> int:
        """Forget guests' keys that are past their time. Returns how many."""
        now = self.clock()
        with self.lock:
            old = [
                (owner, provider_id)
                for owner, entries in self.session_keys.items()
                for provider_id, entry in entries.items()
                if self._key_expired(entry, now)
            ]
            for owner, provider_id in old:
                self._drop_key(owner, provider_id)
        return len(old)

    @staticmethod
    def _key_expired(entry: _SessionKey, now: float) -> bool:
        return now - entry.used > SESSION_KEY_IDLE_SECONDS or now - entry.saved > SESSION_KEY_MAX_SECONDS

    def _drop_key(self, owner: str, provider_id: str) -> None:
        entries = self.session_keys.get(owner, {})
        entries.pop(provider_id, None)
        if not entries:
            self.session_keys.pop(owner, None)

    def lan_addresses(self, fresh: bool = False) -> list[str]:
        """This computer's addresses on the network: looked up at most once a minute, or now if `fresh`."""
        cached = self._lan
        if fresh or cached is None or time.monotonic() - cached[0] > LAN_CACHE_SECONDS:
            cached = self._lan = (time.monotonic(), lan_ipv4_addresses())
        return list(cached[1])

    def lan_urls(self) -> list[str]:
        """The addresses other devices can open while sharing is on."""
        return [f"http://{ip}:{self.sharing.port}" for ip in self.lan_addresses()]

    def qr_code(self, url: str) -> str:
        cached = self._qr
        if cached is None or cached[0] != url:
            cached = self._qr = (url, qr_svg(url).decode("utf-8"))
        return cached[1]

    def share_status(self, viewer: Viewer, settings: Settings) -> dict:
        """What the Share on Wi-Fi button and panel show. The note and the warning are for the person
        at the computer, who is the only one who can act on them."""
        enabled = self.sharing.enabled
        urls = self.lan_urls() if enabled else []
        password_set = settings.access_password_hash is not None
        warning = None
        if enabled and viewer.is_host and not urls:
            warning = NO_NETWORK_WHILE_SHARING
        elif enabled and viewer.is_host and not password_set:
            warning = NO_PASSWORD_WARNING
        return {
            "enabled": enabled,
            "port": self.sharing.port if enabled else None,
            "urls": urls,
            "qr_svg": self.qr_code(urls[0]) if urls else None,  # the first address, for a phone's camera
            "password_set": password_set,
            "can_toggle": viewer.is_host,
            "note": firewall_note() if enabled and viewer.is_host else None,
            "warning": warning,
        }


def _server_secret() -> bytes:
    """A random key, made once per computer, for signing session cookies."""
    path = config.data_dir() / "server-secret"
    try:
        value = bytes.fromhex(path.read_text(encoding="ascii").strip())
        if len(value) >= 32:
            return value
    except (OSError, ValueError):
        pass
    value = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(value.hex())
    return value


def _context(request: Request) -> _Context:
    return request.app.state.upscaler


_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,100}")


def current_viewer(request: Request, response: Response) -> Viewer:
    client = request.client.host if request.client else ""
    visit = request.cookies.get(VISIT_COOKIE, "")
    visit = visit if _TOKEN.fullmatch(visit) else None
    if client in HOST_ADDRESSES and not request.scope.get(VIA_WIFI):
        return Viewer(is_host=True, owner=HOST, token=None, logged_in=True, visit=visit)
    ctx = _context(request)
    token = request.cookies.get(SESSION_COOKIE, "")
    if not _TOKEN.fullmatch(token):
        token = secrets.token_urlsafe(32)
        response.set_cookie(SESSION_COOKIE, token, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    password_hash = load_settings().access_password_hash
    proof = request.cookies.get(AUTH_COOKIE, "")
    logged_in = password_hash is None or hmac.compare_digest(
        proof.encode("utf-8", "replace"), ctx.auth_value(token, password_hash).encode()
    )
    owner = GUEST_PREFIX + ctx.guest_id(token)
    return Viewer(is_host=False, owner=owner, token=token, logged_in=logged_in, visit=visit)


def require_member(viewer: Annotated[Viewer, Depends(current_viewer)]) -> Viewer:
    if not viewer.logged_in:
        raise HTTPException(401, NEEDS_PASSWORD)
    return viewer


def require_host(viewer: Annotated[Viewer, Depends(require_member)]) -> Viewer:
    if not viewer.is_host:
        raise HTTPException(403, HOST_ONLY)
    return viewer


AnyViewer = Annotated[Viewer, Depends(current_viewer)]
Member = Annotated[Viewer, Depends(require_member)]
HostOnly = Annotated[Viewer, Depends(require_host)]


# --- Passwords -------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    return f"{salt.hex()}:{_scrypt(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split(":")
        salt, digest = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except ValueError:
        return False
    return hmac.compare_digest(_scrypt(password, salt), digest)


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)


_EVERYONE = "*"  # LoginThrottle's count of wrong passwords from every device together


class LoginThrottle:
    """Slows down anyone guessing the guests' password, however many requests they send at once.
    Passwords are checked one at a time. After LOGIN_FREE_TRIES wrong ones in a row from a device
    (or LOGIN_FREE_TRIES_EVERYONE from all of them together), the next try waits LOGIN_WAIT_SECONDS
    before it is even checked, and each wrong one after that doubles the wait."""

    def __init__(self, clock: Any):
        self.clock = clock
        self._lock = threading.Lock()
        self._wrong: dict[str, tuple[int, float]] = {}  # device (or _EVERYONE) -> (wrong in a row, last at)

    def attempt(self, device: str, password: str, stored: str) -> tuple[bool, float]:
        """Check a password from `device`: (right, 0), or (False, seconds to wait) without checking it."""
        with self._lock:  # one at a time, so guesses sent all at once still count one by one
            now = self.clock()
            wait = max(self._wait(device, LOGIN_FREE_TRIES, now), self._wait(_EVERYONE, LOGIN_FREE_TRIES_EVERYONE, now))
            if wait > 0:
                return False, wait
            if verify_password(password, stored):
                self._wrong.pop(device, None)
                return True, 0.0
            for key in (device, _EVERYONE):
                count, last = self._wrong.get(key, (0, now))
                self._wrong[key] = (0 if now - last > LOGIN_FORGET_SECONDS else count) + 1, now
            self._forget_old(now)
            return False, 0.0

    def _wait(self, key: str, free_tries: int, now: float) -> float:
        count, last = self._wrong.get(key, (0, now))
        if count < free_tries or now - last > LOGIN_FORGET_SECONDS:
            return 0.0
        wait = min(LOGIN_WAIT_SECONDS * 2 ** (count - free_tries), LOGIN_MAX_WAIT_SECONDS)
        return max(last + wait - now, 0.0)

    def _forget_old(self, now: float) -> None:
        for key in [key for key, (_, last) in self._wrong.items() if now - last > LOGIN_FORGET_SECONDS]:
            del self._wrong[key]


def _wait_message(seconds: float) -> str:
    whole = max(math.ceil(seconds), 1)
    if whole < 60:
        wait = "1 second" if whole == 1 else f"{whole} seconds"
    else:
        minutes = math.ceil(whole / 60)
        wait = "1 minute" if minutes == 1 else f"{minutes} minutes"
    return f"Too many wrong passwords. Wait {wait}, then try again."


# --- The app -----------------------------------------------------------------------------


def create_app(
    queue: JobQueue, share: bool, *, port: int = DEFAULT_PORT, wifi_socket: socket.socket | None = None
) -> FastAPI:
    """The app, for a listener on 127.0.0.1:`port`. With `share`, sharing on Wi-Fi switches on as the
    app starts: on `wifi_socket` when the caller has bound it already, otherwise on port + 1."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        queue.start()
        if share:
            await run_in_threadpool(_share_from_the_start, sharing, wifi_socket)
        housekeeping = asyncio.create_task(_housekeeping(app.state.upscaler))
        try:
            yield
        finally:
            housekeeping.cancel()
            # The queue first: a running job is saved as waiting at once, in case the app is gone
            # before everything else has stopped (Windows allows a few seconds when a window closes).
            await run_in_threadpool(queue.stop)
            await run_in_threadpool(sharing.stop)

    app = FastAPI(
        title=PRODUCT_NAME, version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    sharing = WifiSharing(app, port + SHARE_PORT_OFFSET)
    app.state.upscaler = _Context(queue, sharing)
    app.include_router(router)
    app.add_exception_handler(RequestValidationError, _bad_request)
    app.mount("/", _Pages(directory=WEB_DIR, html=True, check_dir=False), name="pages")
    app.add_middleware(_Guard)
    return app


async def _housekeeping(ctx: _Context) -> None:
    """Tidy up now, and then every hour, while the app runs."""
    while True:
        await asyncio.to_thread(_tidy_up, ctx)
        await asyncio.sleep(HOUSEKEEPING_SECONDS)


def _tidy_up(ctx: _Context) -> None:
    """Delete what is past its time: guests' results and staged pictures (24 hours after they were
    made), finished jobs' records and folders with their previews (jobs.JobQueue._expired), and
    guests' own cloud keys."""
    chores = (
        ("guests' results", lambda: delete_old_files(config.guest_outputs_dir(), GUEST_RESULTS_SECONDS)),
        ("staged pictures", lambda: tidy_staging(staging_dir(), STAGING_SECONDS)),
        ("old jobs", ctx.queue.prune_expired),
        ("guests' keys", ctx.prune_session_keys),
    )
    for what, chore in chores:
        try:
            chore()
        except Exception:
            log.exception("Couldn't tidy up %s", what)


async def _bad_request(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": BAD_REQUEST}, status_code=400)


class _Pages(StaticFiles):
    """The app's own pages. Browsers re-check them on each visit, so an updated app shows at once."""

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class _Guard:
    """Refuses requests that only a malicious web page would make:
    - a Host header naming some other website (DNS rebinding: a page re-pointing its own domain
      at this computer to talk to the app as the host)
    - a change (POST, DELETE…) sent from another website's page (cross-site request forgery).
    Refuses a body bigger than MAX_FIELD_BYTES, except for uploads (BIG_BODY_PATHS), before reading it,
    so no device on the Wi-Fi can fill the computer's memory, password or not.
    Also tells browsers never to show the app inside another site's frame."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        if not allowed_host(host):
            await PlainTextResponse("Unknown host name.", status_code=400)(scope, receive, send)
            return
        if scope["method"] not in SAFE_METHODS and not same_origin(headers.get("origin"), host):
            await JSONResponse({"detail": OTHER_WEBSITE}, status_code=403)(scope, receive, send)
            return
        limited = scope["path"] not in BIG_BODY_PATHS
        if limited and _declared_length(headers) > MAX_FIELD_BYTES:
            await JSONResponse({"detail": REQUEST_TOO_BIG}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def receive_at_most_a_little() -> Any:
            # For bodies sent without a length (or a wrong one): stop as soon as there's too much.
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_FIELD_BYTES:
                    raise HTTPException(413, REQUEST_TOO_BIG)
            return message

        async def send_with_safety_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                extra = MutableHeaders(scope=message)
                extra.setdefault("X-Frame-Options", "DENY")
                extra.setdefault("X-Content-Type-Options", "nosniff")
                extra.setdefault("Referrer-Policy", "no-referrer")
            await send(message)

        await self.app(scope, receive_at_most_a_little if limited else receive, send_with_safety_headers)


def _declared_length(headers: Headers) -> int:
    """The body length a request declares; 0 when it doesn't say (or says something odd)."""
    try:
        return max(int(headers.get("content-length") or 0), 0)
    except ValueError:
        return 0


def allowed_host(host_header: str) -> bool:
    """IP addresses, localhost and this computer's own name. We only ever hand out IP addresses."""
    hostname = _hostname(host_header)
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    return hostname == "localhost" or hostname.endswith(".localhost") or hostname in _own_names()


def _hostname(host_header: str) -> str:
    value = host_header.strip().lower()
    if value.startswith("["):  # [::1]:7860
        return value[1 : value.find("]")] if "]" in value else ""
    return value.rsplit(":", 1)[0].rstrip(".") if value.count(":") <= 1 else value


def _own_names() -> set[str]:
    name = socket.gethostname().lower().rstrip(".")
    bare = name.removesuffix(".local")
    return {name, bare, bare + ".local"}


def same_origin(origin: str | None, host_header: str) -> bool:
    if origin is None:
        return True  # not sent by a web page from another site (browsers always send Origin then)
    return urlsplit(origin).netloc.lower() == host_header.strip().lower()


router = APIRouter(prefix="/api")


# --- Status and login ------------------------------------------------------------------


@router.get("/status")
def status(request: Request, viewer: AnyViewer) -> dict:
    app_info = {"name": PRODUCT_NAME}
    if not viewer.logged_in:
        return {"app": app_info, "version": __version__, "is_host": False, "needs_login": True}
    ctx = _context(request)
    settings = load_settings()
    hardware = detect_hardware()
    photoreal = _photoreal_status(hardware)
    providers = [_provider_status(ctx, viewer, provider, settings) for provider in PROVIDERS.values()]
    cpu_only = hardware.device == "cpu"
    body = {
        "app": app_info,
        "version": __version__,
        "is_host": viewer.is_host,
        "needs_login": False,
        "hardware": {
            "device": hardware.device,
            "device_name": hardware.device_name,
            "memory_gb": hardware.memory_gb,
            "cpu_only": cpu_only,
            "note": hardware.note or (CPU_ONLY_NOTE if cpu_only else None),
        },
        "photoreal": photoreal,
        "image_modes": _image_modes(photoreal["supported"], providers),
        "sizes": [{"id": size, "label": label} for size, label in PRESETS.items()],
        "formats": [_output_format_entry(fmt) for fmt in OUTPUT_FORMATS],
        "dpi_choices": list(DPI_CHOICES),
        "presets": [dict(preset) for preset in USE_CASE_PRESETS],
        "samples": [sample for sample, _ in _bundled_samples().values()],
        "accept": {"images": sorted(SUPPORTED_IMAGE_EXTS), "videos": sorted(SUPPORTED_VIDEO_EXTS)},
        "video_modes": _video_modes(),
        "providers": providers,
        "share": ctx.share_status(viewer, settings),
    }
    if viewer.is_host:
        body["settings"] = public_settings(settings)
    return body


def _output_format_entry(fmt: str) -> dict:
    entry: dict[str, Any] = {"id": fmt}
    if fmt in FORMAT_MAX_LONG_EDGE:
        entry["max_long_edge"] = FORMAT_MAX_LONG_EDGE[fmt]
    return entry


def _photoreal_status(hardware: Hardware) -> dict:
    supported, reason = photoreal_support(hardware)
    engine = get_seedvr2_engine(hardware) if supported else None
    if supported and engine is None:
        supported, reason = False, "Photo-real can't run on this computer"
    return {
        "supported": supported,
        "reason": reason,
        "downloaded": bool(engine is not None and engine.weights_present()),
        "engine": engine.name if engine is not None else None,
    }


def _image_modes(photoreal_supported: bool, providers: list[dict]) -> list[dict]:
    modes = [
        {
            "id": mode,
            "label": PHOTOREAL_LABEL if mode == PHOTOREAL else MODELS[mode].label,
            "description": description,
            "eta_hint": eta_hint,
            "available": photoreal_supported if mode == PHOTOREAL else True,
            "needs_key": False,
        }
        for mode, (description, eta_hint) in IMAGE_MODES.items()
    ]
    modes[0]["levels"] = [
        {"id": level, "label": label, "description": description, "eta_hint": eta_hint}
        for level, (label, description, eta_hint) in PHOTOREAL_LEVEL_TEXT.items()
    ]
    for provider in providers:
        for model in provider["models"]:
            modes.append(
                {
                    "id": f"{CLOUD_PREFIX}{provider['id']}:{model['id']}",
                    "label": model["label"],
                    "description": model["description"],
                    "eta_hint": CLOUD_ETA_HINT,
                    "available": provider["connected"],
                    "needs_key": not provider["connected"],
                    "kind": model["kind"],
                    "est_cost_8k_usd": model["est_cost_8k_usd"],
                }
            )
    return modes


_SAMPLE_TEXT_FIELDS = ("title", "caption", "preset", "credit", "license", "source_url")


def _bundled_samples() -> dict[str, tuple[dict, Path]]:
    """The sample pictures listed in web/samples/samples.json: id -> (what the page shows, the file).

    Read on every call, so the list is never stale; a missing or damaged list just means no samples.
    """
    try:
        entries = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    samples: dict[str, tuple[dict, Path]] = {}
    for entry in entries if isinstance(entries, list) else []:
        try:
            sample_id, file, thumb = entry["id"], _sample_path(entry["file"]), _sample_path(entry["thumb"])
            shown = {
                "id": str(sample_id),
                **{name: str(entry.get(name) or "") for name in _SAMPLE_TEXT_FIELDS},
                "thumb_url": f"/samples/{quote(thumb.name)}",
                "url": f"/samples/{quote(file.name)}",
                "width": int(entry["width"]),
                "height": int(entry["height"]),
            }
        except (KeyError, TypeError, ValueError):
            log.warning("Skipping a damaged entry in samples.json: %r", entry)
            continue
        samples[shown["id"]] = (shown, file)
    return samples


def _sample_path(name: Any) -> Path:
    """A file in the samples folder, by its plain name; ValueError for anything else."""
    if not isinstance(name, str) or not name or Path(name).name != name or name.startswith("."):
        raise ValueError(f"Not a sample file name: {name!r}")
    path = SAMPLES_DIR / name
    if not path.is_file():
        raise ValueError(f"Missing sample file: {name}")
    return path


def _video_modes() -> list[dict]:
    modes = []
    for mode in VIDEO_MODES:
        label, description, eta_hint = VIDEO_MODE_TEXT.get(mode, (mode.title(), "", ""))
        modes.append({"id": mode, "label": label, "description": description, "eta_hint": eta_hint, "available": True})
    return modes


def _provider_status(ctx: _Context, viewer: Viewer, provider: CloudProvider, settings: Settings) -> dict:
    return {
        "id": provider.id,
        "name": provider.name,
        "connected": _can_use_cloud(ctx, viewer, provider.id, settings),
        # The key this viewer pasted for their visit ("Forget my key" removes it), not the host's.
        "own_key": ctx.session_key(viewer, provider.id) is not None,
        "models": [asdict(model) for model in provider.models],
    }


def _can_use_cloud(ctx: _Context, viewer: Viewer, provider_id: str, settings: Settings) -> bool:
    if ctx.session_key(viewer, provider_id):
        return True
    if not ctx.host_has_key(provider_id):
        return False
    return viewer.is_host or guests_may_use_host_cloud(settings)


def public_settings(settings: Settings) -> dict:
    has_password = settings.access_password_hash is not None
    return {
        "has_password": has_password,
        "allow_guests_cloud": settings.allow_guests_cloud and has_password,
        "default_image_mode": settings.default_image_mode,
        "output_format": settings.output_format,
        "grain": settings.grain,
    }


class LoginRequest(BaseModel):
    password: str = ""


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response, viewer: AnyViewer) -> dict:
    stored = load_settings().access_password_hash
    if viewer.is_host or stored is None:
        return {"ok": True}
    ctx = _context(request)
    device = request.client.host if request.client else ""
    right, wait = await run_in_threadpool(ctx.logins.attempt, device, body.password, stored)
    if wait:
        raise HTTPException(429, _wait_message(wait))
    if not right:
        await asyncio.sleep(LOGIN_FAILURE_DELAY_SECONDS)
        raise HTTPException(401, WRONG_PASSWORD)
    proof = ctx.auth_value(viewer.token, stored)
    response.set_cookie(AUTH_COOKIE, proof, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return {"ok": True}


# --- Settings (host only) ------------------------------------------------------------------


class SettingsChange(BaseModel):
    access_password: str | None = None  # "" removes the password
    allow_guests_cloud: bool | None = None
    default_image_mode: str | None = None
    output_format: str | None = None
    grain: str | None = None


@router.get("/settings")
def get_settings(viewer: HostOnly) -> dict:
    return {"settings": public_settings(load_settings())}


@router.post("/settings")
def change_settings(change: SettingsChange, request: Request, viewer: HostOnly) -> dict:
    new_hash = None
    if change.access_password:
        if len(change.access_password) < MIN_PASSWORD_LENGTH:
            raise HTTPException(400, f"Use at least {MIN_PASSWORD_LENGTH} characters.")
        new_hash = hash_password(change.access_password)  # slow on purpose, so not while holding the settings
    # Load, change and save as one step: two changes at once (quick clicks) never undo each other.
    with config.changing_settings() as settings:
        if change.access_password == "":
            settings.access_password_hash = None
            settings.allow_guests_cloud = False
        elif new_hash is not None:
            settings.access_password_hash = new_hash
        if change.allow_guests_cloud is not None:
            if change.allow_guests_cloud and settings.access_password_hash is None:
                raise HTTPException(
                    400, "Set an access password first, so nobody else on your Wi-Fi can spend your money."
                )
            settings.allow_guests_cloud = change.allow_guests_cloud
        if change.default_image_mode is not None:
            if change.default_image_mode not in IMAGE_MODES:
                raise HTTPException(400, "That isn't one of the app's quality modes.")
            settings.default_image_mode = change.default_image_mode
        if change.output_format is not None:
            if change.output_format not in OUTPUT_FORMATS:
                raise HTTPException(400, "That isn't one of the app's file formats.")
            settings.output_format = change.output_format
        if change.grain is not None:
            if change.grain not in GRAIN_CHOICES:
                raise HTTPException(400, "Grain can be off, subtle or medium.")
            settings.grain = change.grain
    _context(request).queue.publish_status()
    return {"settings": public_settings(settings)}


# --- Uploads -------------------------------------------------------------------------------


_UNSAFE_CHARACTERS = re.compile(r'[<>:"/\\|?*]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}


def safe_filename(name: str) -> str:
    """A file name that is safe on Mac and Windows and can't point outside its folder.
    Spaces, accents and emoji are kept, so results keep the names people recognise."""
    name = unicodedata.normalize("NFC", name)
    name = re.split(r"[/\\]", name)[-1]  # never a path
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Cc")
    name = _UNSAFE_CHARACTERS.sub("_", name).strip().lstrip(".").rstrip(". ")
    if not name:
        return "upload"
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    if stem.upper() in _WINDOWS_RESERVED:
        stem = "_" + stem
    suffix = suffix.encode()[:16].decode("utf-8", "ignore")
    room = MAX_NAME_BYTES - len(suffix.encode()) - (1 if dot else 0)
    stem = stem.encode()[:room].decode("utf-8", "ignore").rstrip(". ") or "upload"
    return f"{stem}.{suffix}" if dot else stem


class _UploadTooBig(Exception):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class _BadUpload(Exception):
    pass


class _MultipartReceiver:
    """Writes each uploaded file straight to disk, each in its own numbered folder (in/0/, in/1/, …)
    so same-named files never collide. The expected text fields are kept in memory, at most
    MAX_FIELD_BYTES of them in all; any other field is refused."""

    def __init__(
        self,
        inbox: Path,
        file_fields: frozenset[str],
        max_files: int,
        max_file_bytes: int | None,
        too_many: str | None = None,
        text_fields: frozenset[str] = frozenset(),
    ):
        self.inbox = inbox
        self.file_fields = file_fields
        self.text_fields = text_fields
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.too_many = too_many or f"Send at most {max_files} pictures at once."
        self.files: list[Path] = []
        self.fields: dict[str, str] = {}
        self._text_bytes = 0
        self._header_bytes = 0
        self._headers: dict[bytes, bytes] = {}
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._file: Any = None
        self._written = 0
        self._field_name: str | None = None
        self._field_value = bytearray()

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._part_begin,
            "on_header_field": lambda data, start, end: self._header_part(self._header_name, data[start:end]),
            "on_header_value": lambda data, start, end: self._header_part(self._header_value, data[start:end]),
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _part_begin(self) -> None:
        self._headers, self._field_name, self._header_bytes = {}, None, 0

    def _header_part(self, buffer: bytearray, chunk: bytes) -> None:
        self._header_bytes += len(chunk)
        if self._header_bytes > MAX_PART_HEADER_BYTES:
            raise _BadUpload(BAD_REQUEST)
        buffer.extend(chunk)

    def _header_end(self) -> None:
        self._headers[bytes(self._header_name).lower()] = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def _headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = params.get(b"name", b"").decode("utf-8", "replace")
        filename = params.get(b"filename")
        if filename is None:
            if name not in self.text_fields:
                raise _BadUpload(BAD_REQUEST)
            self._field_name, self._field_value = name, bytearray()
            return
        if name not in self.file_fields:
            raise _BadUpload(BAD_REQUEST)
        if len(self.files) >= self.max_files:
            raise _BadUpload(self.too_many)
        folder = self.inbox / str(len(self.files))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / safe_filename(filename.decode("utf-8", "replace"))
        self._file = path.open("wb")
        self._written = 0
        self.files.append(path)

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._file is not None:
            self._written += len(chunk)
            if self.max_file_bytes is not None and self._written > self.max_file_bytes:
                raise _UploadTooBig(self.files[-1].name)
            self._file.write(chunk)
        elif self._field_name is not None:
            self._field_value.extend(chunk)
            self._text_bytes += len(chunk)
            if self._text_bytes > MAX_FIELD_BYTES:
                raise _BadUpload(BAD_REQUEST)

    def _part_end(self) -> None:
        if self._file is not None:
            self.close()
        elif self._field_name is not None:
            self.fields[self._field_name] = self._field_value.decode("utf-8", "replace")
            self._field_name = None


_WRITE_BATCH_BYTES = 1024 * 1024


async def _receive_upload(
    request: Request,
    inbox: Path,
    file_fields: frozenset[str],
    max_files: int,
    max_file_bytes: int | None,
    *,
    too_many: str | None = None,
    text_fields: frozenset[str] = OPTIONS_FIELDS,
    what: str = "these pictures",
) -> _MultipartReceiver:
    """Receive a multipart upload into `inbox`, stopping (507) if the disk gets so full that less
    than SPACE_RESERVE_BYTES would be left. `what` names the upload in that message."""
    kind, params = parse_options_header(request.headers.get("content-type", ""))
    if kind != b"multipart/form-data" or b"boundary" not in params:
        raise _BadUpload(BAD_REQUEST)
    receiver = _MultipartReceiver(inbox, file_fields, max_files, max_file_bytes, too_many, text_fields)
    parser = MultipartParser(params[b"boundary"], receiver.callbacks())
    pending = bytearray()
    unchecked = 0  # bytes written since free space was last checked
    try:
        async for chunk in request.stream():
            pending.extend(chunk)
            if len(pending) >= _WRITE_BATCH_BYTES:
                # Disk writes happen off the event loop, so other people's pages stay responsive.
                await run_in_threadpool(parser.write, bytes(pending))
                unchecked += len(pending)
                pending.clear()
                if unchecked >= SPACE_CHECK_BYTES:
                    _check_space(config.jobs_dir(), SPACE_RESERVE_BYTES, what)
                    unchecked = 0
        await run_in_threadpool(parser.write, bytes(pending))
        parser.finalize()
        _check_space(config.jobs_dir(), SPACE_RESERVE_BYTES, what)
    except MultipartParseError as exc:
        raise _BadUpload(BAD_REQUEST) from exc
    finally:
        receiver.close()
    return receiver


@contextmanager
def _upload_folder(folder: Path) -> Iterator[Path]:
    """A reserved folder for an upload (a job's, or a staged picture's), removed again if anything
    goes wrong before the job or the staged picture is complete."""
    try:
        yield folder
    except _UploadTooBig as exc:
        shutil.rmtree(folder, ignore_errors=True)
        size = f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        raise HTTPException(413, f"{exc.name} is bigger than {size}. Try a smaller copy.") from None
    except _BadUpload as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, str(exc)) from None
    except ClientDisconnect:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, BAD_REQUEST) from None
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        if exc.errno == errno.ENOSPC:
            raise HTTPException(507, NO_SPACE) from None
        raise
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _options_field(fields: dict[str, str]) -> dict:
    try:
        options = json.loads(fields.get("options") or "{}")
    except ValueError:
        raise HTTPException(400, BAD_REQUEST) from None
    if not isinstance(options, dict):
        raise HTTPException(400, BAD_REQUEST)
    return options


def _fraction(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise HTTPException(400, f"{what} must be between 0% and 100%.")
    return float(value)


def image_options(raw: dict, settings: Settings) -> dict:
    """Checked options for a picture job, with the host's saved defaults filled in."""
    default_mode = settings.default_image_mode if settings.default_image_mode in IMAGE_MODES else DEFAULT_MODE
    mode = raw.get("mode") or default_mode
    if not _known_image_mode(mode):
        raise HTTPException(400, "That isn't one of the app's quality modes.")
    size = raw.get("size", "8k")
    if not isinstance(size, str) or size not in PRESETS:  # a list would make the lookup itself fail
        raise HTTPException(400, "That isn't one of the app's sizes.")
    grain = raw.get("grain", settings.grain)
    if not isinstance(grain, str) or grain not in GRAIN_CHOICES:
        raise HTTPException(400, "Grain can be off, subtle or medium.")
    output_format = raw.get("output_format", settings.output_format)
    if not isinstance(output_format, str) or output_format not in OUTPUT_FORMATS:
        raise HTTPException(400, "That isn't one of the app's file formats.")
    dpi = raw.get("dpi", DEFAULT_DPI)
    if isinstance(dpi, bool) or not isinstance(dpi, (int, float)) or dpi not in DPI_CHOICES:
        raise HTTPException(400, f"Choose {', '.join(map(str, DPI_CHOICES[:-1]))} or {DPI_CHOICES[-1]} DPI.")
    level = raw.get("photoreal_level", DEFAULT_PHOTOREAL_LEVEL)
    if not isinstance(level, str) or level not in PHOTOREAL_LEVELS:
        raise HTTPException(400, "Photo-real detail can be Balanced, Rich or Gentle.")
    return {
        "mode": mode,
        "size": size,
        "strength": _fraction(raw.get("strength", 1.0), "Detail strength"),
        "grain": grain,
        "output_format": output_format,
        "dpi": int(dpi),
        "photoreal_level": level,
        "creativity": _fraction(raw.get("creativity", 0.0), "Creativity"),
        "resemblance": _fraction(raw.get("resemblance", 1.0), "Stay close to the original"),
    }


def _known_image_mode(mode: Any) -> bool:
    if not isinstance(mode, str):
        return False
    if mode in IMAGE_MODES:
        return True
    parts = mode.split(":")
    if len(parts) != 3 or parts[0] + ":" != CLOUD_PREFIX or parts[1] not in PROVIDERS:
        return False
    return any(model.id == parts[2] for model in PROVIDERS[parts[1]].models)


def video_options(raw: dict) -> dict:
    mode = raw.get("mode", "fast")
    if not isinstance(mode, str) or mode not in VIDEO_MODES:
        raise HTTPException(400, "That isn't one of the app's video modes.")
    preview = raw.get("preview_seconds")
    if preview is not None and (
        isinstance(preview, bool) or not isinstance(preview, (int, float)) or not 0 < preview <= MAX_PREVIEW_SECONDS
    ):
        raise HTTPException(400, "A preview can be up to 10 minutes long.")
    strength = _fraction(raw.get("strength", 1.0), "Detail strength")
    return {"mode": mode, "preview_seconds": None if preview is None else float(preview), "strength": strength}


def _cloud_key_for(ctx: _Context, viewer: Viewer, mode: str, settings: Settings) -> str | None:
    """The guest's own key for a cloud job (kept in memory with the job), or None to use the
    host's saved key. Refuses when the person may not use a cloud service."""
    if not mode.startswith(CLOUD_PREFIX):
        return None
    provider_id = mode.split(":")[1]
    if own := ctx.session_key(viewer, provider_id, use=True):
        return own
    if viewer.is_host:
        if ctx.host_has_key(provider_id):
            return None
        raise HTTPException(400, f"To use this mode, connect {PROVIDERS[provider_id].name} in Settings first.")
    if guests_may_use_host_cloud(settings) and ctx.host_has_key(provider_id):
        return None
    raise HTTPException(403, GUEST_CLOUD_NOT_ALLOWED)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _check_space(path: Path, needed: int, what: str = "this video") -> None:
    if needed and free_bytes(path) < needed:
        gigabytes = max(needed / 1e9, 0.1)
        raise HTTPException(
            507,
            f"There isn't enough free disk space on the main computer for {what}. It needs about "
            f"{gigabytes:.1f} GB free. Free up some space and try again.",
        )


@router.post("/images")
async def upload_images(request: Request, viewer: Member) -> dict:
    """A picture job: multipart `files` plus `options`, or JSON {tokens, options} naming pictures
    already checked by /api/inspect (they aren't sent again)."""
    ctx = _context(request)
    kind, _ = parse_options_header(request.headers.get("content-type", ""))
    if kind == b"application/json":
        body = await _json_object(request)
        return await run_in_threadpool(_images_from_tokens, ctx, viewer, body)
    job_id = ctx.queue.new_job_id()
    with _upload_folder(ctx.queue.job_dir(job_id)) as folder:
        declared = _declared_length(request.headers)
        _check_space(folder.parent, declared * PICTURE_SPACE_FACTOR + SPACE_RESERVE_BYTES, "these pictures")
        uploaded = await _check_guest_quota(ctx, viewer, declared)
        upload = await _receive_upload(request, folder / "in", IMAGE_FIELDS, MAX_IMAGES_PER_BATCH, MAX_IMAGE_BYTES)
        if not upload.files:
            raise HTTPException(400, NO_PICTURE)
        await _check_guest_quota(ctx, viewer, sum(path.stat().st_size for path in upload.files), uploaded)
        settings = load_settings()
        options = image_options(_options_field(upload.fields), settings)
        key = _cloud_key_for(ctx, viewer, options["mode"], settings)
        sizes = await run_in_threadpool(_check_pictures, upload.files)
        _check_room_for_results(viewer, options, sizes)
        inputs = [str(path) for path in upload.files]
        job = ctx.queue.submit("image", viewer.owner, inputs, options, job_id=job_id, secret=key)
    return {"job": job_view(job, viewer)}


def _check_pictures(paths: list[Path]) -> list[ImageInfo | None]:
    """Refuse, in plain words, any file that isn't a picture the app can read. Files named like a
    supported picture are taken as they are (a damaged one gets its own message from its job);
    anything else, such as a file with no extension, is checked by its content. Returns what each
    picture is (its size, for _check_room_for_results), or None where that couldn't be read."""
    found: list[ImageInfo | None] = []
    for path in paths:
        try:
            found.append(inspect_image(path))
        except Exception as exc:
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                found.append(None)  # its job explains any trouble with it
            elif isinstance(exc, ImageFileError):
                raise HTTPException(400, str(exc)) from None
            else:
                raise
    return found


def _check_room_for_results(viewer: Viewer, options: dict, pictures: list[ImageInfo | None]) -> None:
    """Refuse (507) a picture job whose finished pictures won't fit on the disk they're saved to with
    SPACE_RESERVE_BYTES to spare: said before it starts, rather than found out picture by picture (each
    picture checks again before it's made, as other jobs and apps may use the space meanwhile). Pictures
    whose size isn't known count as nothing."""
    total = 0
    for info in pictures:
        if info is None:
            continue
        try:
            out = target_size(info.width, info.height, options["size"], exact=info.is_vector)
            fmt = estimate_format(options["output_format"], info.format)
        except ValueError:  # the job itself explains it
            continue
        total += estimated_file_bytes(*out, fmt, alpha=info.has_alpha)
    out_dir = config.host_outputs_dir() if viewer.is_host else config.guest_outputs_dir()
    free = free_bytes(out_dir)
    if free >= total + SPACE_RESERVE_BYTES:
        return
    count = sum(info is not None for info in pictures)
    what = "This picture" if count == 1 else f"These {count} pictures"
    advice = "choose a smaller size." if options["output_format"] == "jpeg" else "choose JPEG or a smaller size."
    raise HTTPException(
        507,
        f"{what} will take about {describe_size(total)} once finished, and the main computer has "
        f"{describe_size(free)} free. {PRODUCT_NAME} always leaves {SPACE_RESERVE_TEXT} free so the computer's "
        f"other apps can still save their work. Free up some space, or {advice}",
    )


async def _json_object(request: Request) -> dict:
    """The request's JSON body, which must be a small object."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_FIELD_BYTES:
            raise HTTPException(400, BAD_REQUEST)
    try:
        data = json.loads(bytes(body))
    except ValueError:
        raise HTTPException(400, BAD_REQUEST) from None
    if not isinstance(data, dict):
        raise HTTPException(400, BAD_REQUEST)
    return data


def _images_from_tokens(ctx: _Context, viewer: Viewer, body: dict) -> dict:
    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        raise HTTPException(400, BAD_REQUEST)
    if not tokens:
        raise HTTPException(400, NO_PICTURE)
    if len(tokens) > MAX_IMAGES_PER_BATCH:
        raise HTTPException(400, f"Send at most {MAX_IMAGES_PER_BATCH} pictures at once.")
    staged = [_staged(token, viewer) for token in tokens]
    if any(picture is None for picture in staged):
        raise HTTPException(400, STAGED_GONE)
    raw = body.get("options", {})
    if not isinstance(raw, dict):
        raise HTTPException(400, BAD_REQUEST)
    settings = load_settings()
    options = image_options(raw, settings)
    key = _cloud_key_for(ctx, viewer, options["mode"], settings)
    _check_room_for_results(viewer, options, [_staged_info(picture) for picture in staged])
    job_id = ctx.queue.new_job_id()
    with _upload_folder(ctx.queue.job_dir(job_id)) as folder:
        # Each in its own folder, as uploads are, so same-named pictures never collide.
        inputs = [str(_link_into(picture.file, folder / "in" / str(n))) for n, picture in enumerate(staged)]
        job = ctx.queue.submit("image", viewer.owner, inputs, options, job_id=job_id, secret=key)
    return {"job": job_view(job, viewer)}


def _link_into(source: Path, folder: Path) -> Path:
    """A job's own copy of a staged picture: a hard link where the disk allows (instant, and no extra
    space), so tidying either one away never touches the other."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def video_space_needed(seconds: float, source_bytes: int) -> int:
    """Free space a video job needs: its 4K segments and the joined result for `seconds` of video,
    plus `source_bytes` for the video itself while it isn't on the disk yet."""
    return math.ceil(seconds * VIDEO_OUTPUT_BITS_PER_SECOND / 8 * VIDEO_WORK_FACTOR) + source_bytes


@router.post("/videos")
async def upload_video(request: Request, viewer: Member) -> dict:
    ctx = _context(request)
    job_id = ctx.queue.new_job_id()
    with _upload_folder(ctx.queue.job_dir(job_id)) as folder:
        declared = _declared_length(request.headers)
        _check_space(folder.parent, video_space_needed(0, declared) + SPACE_RESERVE_BYTES)  # the video itself
        uploaded = await _check_guest_quota(ctx, viewer, declared)
        upload = await _receive_upload(
            request, folder / "in", VIDEO_FIELDS, 1, None, too_many="One video at a time, please.", what="this video"
        )
        if not upload.files:
            raise HTTPException(400, "Add a video first.")
        video = upload.files[0]
        await _check_guest_quota(ctx, viewer, video.stat().st_size, uploaded)
        options = video_options(_options_field(upload.fields))
        try:
            info = await run_in_threadpool(probe_video, video)
        except VideoError as exc:
            raise HTTPException(400, str(exc)) from None
        if info.frames <= 1:  # e.g. a photo; the video reader opens those too
            raise HTTPException(400, "This file is a single picture, not a video. Add pictures on the Photos page.")
        # The video is on the disk now; what it becomes depends on how long it is, not on its file size.
        preview = options["preview_seconds"]
        seconds = info.duration if preview is None else min(info.duration, preview)
        _check_space(folder.parent, video_space_needed(seconds, 0) + SPACE_RESERVE_BYTES)
        job = ctx.queue.submit("video", viewer.owner, [str(video)], options, job_id=job_id)
    return {"job": job_view(job, viewer)}


class EstimateRequest(BaseModel):
    width: int
    height: int
    size: Any = None
    mode: Any = None
    photoreal_level: Any = None
    output_format: Any = None
    dpi: Any = None
    is_vector: Any = None  # an SVG (inspect's is_vector): drawn at full size, with no AI


@router.post("/estimate")
def estimate_job(body: EstimateRequest, viewer: Member) -> dict:
    """What a picture of this size will give and cost with these options, before it is sent. For a
    cloud mode, cost_usd is the service's price for this one picture (None when it comes out of a
    plan's credits); the page multiplies it by the number of pictures."""
    if not (0 < body.width <= MAX_ESTIMATE_SIDE and 0 < body.height <= MAX_ESTIMATE_SIDE):
        raise HTTPException(400, "That picture size doesn't look right.")
    if body.is_vector is not None and not isinstance(body.is_vector, bool):
        raise HTTPException(400, BAD_REQUEST)
    vector = bool(body.is_vector)
    raw = body.model_dump(exclude={"width", "height", "is_vector"}, exclude_none=True)
    options = image_options(raw, load_settings())
    out_dir = config.host_outputs_dir() if viewer.is_host else config.guest_outputs_dir()
    try:
        result = estimate(
            body.width,
            body.height,
            options["size"],
            options["mode"],
            estimate_format(options["output_format"]),
            detect_hardware(),
            dpi=options["dpi"],
            out_dir=out_dir,
            level=options["photoreal_level"],
            vector=vector,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if problem := output_size_problem(result["out"], options["size"]):
        result["warnings"].insert(0, problem)
    elif problem := _format_problem(options["output_format"], result["out"]):
        result["warnings"].insert(0, problem)
    if options["mode"].startswith(CLOUD_PREFIX):
        result["cost_usd"] = cloud_cost_usd(options["mode"], (body.width, body.height), tuple(result["out"]), vector)
    return result


def cloud_cost_usd(mode: str, src: tuple[int, int], out: tuple[int, int], vector: bool = False) -> float | None:
    """About what the cloud service charges for one picture, worked out the service's own way from
    the scale the job will ask it for (pipeline_image.cloud_scale: when one pass isn't enough, 16K or a
    small picture, it makes 4x and the Fast model on this computer does the rest, for free). 0 when
    nothing is sent (a vector, or a picture already big enough); None when the price comes out of a
    plan's credits and isn't published."""
    _, provider_id, model_id = mode.split(":")
    if vector or plan_passes(src, out).passes == 0:
        return 0.0
    price = PROVIDERS[provider_id].price_usd(model_id, src, cloud_scale(src, out))
    return None if price is None else round(price, 3)


def _format_problem(fmt: str, out: tuple[int, int]) -> str | None:
    """Why a result this big can't be saved as `fmt` (WebP above 16,383 px), or None."""
    try:
        resolve_format(fmt, None, tuple(out))
    except ValueError as exc:
        return str(exc)
    return None


class SampleRequest(BaseModel):
    sample_id: str = ""
    options: dict = {}


@router.post("/images/sample")
def sample_job(body: SampleRequest, request: Request, viewer: Member) -> dict:
    """A picture job for one of the bundled samples. The server copies the sample into the job, so
    the browser never has to download and upload it."""
    found = _bundled_samples().get(body.sample_id)
    if found is None:
        raise HTTPException(404, "That sample picture isn't here.")
    ctx = _context(request)
    settings = load_settings()
    options = image_options(body.options, settings)
    key = _cloud_key_for(ctx, viewer, options["mode"], settings)
    _check_room_for_results(viewer, options, _check_pictures([found[1]]))
    job_id = ctx.queue.new_job_id()
    with _upload_folder(ctx.queue.job_dir(job_id)) as folder:
        target = folder / "in" / "0" / found[1].name
        target.parent.mkdir(parents=True)
        shutil.copyfile(found[1], target)
        job = ctx.queue.submit("image", viewer.owner, [str(target)], options, job_id=job_id, secret=key)
    return {"job": job_view(job, viewer)}


# --- Checking pictures on drop (staging) ----------------------------------------------------

_STAGING_TOKEN = re.compile(r"[A-Za-z0-9_-]{32}")
_STAGED_RECORD = "info.json"
_STAGED_THUMB = "thumb.jpg"
_inspections = threading.BoundedSemaphore(MAX_INSPECTIONS_AT_ONCE)


def staging_dir() -> Path:
    """Where dropped pictures wait for their job (not created until a picture arrives)."""
    return config.jobs_dir() / "staging"


@dataclass(frozen=True)
class _Staged:
    folder: Path
    file: Path
    info: dict | None = None  # inspect_image's answer, kept with the picture (older records have none)


def _staged(token: Any, viewer: Viewer) -> _Staged | None:
    """The picture this viewer staged under `token`; None when there's no such picture, it has
    expired, or it's someone else's (all look the same, so tokens can't be probed)."""
    if not isinstance(token, str) or not _STAGING_TOKEN.fullmatch(token):
        return None
    folder = staging_dir() / token
    try:
        record = json.loads((folder / _STAGED_RECORD).read_text(encoding="utf-8"))
        name, owner, created = record["file"], record["owner"], float(record["created"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(name, str) or Path(name).name != name or owner != viewer.owner:
        return None
    file = folder / "in" / "0" / name
    if time.time() - created > STAGING_SECONDS or not file.is_file():
        return None
    info = record.get("info")
    return _Staged(folder, file, info if isinstance(info, dict) else None)


def _staged_info(picture: _Staged) -> ImageInfo | None:
    """What a staged picture is, as /api/inspect found it (looked at again for older records)."""
    if picture.info is not None:
        try:
            return ImageInfo(**picture.info)
        except TypeError:
            pass
    try:
        return inspect_image(picture.file)
    except Exception:  # its job explains any trouble with it
        return None


def tidy_staging(folder: Path, max_age_seconds: float, now: float | None = None) -> int:
    """Delete staged pictures older than `max_age_seconds`, each with its thumbnail. Returns how many."""
    if not folder.is_dir():
        return 0
    now = time.time() if now is None else now
    removed = 0
    for entry in folder.iterdir():
        try:
            created = float(json.loads((entry / _STAGED_RECORD).read_text(encoding="utf-8"))["created"])
        except (OSError, ValueError, KeyError, TypeError):
            try:  # no record: an upload in progress (recent) or one a crash interrupted (old)
                created = entry.lstat().st_mtime
            except OSError:
                continue
        if now - created <= max_age_seconds:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
        removed += 1
    return removed


@router.post("/inspect")
async def inspect_upload(request: Request, viewer: Member) -> dict:
    """Check one dropped picture (multipart `file`) and keep it for the job that follows: its size,
    format and warnings, a thumbnail (browsers can't show HEIC, RAW or TIFF) and a suggested mode."""
    token = secrets.token_urlsafe(24)  # 32 characters
    declared = _declared_length(request.headers)
    _check_space(config.jobs_dir(), declared * PICTURE_SPACE_FACTOR + SPACE_RESERVE_BYTES, "this picture")
    ctx = _context(request)
    uploaded = await _check_guest_quota(ctx, viewer, 0)
    with _upload_folder(staging_dir() / token) as folder:
        upload = await _receive_upload(
            request,
            folder / "in",
            INSPECT_FIELDS,
            1,
            MAX_IMAGE_BYTES,
            too_many="One picture at a time, please.",
            text_fields=frozenset(),
            what="this picture",
        )
        if not upload.files:
            raise HTTPException(400, NO_PICTURE)
        path = upload.files[0]
        await _check_guest_quota(ctx, viewer, path.stat().st_size, uploaded)
        try:
            info, suggestion = await run_in_threadpool(_inspect_staged, path, folder / _STAGED_THUMB)
        except ImageFileError as exc:
            raise HTTPException(400, str(exc)) from None
        record = {"owner": viewer.owner, "created": time.time(), "file": path.name, "info": asdict(info)}
        (folder / _STAGED_RECORD).write_text(json.dumps(record), encoding="utf-8")
    return {
        "token": token,
        "name": path.name,
        "width": info.width,
        "height": info.height,
        "format": info.format,
        "has_alpha": info.has_alpha,
        "frames": info.frames,
        "is_vector": info.is_vector,
        "warnings": list(info.warnings),
        "thumb_url": f"/api/staging/{token}/thumb.jpg",
        "suggestion": suggestion,
    }


async def _check_guest_quota(ctx: _Context, viewer: Viewer, adding: int, uploaded: int | None = None) -> int:
    """Refuse (507) a guest's upload of `adding` bytes that would take their uploads past
    GUEST_UPLOAD_QUOTA_BYTES. `uploaded` is what they already had, from an earlier call for the same
    upload (measured again when None). Returns that figure. The host's own disk is theirs to fill."""
    if viewer.is_host:
        return 0
    if uploaded is None:
        uploaded = await run_in_threadpool(_guest_upload_bytes, ctx.queue, viewer.owner)
    if uploaded + adding > GUEST_UPLOAD_QUOTA_BYTES or uploaded >= GUEST_UPLOAD_QUOTA_BYTES:
        raise HTTPException(507, STAGING_FULL)
    return uploaded


def _guest_upload_bytes(queue: JobQueue, owner: str) -> int:
    """How much of the disk `owner`'s uploads take up: the pictures they have waiting (those under 24
    hours old) and the uploads their jobs keep (until the job is forgotten). A file counts once: a
    waiting picture and its job's copy are usually the same file (a hard link)."""
    folders = []
    staging = staging_dir()
    for entry in staging.iterdir() if staging.is_dir() else ():
        try:
            record = json.loads((entry / _STAGED_RECORD).read_text(encoding="utf-8"))
            if record["owner"] == owner and time.time() - float(record["created"]) <= STAGING_SECONDS:
                folders.append(entry / "in")
        except (OSError, ValueError, KeyError, TypeError):
            continue  # an upload still arriving, or one that's being tidied away
    folders += [queue.job_dir(job.id) / "in" for job in queue.list() if job.owner == owner]
    seen: set[tuple[int, int]] = set()
    total = 0
    for folder in folders:
        for path in folder.rglob("*"):
            try:
                found = path.stat()
            except OSError:
                continue  # tidied away meanwhile
            if path.is_file() and (found.st_dev, found.st_ino) not in seen:
                seen.add((found.st_dev, found.st_ino))
                total += found.st_size
    return total


def _inspect_staged(path: Path, thumb: Path) -> tuple[ImageInfo, dict | None]:
    with _inspections:
        info = inspect_image(path)
        thumb.write_bytes(make_thumbnail(path))
        return info, _suggestion(path, info)


def _suggestion(path: Path, info: ImageInfo) -> dict | None:
    """The suggested mode and its reasons; None if the picture couldn't be analysed, which never
    stops the picture itself from being used."""
    from upscaler import suggest  # OpenCV; loaded when the first picture is checked

    try:
        if info.is_vector:  # drawn at full size whatever the mode, so its pixels needn't be measured
            signals = suggest.Signals(
                faces=0,
                face_area=0.0,
                mesh_score=0.0,
                illustration_score=0.0,
                text_score=0.0,
                sharpness=0.0,
                blockiness=1.0,
                long_edge=max(info.width, info.height),
                is_vector=True,
            )
        else:
            signals = suggest.analyze(load_image(path).rgb)
        chosen = suggest.suggest(info, signals)
    except Exception:
        log.warning("Couldn't suggest a mode for %s", path.name, exc_info=True)
        return None
    return {
        "mode": chosen.mode,
        "photoreal_level": chosen.photoreal_level,
        "reasons": list(chosen.reasons),
        "faces": int(chosen.signals.faces),  # for the "Has people" filter, whatever drove the suggestion
    }


@router.get("/staging/{token}/thumb.jpg")
def staged_thumbnail(token: str, viewer: Member) -> FileResponse:
    staged = _staged(token, viewer)
    thumb = staged.folder / _STAGED_THUMB if staged else None
    if thumb is None or not thumb.is_file():
        raise HTTPException(404, STAGED_GONE)
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


# --- Preview styles ---------------------------------------------------------------------------


class PreviewRequest(BaseModel):
    token: Any = None
    sample_id: Any = None
    routes: Any = None  # None: every style this computer can run
    region: Any = None  # [x, y, width, height] in the picture; None: its most detailed spot
    size: Any = None  # the size the whole picture would go to, for the time estimates
    creativity: Any = None


@router.post("/preview")
def start_preview(body: PreviewRequest, request: Request, viewer: Member) -> dict:
    """Start a "Preview styles" job: one crop of a staged picture or a sample through each style. It
    goes ahead of waiting jobs, though never interrupts a running one."""
    options = preview_options(body)
    if (body.token is None) == (body.sample_id is None):
        raise HTTPException(400, BAD_REQUEST)
    if body.token is not None:
        staged = _staged(body.token, viewer)
        if staged is None:
            raise HTTPException(400, STAGED_GONE)
        source, copy = staged.file, _link_into
    else:
        found = _bundled_samples().get(body.sample_id) if isinstance(body.sample_id, str) else None
        if found is None:
            raise HTTPException(404, "That sample picture isn't here.")
        source, copy = found[1], _copy_into  # never a link into the app's own files
    ctx = _context(request)
    job_id = ctx.queue.new_job_id()
    with _upload_folder(ctx.queue.job_dir(job_id)) as folder:
        target = copy(source, folder / "in" / "0")
        job = ctx.queue.submit("preview", viewer.owner, [str(target)], options, job_id=job_id)
    return {"job": job_view(job, viewer)}


def preview_options(body: PreviewRequest) -> dict:
    routes = body.routes
    if routes is not None:
        known = isinstance(routes, list) and all(isinstance(r, str) and r in UPSCALE_ROUTES for r in routes)
        if not known or not routes:
            raise HTTPException(400, "Those styles can't be previewed.")
        routes = list(dict.fromkeys(routes))  # each style once, in the order asked
    region = body.region
    if region is not None and not (
        isinstance(region, list)
        and len(region) == 4
        and all(isinstance(v, int) and not isinstance(v, bool) and abs(v) <= MAX_ESTIMATE_SIDE for v in region)
        and region[2] > 0
        and region[3] > 0
    ):
        raise HTTPException(400, "That spot on the picture wasn't understood. Click on the picture again.")
    size = "8k" if body.size is None else body.size
    if not isinstance(size, str) or size not in PRESETS:
        raise HTTPException(400, "That isn't one of the app's sizes.")
    creativity = _fraction(0.0 if body.creativity is None else body.creativity, "Creativity")
    return {"routes": routes, "region": region, "size": size, "creativity": creativity}


def _copy_into(source: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / source.name
    shutil.copyfile(source, target)
    return target


# --- Jobs and results ----------------------------------------------------------------------

# What each result shows: pictures' sizes and notes, and preview styles' route, label and timings.
_PUBLIC_RESULT_KEYS = (
    "filename", "width", "height", "source_width", "source_height", "mode_used", "notes", "seconds", "dpi",
    "print", "file_bytes", "route", "label", "est_full_seconds", "error",
)  # fmt: skip


def job_view(job: Job, viewer: Viewer) -> dict:
    """A job as this viewer may see it: everything for the host and the job's owner; for anyone
    else only that a job is waiting or running, so the queue makes sense to them.

    Never a guest's cookie or key: a guest's owner id is derived from their cookie but can't be
    turned back into it, and other guests only see "guest"."""
    mine = job.owner == viewer.owner
    view: dict[str, Any] = {
        "id": job.id,
        "kind": job.kind,
        "owner": job.owner if viewer.is_host or mine or job.owner == HOST else GUEST_PREFIX.rstrip(":"),
        "status": job.status,
        "progress": job.progress,
        "eta_seconds": job.eta_seconds,
        "created": job.created,
        "mine": mine,
    }
    if viewer.is_host or mine:
        view.update(
            message=job.message,
            inputs=[Path(path).name for path in job.inputs],
            options=job.options,
            results=[_public_result(job, n, result) for n, result in enumerate(job.results)],
            error=job.error,
            retryable=job.retryable,  # failed but kept its work: POST .../retry carries on from it
        )
    else:
        view.update(message="", inputs=[], options={}, results=[], error=None, retryable=False)
    return view


def _public_result(job: Job, n: int, result: dict) -> dict:
    """A result without file system paths. Preview crops also say where to fetch them (None for a
    style that couldn't run)."""
    shown = {key: result[key] for key in _PUBLIC_RESULT_KEYS if key in result}
    if job.kind == "preview":
        shown["url"] = f"/api/jobs/{job.id}/results/{n}/file" if result.get("path") else None
    return shown


@router.get("/jobs")
def list_jobs(request: Request, viewer: Member) -> dict:
    return {"jobs": [job_view(job, viewer) for job in _context(request).queue.list()]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, viewer: Member) -> dict:
    job = _context(request).queue.get(job_id)
    if job is None:
        raise HTTPException(404, "That job isn't here any more.")
    return {"job": job_view(job, viewer)}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, viewer: Member) -> dict:
    queue = _context(request).queue
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "That job isn't here any more.")
    if not viewer.is_host and job.owner != viewer.owner:
        raise HTTPException(403, "You can only stop your own jobs.")
    ok = queue.cancel(job_id, viewer.owner)
    return {"ok": ok, "job": job_view(queue.get(job_id) or job, viewer)}


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request, viewer: Member) -> dict:
    """Carry on with a job that failed but kept its work (a video that ran out of disk space), from
    where it stopped. 409 when there's nothing to carry on from."""
    queue = _context(request).queue
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(404, "That job isn't here any more.")
    if not viewer.is_host and job.owner != viewer.owner:
        raise HTTPException(403, "You can only retry your own jobs.")
    if not queue.retry(job_id, viewer.owner):
        raise HTTPException(409, "This job can't carry on from where it stopped. Please start it again.")
    return {"ok": True, "job": job_view(queue.get(job_id) or job, viewer)}


def _readable_job(request: Request, job_id: str, viewer: Viewer) -> Job:
    job = _context(request).queue.get(job_id)
    if job is None:
        raise HTTPException(404, "That job isn't here any more.")
    if not viewer.is_host and job.owner != viewer.owner:
        raise HTTPException(403, NOT_YOURS)
    return job


def _result(job: Job, n: int) -> dict:
    if not 0 <= n < len(job.results):
        raise HTTPException(404, "That result doesn't exist.")
    return job.results[n]


def _servable(request: Request, job: Job, value: Any) -> Path:
    """A result file, but only from the job's own folder or the app's output folders."""
    if not isinstance(value, str) or not value:
        raise HTTPException(404, FILE_GONE)
    roots = (_context(request).queue.job_dir(job.id), config.host_outputs_dir(), config.guest_outputs_dir())
    path = Path(value).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in roots) or not path.is_file():
        raise HTTPException(404, FILE_GONE)
    return path


@router.get("/jobs/{job_id}/results/{n}/file")
def result_file(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    job = _readable_job(request, job_id, viewer)
    result = _result(job, n)
    path = _servable(request, job, result.get("path"))
    return FileResponse(path, filename=Path(result.get("filename") or path.name).name)


def _picture(request: Request, job_id: str, n: int, viewer: Viewer, which: str) -> FileResponse:
    job = _readable_job(request, job_id, viewer)
    path = _servable(request, job, _result(job, n).get(which))
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/jobs/{job_id}/results/{n}/preview.jpg")
def result_preview(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    return _picture(request, job_id, n, viewer, "preview")


@router.get("/jobs/{job_id}/results/{n}/before.jpg")
def result_before(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    return _picture(request, job_id, n, viewer, "before")


@router.get("/jobs/{job_id}/results/{n}/after.jpg")
def result_after(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    return _picture(request, job_id, n, viewer, "after")


# --- The compare viewer: zoomable tiles of a finished picture ------------------------------

NOT_ZOOMABLE = "This result can't be opened in the zoom viewer."
ORIGINAL_NOT_KEPT = "The original picture isn't kept for this result any more, so only the upscaled one can be shown."
NOT_A_VIDEO = "Only videos can be compared side by side."
COMPARISON_NOT_READY = "The comparison isn't ready yet."
TILE_CACHE = "private, max-age=604800, immutable"  # a result's tiles never change
ASK_AGAIN = "private, no-cache"
BEFORE_SOURCES_KEPT = 4  # decoded originals kept for the Before tiles...
BEFORE_PIXELS_KEPT = 100_000_000  # ...up to about this many pixels together (about 400 MB)
PYRAMID_RETRY_SECONDS = 300  # a result whose tiles couldn't be made isn't tried again for this long


@dataclass
class _Source:
    image: Image.Image  # upright RGB, see-through areas on white
    icc: bytes | None

    @property
    def pixels(self) -> int:
        return self.image.width * self.image.height


class _SourceCache:
    """Decoded originals for the viewer's Before tiles. Opening a picture takes a moment (a camera
    RAW, seconds) and a screenful of tiles comes from the same one, so the last few are kept. Each
    is opened once, however many tiles ask for it at the same time."""

    def __init__(self, max_items: int = BEFORE_SOURCES_KEPT, max_pixels: int = BEFORE_PIXELS_KEPT):
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, _Source] = OrderedDict()
        self._opening: dict[tuple, threading.Lock] = {}
        self._max_items = max_items
        self._max_pixels = max_pixels

    def get(self, path: Path) -> _Source:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            opening = self._opening.setdefault(key, threading.Lock())
        with opening:
            with self._lock:
                if key in self._items:
                    return self._items[key]
            loaded = load_image(path)
            source = _Source(Image.fromarray(on_white(loaded.rgb, loaded.alpha)), loaded.icc)
            del loaded
            with self._lock:
                self._items[key] = source
                self._opening.pop(key, None)
                while len(self._items) > 1 and (
                    len(self._items) > self._max_items or sum(s.pixels for s in self._items.values()) > self._max_pixels
                ):
                    self._items.popitem(last=False)
        return source


_before_sources = _SourceCache()
_pyramid_guard = threading.Lock()
_pyramid_locks: dict[Path, threading.Lock] = {}
_pyramid_failed: dict[Path, float] = {}  # folder -> when its tiles last couldn't be made
_lazy_builds = threading.Semaphore(1)  # older results' tiles are made one at a time (a 16K picture is 0.6 GB)
_manifests: OrderedDict[Path, dict] = OrderedDict()
_MANIFESTS_KEPT = 256


def _zoomable(request: Request, job_id: str, n: int, viewer: Viewer) -> tuple[Job, dict]:
    job = _readable_job(request, job_id, viewer)
    result = _result(job, n)
    if job.kind != "image":
        raise HTTPException(404, NOT_ZOOMABLE)
    return job, result


def _job_folder(request: Request, job: Job) -> Path:
    return _context(request).queue.job_dir(job.id).resolve()


def _viewer_folder(request: Request, job: Job, n: int, result: dict) -> Path:
    """The result's tile pyramid: where the job wrote it, or where it is made for an older result."""
    job_dir = _job_folder(request, job)
    stored = result.get("viewer")
    if isinstance(stored, str) and stored:
        folder = Path(stored).resolve()
        if folder.is_relative_to(job_dir):  # never anywhere else, whatever the record says
            return folder
    return job_dir / "results" / f"{n}-viewer"


def _manifest(folder: Path) -> dict | None:
    with _pyramid_guard:
        if folder in _manifests:
            return _manifests[folder]
    manifest = read_viewer_manifest(folder)
    if manifest is not None:
        with _pyramid_guard:
            _manifests[folder] = manifest
            while len(_manifests) > _MANIFESTS_KEPT:
                _manifests.popitem(last=False)
    return manifest


def _pyramid(request: Request, job: Job, n: int, result: dict) -> tuple[Path, dict] | None:
    """The result's tile pyramid and its viewer.json. For a result made before the viewer existed,
    it is made now, once, from the saved picture. None when it can't be (the picture is gone)."""
    folder = _viewer_folder(request, job, n, result)
    manifest = _manifest(folder)
    if manifest is not None:
        return folder, manifest
    with _pyramid_guard:
        lock = _pyramid_locks.setdefault(folder, threading.Lock())
    with lock:
        manifest = _manifest(folder)
        failed = _pyramid_failed.get(folder)
        if manifest is None and (failed is None or time.monotonic() - failed > PYRAMID_RETRY_SECONDS):
            with _lazy_builds:
                manifest = _build_pyramid(request, job, result, folder)
            if manifest is None:
                _pyramid_failed[folder] = time.monotonic()
            else:
                _pyramid_failed.pop(folder, None)
    return (folder, manifest) if manifest is not None else None


def _build_pyramid(request: Request, job: Job, result: dict, folder: Path) -> dict | None:
    try:
        path = _servable(request, job, result.get("path"))
    except HTTPException:
        return None
    try:
        loaded = load_image(path)
        source = (result.get("source_width"), result.get("source_height"))
        known = all(isinstance(side, int) and side > 0 for side in source)
        manifest = write_tile_pyramid(
            on_white(loaded.rgb, loaded.alpha), folder, icc=loaded.icc, source_size=source if known else None
        )
    except Exception:
        log.warning("Couldn't make the viewer's tiles for job %s", job.id, exc_info=True)
        return None
    with _pyramid_guard:
        _manifests[folder] = manifest
    return manifest


def _pyramid_or_404(request: Request, job: Job, n: int, result: dict) -> tuple[Path, dict]:
    found = _pyramid(request, job, n, result)
    if found is None:
        raise HTTPException(404, FILE_GONE)
    return found


def _level(manifest: dict, level: int, col: int, row: int) -> dict:
    levels = manifest["levels"]
    if not 0 <= level < len(levels) or tile_bounds(levels[level], col, row, manifest["tile_size"]) is None:
        raise HTTPException(404, "That part of the picture doesn't exist.")
    return levels[level]


def _original_picture(request: Request, job: Job, result: dict) -> Path | None:
    """The job's original of this result, while it's kept (only ever from the job's own folder)."""
    index = result.get("input")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(job.inputs):
        return None
    path = Path(job.inputs[index]).resolve()
    if not path.is_relative_to(_job_folder(request, job)) or not path.is_file():
        return None
    return path


@router.get("/jobs/{job_id}/results/{n}/viewer.json")
def result_viewer(job_id: str, n: int, request: Request, viewer: Member) -> JSONResponse:
    """The result's tile pyramid ({width, height, tile_size, levels, source_width, source_height})
    and whether its Before side can be shown; {unavailable: true} when there's no picture to zoom."""
    job, result = _zoomable(request, job_id, n, viewer)
    found = _pyramid(request, job, n, result)
    if found is None:
        return JSONResponse({"unavailable": True}, headers={"Cache-Control": ASK_AGAIN})
    _, manifest = found
    body = {**manifest, "before": _original_picture(request, job, result) is not None}
    return JSONResponse(body, headers={"Cache-Control": ASK_AGAIN})


@router.get("/jobs/{job_id}/results/{n}/tiles/{level:int}/{col:int}_{row:int}.jpg")
def result_tile(job_id: str, n: int, level: int, col: int, row: int, request: Request, viewer: Member) -> FileResponse:
    """One tile of the upscaled picture."""
    job, result = _zoomable(request, job_id, n, viewer)
    folder, manifest = _pyramid_or_404(request, job, n, result)
    _level(manifest, level, col, row)
    path = folder / str(level) / f"{col}_{row}.jpg"
    if not path.is_file():
        raise HTTPException(404, FILE_GONE)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": TILE_CACHE})


@router.get("/jobs/{job_id}/results/{n}/before/{level:int}/{col:int}_{row:int}.jpg")
def result_before_tile(
    job_id: str, n: int, level: int, col: int, row: int, request: Request, viewer: Member
) -> Response:
    """The same tile of the original, simply enlarged (Lanczos), made from the job's original."""
    job, result = _zoomable(request, job_id, n, viewer)
    _, manifest = _pyramid_or_404(request, job, n, result)
    spec = _level(manifest, level, col, row)
    original = _original_picture(request, job, result)
    if original is None:
        raise HTTPException(404, ORIGINAL_NOT_KEPT)
    try:
        source = _before_sources.get(original)
    except (ImageFileError, OSError):
        raise HTTPException(404, ORIGINAL_NOT_KEPT) from None
    tile = before_tile(source.image, spec, col, row, manifest["tile_size"])
    return Response(tile_jpeg(tile, source.icc), media_type="image/jpeg", headers={"Cache-Control": TILE_CACHE})


BROWSER_PICTURES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})  # every browser shows these in a tab


@router.get("/jobs/{job_id}/results/{n}/view")
def result_view(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    """The finished picture to look at in a browser tab ("Open in new tab"): the file itself, at full
    size, when browsers can show its format; otherwise (TIFF, HEIC, AVIF) the screen-sized preview."""
    job, result = _zoomable(request, job_id, n, viewer)
    try:
        path = _servable(request, job, result.get("path"))
    except HTTPException:
        path = None
    if path is not None and path.suffix.lower() in BROWSER_PICTURES:
        name = Path(result.get("filename") or path.name).name
        return FileResponse(
            path, filename=name, content_disposition_type="inline", headers={"Cache-Control": ASK_AGAIN}
        )
    return _picture(request, job_id, n, viewer, "preview")


# --- Side-by-side video comparison ------------------------------------------------------------


def _video_result(request: Request, job_id: str, n: int, viewer: Viewer) -> tuple[Job, dict, Path]:
    job = _readable_job(request, job_id, viewer)
    result = _result(job, n)
    if job.kind != "video":
        raise HTTPException(404, NOT_A_VIDEO)
    return job, result, video_compare.compare_folder(_job_folder(request, job), n)


def _original_video(request: Request, job: Job) -> Path | None:
    if not job.inputs:
        return None
    path = Path(job.inputs[0]).resolve()
    if not path.is_relative_to(_job_folder(request, job)) or not path.is_file():
        return None
    return path


def _comparison(job: Job, n: int, state: dict) -> dict:
    view = dict(state)
    if state.get("status") == video_compare.READY:
        base = f"/api/jobs/{job.id}/results/{n}/compare"
        view.update(original_url=f"{base}/original.mp4", upscaled_url=f"{base}/upscaled.mp4")
    return view


@router.post("/jobs/{job_id}/results/{n}/compare")
def start_video_compare(job_id: str, n: int, request: Request, viewer: Member) -> dict:
    """Get the side-by-side comparison ready (asking again while it's being made is fine):
    {status: idle|waiting|preparing|ready|failed|cancelled, progress, message, error?}, plus the
    sizes, frame rate, length and the two copies' addresses once it's ready."""
    job, result, folder = _video_result(request, job_id, n, viewer)
    state = video_compare.COMPARISONS.status(folder)
    if state["status"] in (video_compare.READY, video_compare.WAITING, video_compare.PREPARING):
        return _comparison(job, n, state)
    try:
        upscaled = _servable(request, job, result.get("path"))
    except HTTPException:
        raise HTTPException(404, video_compare.RESULT_GONE) from None
    state = video_compare.COMPARISONS.start(folder, _original_video(request, job), upscaled)
    return _comparison(job, n, state)


@router.get("/jobs/{job_id}/results/{n}/compare/status")
def video_compare_status(job_id: str, n: int, request: Request, viewer: Member) -> dict:
    job, _, folder = _video_result(request, job_id, n, viewer)
    return _comparison(job, n, video_compare.COMPARISONS.status(folder))


@router.post("/jobs/{job_id}/results/{n}/compare/cancel")
def cancel_video_compare(job_id: str, n: int, request: Request, viewer: Member) -> dict:
    job, _, folder = _video_result(request, job_id, n, viewer)
    stopped = video_compare.COMPARISONS.cancel(folder)
    return {"ok": stopped, **_comparison(job, n, video_compare.COMPARISONS.status(folder))}


@router.get("/jobs/{job_id}/results/{n}/poster.jpg")
def video_poster(job_id: str, n: int, request: Request, viewer: Member) -> FileResponse:
    """A still from a finished video, for its card (made the first time it's asked for). The card
    plays the comparison's browser-friendly copy, never the 10-bit HEVC result itself."""
    job, result, folder = _video_result(request, job_id, n, viewer)
    try:
        upscaled = _servable(request, job, result.get("path"))
    except HTTPException:
        raise HTTPException(404, video_compare.RESULT_GONE) from None
    try:
        path = video_compare.poster(folder, upscaled)
    except VideoError as exc:
        raise HTTPException(404, str(exc)) from None
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/jobs/{job_id}/results/{n}/compare/{which}.mp4")
def video_compare_file(job_id: str, n: int, which: str, request: Request, viewer: Member) -> FileResponse:
    """One of the comparison's copies (the browser asks for parts of it as it plays)."""
    if which not in (video_compare.ORIGINAL, video_compare.UPSCALED):
        raise HTTPException(404, FILE_GONE)
    _, _, folder = _video_result(request, job_id, n, viewer)
    if video_compare.ready_info(folder) is None:
        raise HTTPException(404, COMPARISON_NOT_READY)
    path = video_compare.copy_path(folder, which)
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": ASK_AGAIN})


@router.get("/jobs/{job_id}/zip")
def results_zip(job_id: str, request: Request, viewer: Member) -> StreamingResponse:
    job = _readable_job(request, job_id, viewer)
    files: list[tuple[str, Path]] = []
    used: set[str] = set()
    for result in job.results:
        try:
            path = _servable(request, job, result.get("path"))
        except HTTPException:
            continue  # e.g. a guest's file that was already tidied away
        files.append((_unique_name(Path(result.get("filename") or path.name).name, used), path))
    if not files:
        raise HTTPException(404, FILE_GONE)
    headers = {"Content-Disposition": f'attachment; filename="upscaled-{job.id}.zip"'}
    return StreamingResponse(_zip_stream(files), media_type="application/zip", headers=headers)


def _unique_name(name: str, used: set[str]) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 2
    while candidate in used:
        candidate, n = f"{stem}-{n}{suffix}", n + 1
    used.add(candidate)
    return candidate


class _ZipSink:
    """Collects what zipfile writes, so the archive can be sent while it is being made."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


def _zip_stream(files: list[tuple[str, Path]]) -> Iterator[bytes]:
    sink = _ZipSink()
    # Pictures and videos are already compressed, so they are stored as they are.
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as archive:  # type: ignore[arg-type]
        for name, path in files:
            info = zipfile.ZipInfo.from_file(path, name)
            with path.open("rb") as source, archive.open(info, "w", force_zip64=info.file_size > 2**31) as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
                    yield sink.take()
            yield sink.take()
    yield sink.take()


# --- Live updates --------------------------------------------------------------------------


@router.get("/events")
async def events(request: Request, viewer: Member) -> StreamingResponse:
    ctx = _context(request)
    queue = ctx.queue
    via_wifi = bool(request.scope.get(VIA_WIFI))
    subscription = queue.subscribe()

    def still_open() -> bool:
        # A guest's stream through the Wi-Fi listener ends as soon as sharing stops, so the
        # listener can close straight away.
        return not queue.stopping and not (via_wifi and not ctx.sharing.enabled)

    async def stream() -> AsyncIterator[str]:
        try:
            yield "retry: 3000\n\n"
            quiet_since = time.monotonic()
            while still_open():
                try:
                    event = subscription.get_nowait()
                except queue_module.Empty:
                    if await request.is_disconnected():
                        break
                    if time.monotonic() - quiet_since > KEEPALIVE_SECONDS:
                        yield ": still here\n\n"
                        quiet_since = time.monotonic()
                    await asyncio.sleep(EVENT_POLL_SECONDS)
                    continue
                yield _server_sent_event(event, viewer)
                quiet_since = time.monotonic()
        finally:
            queue.unsubscribe(subscription)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


def _server_sent_event(event: dict, viewer: Viewer) -> str:
    data = job_view(event["job"], viewer) if event["event"] == "job" else {}
    return f"event: {event['event']}\ndata: {json.dumps(data)}\n\n"


# --- Photo-real download ---------------------------------------------------------------------


_download_lock = threading.Lock()


@router.post("/photoreal/download")
def start_photoreal_download(request: Request, viewer: Member) -> dict:
    """Start the one-time Photo-real download (about 7 GB of the host's disk and internet): the
    host's decision, like installing anything else."""
    if not viewer.is_host:
        raise HTTPException(403, PHOTOREAL_HOST_ONLY)
    photoreal = _photoreal_status(detect_hardware())
    if not photoreal["supported"]:
        raise HTTPException(400, f"{photoreal['reason']}, so Photo-real can't be used here.")
    queue = _context(request).queue
    with _download_lock:  # two clicks (or two people) still start just one download
        job = next((j for j in queue.list() if j.kind == "download" and j.active), None)
        if job is None:
            job = queue.submit("download", viewer.owner, [], {})
    return {"job": job_view(job, viewer)}


# --- Cloud services ----------------------------------------------------------------------------


class KeyRequest(BaseModel):
    key: str = ""


def _provider(provider_id: str) -> CloudProvider:
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise HTTPException(404, f"{PRODUCT_NAME} doesn't know that cloud service.")
    return provider


@router.get("/providers/{provider_id}/guide")
def provider_guide(provider_id: str, viewer: Member) -> dict:
    _provider(provider_id)
    guide = GUIDES.get(provider_id)
    if guide is None:
        raise HTTPException(404, "There's no guide for that service yet.")
    return asdict(guide)


def _checked_key(provider: CloudProvider, key: str) -> str:
    """Ask the service whether `key` works (never spending money). A key it turns down, or one that
    can't be checked, is refused with a friendly 400 and kept nowhere. Returns the service's message."""
    try:
        ok, message = provider.check_key(key)
    except Exception:  # a key check must never look like a crash
        log.exception("Checking a %s key failed", provider.id)
        raise HTTPException(
            400, f"Couldn't check this key with {provider.name} right now. Check the internet connection and try again."
        ) from None
    if not ok:
        raise HTTPException(400, message or f"{provider.name} didn't accept this key.")
    return message


@router.post("/providers/{provider_id}/key")
def save_provider_key(provider_id: str, body: KeyRequest, request: Request, viewer: HostOnly) -> dict:
    provider = _provider(provider_id)
    message = _checked_key(provider, body.key)
    try:
        keys.set_key(provider_id, body.key)
    except CloudError as exc:
        raise HTTPException(500, exc.user_message) from None
    ctx = _context(request)
    ctx.remember_host_key(provider_id, True)
    ctx.queue.publish_status()
    return {"ok": True, "message": message}


@router.delete("/providers/{provider_id}/key")
def delete_provider_key(provider_id: str, request: Request, viewer: HostOnly) -> dict:
    _provider(provider_id)
    try:
        keys.delete_key(provider_id)
    except CloudError as exc:
        raise HTTPException(500, exc.user_message) from None
    ctx = _context(request)
    ctx.remember_host_key(provider_id, False)
    ctx.queue.publish_status()
    return {"ok": True}


@router.post("/providers/{provider_id}/session-key")
def save_session_key(
    provider_id: str, body: KeyRequest, request: Request, response: Response, viewer: Member
) -> dict:
    """A key for this visit only. It stays in the server's memory, is never saved, and is forgotten
    when the browser closes, after SESSION_KEY_IDLE_SECONDS unused, or after SESSION_KEY_MAX_SECONDS."""
    provider = _provider(provider_id)
    message = _checked_key(provider, body.key)
    visit = viewer.visit or secrets.token_urlsafe(32)
    _context(request).keep_session_key(viewer.owner, provider_id, clean_key(body.key), visit)
    if visit != viewer.visit:  # no max_age: the browser forgets it (and so the key) when it closes
        response.set_cookie(VISIT_COOKIE, visit, httponly=True, samesite="lax")
    return {"ok": True, "message": message}


@router.delete("/providers/{provider_id}/session-key")
def forget_session_key(provider_id: str, request: Request, viewer: Member) -> dict:
    """Forget the key this person pasted for their visit, straight away: their jobs that would still
    send pictures with it stop too (`stopped` says how many), so it's charged no more."""
    _provider(provider_id)
    ctx = _context(request)
    ctx.forget_session_key(viewer.owner, provider_id)
    return {"ok": True, "stopped": ctx.queue.stop_own_key_jobs(viewer.owner, provider_id)}


# --- Open folder (host only) -------------------------------------------------------------------


class OpenFolderRequest(BaseModel):
    job_id: str | None = None


@router.post("/open-folder")
def open_folder(request: Request, viewer: HostOnly, body: OpenFolderRequest | None = None) -> dict:
    target, select = config.host_outputs_dir(), False
    job = _context(request).queue.get(body.job_id) if body and body.job_id else None
    if job is not None and job.results:
        try:
            target, select = _servable(request, job, job.results[0].get("path")), True
        except HTTPException:
            pass  # the file was moved or deleted: open the folder instead
    try:
        reveal_in_file_manager(target, select)
    except OSError:
        log.exception("Couldn't open %s", target)
        raise HTTPException(500, "Couldn't open the folder.") from None
    return {"ok": True}


def reveal_in_file_manager(path: Path, select: bool) -> None:
    """Open a folder in Finder or File Explorer, or show a file selected in its folder."""
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if select else ["open", str(path)], **quiet)
    elif sys.platform == "win32":
        if select:
            subprocess.Popen(["explorer", f"/select,{path}"], **quiet)
        else:
            os.startfile(path)  # type: ignore[attr-defined]  # Windows only
    else:
        subprocess.Popen(["xdg-open", str(path.parent if select else path)], **quiet)


# --- Sharing on Wi-Fi ----------------------------------------------------------------------------


class WifiSharing:
    """Lets phones and computers on the Wi-Fi use the app: a second uvicorn server for the same app,
    on every network at `port` (or the next free port), in a thread of its own, so it can be switched
    on and off while the app runs. Whoever comes in through it is a guest (_ViaWifi)."""

    def __init__(self, app: Any, port: int):
        self.app = _ViaWifi(app)
        self.host = WIFI_HOST  # tests point it at 127.0.0.1, so the computer's firewall never asks
        self.preferred_port = port  # tried first every time sharing starts
        self.port = port  # the port in use while sharing is on
        self._lock = threading.Lock()  # one switch at a time
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        thread = self._thread
        return self._server is not None and thread is not None and thread.is_alive()

    def start(self, sock: socket.socket | None = None) -> None:
        """Start sharing on `sock`, or on a new socket for every network at the preferred port or the
        next free one. Does nothing if sharing is already on. Raises OSError when no port is free."""
        with self._lock:
            if self.enabled:
                if sock is not None:
                    sock.close()
                return
            if sock is None:
                sock = share_socket(self.host, self.preferred_port)
            self.port = sock.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(
                    self.app,
                    lifespan="off",  # the app's own listener runs the job queue
                    proxy_headers=False,  # who is asking is the device itself, never what a header claims
                    log_config=None,  # the app's own listener has set up logging already
                    log_level="warning",
                    access_log=False,
                    timeout_graceful_shutdown=WIFI_GRACE_SECONDS,
                )
            )
            thread = threading.Thread(target=self._serve, args=(server, sock), name="upscaler-wifi", daemon=True)
            self._server, self._thread = server, thread
            thread.start()

    def stop(self) -> None:
        """Stop sharing. Guests' pages lose the connection; their jobs carry on."""
        with self._lock:
            server, thread = self._server, self._thread
            self._server = self._thread = None  # from now on `enabled` is False, which ends guests' live updates
            if server is None or thread is None:
                return
            server.should_exit = True
            thread.join(WIFI_STOP_SECONDS)

    @staticmethod
    def _serve(server: uvicorn.Server, sock: socket.socket) -> None:
        try:
            server.run(sockets=[sock])  # outside the main thread, uvicorn leaves Ctrl+C to the app's listener
        except Exception:
            log.exception("Sharing on Wi-Fi stopped unexpectedly")
        finally:
            sock.close()


class _ViaWifi:
    """The app as the Wi-Fi listener serves it: every request is marked as coming from the network."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, VIA_WIFI: True}
        await self.app(scope, receive, send)


def listen_socket(host: str, port: int) -> socket.socket:
    """A socket bound to host:port for uvicorn to serve. Raises OSError when the port is taken."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":  # lets the app restart at once; on Windows this would allow two apps per port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    sock.set_inheritable(True)
    return sock


def share_ports(port: int) -> range:
    """The ports sharing tries, in order: `port`, then the next ones while other programs have them.
    0 means any free port."""
    return range(0, 1) if port == 0 else range(port, min(port + SHARE_PORT_TRIES, 65_536))


def share_socket(host: str, port: int) -> socket.socket:
    """A socket for sharing on Wi-Fi at the first free port of `share_ports(port)`. Raises OSError
    when none is free."""
    error = OSError(errno.EADDRINUSE, f"No free port from {port}")
    for candidate in share_ports(port):
        try:
            return listen_socket(host, candidate)
        except OSError as exc:  # taken, or reserved (Windows reserves whole ranges): try the next one
            error = exc
    raise error


def firewall_note(platform: str = sys.platform) -> str | None:
    """What to do when the computer's firewall asks about sharing, or None where it doesn't ask."""
    return FIREWALL_NOTES.get(platform)


def _share_from_the_start(sharing: WifiSharing, sock: socket.socket | None) -> None:
    """Sharing as the app starts (the Share launchers). The app still starts when it can't share."""
    try:
        sharing.start(sock)
    except OSError:
        log.warning(SHARE_PORT_TAKEN.format(port=sharing.preferred_port))


class ShareChange(BaseModel):
    enabled: StrictBool


@router.post("/share")
def change_sharing(change: ShareChange, request: Request, viewer: HostOnly) -> dict:
    """Switch sharing on Wi-Fi on or off, without a restart."""
    ctx = _context(request)
    sharing = ctx.sharing
    was_on = sharing.enabled
    if change.enabled and not was_on:
        # Looked up afresh, as the computer may only just have joined the Wi-Fi. With no address to
        # give out, nothing starts listening, so the firewall doesn't ask for nothing.
        if not ctx.lan_addresses(fresh=True):
            raise HTTPException(409, NO_NETWORK_TO_SHARE)
        try:
            sharing.start()
        except OSError:
            log.warning("No free port from %s to share on Wi-Fi", sharing.preferred_port, exc_info=True)
            raise HTTPException(409, SHARE_PORT_TAKEN.format(port=sharing.preferred_port)) from None
    elif not change.enabled:
        sharing.stop()
    if sharing.enabled != was_on:
        ctx.queue.publish_status()  # every open page shows the new state
    return ctx.share_status(viewer, load_settings())


@router.get("/share/qr.svg")
def share_qr_code(url: str, request: Request, viewer: HostOnly) -> Response:
    """A QR code for one of the addresses being shared, for a phone's camera."""
    ctx = _context(request)
    if not ctx.sharing.enabled or url not in ctx.lan_urls():
        raise HTTPException(404, NOT_SHARED)
    return Response(qr_svg(url), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


def qr_svg(text: str) -> bytes:
    """`text` as a QR code: black on white with the standard quiet zone, in an SVG that scales to any size."""
    import segno  # loaded only when the Share panel asks for a code

    buffer = io.BytesIO()
    code = segno.make(text, error="m", micro=False)
    code.save(buffer, kind="svg", border=4, omitsize=True, xmldecl=False, dark="#000000", light="#ffffff")
    return buffer.getvalue()


# --- This computer's Wi-Fi addresses -------------------------------------------------------------

_INET = re.compile(r"\binet (?:addr:)?(\d{1,3}(?:\.\d{1,3}){3})")  # ifconfig / ip addr (Mac, Linux)
_IPCONFIG = re.compile(r"IPv4[^:\r\n]*:\s*(\d{1,3}(?:\.\d{1,3}){3})")  # ipconfig (Windows)


def lan_ipv4_addresses() -> list[str]:
    """Every IPv4 address other devices on the network might reach this computer at, the one
    used for the internet first."""
    found: list[str] = []

    def add(ip: str) -> None:
        try:
            address = ipaddress.IPv4Address(ip)
        except ValueError:
            return
        usable = not (address.is_loopback or address.is_link_local or address.is_unspecified or address.is_multicast)
        if usable and ip not in found:
            found.append(ip)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # a documentation address; connecting a UDP socket sends nothing
            add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(str(info[4][0]))
    except OSError:
        pass
    for ip in _interface_addresses():
        add(ip)
    return found


def _interface_addresses() -> list[str]:
    if sys.platform == "win32":
        commands, pattern = [["ipconfig"]], _IPCONFIG
    elif sys.platform == "darwin":
        commands, pattern = [["ifconfig"]], _INET
    else:
        commands, pattern = [["ip", "-4", "-o", "addr", "show"], ["ifconfig"]], _INET
    for command in commands:
        try:
            output = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=3,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if addresses := pattern.findall(output):
            return addresses
    return []


def lan_urls(port: int) -> list[str]:
    return [f"http://{ip}:{port}" for ip in lan_ipv4_addresses()]
