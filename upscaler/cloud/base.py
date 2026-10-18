"""What every cloud provider shares: the model/provider types, friendly errors, and the HTTP
plumbing (retries, polling, cancelling, preparing the upload, reading the result)."""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from PIL import Image

from upscaler.config import Cancelled, ProgressFn
from upscaler.imageio_utils import LoadedImage, load_image

TIMEOUT = httpx.Timeout(30.0)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# The answers that mean the service turned a request away without starting anything. A request that
# starts a paid job is only sent again after one of these: after a 500, 502 or 504 the job may have
# started anyway.
REFUSED_STATUSES = frozenset({429, 503})
RETRY_DELAYS: tuple[float, ...] = (2.0, 6.0)  # three tries in total
POLL_SECONDS = 2.0
MAX_WAIT_SECONDS = 30 * 60
# Once a job has been sent it may be charged for, so losing the connection (or the service being
# busy) doesn't end the wait at once: the app keeps asking for this long in a row before giving up.
LOST_TOUCH_SECONDS = 5 * 60

# Progress milestones inside one cloud upscale.
UPLOADING_AT = 0.05
WAITING_AT = 0.10
DOWNLOADING_AT = 0.88

# A result may be this much bigger than what we asked for (services round sizes their own way),
# and never needs to be refused below RESULT_PIXEL_FLOOR.
RESULT_SIZE_MARGIN = 1.5
RESULT_PIXEL_FLOOR = 100_000_000

# Pictures are sent as PNG unless that would be bigger than this (a big photo): then as a JPEG that
# looks the same at a third of the size. Most services take the picture inline, in the request itself.
PNG_UPLOAD_MAX_BYTES = 10_000_000
UPLOAD_CHUNK_BYTES = 256 * 1024  # how often an upload checks for Cancel and reports progress

UNUSUAL_KEY_MESSAGE = (
    "That doesn't look like a whole key: it has characters keys never contain, like … or a special space. "
    "Copy the key again with the site's Copy button."
)

_LOSSY_FORMATS = frozenset({"JPEG", "HEIC", "AVIF", "WebP"})  # as load_image names them


@dataclass(frozen=True)
class CloudModel:
    id: str
    label: str
    kind: str  # "faithful" | "generative"
    est_cost_8k_usd: float  # 0.0 means "comes out of your plan's credits"; see description
    description: str


class CloudError(Exception):
    """A cloud problem, with a message written for the person using the app."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class JobFailed(CloudError):
    """The service ended the job without a usable picture, so there is nothing left to stop."""


class TemporaryError(CloudError):
    """A problem that may clear up by itself: no connection, or the service busy (429/5xx)."""


# What a provider can upscale: see CloudProvider.upscale.
type Source = LoadedImage | np.ndarray | Path


class CloudProvider(Protocol):
    id: str
    name: str
    models: list[CloudModel]  # the first one is the provider's default

    def check_key(self, key: str) -> tuple[bool, str]:
        """Test the key without spending any money."""
        ...

    def upscale(
        self,
        image: Source,
        model_id: str,
        scale: float,
        key: str,
        *,
        progress: ProgressFn,
        cancel: threading.Event,
    ) -> np.ndarray:
        """Return the upscaled image as RGB uint8 (at the provider's size).

        `image` is the picture as the pipeline opened it (best: nothing is decoded twice), its upright
        RGB uint8 pixels, or its file (decoded exactly as the pipeline decodes it)."""
        ...

    def price_usd(self, model_id: str, size: tuple[int, int], scale: float) -> float | None:
        """About what the service charges, in US dollars, to upscale a picture of `size` (width,
        height) when asked for `scale` (pipeline_image.cloud_scale), worked out the way the service
        prices it: from what it really makes (it may only offer certain factors, or stop at a size) and
        its own steps. None when the price comes out of a plan's credits and isn't published."""
        ...


def megapixels(size: tuple[int, int], factor: float) -> float:
    """Megapixels in a picture of `size` (width, height) enlarged `factor` times."""
    return round(size[0] * factor) * round(size[1] * factor) / 1e6


def started_blocks(megapixels: float, block: float) -> int:
    """How many blocks of `block` megapixels a price "per started block" counts (at least one)."""
    return max(1, math.ceil(megapixels / block - 1e-9))


def find_model(models: list[CloudModel], model_id: str, service: str) -> CloudModel:
    for model in models:
        if model.id == model_id:
            return model
    raise CloudError(f"{service} doesn't have a mode called “{model_id}”.")


# --- Keys ---------------------------------------------------------------------------------


def clean_key(key: str) -> str:
    """Keys never contain spaces, so drop every kind: line breaks, and the non-breaking spaces
    that copying from an email or a note can add."""
    return "".join(key.split())


def _sendable(key: str) -> bool:
    """Only printable ASCII can go in a request header, and every real key is printable ASCII."""
    return key.isascii() and key.isprintable()


def key_complaint(key: str, empty_message: str) -> str | None:
    """Why a cleaned key can't be right, found without asking the service."""
    if not key:
        return empty_message
    if not _sendable(key):
        return UNUSUAL_KEY_MESSAGE
    return None


def header_key(key: str, service: str) -> str:
    """The key, ready to put in a request header. Keys are checked before they're saved, so
    this only refuses one that slipped through some other way."""
    key = clean_key(key)
    if not key or not _sendable(key):
        raise CloudError(f"The {service} key isn't usable. Connect {service} again with a fresh key.")
    return key


def new_client() -> httpx.Client:
    """A client with no default headers: keys are passed per request, so downloading the
    result from a file host never sends the key along."""
    return httpx.Client(timeout=TIMEOUT)


# --- Waiting and cancelling ---------------------------------------------------------------


def raise_if_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def pause(cancel: threading.Event, seconds: float) -> None:
    """Sleep, but wake up straight away (and stop) if the user cancels."""
    cancel.wait(seconds)
    raise_if_cancelled(cancel)


# --- Requests with retries ----------------------------------------------------------------


def send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    service: str,
    cancel: threading.Event | None = None,
    idempotent: bool | None = None,
    **kwargs,
) -> httpx.Response:
    """Make a request, retrying when the service is busy (429/5xx) or unreachable.

    A request that isn't idempotent (by default a POST, which starts a paid job) is only sent again
    when we know it never arrived (the connection failed) or the service turned it away (429/503).
    After a slow answer or a 500/502/504 the job may have started anyway, so the app stops and says so
    rather than start, and pay for, a second one. A connection that never opened raises
    TemporaryError."""
    cancel = cancel or threading.Event()
    if idempotent is None:
        idempotent = method.upper() != "POST"
    for attempt in range(len(RETRY_DELAYS) + 1):
        raise_if_cancelled(cancel)
        last_try = attempt == len(RETRY_DELAYS)
        try:
            response = client.request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if last_try:
                raise TemporaryError(unreachable_message(service)) from exc
        except httpx.TransportError as exc:
            if not idempotent:
                raise CloudError(unknown_outcome_message(service)) from exc
            if last_try:
                raise TemporaryError(unreachable_message(service)) from exc
        else:
            code = response.status_code
            if not idempotent and code in RETRY_STATUSES and code not in REFUSED_STATUSES:
                raise CloudError(unknown_outcome_message(service))
            if code not in RETRY_STATUSES or last_try:
                return response
        pause(cancel, RETRY_DELAYS[attempt])
    raise AssertionError("unreachable")


def unreachable_message(service: str) -> str:
    return f"Couldn't reach {service}. Check your internet connection and try again."


def busy_message(service: str) -> str:
    return f"{service} is busy right now. Please try again in a few minutes."


def check_account(service: str) -> str:
    """The end of a message about a job the service finished, which it has probably charged for."""
    return f"It may still have been charged, so check your {service} account before trying again."


def may_still_finish(service: str) -> str:
    """The end of a message about a job the app stopped waiting for: asking the service to stop it is
    only best effort (and Magnific can't), so it may finish anyway."""
    return f"The picture may still finish and be charged, so check your {service} account before trying again."


def unknown_outcome_message(service: str) -> str:
    return (
        f"{service} had a problem and may or may not have started on this picture. "
        f"Check your {service} account before trying again, so you don't pay twice."
    )


def ok_or_raise(response: httpx.Response, message: Callable[[httpx.Response], str]) -> httpx.Response:
    """The response if it succeeded. Otherwise a CloudError saying `message(response)`: a
    TemporaryError when the service was only busy (429/5xx), so asking again later may work."""
    if response.is_success:
        return response
    busy = response.status_code == 429 or response.status_code >= 500
    raise (TemporaryError if busy else CloudError)(message(response))


def keep_asking[T](request: Callable[[], T], cancel: threading.Event) -> T:
    """`request()`, asked again every POLL_SECONDS while it fails with a TemporaryError, for up to
    LOST_TOUCH_SECONDS. For requests made after a job was sent, when giving up could waste a payment."""
    lost_since = time.monotonic()
    while True:
        try:
            return request()
        except TemporaryError:
            if time.monotonic() - lost_since >= LOST_TOUCH_SECONDS:
                raise
        pause(cancel, POLL_SECONDS)


def error_detail(response: httpx.Response, limit: int = 200) -> str:
    """The service's own short explanation, if it gave one."""
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body
    if isinstance(body, dict):
        detail = next((body[field] for field in ("detail", "message", "error", "title") if body.get(field)), "")
    if isinstance(detail, list) and detail and isinstance(detail[0], dict):
        detail = detail[0].get("msg", "")
    if isinstance(detail, dict):
        detail = detail.get("message", "")
    text = " ".join(str(detail or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def said(detail: str) -> str:
    return f" It said: “{detail}”" if detail else ""


def json_body(response: httpx.Response, service: str) -> dict:
    try:
        body = response.json()
    except ValueError as exc:
        raise CloudError(f"{service} sent back something unexpected. Please try again.") from exc
    if not isinstance(body, dict):
        raise CloudError(f"{service} sent back something unexpected. Please try again.")
    return body


# --- Polling ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pending:
    """Returned by a poll check while the job is still running."""

    message: str


def waiting_fraction(elapsed: float) -> float:
    """We can't know how far along a cloud job is, so ease towards the download milestone."""
    return WAITING_AT + (DOWNLOADING_AT - WAITING_AT) * (1 - math.exp(-elapsed / 60.0)) * 0.95


def poll_until_done[T](
    check: Callable[[], T | Pending],
    *,
    service: str,
    progress: ProgressFn,
    cancel: threading.Event,
    stop_job: Callable[[], None] | None = None,
) -> T:
    """Call `check` every POLL_SECONDS until it returns something other than Pending.

    The job has been sent, and may be charged for, by now. So a status check that fails with a
    TemporaryError (Wi-Fi drops, the service is busy) just means "still waiting", until that has
    lasted LOST_TOUCH_SECONDS in a row. If we stop waiting for any reason (the user cancels, a
    definite error, lost touch for too long, or it takes too long), `stop_job` asks the service to
    stop the job (best effort) so it doesn't keep running, and possibly charging, while the user
    starts another. `check` raises JobFailed when the service has already ended the job."""
    start = time.monotonic()
    lost_since: float | None = None
    try:
        while True:
            raise_if_cancelled(cancel)
            try:
                outcome = check()
            except TemporaryError as exc:
                now = time.monotonic()
                lost_since = now if lost_since is None else lost_since
                if now - lost_since >= LOST_TOUCH_SECONDS:
                    raise CloudError(
                        f"Lost touch with {service} for too long, so the app stopped waiting. "
                        + may_still_finish(service)
                    ) from exc
                outcome = Pending(f"Lost touch with {service} for a moment. Still waiting…")
            else:
                lost_since = None
            if not isinstance(outcome, Pending):
                return outcome
            elapsed = time.monotonic() - start
            if elapsed > MAX_WAIT_SECONDS:
                raise CloudError(
                    f"{service} is taking much longer than usual, so the app stopped waiting. "
                    + may_still_finish(service)
                )
            progress(waiting_fraction(elapsed), outcome.message)
            cancel.wait(POLL_SECONDS)
    except JobFailed:
        raise
    except Exception:
        if stop_job is not None:
            try:
                stop_job()
            except (CloudError, httpx.HTTPError):
                pass  # we stop waiting either way
        raise


# --- Images in and out --------------------------------------------------------------------


@dataclass(frozen=True)
class Upload:
    data: bytes
    mime: str
    width: int
    height: int

    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.base64()}"

    @property
    def filename(self) -> str:
        return "image.jpg" if self.mime == "image/jpeg" else "image.png"


def source_pixels(image: Source) -> tuple[np.ndarray, bool]:
    """The picture as upright 8-bit RGB, and whether its file was already lossy (JPEG, HEIC...).

    A file is opened with the pipeline's own load_image, never plain Pillow: Pillow would see only
    the small preview inside many camera RAW files (and the service would be paid to upscale that),
    and can't open EXR, Radiance HDR or most RAW at all. Upright matters: the app sizes the result
    from the rotated photo, and not every service honours the EXIF orientation tag. No ICC profile is
    sent, so the service works on the same pixel values as the local engines; the pipeline puts the
    original profile back on save. Alpha is handled locally by the pipeline too.
    A picture or pixels the pipeline has already opened are used as they are."""
    if isinstance(image, np.ndarray):
        return image, False
    loaded = image if isinstance(image, LoadedImage) else load_image(Path(image))
    return loaded.rgb, loaded.format in _LOSSY_FORMATS


def encode_upload(pixels: np.ndarray, lossy: bool) -> Upload:
    """JPEG q95 for already-lossy sources, PNG otherwise, unless that PNG would be over
    PNG_UPLOAD_MAX_BYTES (a big photo): then JPEG q95 too, which looks the same."""
    image = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8))
    buffer = io.BytesIO()
    if not lossy:
        image.save(buffer, "PNG")
        if buffer.tell() <= PNG_UPLOAD_MAX_BYTES:
            return Upload(buffer.getvalue(), "image/png", image.width, image.height)
        buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95, subsampling=0)
    return Upload(buffer.getvalue(), "image/jpeg", image.width, image.height)


def prepare_upload(image: Source) -> Upload:
    """The picture ready to send: see source_pixels and encode_upload."""
    return encode_upload(*source_pixels(image))


def resize_pixels(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`pixels` at `size` (width, height), with Lanczos."""
    if (pixels.shape[1], pixels.shape[0]) == size:
        return pixels
    image = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8))
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS))


class _UploadBody:
    """A request body sent a chunk at a time, so Cancel works part way through a big upload and
    progress can say how far it has got. It can be sent more than once, as send() may retry."""

    def __init__(
        self, data: bytes, message: str, *, progress: ProgressFn, cancel: threading.Event, span: tuple[float, float]
    ) -> None:
        self.data = data
        self.message = message
        self.progress = progress
        self.cancel = cancel
        self.span = span
        self.reported = span[0]  # never goes backwards, even when the body is sent again

    def __iter__(self) -> Iterator[bytes]:
        total = len(self.data)
        start, end = self.span
        for offset in range(0, total, UPLOAD_CHUNK_BYTES):
            raise_if_cancelled(self.cancel)
            yield self.data[offset : offset + UPLOAD_CHUNK_BYTES]
            sent = min(total, offset + UPLOAD_CHUNK_BYTES)
            fraction = start + (end - start) * sent / total
            if fraction > self.reported:
                self.reported = fraction
                self.progress(fraction, f"{self.message} ({100 * sent // total}%)")


def json_upload(
    body: dict, message: str, *, progress: ProgressFn, cancel: threading.Event
) -> tuple[_UploadBody, dict[str, str]]:
    """A JSON body that carries a picture, ready for send(content=..., headers=...): it reports
    progress between UPLOADING_AT and WAITING_AT, as "<message> (40%)", and stops if the user cancels."""
    data = json.dumps(body).encode()
    content = _UploadBody(data, message, progress=progress, cancel=cancel, span=(UPLOADING_AT, WAITING_AT))
    return content, {"Content-Type": "application/json", "Content-Length": str(len(data))}


def file_upload(
    upload: Upload, field: str, message: str, *, progress: ProgressFn, cancel: threading.Event
) -> tuple[_UploadBody, dict[str, str]]:
    """Like json_upload, for a form with the picture as file `field` (multipart/form-data)."""
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{upload.filename}"\r\n'
        f"Content-Type: {upload.mime}\r\n\r\n"
    )
    data = head.encode() + upload.data + f"\r\n--{boundary}--\r\n".encode()
    content = _UploadBody(data, message, progress=progress, cancel=cancel, span=(UPLOADING_AT, WAITING_AT))
    return content, {"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(data))}


def result_pixel_limit(upload: Upload, factor: float | str) -> int:
    """The most pixels a result can sensibly have: the size we asked for, plus room for rounding.
    `factor` is a number or a service's own spelling such as "4x"."""
    scale = float(str(factor).removesuffix("x"))
    expected = upload.width * upload.height * scale * scale
    return max(RESULT_PIXEL_FLOOR, math.ceil(expected * RESULT_SIZE_MARGIN))


@contextmanager
def _pillow_pixel_limit(max_pixels: int) -> Iterator[None]:
    """Let Pillow open up to `max_pixels`, never lowering a limit the app has already raised.

    Pillow only has a global setting. Changing it briefly is safe because cloud results are only
    ever decoded by the single job worker."""
    previous = Image.MAX_IMAGE_PIXELS
    if previous is not None and previous < max_pixels:
        Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def decode_image(data: bytes, service: str, *, max_pixels: int) -> np.ndarray:
    """Open a finished picture, which can be far over Pillow's default limit (4x of a 12 MP photo
    is 195 MP). We check against what this job asked for instead.

    The user has already paid by now, so the messages don't suggest simply trying again."""
    too_big = (
        f"{service} sent back a much bigger picture than the app asked for, so the app didn't use it. "
        "Try a different mode."
    )
    try:
        with _pillow_pixel_limit(max_pixels), Image.open(io.BytesIO(data)) as result:
            if result.width * result.height > max_pixels:
                raise CloudError(too_big)
            rgb = result if result.mode == "RGB" else result.convert("RGB")  # a 195 MP copy is 585 MB
            return np.array(rgb, dtype=np.uint8)
    except Image.DecompressionBombError as exc:  # over twice max_pixels: Pillow refuses on its own
        raise CloudError(too_big) from exc
    except (OSError, ValueError) as exc:
        raise CloudError(f"{service} sent back a picture the app couldn't open. Try a different mode.") from exc


def finished_but(service: str, problem: str, note: str = "") -> str:
    """A message for a job the service finished, and has probably charged for, that the app then
    couldn't use. It never says to simply try again: that would pay again."""
    return f"{service} finished your picture, but {problem}. {check_account(service)}" + (f" {note}" if note else "")


def download_image(
    client: httpx.Client,
    url: str,
    *,
    service: str,
    cancel: threading.Event,
    max_pixels: int,
    headers: dict[str, str] | None = None,
    note: str = "",
) -> np.ndarray:
    """Download the finished picture, waiting through a dropped connection (see keep_asking).
    `note` is added when the download fails, e.g. how long the service keeps the picture."""
    if url.startswith("data:"):
        return decode_image(base64.b64decode(url.split(",", 1)[1]), service, max_pixels=max_pixels)
    not_downloaded = finished_but(service, "the app couldn't download it", note)
    if not url.startswith("https://"):
        raise CloudError(not_downloaded)

    def fetch() -> httpx.Response:
        return ok_or_raise(
            send(client, "GET", url, service=service, cancel=cancel, headers=headers or {}),
            lambda response: not_downloaded,
        )

    try:
        response = keep_asking(fetch, cancel)
    except CloudError as exc:
        raise CloudError(not_downloaded) from exc
    if response.status_code != 200:
        raise CloudError(not_downloaded)
    return decode_image(response.content, service, max_pixels=max_pixels)
