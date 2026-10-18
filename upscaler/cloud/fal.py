"""fal.ai: one key for SeedVR2, Topaz (precision and generative) and Crystal.

Checked against fal's official docs on 2026-10-17:

Queue API (https://fal.ai/docs/model-apis/model-endpoints/queue)
- POST https://queue.fal.run/{endpoint_id}, header "Authorization: Key <key>", JSON body = the
  model's input fields (no envelope) -> {request_id, response_url, status_url, cancel_url,
  queue_position}.
- GET status_url -> {"status": "IN_QUEUE" (+queue_position) | "IN_PROGRESS" | "COMPLETED"}.
  A failed request is COMPLETED with "error" and "error_type".
- GET response_url -> the model's output JSON. PUT cancel_url -> 202 CANCELLATION_REQUESTED.
- We poll the URLs fal returns rather than building them: the docs don't say how they're
  formed for endpoints with a sub-path such as fal-ai/seedvr/upscale/image.
Privacy (https://fal.ai/docs/model-apis/data-retention, .../media-expiration)
- Request inputs and outputs are kept 90 days unless the request sends "X-Fal-Store-IO: 0".
- Output files live on fal's CDN (at least 7 days by default, per the FAQ) unless the request sets
  X-Fal-Object-Lifecycle-Preference: {"expiration_duration_seconds": <n>}.
Errors (https://fal.ai/docs/documentation/model-apis/errors)
- {"detail": [{"loc", "msg", "type", ...}]}; e.g. content_policy_violation, image_too_large,
  image_too_small are 422. Server errors (5xx) are never billed (pricing docs).
Files: each model's API page says base64 data URIs are accepted for file inputs. fal's CDN docs
(https://fal.ai/docs/documentation/model-apis/fal-cdn) advise against data URIs for big files, as
they make the request much bigger and slower. We still use one (not yet checked: uploading to fal's
storage first, as fal's own client libraries do), but keep it small (JPEG once a PNG would pass
10 MB, see base.encode_upload) and send it a chunk at a time, so Cancel works mid-upload.
Endpoints (https://fal.ai/models/<endpoint_id>/api and the model page for price). The cards show the
price of an 8K picture made from a 12 MP phone photo: 8192x6144, 50.3 MP of output.
- fal-ai/seedvr/upscale/image: image_url, upscale_mode "factor"|"target", upscale_factor (float),
  seed, noise_scale, output_format png|jpg|webp -> {image: {url, width, height, ...}, seed}.
  $0.001 per megapixel. The model page says SeedVR2 but not which size (3B or 7B).
- topaz/upscale/image/precision: image_url, model ("High Fidelity V2", "High Fidelity V3",
  "Standard V2", ...), upscale_factor (float, "up to 4x"), output_format jpeg|png,
  face_enhancement (DEFAULT TRUE), face_enhancement_strength (default 0.8),
  face_enhancement_creativity -> {image: {url, ...}}. $0.08 per started 24 MP of output.
- topaz/upscale/image/generative: same fields, model "Wonder 3.5" etc. -> {image: {url, ...}}.
  Wonder 3/3.5: $0.08 per started 8 MP of output.
- clarityai/crystal-upscaler: image_url, scale_factor (1-200), creativity (0-10, default 0),
  output_format png|jpg -> {images: [...]}. $0.016 per megapixel.
Key check without spending: GET the status of a random request id. Tried on 2026-10-17 with a
made-up key: 401 {"detail": "invalid key credentials"}; with no key at all: 404 NOT_FOUND. A real
key should also get 404 (couldn't be tried without one), so an empty key is never sent.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass

import httpx
import numpy as np

from upscaler.cloud.base import (
    DOWNLOADING_AT,
    UPLOADING_AT,
    WAITING_AT,
    CloudError,
    CloudModel,
    JobFailed,
    Pending,
    Source,
    busy_message,
    clean_key,
    download_image,
    error_detail,
    find_model,
    finished_but,
    header_key,
    json_body,
    json_upload,
    keep_asking,
    key_complaint,
    megapixels,
    new_client,
    ok_or_raise,
    poll_until_done,
    prepare_upload,
    result_pixel_limit,
    said,
    send,
    started_blocks,
)
from upscaler.config import ProgressFn

SERVICE = "fal.ai"
QUEUE = "https://queue.fal.run"
KEY_CHECK_APP = "fal-ai/seedvr"
TOPAZ_MAX_FACTOR = 4.0

PRIVACY_HEADERS = {
    "X-Fal-Store-IO": "0",
    "X-Fal-Object-Lifecycle-Preference": json.dumps({"expiration_duration_seconds": 3600}),
}
# fal's Topaz endpoints enhance faces by default, which can change who someone looks like.
NO_FACE_ENHANCEMENT = {"face_enhancement": False, "face_enhancement_strength": 0.0, "face_enhancement_creativity": 0.0}


def fal_price_usd(model_id: str, size: tuple[int, int], scale: float) -> float:
    """fal's price for upscaling a picture of `size` by `scale` (see the notes at the top): per
    megapixel made for SeedVR2 and Crystal; per started 24 MP (Precision) or 8 MP (Wonder) for Topaz,
    which makes at most 4x."""
    match model_id:
        case "seedvr2":
            return 0.001 * megapixels(size, scale)
        case "topaz_precision":
            return 0.08 * started_blocks(megapixels(size, min(scale, TOPAZ_MAX_FACTOR)), 24)
        case "topaz_generative":
            return 0.08 * started_blocks(megapixels(size, min(scale, TOPAZ_MAX_FACTOR)), 8)
        case "crystal":
            return 0.016 * megapixels(size, scale)
    raise CloudError(f"{SERVICE} doesn't have a mode called “{model_id}”.")


# The cards show the price of an 8K picture made from a 12 MP phone photo (4032x3024 to 8192x6144).
_PHONE_PHOTO, _TO_8K = (4032, 3024), 8192 / 4032

# SeedVR2 comes first: it is fal's default here, like Photo-real is locally.
MODELS = [
    CloudModel(
        id="seedvr2",
        label="SeedVR2 (cloud)",
        kind="generative",
        est_cost_8k_usd=round(fal_price_usd("seedvr2", _PHONE_PHOTO, _TO_8K), 2),
        description="SeedVR2, the model behind Photo-real, run at full size on fast cloud computers. "
        "Adds convincing photo detail; check faces and text.",
    ),
    CloudModel(
        id="topaz_precision",
        label="Topaz Precision",
        kind="faithful",
        est_cost_8k_usd=round(fal_price_usd("topaz_precision", _PHONE_PHOTO, _TO_8K), 2),
        description="Topaz High Fidelity: keeps faces, furniture and logos exactly as they are, "
        "just sharper and bigger.",
    ),
    CloudModel(
        id="topaz_generative",
        label="Topaz Wonder",
        kind="generative",
        est_cost_8k_usd=round(fal_price_usd("topaz_generative", _PHONE_PHOTO, _TO_8K), 2),
        description="Topaz's creative upscaler: rich, photographic detail, but small things can change.",
    ),
    CloudModel(
        id="crystal",
        label="Crystal (portraits)",
        kind="faithful",
        est_cost_8k_usd=round(fal_price_usd("crystal", _PHONE_PHOTO, _TO_8K), 2),
        description="Tuned for faces and skin, with creativity at zero so people still look like themselves.",
    ),
]


def fal_request(model_id: str, image_url: str, scale: float) -> tuple[str, dict]:
    """The endpoint id and input body for one upscale."""
    factor = round(float(scale), 3)
    match model_id:
        case "seedvr2":
            return "fal-ai/seedvr/upscale/image", {
                "image_url": image_url,
                "upscale_mode": "factor",
                "upscale_factor": factor,
                "seed": 42,
                "output_format": "png",
            }
        case "topaz_precision":
            return "topaz/upscale/image/precision", {
                "image_url": image_url,
                "model": "High Fidelity V2",
                "upscale_factor": min(factor, TOPAZ_MAX_FACTOR),
                "output_format": "png",
                **NO_FACE_ENHANCEMENT,
            }
        case "topaz_generative":
            return "topaz/upscale/image/generative", {
                "image_url": image_url,
                "model": "Wonder 3.5",
                "upscale_factor": min(factor, TOPAZ_MAX_FACTOR),
                "output_format": "png",
                **NO_FACE_ENHANCEMENT,
            }
        case "crystal":
            return "clarityai/crystal-upscaler", {
                "image_url": image_url,
                "scale_factor": factor,
                "creativity": 0,
                "output_format": "png",
            }
    raise CloudError(f"{SERVICE} doesn't have a mode called “{model_id}”.")


@dataclass(frozen=True)
class _JobUrls:
    status: str
    response: str
    cancel: str


class FalProvider:
    id = "fal"
    name = SERVICE
    models = MODELS

    def price_usd(self, model_id: str, size: tuple[int, int], scale: float) -> float:
        return fal_price_usd(model_id, size, scale)

    def check_key(self, key: str) -> tuple[bool, str]:
        key = clean_key(key)
        if complaint := key_complaint(key, "Paste your fal.ai key first."):
            return False, complaint
        url = f"{QUEUE}/{KEY_CHECK_APP}/requests/{uuid.uuid4()}/status"
        try:
            with new_client() as client:
                response = send(client, "GET", url, service=SERVICE, headers=_auth(key))
        except CloudError as exc:
            return False, exc.user_message
        if response.status_code == 404:  # the key works; the made-up request just doesn't exist
            return True, "Connected to fal.ai."
        if response.status_code in (401, 403):
            return False, "fal.ai didn't accept this key. Check that you copied the whole key and try again."
        return False, busy_message(SERVICE)

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
        find_model(self.models, model_id, SERVICE)
        progress(0.0, "Getting your picture ready for fal.ai…")
        upload = prepare_upload(image)
        app, body = fal_request(model_id, upload.data_uri(), scale)
        auth = _auth(key)
        with new_client() as client:
            message = "Sending your picture to fal.ai…"
            progress(UPLOADING_AT, message)
            content, content_headers = json_upload(body, message, progress=progress, cancel=cancel)
            submitted = send(
                client,
                "POST",
                f"{QUEUE}/{app}",
                service=SERVICE,
                cancel=cancel,
                content=content,
                headers=auth | PRIVACY_HEADERS | content_headers,
            )
            urls = _job_urls(json_body(_ok(submitted), SERVICE))
            progress(WAITING_AT, "Waiting for fal.ai…")
            poll_until_done(
                lambda: _status(client, urls, auth, cancel),
                service=SERVICE,
                progress=progress,
                cancel=cancel,
                stop_job=lambda: client.put(urls.cancel, headers=auth),
            )
            result = _result(client, urls, auth, cancel)
            progress(DOWNLOADING_AT, "Downloading the finished picture from fal.ai…")
            pixels = download_image(
                client,
                _output_url(result),
                service=SERVICE,
                cancel=cancel,
                max_pixels=result_pixel_limit(upload, scale),  # no fal mode goes past `scale`
            )
        progress(1.0, "fal.ai finished.")
        return pixels


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Key {header_key(key, SERVICE)}"}


def _job_urls(job: dict) -> _JobUrls:
    urls = [job.get(field) for field in ("status_url", "response_url", "cancel_url")]
    # The key goes along with every one of these requests, so only ever send it to fal's queue.
    if not all(isinstance(url, str) and url.startswith(f"{QUEUE}/") for url in urls):
        raise CloudError("fal.ai didn't start the job. Please try again.")
    return _JobUrls(*urls)


def _result(client: httpx.Client, urls: _JobUrls, auth: dict[str, str], cancel: threading.Event) -> dict:
    """The finished job's output. fal has done (and charged for) the job by now."""
    try:
        return keep_asking(
            lambda: json_body(
                _ok(send(client, "GET", urls.response, service=SERVICE, cancel=cancel, headers=auth)), SERVICE
            ),
            cancel,
        )
    except CloudError as exc:
        raise CloudError(finished_but(SERVICE, "the app couldn't fetch it")) from exc


def _status(client: httpx.Client, urls: _JobUrls, auth: dict[str, str], cancel: threading.Event) -> bool | Pending:
    status = json_body(_ok(send(client, "GET", urls.status, service=SERVICE, cancel=cancel, headers=auth)), SERVICE)
    match status.get("status"):
        case "COMPLETED":
            if status.get("error"):
                raise JobFailed(_failure_message(status.get("error_type"), str(status["error"])))
            return True
        case "IN_QUEUE":
            position = status.get("queue_position")
            place = f" (number {position + 1} in line)" if isinstance(position, int) else ""
            return Pending(f"Waiting in fal.ai's queue{place}…")
    return Pending("fal.ai is upscaling your picture…")


def _output_url(result: dict) -> str:
    image = result.get("image")
    if image is None and result.get("images"):
        image = result["images"][0]  # Crystal returns a list
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str):
        raise CloudError(finished_but(SERVICE, "it didn't send the picture back"))
    return url


def _ok(response: httpx.Response) -> httpx.Response:
    return ok_or_raise(response, _http_error_message)


def _http_error_message(response: httpx.Response) -> str:
    code = response.status_code
    if code == 401:
        return "fal.ai didn't accept your key. Connect fal.ai again in Settings with a fresh key."
    if code in (402, 403):
        return (
            "fal.ai says your account has no credit (or the key isn't allowed to do this). "
            "Add credit at fal.ai → Billing, then try again."
        )
    if code == 422:
        return _failure_message(_error_type(response), error_detail(response))
    if code == 429 or code >= 500:
        return busy_message(SERVICE)
    return f"fal.ai couldn't finish the upscale.{said(error_detail(response))}"


def _error_type(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, list) and detail and isinstance(detail[0], dict):
            return detail[0].get("type")
        if body.get("error_type"):
            return body["error_type"]
    return response.headers.get("X-Fal-Error-Type")


def _failure_message(error_type: str | None, detail: str) -> str:
    match error_type:
        case "content_policy_violation":
            return "fal.ai's safety filter turned this picture down. Try a different picture or mode."
        case "image_too_large" | "file_too_large":
            return "This picture is too big for this fal.ai mode. Try a smaller size or a different mode."
        case "image_too_small":
            return "This picture is too small for this fal.ai mode. Try a different mode."
        case "generation_timeout" | "request_timeout":
            return "fal.ai took too long with this picture. Please try again."
    return f"fal.ai couldn't upscale this picture.{said(detail)}"
