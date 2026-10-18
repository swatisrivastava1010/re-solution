"""Replicate: Topaz Labs' faithful upscaler (High Fidelity V2), face enhancement off.

Checked against Replicate's official docs on 2026-10-17:

Model (https://replicate.com/topazlabs/image-upscale/api/schema, OpenAPI "Input" / "Output")
- image (uri, required); enhance_model: "Standard V2" | "Low Resolution V2" | "CGI" |
  "High Fidelity V2" | "Text Refine"; upscale_factor: "None" | "2x" | "4x" | "6x";
  output_format: "jpg" | "png"; subject_detection; face_enhancement (bool, default false);
  face_enhancement_strength (0-1, default 0.8); face_enhancement_creativity (0-1, default 0).
- Output: one URI string.
- Price (https://replicate.com/topazlabs/image-upscale/llms.txt, re-checked 2026-10-18): by units of
  output size, in this table of output MP -> price: 12 and 24 -> $0.05, 36 and 48 -> $0.10, 60 -> $0.15,
  96 -> $0.20, 132 -> $0.24, 168 -> $0.29, 336 -> $0.53, 512 -> $0.82. Sizes between rows aren't
  published; we count the next row up. Only 2x/4x/6x exist, so an 8K picture is $0.10-0.25: a 12 MP
  phone photo comes back at 2x, 8064x6048 (48.8 MP, $0.15), which the card shows; a 24 MP camera photo
  at 2x is about 96 MP ($0.20).
HTTP API (https://replicate.com/docs/reference/http and https://api.replicate.com/openapi.json)
- "Authorization: Bearer <token>"; tokens look like r8_....
- POST /v1/models/{owner}/{name}/predictions {"input": {...}} -> 201/202 prediction
  {id, status, output, error, urls: {get, cancel, web}}. Status: starting | processing |
  succeeded | failed | canceled | aborted. Poll urls.get; POST urls.cancel to stop.
- File inputs: data URLs only for files up to 256 KB; bigger files need an HTTP URL. So we
  upload with POST /v1/files (multipart field "content", 100 MB max) -> {id, urls: {get}}, pass
  urls.get as the input, and DELETE /v1/files/{id} when done.
- Output files: "You'll need to pass the Authorization header to request them."
- GET /v1/account -> {type, username, name}: a free way to test a token. Tried on 2026-10-17
  with a made-up token: 401 "You did not pass a valid authentication token".
- 429 body: {"detail": "Request was throttled. Expected available in 1 second."}
Retention (https://replicate.com/docs/topics/predictions/data-retention): an API prediction's
inputs, outputs, output files and logs are removed after an hour.
Billing (https://replicate.com/docs/topics/billing, .../prepaid-credit): prepaid credit bought at
https://replicate.com/account/billing; failed runs aren't charged. The API error returned when
credit runs out isn't documented, so 402 and 403 both point people to Billing.
"""

from __future__ import annotations

import threading

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
    Upload,
    busy_message,
    clean_key,
    download_image,
    error_detail,
    file_upload,
    find_model,
    finished_but,
    header_key,
    json_body,
    key_complaint,
    megapixels,
    new_client,
    ok_or_raise,
    poll_until_done,
    prepare_upload,
    result_pixel_limit,
    said,
    send,
)
from upscaler.config import ProgressFn

SERVICE = "Replicate"
API = "https://api.replicate.com/v1"
MODEL_PATH = "topazlabs/image-upscale"
FINISHED = frozenset({"succeeded", "failed", "canceled", "aborted"})
# The app's exact resize covers up to 1.25x on top of the model, like the local pipeline.
LANCZOS_HEADROOM = 1.25
# Replicate's price table: (output megapixels, US dollars), see the notes at the top.
PRICES = (
    (12, 0.05), (24, 0.05), (36, 0.10), (48, 0.10), (60, 0.15),
    (96, 0.20), (132, 0.24), (168, 0.29), (336, 0.53), (512, 0.82),
)  # fmt: skip


def replicate_scale_factor(scale: float) -> str:
    """Replicate offers only 2x, 4x and 6x: use the smallest the exact resize can finish from."""
    for factor in (2, 4):
        if scale <= factor * LANCZOS_HEADROOM:
            return f"{factor}x"
    return "6x"


def replicate_price_usd(size: tuple[int, int], scale: float) -> float:
    """Replicate's price for a picture of `size` asked for `scale`: what the factor it really uses
    makes, priced by the next row up in its table (beyond the table, at the last row's rate)."""
    made = megapixels(size, int(replicate_scale_factor(scale).removesuffix("x")))
    for limit, price in PRICES:
        if made <= limit:
            return price
    limit, price = PRICES[-1]
    return price * made / limit


MODELS = [
    CloudModel(
        id="topaz_hf",
        label="Topaz High Fidelity",
        kind="faithful",
        est_cost_8k_usd=replicate_price_usd((4032, 3024), 8192 / 4032),  # a phone photo to 8K
        description="Topaz's most faithful model: keeps faces, furniture and logos exactly as they are. "
        "Replicate deletes your picture after an hour.",
    ),
]


class ReplicateProvider:
    id = "replicate"
    name = SERVICE
    models = MODELS

    def price_usd(self, model_id: str, size: tuple[int, int], scale: float) -> float:
        find_model(self.models, model_id, SERVICE)
        return replicate_price_usd(size, scale)

    def check_key(self, key: str) -> tuple[bool, str]:
        key = clean_key(key)
        if complaint := key_complaint(key, "Paste your Replicate token first."):
            return False, complaint
        try:
            with new_client() as client:
                response = send(client, "GET", f"{API}/account", service=SERVICE, headers=_auth(key))
            if response.status_code == 200:
                username = json_body(response, SERVICE).get("username")
                return True, f"Connected to Replicate as {username}." if username else "Connected to Replicate."
        except CloudError as exc:
            return False, exc.user_message
        if response.status_code in (401, 403):
            return (
                False,
                "Replicate didn't accept this token. Check that you copied the whole token (it starts with r8_).",
            )
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
        progress(0.0, "Getting your picture ready for Replicate…")
        upload = prepare_upload(image)
        auth = _auth(key)
        with new_client() as client:
            progress(UPLOADING_AT, "Sending your picture to Replicate…")
            file_id, file_url = _upload_file(client, upload, auth, progress, cancel)
            try:
                model_input = _model_input(file_url, scale)
                created = send(
                    client,
                    "POST",
                    f"{API}/models/{MODEL_PATH}/predictions",
                    service=SERVICE,
                    cancel=cancel,
                    json={"input": model_input},
                    headers=auth,
                )
                progress(WAITING_AT, "Waiting for Replicate…")
                output_url = _wait_for_output(client, json_body(_ok(created), SERVICE), auth, progress, cancel)
                progress(DOWNLOADING_AT, "Downloading the finished picture from Replicate…")
                pixels = download_image(
                    client,
                    output_url,
                    service=SERVICE,
                    cancel=cancel,
                    max_pixels=result_pixel_limit(upload, model_input["upscale_factor"]),
                    headers=auth if _is_replicate_host(output_url) else None,
                )
            finally:
                _delete_file(client, file_id, auth)
        progress(1.0, "Replicate finished.")
        return pixels


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {header_key(key, SERVICE)}"}


def _model_input(image_url: str, scale: float) -> dict:
    return {
        "image": image_url,
        "enhance_model": "High Fidelity V2",
        "upscale_factor": replicate_scale_factor(scale),
        "output_format": "png",
        "face_enhancement": False,  # it can change who someone looks like
        "face_enhancement_strength": 0,
        "face_enhancement_creativity": 0,
    }


def _upload_file(
    client: httpx.Client, upload: Upload, auth: dict[str, str], progress: ProgressFn, cancel: threading.Event
) -> tuple[str, str]:
    content, content_headers = file_upload(
        upload, "content", "Sending your picture to Replicate…", progress=progress, cancel=cancel
    )
    response = send(
        client,
        "POST",
        f"{API}/files",
        service=SERVICE,
        cancel=cancel,
        idempotent=True,  # uploading the picture costs nothing, so sending it again is safe
        content=content,
        headers=auth | content_headers,
    )
    if response.status_code == 413:
        raise CloudError("This picture is too big to send to Replicate (the limit is 100 MB).")
    body = json_body(_ok(response), SERVICE)
    file_url = (body.get("urls") or {}).get("get")
    if not body.get("id") or not isinstance(file_url, str):
        raise CloudError("Replicate didn't accept the upload. Please try again.")
    return body["id"], file_url


def _delete_file(client: httpx.Client, file_id: str, auth: dict[str, str]) -> None:
    try:
        client.delete(f"{API}/files/{file_id}", headers=auth)
    except httpx.HTTPError:
        pass  # Replicate removes it on its own later; nothing else to do


def _wait_for_output(
    client: httpx.Client, prediction: dict, auth: dict[str, str], progress: ProgressFn, cancel: threading.Event
) -> str:
    latest = prediction
    urls = prediction.get("urls") or {}
    get_url = _api_url(urls.get("get"), f"{API}/predictions/{prediction.get('id')}")
    cancel_url = _api_url(urls.get("cancel"), f"{get_url}/cancel")

    def check() -> str | Pending:
        nonlocal latest
        if latest.get("status") not in FINISHED:
            latest = json_body(_ok(send(client, "GET", get_url, service=SERVICE, cancel=cancel, headers=auth)), SERVICE)
        return _outcome(latest)

    return poll_until_done(
        check,
        service=SERVICE,
        progress=progress,
        cancel=cancel,
        stop_job=lambda: client.post(cancel_url, headers=auth),
    )


def _outcome(prediction: dict) -> str | Pending:
    match prediction.get("status"):
        case "succeeded":
            output = prediction.get("output")
            url = output[0] if isinstance(output, list) and output else output
            if not isinstance(url, str):
                raise JobFailed(finished_but(SERVICE, "it didn't send the picture back"))
            return url
        case "failed":
            raise JobFailed(f"Replicate couldn't upscale this picture.{said(str(prediction.get('error') or ''))}")
        case "canceled" | "aborted":
            raise JobFailed("Replicate stopped this job before it finished. Please try again.")
        case "starting":
            return Pending("Replicate is starting the upscaler (this can take a minute the first time)…")
    return Pending("Replicate is upscaling your picture…")


def _api_url(url: object, fallback: str) -> str:
    return url if isinstance(url, str) and url.startswith(f"{API}/") else fallback  # keep the token on Replicate


def _is_replicate_host(url: str) -> bool:
    host = httpx.URL(url).host
    return host in ("replicate.delivery", "api.replicate.com") or host.endswith(".replicate.delivery")


def _ok(response: httpx.Response) -> httpx.Response:
    return ok_or_raise(response, _http_error_message)


def _http_error_message(response: httpx.Response) -> str:
    code = response.status_code
    detail = error_detail(response)
    if code == 401:
        return "Replicate didn't accept your token. Connect Replicate again in Settings with a fresh token."
    if code in (402, 403):
        return (
            "Replicate says your account can't run this yet, usually because there's no credit. "
            f"Add credit at replicate.com → Account → Billing, then try again.{said(detail)}"
        )
    if code == 422:
        return f"Replicate didn't accept this picture.{said(detail)}"
    if code == 429 or code >= 500:
        return busy_message(SERVICE)
    return f"Replicate couldn't finish the upscale.{said(detail)}"
