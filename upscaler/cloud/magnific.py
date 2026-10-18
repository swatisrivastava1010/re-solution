"""Magnific (formerly Freepik): Precision V2 (faithful) and Creative upscalers.

Checked against Magnific's official docs on 2026-10-17 (docs.magnific.com and its OpenAPI file
https://storage.googleapis.com/fc-freepik-pro-rev1-eu-api-specs/magnific-api-v1-openapi.yaml):

- Auth: header "x-magnific-api-key: <key>" (https://docs.magnific.com/authentication). The old
  api.freepik.com host now gives the same Magnific answers.
- Precision V2 (https://docs.magnific.com/api-reference/image-upscaler-precision-v2/post-image-upscaler-precision-v2)
  POST https://api.magnific.com/v1/ai/image-upscaler-precision-v2 {image: HTTPS URL or base64
  string (required), scale_factor: integer 2-16, flavor: "sublime" | "photo" | "photo_denoiser",
  sharpen (0-100, default 7), smart_grain (0-100, default 7), ultra_detail (0-100, default 30),
  filter_nsfw, webhook_url} -> {"data": {"task_id", "status", "generated": []}}.
  GET .../image-upscaler-precision-v2/{task-id} -> the same shape; status CREATED | IN_PROGRESS |
  COMPLETED | FAILED; "generated" holds the result URL once COMPLETED.
  GET .../image-upscaler-precision-v2 lists your tasks, which is our free key check.
  Price: by output area, not published per picture.
- Creative (https://docs.magnific.com/api-reference/image-upscaler-creative/post-image-upscaler)
  POST https://api.magnific.com/v1/ai/image-upscaler {image: base64 (required), scale_factor:
  "2x" | "4x" | "8x" | "16x", optimized_for ("standard", "soft_portraits", ...), prompt,
  creativity / hdr / resemblance / fractality (-10..10, default 0), engine ("automatic", ...)}.
  "The resulted image can't exceed maximum allowed size of 25.3 million pixels."
  Poll GET .../image-upscaler/{task-id}. Price examples (image-upscaler overview page):
  2560x1920 €0.20, 5120x2880 €0.40, 5120x3840 €0.50. Those three fit €0.10 plus €0.02 per megapixel
  made to within a cent, so the app works other sizes out that way (our reading, not a published rule).
- Errors: 400 {"message": ...}, 401 {"message": "Invalid API key"}, 429 "Too many requests",
  500, 503. Tried on 2026-10-17 with a made-up key: 401 "Unauthorized: The provided API key is
  invalid...". There is no cancel endpoint.
- Output links "expire 30 minutes after your content is generated" (https://docs.magnific.com/quickstart).
- Neither endpoint has a face-enhancement option; Creative keeps creativity at 0.
- Billing (https://docs.magnific.com/pricing): every API call spends credits from the
  organisation's balance on every paid plan; "Unlimited" web allowances don't cover the API.
"""

from __future__ import annotations

import math
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
    encode_upload,
    error_detail,
    find_model,
    finished_but,
    header_key,
    json_body,
    json_upload,
    key_complaint,
    new_client,
    ok_or_raise,
    poll_until_done,
    resize_pixels,
    result_pixel_limit,
    said,
    send,
    source_pixels,
)
from upscaler.config import ProgressFn

SERVICE = "Magnific"
API = "https://api.magnific.com/v1/ai"
PATHS = {"precision_v2": "image-upscaler-precision-v2", "creative": "image-upscaler"}
CREATIVE_MAX_PIXELS = 25_300_000
CREATIVE_FACTORS = (2, 4, 8, 16)
# The app's exact resize covers up to 1.25x on top of the model, like the local pipeline.
LANCZOS_HEADROOM = 1.25
# Creative sends a smaller copy for a bigger factor only when the result comes out at least this much
# bigger than the factor that fits would make it.
CREATIVE_MIN_GAIN = 1.15
LINK_NOTE = "Magnific keeps a finished picture for only 30 minutes."
# Creative's price, read from Magnific's examples (see the notes at the top), and roughly in dollars.
CREATIVE_EUR_BASE = 0.10
CREATIVE_EUR_PER_MEGAPIXEL = 0.02
USD_PER_EUR = 1.15


def creative_price_usd(megapixels: float) -> float:
    """About what a Creative result of `megapixels` costs in plan credits, in US dollars."""
    return (CREATIVE_EUR_BASE + CREATIVE_EUR_PER_MEGAPIXEL * megapixels) * USD_PER_EUR


MODELS = [
    CloudModel(
        id="precision_v2",
        label="Magnific Precision",
        kind="faithful",
        est_cost_8k_usd=0.0,
        description="Crisp, natural detail without redrawing things. Uses credits from your Magnific plan; "
        "Magnific doesn't publish a price per picture.",
    ),
    CloudModel(
        id="creative",
        label="Magnific Creative",
        kind="generative",
        est_cost_8k_usd=round(creative_price_usd(CREATIVE_MAX_PIXELS / 1e6), 2),  # 8K is past its limit
        description="Magnific's creative upscaler adds lots of new detail. It goes up to about 25 megapixels "
        "(around 6K) and the app finishes the rest. Uses plan credits (about €0.50 for a 20-megapixel result).",
    ),
]


def precision_scale_factor(scale: float) -> int:
    return max(2, min(16, math.ceil(scale - 0.01)))


def creative_plan(width: int, height: int, scale: float) -> tuple[str, tuple[int, int]]:
    """The Creative factor for `scale`, and the size (width, height) to send the picture at.

    Creative's results stop at 25.3 MP. When the factor the job wants would pass that, the biggest
    factor that fits can come back far too small: 2x of a 1080p picture is 3840 wide, which the app
    would have to stretch 2.1x for 8K. So the app sends a slightly smaller copy that the next factor
    takes right up to the limit instead (4x of 1676x943 is 6704x3772), when that comes out clearly
    bigger. A picture that's too big even for 2x is refused, as Precision suits it better."""
    wanted = next((f for f in CREATIVE_FACTORS if f * LANCZOS_HEADROOM >= scale), CREATIVE_FACTORS[-1])
    fitting = [f for f in CREATIVE_FACTORS if f <= wanted and width * height * f * f <= CREATIVE_MAX_PIXELS]
    if not fitting:
        raise CloudError(
            "This picture is already too big for Magnific Creative, which makes pictures of up to about "
            "25 megapixels. Try Magnific Precision instead."
        )
    factor = max(fitting)
    if factor < wanted:
        bigger = CREATIVE_FACTORS[CREATIVE_FACTORS.index(factor) + 1]
        shrink = math.sqrt(CREATIVE_MAX_PIXELS / (width * height * bigger * bigger))
        if shrink * bigger >= factor * CREATIVE_MIN_GAIN:
            return f"{bigger}x", (max(1, math.floor(width * shrink)), max(1, math.floor(height * shrink)))
    return f"{factor}x", (width, height)


def upload_plan(model_id: str, width: int, height: int, scale: float) -> tuple[int | str, tuple[int, int]]:
    """The scale_factor to ask for and the size (width, height) to send the picture at."""
    if model_id == "precision_v2":
        return precision_scale_factor(scale), (width, height)
    if model_id == "creative":
        return creative_plan(width, height, scale)
    raise CloudError(f"{SERVICE} doesn't have a mode called “{model_id}”.")


def magnific_request(model_id: str, upload: Upload, factor: int | str) -> tuple[str, dict]:
    """The endpoint path and JSON body. Magnific wants plain base64, not a data URI."""
    if model_id == "precision_v2":
        return PATHS[model_id], {
            "image": upload.base64(),
            "scale_factor": factor,
            "flavor": "photo",
        }
    if model_id == "creative":
        return PATHS[model_id], {
            "image": upload.base64(),
            "scale_factor": factor,
            "optimized_for": "standard",
            "creativity": 0,
            "hdr": 0,
            "resemblance": 0,
            "fractality": 0,
            "engine": "automatic",
        }
    raise CloudError(f"{SERVICE} doesn't have a mode called “{model_id}”.")


class MagnificProvider:
    id = "magnific"
    name = SERVICE
    models = MODELS

    def price_usd(self, model_id: str, size: tuple[int, int], scale: float) -> float | None:
        """Precision's price isn't published (None). Creative's follows from the size it really makes
        (it stops at about 25 megapixels); None for a picture too big for it, which it refuses."""
        find_model(self.models, model_id, SERVICE)
        if model_id != "creative":
            return None
        try:
            factor, (width, height) = creative_plan(*size, scale)
        except CloudError:
            return None
        made = min(width * height * int(factor.removesuffix("x")) ** 2, CREATIVE_MAX_PIXELS)
        return creative_price_usd(made / 1e6)

    def check_key(self, key: str) -> tuple[bool, str]:
        key = clean_key(key)
        if complaint := key_complaint(key, "Paste your Magnific API key first."):
            return False, complaint
        try:
            with new_client() as client:
                response = send(client, "GET", f"{API}/{PATHS['precision_v2']}", service=SERVICE, headers=_auth(key))
        except CloudError as exc:
            return False, exc.user_message
        if response.status_code == 200:
            return True, "Connected to Magnific."
        if response.status_code == 401:
            return False, "Magnific didn't accept this key. Check that you copied the whole key and try again."
        if response.status_code in (402, 403):
            return (
                False,
                "Magnific knows this key, but your plan can't use the API. Step 1 of the guide explains the options.",
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
        progress(0.0, "Getting your picture ready for Magnific…")
        pixels, lossy = source_pixels(image)
        factor, size = upload_plan(model_id, pixels.shape[1], pixels.shape[0], scale)
        upload = encode_upload(resize_pixels(pixels, size), lossy)
        del pixels
        path, body = magnific_request(model_id, upload, factor)
        auth = _auth(key)
        with new_client() as client:
            message = "Sending your picture to Magnific…"
            progress(UPLOADING_AT, message)
            content, content_headers = json_upload(body, message, progress=progress, cancel=cancel)
            created = send(
                client,
                "POST",
                f"{API}/{path}",
                service=SERVICE,
                cancel=cancel,
                content=content,
                headers=auth | content_headers,
            )
            task = _task(json_body(_ok(created), SERVICE))
            progress(WAITING_AT, "Waiting for Magnific…")
            output_url = _wait_for_output(client, path, task, auth, progress, cancel)
            progress(DOWNLOADING_AT, "Downloading the finished picture from Magnific…")
            result = download_image(
                client,
                output_url,
                service=SERVICE,
                cancel=cancel,
                max_pixels=result_pixel_limit(upload, factor),
                note=LINK_NOTE,
            )
        progress(1.0, "Magnific finished.")
        return result


def _auth(key: str) -> dict[str, str]:
    return {"x-magnific-api-key": header_key(key, SERVICE)}


def _task(body: dict) -> dict:
    task = body.get("data")
    if not isinstance(task, dict) or not task.get("task_id"):
        raise CloudError("Magnific didn't start the job. Please try again.")
    return task


def _wait_for_output(
    client: httpx.Client, path: str, task: dict, auth: dict[str, str], progress: ProgressFn, cancel: threading.Event
) -> str:
    latest = task
    status_url = f"{API}/{path}/{task['task_id']}"

    def check() -> str | Pending:
        nonlocal latest
        if latest.get("status") not in ("COMPLETED", "FAILED"):
            response = send(client, "GET", status_url, service=SERVICE, cancel=cancel, headers=auth)
            latest = _task(json_body(_ok(response), SERVICE))
        return _outcome(latest)

    # No cancel endpoint: if the user cancels, Magnific may still finish (and charge for) the job.
    # The guide's cost note tells people so.
    return poll_until_done(check, service=SERVICE, progress=progress, cancel=cancel)


def _outcome(task: dict) -> str | Pending:
    match task.get("status"):
        case "COMPLETED":
            generated = task.get("generated") or []
            if not generated or not isinstance(generated[0], str):
                raise JobFailed(finished_but(SERVICE, "it didn't send the picture back"))
            return generated[0]
        case "FAILED":
            raise JobFailed("Magnific couldn't upscale this picture. Please try again, or try a different mode.")
        case "CREATED":
            return Pending("Magnific has your picture and will start soon…")
    return Pending("Magnific is upscaling your picture…")


def _ok(response: httpx.Response) -> httpx.Response:
    return ok_or_raise(response, _http_error_message)


def _http_error_message(response: httpx.Response) -> str:
    code = response.status_code
    detail = error_detail(response)
    if code == 401:
        return "Magnific didn't accept your key. Connect Magnific again in Settings with a fresh key."
    if code in (402, 403):
        return (
            "Magnific says your plan can't use the API right now, or it has run out of credits. "
            "Check your plan and credits on magnific.com."
        )
    if code in (400, 422):
        return f"Magnific didn't accept this picture.{said(detail)}"
    if code == 429 or code >= 500:
        return busy_message(SERVICE)
    return f"Magnific couldn't finish the upscale.{said(detail)}"
