"""Cloud providers, keys and guides. Every HTTP call is mocked with respx; the response
shapes come from each provider's official docs (see the comments at the top of each module)."""

import base64
import email.parser
import io
import json
import math
import struct
import threading
import zlib

import httpx
import keyring
import keyring.backend
import numpy as np
import pytest
import respx
from PIL import Image

from upscaler.cloud import PROVIDERS, base, get_provider, keys, magnific
from upscaler.cloud.base import CloudError
from upscaler.cloud.fal import fal_request
from upscaler.cloud.guides import GUIDES
from upscaler.cloud.magnific import CREATIVE_MAX_PIXELS, creative_plan
from upscaler.cloud.replicate import replicate_scale_factor
from upscaler.config import Cancelled
from upscaler.imageio_utils import load_image

KEY = "test-key-123"
OUTPUT_W, OUTPUT_H = 256, 192


def png_bytes(width: int, height: int, color=(200, 120, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(base, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(base, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(base, "LOST_TOUCH_SECONDS", 2.0)  # a test that never reconnects still ends quickly


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "chair photo.png"
    gradient = np.linspace(0, 255, 64 * 48 * 3).reshape(48, 64, 3).astype(np.uint8)
    Image.fromarray(gradient).save(path)
    return path


class Recorder:
    def __init__(self):
        self.calls: list[tuple[float, str]] = []

    def __call__(self, fraction: float, message: str) -> None:
        self.calls.append((fraction, message))

    def assert_sane(self):
        fractions = [f for f, _ in self.calls]
        assert fractions, "progress was never reported"
        assert all(0.0 <= f <= 1.0 for f in fractions)
        assert fractions == sorted(fractions)
        assert all(message for _, message in self.calls)


def sent_json(route) -> dict:
    return json.loads(route.calls.last.request.content)


# --- fal.ai -------------------------------------------------------------------------------

FAL_RID = "764cabcf-b745-4b3e-ae38-1200304cf45b"
FAL_BASE = f"https://queue.fal.run/topaz/upscale/requests/{FAL_RID}"
FAL_MEDIA = "https://v3.fal.media/files/penguin/out.png"


def mock_fal_submit_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": FAL_RID,
            "response_url": FAL_BASE,
            "status_url": f"{FAL_BASE}/status",
            "cancel_url": f"{FAL_BASE}/cancel",
            "queue_position": 0,
        },
    )


def mock_fal_submit(app: str, router=respx):
    return router.post(f"https://queue.fal.run/{app}").mock(return_value=mock_fal_submit_response())


@respx.mock
def test_fal_upscale_flow(photo):
    submit = mock_fal_submit("topaz/upscale/image/precision")
    respx.get(f"{FAL_BASE}/status").mock(
        side_effect=[
            httpx.Response(202, json={"status": "IN_QUEUE", "queue_position": 2}),
            httpx.Response(202, json={"status": "IN_PROGRESS", "logs": []}),
            httpx.Response(200, json={"status": "COMPLETED", "logs": [], "metrics": {"inference_time": 3.1}}),
        ]
    )
    respx.get(FAL_BASE).mock(
        return_value=httpx.Response(
            200,
            json={"image": {"url": FAL_MEDIA, "content_type": "image/png", "file_name": "out.png", "file_size": 1234}},
        )
    )
    media = respx.get(FAL_MEDIA).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    progress = Recorder()

    out = get_provider("fal").upscale(photo, "topaz_precision", 4.0, KEY, progress=progress, cancel=threading.Event())

    assert out.shape == (OUTPUT_H, OUTPUT_W, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (200, 120, 40)
    request = submit.calls.last.request
    assert request.headers["Authorization"] == f"Key {KEY}"
    assert request.headers["X-Fal-Store-IO"] == "0"
    body = sent_json(submit)
    assert body["face_enhancement"] is False
    assert body["model"] == "High Fidelity V2"
    assert body["upscale_factor"] == 4.0
    assert body["image_url"].startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(body["image_url"].split(",", 1)[1])))
    assert decoded.size == (64, 48)
    assert "Authorization" not in media.calls.last.request.headers  # never send the key to the file host
    progress.assert_sane()


@pytest.mark.parametrize("model", [m.id for m in PROVIDERS["fal"].models])
def test_fal_requests_never_enhance_faces(model):
    app, body = fal_request(model, "data:image/png;base64,AAAA", 4.0)
    assert app and body["image_url"] == "data:image/png;base64,AAAA"
    if "face_enhancement" in body or app.startswith("topaz/"):
        assert body["face_enhancement"] is False
    if "creativity" in body:
        assert body["creativity"] == 0


def test_fal_topaz_factor_is_capped_at_4():
    _, body = fal_request("topaz_precision", "data:,", 5.2)
    assert body["upscale_factor"] == 4.0


def test_seedvr2_is_the_default_fal_model():
    assert PROVIDERS["fal"].models[0].id == "seedvr2"


FAL_KEY_CHECK = r"https://queue\.fal\.run/fal-ai/seedvr/requests/[0-9a-f-]{36}/status"


@respx.mock
def test_fal_bad_key():
    route = respx.get(url__regex=FAL_KEY_CHECK).mock(
        return_value=httpx.Response(401, json={"detail": "invalid key credentials"})
    )
    ok, message = get_provider("fal").check_key("nope:nope")
    assert ok is False
    assert "fal.ai" in message
    assert route.calls.last.request.headers["Authorization"] == "Key nope:nope"


@respx.mock
def test_fal_good_key_spends_nothing():
    route = respx.get(url__regex=FAL_KEY_CHECK).mock(return_value=httpx.Response(404, json={"status": "NOT_FOUND"}))
    ok, _ = get_provider("fal").check_key(KEY)
    assert ok is True
    assert len(respx.calls) == 1 and route.called


@respx.mock
def test_fal_no_credit_message(photo):
    respx.post("https://queue.fal.run/fal-ai/seedvr/upscale/image").mock(
        return_value=httpx.Response(403, json={"detail": "User is locked."})
    )
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "credit" in caught.value.user_message
    assert "Billing" in caught.value.user_message


@pytest.mark.respx(assert_all_called=False)
def test_fal_failed_request_reports_error(photo, respx_mock):
    mock_fal_submit("clarityai/crystal-upscaler", respx_mock)
    respx_mock.get(f"{FAL_BASE}/status").mock(
        return_value=httpx.Response(
            200, json={"status": "COMPLETED", "error": "Image too large", "error_type": "image_too_large"}
        )
    )
    cancel_route = respx_mock.put(f"{FAL_BASE}/cancel").mock(return_value=httpx.Response(400))
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "crystal", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "too big" in caught.value.user_message
    assert not cancel_route.called  # fal already ended the job


@respx.mock
def test_polling_failure_stops_the_job(photo, monkeypatch):
    monkeypatch.setattr(base, "LOST_TOUCH_SECONDS", 0.0)  # give up on the first failed status check
    mock_fal_submit("fal-ai/seedvr/upscale/image")
    status = respx.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(500))
    cancel_route = respx.put(f"{FAL_BASE}/cancel").mock(
        return_value=httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
    )
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    message = caught.value.user_message
    assert "fal.ai" in message and "account" in message
    assert "try again." not in message.lower().replace("before trying again", "")  # it may be charged already
    assert status.call_count == 3
    assert cancel_route.called  # so the job can't keep running, and charging, unseen


@respx.mock
def test_polling_keeps_waiting_through_a_dropped_connection(photo):
    """The job is paid for once it's sent, so a few seconds without Wi-Fi mustn't abandon it."""
    mock_fal_submit("fal-ai/seedvr/upscale/image")
    status = respx.get(f"{FAL_BASE}/status").mock(
        side_effect=[
            httpx.ConnectError("Wi-Fi off"),
            httpx.ConnectError("Wi-Fi off"),
            httpx.ConnectError("Wi-Fi off"),  # one whole status check fails
            httpx.Response(502),
            httpx.Response(502),
            httpx.Response(503),  # and another: the service is busy
            httpx.Response(200, json={"status": "COMPLETED"}),
        ]
    )
    respx.get(FAL_BASE).mock(return_value=httpx.Response(200, json={"image": {"url": FAL_MEDIA}}))
    respx.get(FAL_MEDIA).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    cancel_route = respx.put(f"{FAL_BASE}/cancel").mock(return_value=httpx.Response(202))
    progress = Recorder()

    out = get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=progress, cancel=threading.Event())

    assert out.shape == (OUTPUT_H, OUTPUT_W, 3)
    assert status.call_count == 7 and not cancel_route.called
    assert any("Lost touch with fal.ai" in message for _, message in progress.calls)
    progress.assert_sane()


@respx.mock
def test_polling_stops_at_once_on_a_definite_answer(photo):
    mock_fal_submit("fal-ai/seedvr/upscale/image")
    status = respx.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(401, json={"detail": "bad key"}))
    cancel_route = respx.put(f"{FAL_BASE}/cancel").mock(return_value=httpx.Response(202))
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "key" in caught.value.user_message
    assert status.call_count == 1 and cancel_route.called


@respx.mock
def test_cloud_cancel(photo):
    cancel = threading.Event()
    mock_fal_submit("fal-ai/seedvr/upscale/image")

    def still_working(request):
        cancel.set()  # the user clicks Cancel while fal is working
        return httpx.Response(202, json={"status": "IN_PROGRESS", "logs": []})

    respx.get(f"{FAL_BASE}/status").mock(side_effect=still_working)
    cancel_route = respx.put(f"{FAL_BASE}/cancel").mock(
        return_value=httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})
    )
    with pytest.raises(Cancelled):
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=cancel)
    assert cancel_route.called


# --- Replicate ----------------------------------------------------------------------------

REP_API = "https://api.replicate.com/v1"
REP_FILE = f"{REP_API}/files/cneqzikepnug6xezperrr4z55o"
REP_PREDICTION = f"{REP_API}/predictions/gm3qorzdhgbfurvjtvhg6dckhu"
REP_OUTPUT = "https://replicate.delivery/xezq/abc123/output.png"


def replicate_prediction(status: str, output=None, error=None) -> dict:
    return {
        "id": "gm3qorzdhgbfurvjtvhg6dckhu",
        "model": "topazlabs/image-upscale",
        "version": "hidden",
        "input": {},
        "logs": "",
        "output": output,
        "error": error,
        "status": status,
        "data_removed": False,
        "created_at": "2026-10-17T10:00:00.000000Z",
        "urls": {
            "web": "https://replicate.com/p/gm3qorzdhgbfurvjtvhg6dckhu",
            "get": REP_PREDICTION,
            "cancel": f"{REP_PREDICTION}/cancel",
        },
    }


@respx.mock
def test_replicate_upscale_flow(photo):
    upload = respx.post(f"{REP_API}/files").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "cneqzikepnug6xezperrr4z55o",
                "content_type": "image/png",
                "size": 1234,
                "checksums": {"sha256": "f0"},
                "metadata": {},
                "created_at": "2026-10-17T10:00:00Z",
                "expires_at": "2026-10-17T11:00:00Z",
                "urls": {"get": REP_FILE},
            },
        )
    )
    create = respx.post(f"{REP_API}/models/topazlabs/image-upscale/predictions").mock(
        return_value=httpx.Response(201, json=replicate_prediction("starting"))
    )
    respx.get(REP_PREDICTION).mock(
        side_effect=[
            httpx.Response(200, json=replicate_prediction("processing")),
            httpx.Response(200, json=replicate_prediction("succeeded", output=REP_OUTPUT)),
        ]
    )
    respx.get(REP_OUTPUT).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    delete = respx.delete(REP_FILE).mock(return_value=httpx.Response(204))
    progress = Recorder()

    out = get_provider("replicate").upscale(photo, "topaz_hf", 4.0, KEY, progress=progress, cancel=threading.Event())

    assert out.shape == (OUTPUT_H, OUTPUT_W, 3)
    assert upload.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"
    assert b'name="content"' in upload.calls.last.request.content
    form = email.parser.BytesParser().parsebytes(  # the upload is a well-formed form with the picture in it
        f"Content-Type: {upload.calls.last.request.headers['Content-Type']}\r\n\r\n".encode()
        + upload.calls.last.request.content
    )
    (part,) = form.get_payload()
    assert part.get_param("name", header="Content-Disposition") == "content"
    assert part.get_filename() == "image.png" and part.get_content_type() == "image/png"
    assert Image.open(io.BytesIO(part.get_payload(decode=True))).size == (64, 48)
    body = sent_json(create)["input"]
    assert body["face_enhancement"] is False
    assert body["enhance_model"] == "High Fidelity V2"
    assert body["upscale_factor"] == "4x"
    assert body["image"] == REP_FILE
    assert delete.called  # our upload does not linger on Replicate
    progress.assert_sane()


@respx.mock
def test_replicate_check_key_free():
    account = respx.get(f"{REP_API}/account").mock(
        return_value=httpx.Response(200, json={"type": "user", "username": "grandma", "name": "Grandma"})
    )
    ok, message = get_provider("replicate").check_key(KEY)
    assert ok is True and "grandma" in message
    assert account.call_count == 1 and len(respx.calls) == 1
    assert account.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"


@respx.mock
def test_replicate_bad_key():
    respx.get(f"{REP_API}/account").mock(
        return_value=httpx.Response(
            401,
            json={"title": "Unauthenticated", "detail": "You did not pass a valid authentication token", "status": 401},
        )
    )
    ok, message = get_provider("replicate").check_key("r8_wrong")
    assert ok is False and "Replicate" in message


@respx.mock
def test_retries_on_server_error():
    account = respx.get(f"{REP_API}/account").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"type": "user", "username": "sam"})]
    )
    assert get_provider("replicate").check_key(KEY)[0] is True
    assert account.call_count == 2


@respx.mock
def test_replicate_failed_prediction_deletes_upload(photo):
    respx.post(f"{REP_API}/files").mock(
        return_value=httpx.Response(201, json={"id": "cneqzikepnug6xezperrr4z55o", "urls": {"get": REP_FILE}})
    )
    respx.post(f"{REP_API}/models/topazlabs/image-upscale/predictions").mock(
        return_value=httpx.Response(201, json=replicate_prediction("failed", error="Out of memory"))
    )
    delete = respx.delete(REP_FILE).mock(return_value=httpx.Response(204))
    with pytest.raises(CloudError) as caught:
        get_provider("replicate").upscale(photo, "topaz_hf", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "Out of memory" in caught.value.user_message
    assert delete.called


@pytest.mark.parametrize(
    ("scale", "factor"), [(2.0, "2x"), (2.4, "2x"), (4.0, "4x"), (4.9, "4x"), (5.5, "6x"), (9.0, "6x")]
)
def test_replicate_scale_factor(scale, factor):
    assert replicate_scale_factor(scale) == factor


# --- Magnific -----------------------------------------------------------------------------

MAG_API = "https://api.magnific.com/v1/ai"
MAG_TASK = "046b6c7f-0b8a-43b9-b35d-6489e6daee91"
MAG_OUTPUT = "https://ai-statics.freepik.com/completed_task_image.png"


def magnific_task(status: str, generated: list[str] | None = None) -> dict:
    return {"data": {"generated": generated or [], "task_id": MAG_TASK, "status": status}}


@respx.mock
def test_magnific_flow(photo):
    submit = respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("CREATED"))
    )
    respx.get(f"{MAG_API}/image-upscaler-precision-v2/{MAG_TASK}").mock(
        side_effect=[
            httpx.Response(200, json=magnific_task("IN_PROGRESS")),
            httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT])),
        ]
    )
    respx.get(MAG_OUTPUT).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    progress = Recorder()

    out = get_provider("magnific").upscale(photo, "precision_v2", 4.0, KEY, progress=progress, cancel=threading.Event())

    assert out.shape == (OUTPUT_H, OUTPUT_W, 3)
    assert submit.calls.last.request.headers["x-magnific-api-key"] == KEY
    body = sent_json(submit)
    assert body["scale_factor"] == 4
    assert body["flavor"] == "photo"
    assert not body["image"].startswith("data:")  # Magnific wants plain base64
    assert Image.open(io.BytesIO(base64.b64decode(body["image"]))).size == (64, 48)
    progress.assert_sane()


@respx.mock
def test_magnific_creative_keeps_creativity_at_zero(photo):
    submit = respx.post(f"{MAG_API}/image-upscaler").mock(
        return_value=httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT]))
    )
    respx.get(MAG_OUTPUT).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    get_provider("magnific").upscale(photo, "creative", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    body = sent_json(submit)
    assert body["creativity"] == 0
    assert body["scale_factor"] == "4x"
    assert body["optimized_for"] == "standard"


@respx.mock
def test_magnific_failed_task(photo):
    respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("FAILED"))
    )
    with pytest.raises(CloudError) as caught:
        get_provider("magnific").upscale(photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "Magnific" in caught.value.user_message


@respx.mock
def test_magnific_check_key():
    route = respx.get(f"{MAG_API}/image-upscaler-precision-v2").mock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(401, json={"message": "Invalid API key"}),
        ]
    )
    assert get_provider("magnific").check_key(KEY)[0] is True
    ok, message = get_provider("magnific").check_key("wrong")
    assert ok is False and "Magnific" in message
    assert route.calls.last.request.headers["x-magnific-api-key"] == "wrong"


def test_magnific_creative_stays_under_25_megapixels():
    assert creative_plan(1000, 750, 4.0) == ("4x", (1000, 750))  # 4000x3000 = 12 MP
    assert creative_plan(500, 400, 8.0) == ("8x", (500, 400))
    # 2x of the whole picture already comes close to the limit, so it isn't shrunk for 4x.
    assert creative_plan(3000, 2000, 8192 / 3000) == ("2x", (3000, 2000))
    with pytest.raises(CloudError):
        creative_plan(3000, 3000, 2.0)  # even 2x is 36 MP


@pytest.mark.parametrize(("width", "height"), [(2048, 1539), (1920, 1080), (1700, 1275)])
def test_magnific_creative_sends_a_smaller_copy_rather_than_coming_back_far_too_small(width, height):
    """4x of these is over 25 MP, and 2x would leave the app to stretch the result 2-2.4x for 8K.
    A slightly smaller copy at 4x lands just under the limit instead."""
    scale = 8192 / width
    factor, (sent_w, sent_h) = creative_plan(width, height, scale)
    assert factor == "4x"
    assert sent_w < width and sent_h < height
    assert sent_w * sent_h * 16 <= CREATIVE_MAX_PIXELS
    assert sent_w * sent_h * 16 >= 0.98 * CREATIVE_MAX_PIXELS
    assert abs(sent_w / sent_h - width / height) < 0.01  # the same shape
    assert 8192 / (sent_w * 4) < 1.45  # the app's final resize is small (was 2-2.4x)


@respx.mock
def test_magnific_creative_upload_is_shrunk_to_fit(photo, monkeypatch):
    monkeypatch.setattr(magnific, "CREATIVE_MAX_PIXELS", 64 * 48 * 16 // 2)  # 4x of the photo is twice the limit
    submit = respx.post(f"{MAG_API}/image-upscaler").mock(
        return_value=httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT]))
    )
    respx.get(MAG_OUTPUT).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    get_provider("magnific").upscale(photo, "creative", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    body = sent_json(submit)
    width, height = Image.open(io.BytesIO(base64.b64decode(body["image"]))).size
    assert body["scale_factor"] == "4x"
    assert (width, height) == (45, 33)
    assert width * height * 16 <= magnific.CREATIVE_MAX_PIXELS


@respx.mock
def test_magnific_polling_keeps_waiting_through_a_dropped_connection(photo):
    respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("CREATED"))
    )
    status = respx.get(f"{MAG_API}/image-upscaler-precision-v2/{MAG_TASK}").mock(
        side_effect=[httpx.ConnectError("no route")] * 3
        + [httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT]))]
    )
    respx.get(MAG_OUTPUT).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    out = get_provider("magnific").upscale(
        photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event()
    )
    assert out.shape == (OUTPUT_H, OUTPUT_W, 3) and status.call_count == 4


# --- Paid submissions are never sent twice ------------------------------------------------


@pytest.mark.parametrize("code", [500, 502, 504])
@respx.mock
def test_fal_submission_is_not_resent_after_a_server_error(photo, code):
    """After a 500, 502 or 504 the job may have started anyway: sending it again could pay twice."""
    submit = respx.post("https://queue.fal.run/fal-ai/seedvr/upscale/image").mock(
        side_effect=[httpx.Response(code), mock_fal_submit_response()]
    )
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert submit.call_count == 1
    message = caught.value.user_message
    assert "may or may not" in message and "fal.ai account" in message


@pytest.mark.parametrize("code", [429, 503])
@pytest.mark.respx(assert_all_called=False)
def test_fal_submission_is_resent_when_the_service_turned_it_away(photo, code, respx_mock):
    submit = respx_mock.post("https://queue.fal.run/fal-ai/seedvr/upscale/image").mock(
        side_effect=[httpx.Response(code), mock_fal_submit_response()]
    )
    respx_mock.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    respx_mock.get(FAL_BASE).mock(return_value=httpx.Response(200, json={"image": {"url": FAL_MEDIA}}))
    respx_mock.get(FAL_MEDIA).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert submit.call_count == 2
    assert sent_json(submit)["image_url"].startswith("data:image/png;base64,")  # the whole body, sent again


@respx.mock
def test_magnific_submission_is_not_resent_after_a_server_error(photo):
    submit = respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        side_effect=[httpx.Response(502), httpx.Response(500), httpx.Response(200, json=magnific_task("CREATED"))]
    )
    with pytest.raises(CloudError) as caught:
        get_provider("magnific").upscale(photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert submit.call_count == 1
    assert "Magnific account" in caught.value.user_message


@respx.mock
def test_slow_submission_answer_is_not_resent(photo):
    submit = respx.post("https://queue.fal.run/fal-ai/seedvr/upscale/image").mock(
        side_effect=[httpx.ReadTimeout("slow"), mock_fal_submit_response()]
    )
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert submit.call_count == 1
    assert "may or may not" in caught.value.user_message


@respx.mock
def test_replicate_prediction_is_not_resent_after_a_server_error(photo):
    upload = respx.post(f"{REP_API}/files").mock(
        side_effect=[  # uploading the picture again is free, so that is still retried
            httpx.Response(500),
            httpx.Response(201, json={"id": "cneqzikepnug6xezperrr4z55o", "urls": {"get": REP_FILE}}),
        ]
    )
    create = respx.post(f"{REP_API}/models/topazlabs/image-upscale/predictions").mock(
        side_effect=[httpx.Response(504), httpx.Response(201, json=replicate_prediction("starting"))]
    )
    respx.delete(REP_FILE).mock(return_value=httpx.Response(204))
    with pytest.raises(CloudError) as caught:
        get_provider("replicate").upscale(photo, "topaz_hf", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert upload.call_count == 2 and create.call_count == 1
    assert "Replicate account" in caught.value.user_message


# --- After the service has finished (and charged) -----------------------------------------


@pytest.mark.parametrize(
    "answer", [httpx.Response(404), httpx.Response(403), httpx.ConnectError("Wi-Fi off")], ids=["404", "403", "offline"]
)
@pytest.mark.respx(assert_all_called=False)
def test_download_failure_doesnt_invite_a_blind_retry(photo, monkeypatch, respx_mock, answer):
    monkeypatch.setattr(base, "LOST_TOUCH_SECONDS", 0.0)
    respx_mock.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT]))
    )
    respx_mock.get(MAG_OUTPUT).mock(side_effect=[answer] * 3)
    with pytest.raises(CloudError) as caught:
        get_provider("magnific").upscale(photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    message = caught.value.user_message
    assert "finished" in message and "Magnific account" in message
    assert "Please try again" not in message
    assert "30 minutes" in message  # Magnific's links expire


@respx.mock
def test_download_waits_through_a_dropped_connection(photo):
    respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("COMPLETED", [MAG_OUTPUT]))
    )
    media = respx.get(MAG_OUTPUT).mock(
        side_effect=[httpx.ConnectError("Wi-Fi off")] * 3
        + [httpx.Response(503), httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H))]
    )
    out = get_provider("magnific").upscale(
        photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event()
    )
    assert out.shape == (OUTPUT_H, OUTPUT_W, 3) and media.call_count == 5


@pytest.mark.respx(assert_all_called=False)
def test_fal_finished_without_a_picture_doesnt_invite_a_blind_retry(photo, respx_mock):
    mock_fal_submit("fal-ai/seedvr/upscale/image", respx_mock)
    respx_mock.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    respx_mock.get(FAL_BASE).mock(return_value=httpx.Response(200, json={"image": None}))
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "fal.ai account" in caught.value.user_message
    assert "Please try again" not in caught.value.user_message


@pytest.mark.respx(assert_all_called=False)
def test_fal_result_fetch_failure_doesnt_invite_a_blind_retry(photo, monkeypatch, respx_mock):
    monkeypatch.setattr(base, "LOST_TOUCH_SECONDS", 0.0)
    mock_fal_submit("fal-ai/seedvr/upscale/image", respx_mock)
    respx_mock.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    respx_mock.get(FAL_BASE).mock(return_value=httpx.Response(500))
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "fal.ai account" in caught.value.user_message
    assert "Please try again" not in caught.value.user_message


@respx.mock
def test_replicate_finished_without_a_picture_doesnt_invite_a_blind_retry(photo):
    respx.post(f"{REP_API}/files").mock(
        return_value=httpx.Response(201, json={"id": "cneqzikepnug6xezperrr4z55o", "urls": {"get": REP_FILE}})
    )
    respx.post(f"{REP_API}/models/topazlabs/image-upscale/predictions").mock(
        return_value=httpx.Response(201, json=replicate_prediction("succeeded", output=None))
    )
    respx.delete(REP_FILE).mock(return_value=httpx.Response(204))
    with pytest.raises(CloudError) as caught:
        get_provider("replicate").upscale(photo, "topaz_hf", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "Replicate account" in caught.value.user_message
    assert "Please try again" not in caught.value.user_message


@respx.mock
def test_magnific_finished_without_a_picture_doesnt_invite_a_blind_retry(photo):
    respx.post(f"{MAG_API}/image-upscaler-precision-v2").mock(
        return_value=httpx.Response(200, json=magnific_task("COMPLETED", []))
    )
    with pytest.raises(CloudError) as caught:
        get_provider("magnific").upscale(photo, "precision_v2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "Magnific account" in caught.value.user_message
    assert "Please try again" not in caught.value.user_message


@pytest.mark.respx(assert_all_called=False)
def test_taking_too_long_doesnt_invite_a_blind_retry(photo, monkeypatch, respx_mock):
    monkeypatch.setattr(base, "MAX_WAIT_SECONDS", -1.0)
    mock_fal_submit("fal-ai/seedvr/upscale/image", respx_mock)
    respx_mock.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(202, json={"status": "IN_PROGRESS"}))
    cancel_route = respx_mock.put(f"{FAL_BASE}/cancel").mock(return_value=httpx.Response(202))
    with pytest.raises(CloudError) as caught:
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "fal.ai account" in caught.value.user_message
    assert "try again later" not in caught.value.user_message
    assert cancel_route.called


# --- Uploading ----------------------------------------------------------------------------


@respx.mock
def test_fal_upload_can_be_cancelled_part_way(photo, monkeypatch):
    """A big picture can take minutes to send on a home connection; Cancel mustn't wait for all of it."""
    monkeypatch.setattr(base, "UPLOAD_CHUNK_BYTES", 64)
    cancel = threading.Event()
    submit = mock_fal_submit("fal-ai/seedvr/upscale/image")
    seen: list[float] = []

    def progress(fraction: float, message: str) -> None:
        seen.append(fraction)
        if base.UPLOADING_AT < fraction < base.WAITING_AT:
            cancel.set()  # the user clicks Cancel while the picture is on its way

    with pytest.raises(Cancelled):
        get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=progress, cancel=cancel)
    assert not submit.called  # the request never finished, so fal never started the job
    assert any(base.UPLOADING_AT < f < base.WAITING_AT for f in seen)
    assert max(seen) < base.WAITING_AT


@respx.mock
def test_fal_upload_reports_progress(photo, monkeypatch):
    monkeypatch.setattr(base, "UPLOAD_CHUNK_BYTES", 64)
    mock_fal_submit("fal-ai/seedvr/upscale/image")
    respx.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    respx.get(FAL_BASE).mock(return_value=httpx.Response(200, json={"image": {"url": FAL_MEDIA}}))
    respx.get(FAL_MEDIA).mock(return_value=httpx.Response(200, content=png_bytes(OUTPUT_W, OUTPUT_H)))
    progress = Recorder()
    get_provider("fal").upscale(photo, "seedvr2", 4.0, KEY, progress=progress, cancel=threading.Event())
    sending = [f for f, message in progress.calls if base.UPLOADING_AT < f < base.WAITING_AT]
    assert len(sending) >= 3
    assert any("%" in message for _, message in progress.calls)
    progress.assert_sane()


def test_prepare_upload_sends_phone_photos_upright(tmp_path):
    path = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90°", as phones write it
    Image.new("RGB", (40, 20), (90, 90, 90)).save(path, exif=exif)
    upload = base.prepare_upload(path)
    assert upload.mime == "image/jpeg"
    assert (upload.width, upload.height) == (20, 40)
    assert Image.open(io.BytesIO(upload.data)).size == (20, 40)


def test_prepare_upload_16bit_grey_keeps_its_brightness(tmp_path):
    path = tmp_path / "scan.png"
    Image.fromarray(np.full((10, 12), 32768, dtype=np.uint16)).save(path)
    upload = base.prepare_upload(path)
    pixels = np.asarray(Image.open(io.BytesIO(upload.data)))
    assert upload.mime == "image/png" and pixels.shape == (10, 12, 3)
    assert abs(float(pixels.mean()) - 128) <= 1


_SHORT, _LONG, _RATIONAL, _SRATIONAL, _BYTE, _ASCII = 3, 4, 5, 10, 1, 2


def _tiff_directory(entries: list[tuple[int, int, object]], at: int, data: bytes) -> bytes:
    """One little-endian TIFF directory written at offset `at`, then its longer values, then `data`
    (which a StripOffsets entry, tag 273, points at)."""

    def pack(kind: int, values) -> tuple[bytes, int]:
        if kind == _ASCII:
            return values.encode() + b"\0", len(values) + 1
        if kind == _BYTE:
            return bytes(values), len(values)
        if kind in (_RATIONAL, _SRATIONAL):
            pair = "ii" if kind == _SRATIONAL else "II"
            return struct.pack("<" + pair * len(values), *[x for v in values for x in v]), len(values)
        return struct.pack(f"<{len(values)}{'H' if kind == _SHORT else 'I'}", *values), len(values)

    packed = [(tag, kind, *pack(kind, values)) for tag, kind, values in sorted(entries)]
    extra_at = at + 2 + 12 * len(packed) + 4
    extra = b"".join(payload + b"\0" * (len(payload) % 2) for _, _, payload, _ in packed if len(payload) > 4)
    directory, offset = struct.pack("<H", len(packed)), extra_at
    for tag, kind, payload, count in packed:
        if tag == 273:
            payload = struct.pack("<I", extra_at + len(extra))
        if len(payload) <= 4:
            directory += struct.pack("<HHI", tag, kind, count) + payload.ljust(4, b"\0")
        else:
            directory += struct.pack("<HHII", tag, kind, count, offset)
            offset += len(payload) + len(payload) % 2
    return directory + struct.pack("<I", 0) + extra + data


def write_dng_with_preview(path, preview=(40, 30), sensor=(96, 64)) -> None:
    """A DNG laid out like Adobe DNG Converter's and many cameras' RAW files: the first directory is a
    small RGB preview (all Pillow sees) and a sub-directory holds the real sensor picture."""
    (pw, ph), (sw, sh) = preview, sensor
    thumb = np.full((ph, pw, 3), 128, np.uint8).tobytes()
    cfa = np.full((sh, sw), 6000, "<u2")
    cfa[0::2, 0::2] = 50000
    sensor_data = cfa.tobytes()

    def main(sub_at: int) -> list:
        return [
            (254, _LONG, [1]), (256, _LONG, [pw]), (257, _LONG, [ph]), (258, _SHORT, [8, 8, 8]), (259, _SHORT, [1]),
            (262, _SHORT, [2]), (271, _ASCII, "Re-Solution"), (272, _ASCII, "Test DNG"), (273, _LONG, [0]),
            (277, _SHORT, [3]), (278, _LONG, [ph]), (279, _LONG, [len(thumb)]), (284, _SHORT, [1]),
            (330, _LONG, [sub_at]), (50706, _BYTE, [1, 4, 0, 0]), (50708, _ASCII, "Re-Solution Test DNG"),
            (50721, _SRATIONAL, [(1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1)]),
            (50728, _RATIONAL, [(1, 1), (1, 1), (1, 1)]), (50778, _SHORT, [21]),
        ]  # fmt: skip

    sub = [
        (254, _LONG, [0]), (256, _LONG, [sw]), (257, _LONG, [sh]), (258, _SHORT, [16]), (259, _SHORT, [1]),
        (262, _SHORT, [32803]), (273, _LONG, [0]), (277, _SHORT, [1]), (278, _LONG, [sh]),
        (279, _LONG, [len(sensor_data)]), (284, _SHORT, [1]), (33421, _SHORT, [2, 2]), (33422, _BYTE, [0, 1, 1, 2]),
        (50717, _LONG, [65535]),
    ]  # fmt: skip
    size = len(_tiff_directory(main(0), 8, thumb))
    sub_at = 8 + size + size % 2
    first = _tiff_directory(main(sub_at), 8, thumb)
    second = _tiff_directory(sub, sub_at, sensor_data)
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + first + b"\0" * (size % 2) + second)


def write_exr(path, width: int, height: int, value: float) -> None:
    """A tiny uncompressed float RGB EXR, which Pillow can't open."""

    def attribute(name: str, kind: str, payload: bytes) -> bytes:
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(payload)) + payload

    window = struct.pack("<4i", 0, 0, width - 1, height - 1)
    channels = b"".join(name + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for name in (b"B", b"G", b"R")) + b"\0"
    header = b"v/1\x01" + struct.pack("<I", 2) + b"".join([
        attribute("channels", "chlist", channels),
        attribute("compression", "compression", b"\0"),
        attribute("dataWindow", "box2i", window),
        attribute("displayWindow", "box2i", window),
        attribute("lineOrder", "lineOrder", b"\0"),
        attribute("pixelAspectRatio", "float", struct.pack("<f", 1)),
        attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0)),
        attribute("screenWindowWidth", "float", struct.pack("<f", 1)),
    ]) + b"\0"  # fmt: skip
    row = struct.pack(f"<{3 * width}f", *[value] * (3 * width))
    blocks = [struct.pack("<ii", y, len(row)) + row for y in range(height)]
    first_block = len(header) + 8 * height
    offsets = struct.pack(f"<{height}Q", *(first_block + i * len(blocks[0]) for i in range(height)))
    path.write_bytes(header + offsets + b"".join(blocks))


def test_prepare_upload_sends_the_raw_photo_not_its_little_preview(tmp_path):
    """Pillow only sees the 40x30 preview inside this DNG; the service must get the real 96x64 photo,
    exactly as the pipeline decodes it, or it would be paid to upscale a thumbnail."""
    path = tmp_path / "camera.dng"
    write_dng_with_preview(path, preview=(40, 30), sensor=(96, 64))
    with Image.open(path) as pillow_view:
        assert pillow_view.size == (40, 30)
    pipeline_view = load_image(path)

    upload = base.prepare_upload(path)

    assert (upload.width, upload.height) == (96, 64)
    assert Image.open(io.BytesIO(upload.data)).size == (96, 64)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(upload.data)).convert("RGB")), pipeline_view.rgb)


def test_prepare_upload_reads_exr(tmp_path):
    path = tmp_path / "render.exr"
    write_exr(path, 32, 24, 0.5)
    upload = base.prepare_upload(path)
    assert (upload.width, upload.height) == (32, 24) and upload.mime == "image/png"


def test_prepare_upload_takes_pixels_the_app_already_has():
    pixels = np.zeros((24, 32, 3), np.uint8)
    pixels[:, :16] = (250, 10, 10)
    upload = base.prepare_upload(pixels)
    assert upload.mime == "image/png" and (upload.width, upload.height) == (32, 24)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(upload.data))), pixels)


def test_prepare_upload_takes_the_picture_the_pipeline_opened(tmp_path):
    """Nothing is decoded twice, and a JPEG source still goes up as JPEG rather than a bigger PNG."""
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (40, 20), (90, 90, 90)).save(path)
    opened = load_image(path)
    upload = base.prepare_upload(opened)
    assert upload.mime == "image/jpeg" and (upload.width, upload.height) == (40, 20)
    opened.format = "PNG"
    assert base.prepare_upload(opened).mime == "image/png"


def test_prepare_upload_sends_big_photos_as_jpeg(monkeypatch):
    """A big photo as PNG is several times the size of a JPEG that looks the same: slow to send, and
    services advise against huge inline uploads."""
    noisy = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    assert base.prepare_upload(noisy).mime == "image/png"
    monkeypatch.setattr(base, "PNG_UPLOAD_MAX_BYTES", 1000)
    upload = base.prepare_upload(noisy)
    assert upload.mime == "image/jpeg" and (upload.width, upload.height) == (64, 48)


@respx.mock
def test_fal_upscale_of_a_raw_photo_sends_the_real_photo(tmp_path):
    path = tmp_path / "camera.dng"
    write_dng_with_preview(path, preview=(40, 30), sensor=(96, 64))
    submit = mock_fal_submit("fal-ai/seedvr/upscale/image")
    respx.get(f"{FAL_BASE}/status").mock(return_value=httpx.Response(200, json={"status": "COMPLETED"}))
    respx.get(FAL_BASE).mock(return_value=httpx.Response(200, json={"image": {"url": FAL_MEDIA}}))
    respx.get(FAL_MEDIA).mock(return_value=httpx.Response(200, content=png_bytes(192, 128)))
    get_provider("fal").upscale(path, "seedvr2", 2.0, KEY, progress=Recorder(), cancel=threading.Event())
    sent = Image.open(io.BytesIO(base64.b64decode(sent_json(submit)["image_url"].split(",", 1)[1])))
    assert sent.size == (96, 64)


# --- Keys with odd characters, unknown modes ----------------------------------------------

KEY_CHECK_ROUTES = {
    "fal": {"url__regex": FAL_KEY_CHECK},
    "replicate": {"url": f"{REP_API}/account"},
    "magnific": {"url": f"{MAG_API}/image-upscaler-precision-v2"},
}


def sent_key(request: httpx.Request) -> str:
    for name in ("Authorization", "x-magnific-api-key"):
        if name in request.headers:
            return request.headers[name].split()[-1]
    raise AssertionError("no key was sent")


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
@pytest.mark.parametrize("pasted", ["abcd…wxyz", "abc\u200bdef", "r8_abc\u00a0def"])
@pytest.mark.respx(assert_all_called=False)
def test_check_key_with_odd_characters_answers_kindly(provider_id, pasted, respx_mock):
    route = respx_mock.get(**KEY_CHECK_ROUTES[provider_id]).mock(
        return_value=httpx.Response(401, json={"detail": "invalid key credentials"})
    )
    ok, message = get_provider(provider_id).check_key(pasted)
    assert ok is False and message
    if "\u00a0" in pasted:  # a non-breaking space from an email or a note is simply dropped
        assert sent_key(route.calls.last.request) == "r8_abcdef"
    else:  # can't be part of any key, and can't even be sent
        assert "Copy the key again" in message
        assert not route.called


@pytest.mark.parametrize("provider_id", list(PROVIDERS))
@respx.mock
def test_unknown_mode_is_a_friendly_error(photo, provider_id):
    with pytest.raises(CloudError) as caught:
        get_provider(provider_id).upscale(photo, "nope", 4.0, KEY, progress=Recorder(), cancel=threading.Event())
    assert "doesn't have a mode called" in caught.value.user_message


# --- Big results --------------------------------------------------------------------------


def png_header(width: int, height: int) -> bytes:
    """A PNG that claims to be width x height but holds almost no pixel data, so a test can
    check size handling without decoding hundreds of megapixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    pixels = zlib.compress(b"\0" * 64)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


PHONE_4X = (16128, 12096)  # a 12 MP phone photo at 4x: 195 MP, over Pillow's default limit


def test_big_results_are_not_mistaken_for_decompression_bombs():
    upload = base.Upload(b"", "image/jpeg", 4032, 3024)
    limit = base.result_pixel_limit(upload, "4x")  # Replicate turns 3x into 4x
    assert limit >= PHONE_4X[0] * PHONE_4X[1]
    pillow_limit = Image.MAX_IMAGE_PIXELS

    with pytest.raises(CloudError) as caught:  # the crafted file has no real pixels, so it can't decode
        base.decode_image(png_header(*PHONE_4X), "Replicate", max_pixels=limit)

    assert not isinstance(caught.value.__cause__, Image.DecompressionBombError)
    assert "try again" not in caught.value.user_message.lower()  # trying again would pay again
    assert Image.MAX_IMAGE_PIXELS == pillow_limit


def test_results_far_bigger_than_asked_for_are_refused():
    with pytest.raises(CloudError) as caught:
        base.decode_image(png_header(*PHONE_4X), "fal.ai", max_pixels=1_000_000)
    message = caught.value.user_message
    assert "fal.ai" in message and "bigger" in message
    assert "try again" not in message.lower()


def test_result_pixel_limit_is_generous_for_small_jobs():
    tiny = base.Upload(b"", "image/png", 64, 48)
    assert base.result_pixel_limit(tiny, 4.0) >= 256 * 192
    assert base.result_pixel_limit(tiny, "2x") == base.result_pixel_limit(tiny, 2)


# --- Keys and guides ----------------------------------------------------------------------


class MemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError(username)
        del self.store[(service, username)]


@pytest.fixture
def memory_keyring():
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


def test_keys_roundtrip(memory_keyring):
    assert keys.get_key("fal") is None
    keys.set_key("fal", "  abc:def \n")
    assert keys.get_key("fal") == "abc:def"
    assert memory_keyring.store == {("Upscaler", "fal"): "abc:def"}
    keys.delete_key("fal")
    assert keys.get_key("fal") is None
    keys.delete_key("fal")  # deleting twice is fine


class RefusingKeyring(MemoryKeyring):
    """Like the macOS Keychain when someone clicks Deny, or it's locked: deleting fails, and keyring
    reports that as the same PasswordDeleteError it uses for "there was no such key"."""

    def delete_password(self, service, username):
        raise keyring.errors.PasswordDeleteError("User interaction is not allowed.")


def test_removing_a_key_the_keychain_wont_let_go_of_says_so():
    previous = keyring.get_keyring()
    backend = RefusingKeyring()
    keyring.set_keyring(backend)
    try:
        backend.store[("Upscaler", "fal")] = "abc:def"
        with pytest.raises(CloudError) as caught:
            keys.delete_key("fal")
        assert "keychain" in caught.value.user_message
        assert keys.get_key("fal") == "abc:def"
        del backend.store[("Upscaler", "fal")]
        keys.delete_key("fal")  # nothing there: that's fine
    finally:
        keyring.set_keyring(previous)


def test_providers_registry():
    assert list(PROVIDERS) == ["fal", "replicate", "magnific"]
    for provider_id, provider in PROVIDERS.items():
        assert provider.id == provider_id and provider.name and provider.models
        for model in provider.models:
            assert model.kind in ("faithful", "generative")
            assert model.est_cost_8k_usd >= 0 and model.label and model.description
    with pytest.raises(KeyError):
        get_provider("nope")


def test_guides_complete():
    assert set(GUIDES) == set(PROVIDERS)
    for provider_id, guide in GUIDES.items():
        assert guide.provider_id == provider_id
        assert len(guide.steps) >= 4
        assert guide.cost_note.strip() and guide.privacy_note.strip()
        assert guide.tagline.strip() and guide.best_for.strip() and guide.key_hint.strip()
        for step in guide.steps:
            assert step.title.strip() and step.body.strip()
            assert (step.link_url is None) == (step.link_label is None)
            if step.link_url is not None:
                assert step.link_url.startswith("https://")


def test_magnific_guide_mentions_business_plan():
    guide = GUIDES["magnific"]
    text = " ".join([guide.tagline, guide.best_for, guide.cost_note] + [s.body for s in guide.steps])
    assert "Business" in text


def test_magnific_guide_says_cancelling_still_spends_credits():
    """Magnific has no way to stop a job, so Cancel can't save the credits."""
    note = GUIDES["magnific"].cost_note.lower()
    assert "cancel" in note and "credits" in note


def test_guides_warn_that_16k_costs_more():
    for provider_id in ("fal", "replicate"):
        assert "16K" in GUIDES[provider_id].cost_note


def _published_price(provider_id: str, model_id: str, megapixels: float) -> float:
    """Each provider's published price for a result of `megapixels` (see the notes in fal.py and replicate.py)."""
    match provider_id, model_id:
        case "fal", "seedvr2":
            return 0.001 * megapixels
        case "fal", "topaz_precision":
            return 0.08 * math.ceil(megapixels / 24)  # per started 24 MP
        case "fal", "topaz_generative":
            return 0.08 * math.ceil(megapixels / 8)  # Wonder: per started 8 MP
        case "fal", "crystal":
            return 0.016 * megapixels
        case "replicate", "topaz_hf":  # its table of output MP -> price, priced by the next row up
            table = ((24, 0.05), (48, 0.10), (60, 0.15), (96, 0.20), (132, 0.24), (168, 0.29), (336, 0.53))
            return next(price for limit, price in table if megapixels <= limit)
    raise AssertionError(model_id)


def test_card_prices_are_for_an_8k_picture_made_from_a_phone_photo():
    """The commonest picture is a 4:3 phone photo (4032x3024). At the default 8K it becomes 8192x6144,
    or 8064x6048 on Replicate, which only offers 2x and 4x."""
    for provider_id, provider in PROVIDERS.items():
        for model in provider.models:
            if provider_id == "magnific":
                continue  # Magnific doesn't publish prices per picture
            if provider_id == "replicate":
                factor = int(replicate_scale_factor(8192 / 4032).removesuffix("x"))
                megapixels = 4032 * factor * 3024 * factor / 1e6
            else:
                megapixels = 8192 * 6144 / 1e6
            price = _published_price(provider_id, model.id, megapixels)
            assert model.est_cost_8k_usd == pytest.approx(price, abs=0.006)


def test_card_prices_match_what_the_estimate_charges_for_the_same_picture():
    """The card's 8K price and the estimate before sending come from the same price rules. Magnific
    Creative can't take a 12 MP photo (it makes 25.3 MP at most), so its card shows that biggest result,
    as an 8K job from a smaller picture makes."""
    for provider in PROVIDERS.values():
        for model in provider.models:
            source = (2048, 1536) if model.id == "creative" else (4032, 3024)
            price = provider.price_usd(model.id, source, 8192 / source[0])
            assert (price or 0.0) == pytest.approx(model.est_cost_8k_usd, abs=0.006)


def test_prices_follow_what_each_service_really_makes():
    fal_ai, replicate_com, magnific_com = PROVIDERS["fal"], PROVIDERS["replicate"], PROVIDERS["magnific"]
    photo = (2048, 1536)
    # Topaz on fal stops at 4x, so asking for more costs no more.
    assert fal_ai.price_usd("topaz_precision", photo, 6.0) == fal_ai.price_usd("topaz_precision", photo, 4.0)
    # Replicate only makes 2x, 4x or 6x: 3x comes back at 4x and is charged as 4x.
    assert replicate_com.price_usd("topaz_hf", photo, 3.0) == replicate_com.price_usd("topaz_hf", photo, 4.0) == 0.15
    assert replicate_com.price_usd("topaz_hf", (4000, 3000), 6.0) == pytest.approx(0.82)  # 432 MP: the 512 MP row
    # Magnific Precision comes out of the plan's credits at an unpublished rate; Creative stops at 25.3 MP,
    # and a picture already too big for it isn't priced (it's refused).
    assert magnific_com.price_usd("precision_v2", photo, 4.0) is None
    assert magnific_com.price_usd("creative", photo, 4.0) == magnific_com.price_usd("creative", photo, 8.0)
    assert magnific_com.price_usd("creative", (6000, 4500), 2.0) is None
    # Magnific's published examples: 2560x1920 €0.20, 5120x2880 €0.40, 5120x3840 €0.50.
    for size, euros in (((1280, 960), 0.20), ((2560, 1440), 0.40), ((2560, 1920), 0.50)):
        assert magnific_com.price_usd("creative", size, 2.0) == pytest.approx(euros * magnific.USD_PER_EUR, abs=0.01)


PROVIDER_SITES = {"fal": "fal.ai", "replicate": "replicate.com", "magnific": "magnific.com"}


def test_guide_links_stay_on_the_providers_own_site():
    """The wizard shows every link as a big button, so none may lead to a service the app can't use."""
    for provider_id, guide in GUIDES.items():
        for step in guide.steps:
            if step.link_url is not None:
                host = httpx.URL(step.link_url).host
                assert host == PROVIDER_SITES[provider_id] or host.endswith("." + PROVIDER_SITES[provider_id])
