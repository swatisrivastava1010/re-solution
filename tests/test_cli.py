import _thread
import json
import signal
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from upscaler import cli, config, hardware, jobs, pipeline_image, seedvr2, server
from upscaler.config import Cancelled
from upscaler.hardware import Hardware
from upscaler.pipeline_image import ImageResult

MAC_16GB = Hardware("mps", "Apple M4", 16, "mac", True)


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeEngine:
    name = "mlx"

    def __init__(self, downloaded=True):
        self.downloaded = downloaded
        self.events = []

    def weights_present(self):
        return self.downloaded

    def download_weights(self, progress, cancel):
        self.events.append("download")
        progress(0.5, "Downloading Photo-real")
        self.downloaded = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "home" / "Pictures"))  # Windows asks the system
    monkeypatch.setattr(cli, "KeepAwake", _NoKeepAwake)
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)
    monkeypatch.setattr(hardware, "detect_hardware", lambda: MAC_16GB)
    monkeypatch.setattr(pipeline_image, "detect_hardware", lambda: MAC_16GB)
    engine = FakeEngine()
    monkeypatch.setattr(seedvr2, "get_seedvr2_engine", lambda hw: engine)
    calls = []

    def fake_upscale(path, opts, out_dir, *, progress, cancel):
        if Path(path).name.startswith("broken"):
            from upscaler.imageio_utils import ImageFileError

            raise ImageFileError(f"{Path(path).name} couldn't be opened as a picture.")
        engine.events.append("upscale")
        calls.append(SimpleNamespace(path=Path(path), opts=opts, out_dir=out_dir))
        progress(0.5, "Upscaling")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{Path(path).stem}-8K-{opts.mode}.jpg"
        output.write_bytes(b"jpeg")
        return ImageResult(output, 8192, 6156, opts.mode, ["A note."], 75.0, b"", b"", b"")

    monkeypatch.setattr(pipeline_image, "upscale_image", fake_upscale)
    return SimpleNamespace(
        tmp=tmp_path, engine=engine, calls=calls, host_out=tmp_path / "home" / "Pictures" / "Upscaled"
    )


def picture(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"png")
    return path


# --- serve ------------------------------------------------------------------------------


def test_no_arguments_means_serve(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "serve", lambda share, port, open_browser: seen.append((share, port, open_browser)) or 0)
    assert cli.run([]) == 0
    assert cli.run(["--share", "--no-browser", "--port", "7861"]) == 0
    assert cli.run(["serve", "--share"]) == 0
    assert seen == [(False, 7860, True), (True, 7861, False), (True, 7860, True)]


def _fake_uvicorn(monkeypatch, opened: threading.Event):
    def fake_run(self, sockets=None):
        self.started = True
        opened.wait(5)  # stay "running" until the browser has been opened

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)


def test_serve_prints_wifi_addresses_and_opens_browser(env, monkeypatch, capsys):
    opened = threading.Event()
    browsed = []
    _fake_uvicorn(monkeypatch, opened)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: SimpleNamespace(host=host, port=port))
    monkeypatch.setattr(server, "lan_ipv4_addresses", lambda: ["192.168.1.20", "10.0.0.5"])
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: browsed.append(url) or opened.set())

    assert cli.run(["--share"]) == 0
    out = capsys.readouterr().out
    assert "Re:Solution is running." in out
    assert "http://127.0.0.1:7860" in out
    assert "http://192.168.1.20:7861" in out  # other devices come in through the Wi-Fi listener
    assert "http://10.0.0.5:7861" in out
    assert ":7860" not in out.replace("http://127.0.0.1:7860", "")
    assert browsed == ["http://127.0.0.1:7860"]


def test_serve_binds_only_this_computer_without_share(env, monkeypatch, capsys):
    opened = threading.Event()
    bound = []
    _fake_uvicorn(monkeypatch, opened)
    opened.set()
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: bound.append((host, port)) or SimpleNamespace())
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: pytest.fail("--no-browser"))
    assert cli.run(["serve", "--no-browser"]) == 0
    assert bound == [("127.0.0.1", 7860)]
    assert "192.168" not in capsys.readouterr().out


def test_serve_share_adds_a_wifi_listener(env, monkeypatch):
    opened = threading.Event()
    opened.set()
    bound = []
    made = []
    _fake_uvicorn(monkeypatch, opened)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: bound.append((host, port)) or SimpleNamespace())
    monkeypatch.setattr(server, "lan_ipv4_addresses", list)
    real_create_app = server.create_app

    def create_app(queue, share, *, port, wifi_socket=None):
        made.append((share, port, wifi_socket))
        return real_create_app(queue, share, port=port, wifi_socket=wifi_socket)

    monkeypatch.setattr(server, "create_app", create_app)
    assert cli.run(["--share", "--no-browser"]) == 0
    # The app itself stays on this computer only; sharing is a second listener on every network.
    assert bound == [("127.0.0.1", 7860), ("0.0.0.0", 7861)]
    assert [(share, port, wifi is not None) for share, port, wifi in made] == [(True, 7860, True)]


def test_serve_share_port_taken_still_starts_on_this_computer(env, monkeypatch, capsys):
    opened = threading.Event()
    opened.set()
    made = []
    _fake_uvicorn(monkeypatch, opened)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)

    def listen(host, port):
        if host == "0.0.0.0":
            raise OSError(48, "Address already in use")
        return SimpleNamespace()

    monkeypatch.setattr(cli, "_listen_socket", listen)
    real_create_app = server.create_app

    def create_app(queue, share, *, port, wifi_socket=None):
        made.append((share, wifi_socket))
        return real_create_app(queue, share, port=port, wifi_socket=wifi_socket)

    monkeypatch.setattr(server, "create_app", create_app)
    assert cli.run(["--share", "--no-browser"]) == 0
    out = capsys.readouterr().out
    assert "port 7861" in out and "On this computer, open: http://127.0.0.1:7860" in out
    assert made == [(False, None)]


def test_serve_ctrl_c_stops_calmly(env, monkeypatch, capsys):
    def interrupted(self, sockets=None):
        raise KeyboardInterrupt  # uvicorn shuts down gracefully, then passes Ctrl+C on

    monkeypatch.setattr(uvicorn.Server, "run", interrupted)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: SimpleNamespace())
    # 0, not 130: both launchers treat 0 as a normal stop, and the Windows one would call 130 a problem.
    assert cli.run(["--no-browser"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("Re:Solution has stopped.")


def test_serve_when_already_running_just_opens_it(env, monkeypatch, capsys):
    browsed = []
    monkeypatch.setattr(cli, "_probe_running", lambda port: True)
    monkeypatch.setattr(cli.webbrowser, "open", browsed.append)
    assert cli.run([]) == 0
    assert "Re:Solution is already running - open http://127.0.0.1:7860" in capsys.readouterr().out
    assert browsed == ["http://127.0.0.1:7860"]


def _running_copy(monkeypatch, answer):
    """The app is already running on 7860; `answer(url, body)` is how it answers POST /api/share."""
    asked = []

    def post(url, *, json, timeout, trust_env):
        asked.append((url, json, trust_env))
        return answer(url, json)

    monkeypatch.setattr(cli, "_probe_running", lambda port: True)
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    return asked


def test_share_launcher_turns_sharing_on_in_the_copy_already_running(env, monkeypatch, capsys):
    def answer(url, body):
        return httpx.Response(200, json={"enabled": True, "urls": ["http://192.168.1.20:7861"]})

    asked = _running_copy(monkeypatch, answer)
    assert cli.run(["--share"]) == 0
    out = capsys.readouterr().out
    assert "Re:Solution is already running - open http://127.0.0.1:7860" in out
    assert "On phones and other computers on the same Wi-Fi, open:" in out and "http://192.168.1.20:7861" in out
    assert asked == [("http://127.0.0.1:7860/api/share", {"enabled": True}, False)]  # never through a proxy


def test_share_launcher_says_why_sharing_did_not_start(env, monkeypatch, capsys):
    detail = "You don't seem to be connected to Wi-Fi or a network. Connect to your Wi-Fi, then try again."
    _running_copy(monkeypatch, lambda url, body: httpx.Response(409, json={"detail": detail}))
    assert cli.run(["--share", "--no-browser"]) == 0
    assert f"Sharing on Wi-Fi didn't start. {detail}" in capsys.readouterr().out


def test_share_launcher_when_the_running_copy_does_not_answer(env, monkeypatch, capsys):
    def answer(url, body):
        raise httpx.ReadTimeout("busy")

    _running_copy(monkeypatch, answer)
    assert cli.run(["--share", "--no-browser"]) == 0
    assert "Sharing on Wi-Fi is off. To turn it on, click Share on Wi-Fi in the app." in capsys.readouterr().out


def test_plain_launcher_leaves_sharing_alone(env, monkeypatch):
    asked = _running_copy(monkeypatch, lambda url, body: pytest.fail("must not switch sharing"))
    assert cli.run(["--no-browser"]) == 0
    assert asked == []


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="Windows has no SIGHUP")
def test_closing_the_terminal_window_stops_the_app_calmly(env, monkeypatch, capsys):
    installed = []
    monkeypatch.setattr(cli.signal, "signal", lambda sig, handler: installed.append((sig, handler)) or signal.SIG_DFL)
    stopping = []

    def run_until_the_window_closes(self, sockets=None):
        sig, handler = installed[-1]
        assert sig == signal.SIGHUP
        handler(signal.SIGHUP, None)  # the Terminal window is closed
        stopping.append(self.should_exit)  # uvicorn now stops as it does for Ctrl+C

    monkeypatch.setattr(uvicorn.Server, "run", run_until_the_window_closes)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: SimpleNamespace())
    assert cli.run(["--no-browser"]) == 0
    assert stopping == [True]
    assert installed[-1] == (signal.SIGHUP, signal.SIG_DFL)  # put back afterwards
    assert capsys.readouterr().out.rstrip().endswith("Re:Solution has stopped.")


def _record_queue_stops(monkeypatch) -> tuple[list, threading.Event]:
    stops, stopped = [], threading.Event()
    real_stop = jobs.JobQueue.stop

    def stop(self, wait=None):
        stops.append(wait)
        real_stop(self, wait=wait)
        stopped.set()

    monkeypatch.setattr(jobs.JobQueue, "stop", stop)
    return stops, stopped


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="Windows has no SIGHUP")
def test_closing_the_terminal_window_saves_the_jobs_at_once(env, monkeypatch):
    # The window is gone and the Mac may be logging out: the running job is saved as waiting straight
    # away, not only once the web server has wound down.
    installed = []
    monkeypatch.setattr(cli.signal, "signal", lambda sig, handler: installed.append((sig, handler)) or signal.SIG_DFL)
    stops, stopped = _record_queue_stops(monkeypatch)
    saved_while_running = []

    def run_until_the_window_closes(self, sockets=None):
        sig, handler = installed[-1]
        handler(signal.SIGHUP, None)
        saved_while_running.append(stopped.wait(5))

    monkeypatch.setattr(uvicorn.Server, "run", run_until_the_window_closes)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: SimpleNamespace())
    assert cli.run(["--no-browser"]) == 0
    assert saved_while_running == [True]
    assert stops[0] == cli._CLOSE_SAVE_SECONDS


def test_closing_the_window_on_windows_saves_the_jobs_before_windows_ends_the_app(env, monkeypatch):
    # Windows gives a console program about 5 seconds after its window closes, then ends it.
    handlers, removed = [], []

    def install(handler):
        handlers.append(handler)
        return lambda: removed.append(handler)

    monkeypatch.setattr(cli, "_install_console_handler", install)
    stops, _ = _record_queue_stops(monkeypatch)
    answers, exiting = [], []

    def run_until_the_window_closes(self, sockets=None):
        [handler] = handlers
        answers.append(handler(cli._CTRL_C_EVENT))  # Ctrl+C: left to Python and uvicorn
        assert stops == [] and not self.should_exit
        answers.append(handler(cli._CTRL_CLOSE_EVENT))  # returns once the jobs are saved
        exiting.append(self.should_exit)

    monkeypatch.setattr(uvicorn.Server, "run", run_until_the_window_closes)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: SimpleNamespace())
    assert cli.run(["--no-browser"]) == 0
    assert answers == [False, True] and exiting == [True]
    assert stops[0] == cli._CLOSE_SAVE_SECONDS < 5
    assert removed == handlers  # taken away again once the app has stopped


def test_console_handler_is_only_installed_on_windows():
    if cli.sys.platform == "win32":
        pytest.skip("this checks the other systems")
    assert cli._install_console_handler(lambda event: True) is None


def test_serve_port_taken_by_another_copy_starting_up(env, monkeypatch, capsys):
    answers = iter([None, True])  # not answering yet, but it has the port by the time we try it
    monkeypatch.setattr(cli, "_probe_running", lambda port: next(answers))

    def taken(host, port):
        raise OSError(48, "Address already in use")

    monkeypatch.setattr(cli, "_listen_socket", taken)
    assert cli.run(["--no-browser"]) == 0
    assert "Re:Solution is already running - open http://127.0.0.1:7860" in capsys.readouterr().out


def test_serve_port_used_by_another_program(env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_running", lambda port: False)
    assert cli.run(["--no-browser"]) == 1
    out = capsys.readouterr().out
    assert "--port 7861" in out and "Re:Solution" in out


# The real probe, against small stand-in servers on this computer.


@contextmanager
def _http_server(status: int, content_type: str, body: bytes, *, delay: threading.Event | None = None):
    """A tiny web server on a free port that answers every request the same way. Yields
    (port, paths it was asked for). With `delay`, it waits for that event before answering."""
    asked: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            if delay is not None:
                delay.wait(5)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], asked
    finally:
        if delay is not None:
            delay.set()
        httpd.shutdown()
        httpd.server_close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def system_proxy(monkeypatch):
    """A proxy set for the whole computer that answers everything with an error page, as office
    proxies do for addresses they can't reach. Yields the paths it was asked for."""
    page = b"<html><body><h1>502 Bad Gateway</h1></body></html>"
    with _http_server(502, "text/html", page) as (port, asked):
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{port}")
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        yield asked


def test_probe_free_port_is_free_even_behind_a_proxy(system_proxy):
    assert cli._probe_running(_free_port()) is None
    assert system_proxy == []  # 127.0.0.1 never goes to the proxy


def test_serve_starts_behind_a_proxy(env, system_proxy, monkeypatch, capsys):
    port = _free_port()
    bound = []

    def fake_run(self, sockets=None):
        bound.extend(sock.getsockname() for sock in sockets)
        for sock in sockets:
            sock.close()

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    assert cli.run(["--no-browser", "--port", str(port)]) == 0
    assert bound == [("127.0.0.1", port)]
    assert "Re:Solution is running." in capsys.readouterr().out
    assert system_proxy == []


def test_probe_finds_this_app_through_a_proxy_setting(system_proxy):
    status = json.dumps({"version": "1.1.0", "is_host": True}).encode()
    with _http_server(200, "application/json", status) as (port, asked):
        assert cli._probe_running(port) is True
    assert asked == ["/api/status"] and system_proxy == []


def test_probe_another_program_on_the_port():
    with _http_server(200, "text/html", b"<html>someone else's page</html>") as (port, _):
        assert cli._probe_running(port) is False


def test_probe_a_busy_copy_counts_as_running(monkeypatch):
    monkeypatch.setattr(cli, "_PROBE_ANSWER_SECONDS", 0.2)
    with _http_server(200, "application/json", b"{}", delay=threading.Event()) as (port, _):
        assert cli._probe_running(port) is True  # slow to answer: not "another program"


def test_version_names_the_product(capsys):
    with pytest.raises(SystemExit):
        cli.run(["--version"])
    assert capsys.readouterr().out.startswith("Re:Solution ")


# --- image ------------------------------------------------------------------------------


def test_image_defaults_to_photoreal_8k(env, capsys):
    assert cli.run(["image", str(picture(env.tmp, "table.png"))]) == 0
    [call] = env.calls
    assert call.opts.mode == "photoreal"
    assert call.opts.photoreal_level == "balanced"
    assert call.opts.size == "8k"
    assert call.opts.output_format == "jpeg"
    assert call.opts.dpi == 300
    assert call.out_dir == env.host_out
    out = capsys.readouterr().out
    assert "table-8K-photoreal.jpg" in out
    assert "8192 × 6156" in out
    assert "A note." in out


def test_image_uses_the_saved_default_mode(env):
    config.save_settings(config.Settings(default_image_mode="faithful", output_format="png"))
    assert cli.run(["image", str(picture(env.tmp, "table.png"))]) == 0
    assert env.calls[0].opts.mode == "faithful"
    assert env.calls[0].opts.output_format == "png"


def test_image_options(env):
    args = [
        "image",
        str(picture(env.tmp, "a.png")),
        "--mode",
        "fast",
        "--size",
        "6k",
        "--strength",
        "60",
        "--grain",
        "subtle",
        "--format",
        "png",
        "--out",
        str(env.tmp / "out"),
    ]
    assert cli.run(args) == 0
    opts = env.calls[0].opts
    assert (opts.mode, opts.size, opts.strength, opts.grain, opts.output_format) == ("fast", "6k", 0.6, "subtle", "png")
    assert env.calls[0].out_dir == env.tmp / "out"


def test_image_strength_also_applies_to_photoreal(env):
    assert cli.run(["image", str(picture(env.tmp, "a.png")), "--strength", "60"]) == 0
    opts = env.calls[0].opts
    assert opts.mode == "photoreal"
    assert (opts.resemblance, opts.strength) == (0.6, 0.6)  # also right if it falls back to Faithful


def test_image_photoreal_level_tiff_and_dpi(env):
    args = ["image", str(picture(env.tmp, "poster.png")), "--size", "16k", "--level", "rich"]
    assert cli.run([*args, "--format", "tiff", "--dpi", "600"]) == 0
    opts = env.calls[0].opts
    assert (opts.mode, opts.photoreal_level, opts.output_format, opts.dpi) == ("photoreal", "rich", "tiff", 600)


@pytest.mark.parametrize("fmt", ["webp", "original"])
def test_image_webp_and_same_as_original(env, fmt):
    assert cli.run(["image", str(picture(env.tmp, "a.png")), "--format", fmt]) == 0
    assert env.calls[0].opts.output_format == fmt


def test_image_format_help_names_every_format(capsys):
    with pytest.raises(SystemExit):
        cli.run(["image", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "jpeg, png, tiff, webp or original" in help_text


@pytest.mark.parametrize("bad", [["--dpi", "100"], ["--level", "extreme"], ["--format", "gif"]])
def test_image_refuses_unknown_choices(env, bad, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.run(["image", str(picture(env.tmp, "a.png")), *bad])
    assert excinfo.value.code == 2 and env.calls == []


def test_image_downloads_photoreal_first(env, capsys):
    env.engine.downloaded = False
    assert cli.run(["image", str(picture(env.tmp, "a.png"))]) == 0
    assert env.engine.events == ["download", "upscale"]
    assert "7 GB" in capsys.readouterr().out


def test_image_vectors_never_download_photoreal(env, capsys):
    env.engine.downloaded = False
    assert cli.run(["image", str(picture(env.tmp, "logo.svg"))]) == 0
    assert env.engine.events == ["upscale"]
    assert "7 GB" not in capsys.readouterr().out


def test_image_keeps_going_after_a_bad_file(env, capsys):
    files = [str(picture(env.tmp, "broken.png")), str(picture(env.tmp, "good.png"))]
    assert cli.run(["image", *files, "--mode", "fast"]) == 1
    assert [c.path.name for c in env.calls] == ["good.png"]
    assert "broken.png couldn't be opened" in capsys.readouterr().out


def test_image_missing_file_or_bad_mode(env, capsys):
    assert cli.run(["image", str(env.tmp / "nope.png")]) == 2
    assert "nope.png" in capsys.readouterr().out
    assert cli.run(["image", str(picture(env.tmp, "a.png")), "--mode", "magic"]) == 2
    assert env.calls == []


# --- video ------------------------------------------------------------------------------


def test_video_command(env, monkeypatch, capsys):
    from upscaler.video import VideoResult, pipeline

    seen = {}

    def fake_upscale_video(src, out_dir, job_dir, opts, *, progress, cancel):
        seen.update(src=src, out_dir=out_dir, job_dir=job_dir, opts=opts)
        job_dir.mkdir(parents=True, exist_ok=True)
        progress(0.5, "Upscaling frame 1 of 2")
        output = out_dir / "clip-4K-fast-preview3s.mp4"
        out_dir.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        return VideoResult(output, 3840, 2160, 30.0, [])

    monkeypatch.setattr(pipeline, "upscale_video", fake_upscale_video)
    clip = picture(env.tmp, "clip.mov")
    assert cli.run(["video", str(clip), "--preview-seconds", "3"]) == 0
    assert seen["opts"].mode == "fast"
    assert seen["opts"].preview_seconds == 3.0
    assert seen["out_dir"] == env.host_out
    assert seen["job_dir"].is_relative_to(env.tmp / "data")
    assert not seen["job_dir"].exists()  # finished, so nothing to resume
    assert "clip-4K-fast-preview3s.mp4" in capsys.readouterr().out


def _record_video_work_folders(monkeypatch, outcome=None) -> list[Path]:
    from upscaler.video import VideoResult, pipeline

    folders = []

    def fake_upscale_video(src, out_dir, job_dir, opts, *, progress, cancel):
        folders.append(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "segment-00000.mp4").write_bytes(b"finished part")
        if outcome is not None:
            raise outcome
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "clip-4K.mp4"
        output.write_bytes(b"mp4")
        return VideoResult(output, 3840, 2160, 30.0, [])

    monkeypatch.setattr(pipeline, "upscale_video", fake_upscale_video)
    return folders


def test_video_work_folder_depends_on_the_options(env, monkeypatch):
    # Different options make different segments, so they must never share (and resume) a work folder.
    folders = _record_video_work_folders(monkeypatch)
    clip = picture(env.tmp, "clip.mov")
    for args in (
        [], ["--mode", "general"], ["--preview-seconds", "3"], ["--strength", "50"], [],
    ):  # fmt: skip
        assert cli.run(["video", str(clip), *args]) == 0
    assert len(set(folders[:4])) == 4
    assert folders[4] == folders[0]  # the same video with the same options resumes in the same place


def test_video_out_of_disk_space_keeps_the_work_to_carry_on(env, monkeypatch, capsys):
    from upscaler.video import NotEnoughDiskSpace

    message = "There isn't enough free disk space to finish this video. Free up about 2 GB."
    folders = _record_video_work_folders(monkeypatch, NotEnoughDiskSpace(message))
    clip = picture(env.tmp, "clip.mov")
    assert cli.run(["video", str(clip)]) == 1
    out = capsys.readouterr().out
    assert f"clip.mov wasn't upscaled: {message}" in out
    assert "run the same command again" in out
    assert (folders[0] / "segment-00000.mp4").read_bytes() == b"finished part"  # kept for next time


@pytest.mark.parametrize(
    ("failure", "said"),
    [
        ("join", "Couldn't put the finished video together. Check that the computer has free disk space."),
        ("disk full", "The disk is full. Free up some space."),
    ],
)
def test_video_whose_join_failed_or_whose_disk_filled_keeps_the_work(env, monkeypatch, capsys, failure, said):
    import errno

    from upscaler.video.pipeline import JoinFailed

    error = JoinFailed(said) if failure == "join" else OSError(errno.ENOSPC, "No space left on device")
    folders = _record_video_work_folders(monkeypatch, error)
    assert cli.run(["video", str(picture(env.tmp, "clip.mov"))]) == 1
    out = capsys.readouterr().out
    assert f"clip.mov wasn't upscaled: {said} Then run the same command again" in out
    assert (folders[0] / "segment-00000.mp4").read_bytes() == b"finished part"  # kept for next time


def test_video_other_failures_tidy_the_work_away(env, monkeypatch, capsys):
    from upscaler.video import VideoError

    folders = _record_video_work_folders(monkeypatch, VideoError("This file isn't a video the app can read."))
    assert cli.run(["video", str(picture(env.tmp, "clip.mov"))]) == 1
    assert not folders[0].exists()


def test_video_mode_choices_match_the_engine():
    from upscaler.video import VIDEO_MODES

    assert set(cli.VIDEO_MODE_CHOICES) == set(VIDEO_MODES)


def test_video_rejects_bad_preview(env, capsys):
    with pytest.raises(SystemExit):
        cli.run(["video", str(picture(env.tmp, "clip.mov")), "--preview-seconds", "-2"])


# --- Ctrl+C -------------------------------------------------------------------------------


def test_ctrl_c_cancels_the_work(capsys):
    def work(cancel):
        if not cancel.wait(5):
            return "finished"
        raise Cancelled()

    threading.Timer(0.2, _thread.interrupt_main).start()
    with pytest.raises(Cancelled):
        cli.run_cancellable(work)
    assert "Stopping" in capsys.readouterr().out


def test_run_cancellable_returns_the_result():
    assert cli.run_cancellable(lambda cancel: 42) == 42
    with pytest.raises(ValueError):
        cli.run_cancellable(lambda cancel: (_ for _ in ()).throw(ValueError("boom")))


def test_progress_line_on_a_terminal():
    class Terminal:
        def __init__(self):
            self.written = ""

        def isatty(self):
            return True

        def write(self, text):
            self.written += text

        def flush(self):
            pass

    terminal = Terminal()
    line = cli.ProgressLine("a.png", stream=terminal)
    line(0.25, "Upscaling")
    line(0.5, "Upscaling")
    line.end()
    assert "\ra.png:  25%  Upscaling" in terminal.written
    assert " 50%" in terminal.written
    assert terminal.written.endswith("\n")
