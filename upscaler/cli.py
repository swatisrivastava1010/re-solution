"""Command line.

    python -m upscaler [serve] [--share] [--port 7860] [--no-browser]   start the app (the default)
    python -m upscaler image <files…> [--mode] [--size] [--out] …        upscale pictures directly
    python -m upscaler video <file> [--mode] [--preview-seconds] …       upscale a video directly

The engines are imported only when a command needs them, so `--help` answers quickly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import signal
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from upscaler import __version__, config
from upscaler.config import DEFAULT_PORT, GRAIN_CHOICES, OUTPUT_FORMATS, PRODUCT_NAME, Cancelled, load_settings
from upscaler.keepawake import KeepAwake
from upscaler.sizing import DEFAULT_DPI, DEFAULT_PHOTOREAL_LEVEL, DPI_CHOICES, PHOTOREAL_LEVELS, PRESETS

log = logging.getLogger(__name__)

VIDEO_MODE_CHOICES = ("fast", "general", "faithful")  # upscaler.video.VIDEO_MODES
_BROWSER_WAIT_SECONDS = 60
_EXIT_CANCELLED = 130  # what shells report for Ctrl+C
_VECTOR_SUFFIXES = (".svg", ".svgz")
# Looking for a copy that is already running: Windows takes about a second to refuse a connection
# to a free port, and a running copy can be slow to answer while it works on a big picture.
_PROBE_CONNECT_SECONDS = 1.0
_PROBE_ANSWER_SECONDS = 5.0
_SHARE_ANSWER_SECONDS = 10.0  # switching sharing on in a running copy looks up its Wi-Fi addresses
# Closing the window: Windows ends a console program about 5 seconds after its window closes, so the
# jobs are saved first and then given this long to wind down.
_CLOSE_SAVE_SECONDS = 4.0
_CTRL_C_EVENT, _CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT = 0, 2, 5, 6  # Windows console events


def main() -> None:
    sys.exit(run(sys.argv[1:]))


def run(argv: list[str]) -> int:
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help", "--version")):
        argv = ["serve", *argv]  # `serve` is the default command
    args = build_parser().parse_args(argv)
    if args.command == "image":
        return image_command(args)
    if args.command == "video":
        return video_command(args)
    return serve(args.share, args.port, not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m upscaler",
        description=f"{PRODUCT_NAME}: a free image and video upscaler that runs on your computer.",
    )
    parser.add_argument("--version", action="version", version=f"{PRODUCT_NAME} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_parser = commands.add_parser("serve", help="start the app in your web browser (the default)")
    serve_parser.add_argument("--share", action="store_true", help="let phones and computers on your Wi-Fi use it too")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port number (default {DEFAULT_PORT})")
    serve_parser.add_argument("--no-browser", action="store_true", help="don't open the web browser")

    image_parser = commands.add_parser("image", help="upscale pictures")
    image_parser.add_argument("files", nargs="+", help="pictures to upscale")
    image_parser.add_argument(
        "--mode",
        help="photoreal (the default), faithful, faithful_sharp, fast, illustration, or cloud:<service>:<model>",
    )
    image_parser.add_argument(
        "--level",
        choices=PHOTOREAL_LEVELS,
        default=DEFAULT_PHOTOREAL_LEVEL,
        help="Photo-real detail: balanced (the default), rich (the most new detail, slowest) or gentle (the softest)",
    )
    image_parser.add_argument("--size", choices=list(PRESETS), default="8k", help="output size (default 8k)")
    image_parser.add_argument("--out", help="folder for the results (default: Pictures/Upscaled)")
    image_parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        help="jpeg, png, tiff, webp or original (the picture's own format) (default: as in Settings)",
    )
    image_parser.add_argument(
        "--dpi", type=int, choices=DPI_CHOICES, default=DEFAULT_DPI, help=f"print resolution (default {DEFAULT_DPI})"
    )
    image_parser.add_argument("--strength", type=_percent, default=100, help="detail strength in percent (default 100)")
    image_parser.add_argument("--grain", choices=GRAIN_CHOICES, help="film grain (default: as in Settings)")

    video_parser = commands.add_parser("video", help="upscale a video to 4K")
    video_parser.add_argument("file", help="video to upscale")
    video_parser.add_argument("--mode", choices=VIDEO_MODE_CHOICES, default="fast", help="video mode (default fast)")
    video_parser.add_argument("--preview-seconds", type=_positive_seconds, help="only do the first N seconds")
    video_parser.add_argument("--out", help="folder for the result (default: Pictures/Upscaled)")
    video_parser.add_argument("--strength", type=_percent, default=100, help="detail strength in percent (default 100)")
    return parser


def _percent(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("use a number from 0 to 100")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("use a number of seconds above 0")
    return value


# --- serve ------------------------------------------------------------------------------


def serve(share: bool, port: int, open_browser: bool) -> int:
    """Run the app on this computer only (127.0.0.1). With `share`, it also starts sharing on Wi-Fi:
    a second listener on every network at port + 1 (or the next free port), which the host can switch
    off and on in the app."""
    import uvicorn

    from upscaler import server as web
    from upscaler.jobs import JobQueue, default_runners

    local_url = f"http://127.0.0.1:{port}"
    running = _probe_running(port)
    if running:  # e.g. the launcher was double-clicked twice
        return _already_running(port, open_browser, share)
    if running is False:
        return _port_busy(port)
    try:
        sock = _listen_socket("127.0.0.1", port)
    except OSError:
        # Taken since we looked: often another copy of the app that was still starting up.
        if _probe_running(port):
            return _already_running(port, open_browser, share)
        return _port_busy(port)
    wifi = _wifi_socket(port + web.SHARE_PORT_OFFSET) if share else None
    wifi_sock, wifi_port = wifi if wifi is not None else (None, None)

    store = config.jobs_dir()
    queue = JobQueue(store, default_runners(store))
    app = web.create_app(queue, wifi_sock is not None, port=port, wifi_socket=wifi_sock)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            proxy_headers=False,  # who is asking is the connection itself, never what a header claims
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=3,
        )
    )
    _print_addresses(local_url, web.lan_urls(wifi_port) if wifi_port is not None else None)
    # On a Mac, PyAV and OpenCV each bring their own copy of FFmpeg's libavdevice, so the first picture
    # checked can print "objc[…]: Class AVFFrameReceiver is implemented in both …" in this window. The
    # Objective-C runtime writes that straight to the window as the second copy loads, beneath Python,
    # so the app can't hold it back. It is harmless, and the README's troubleshooting section says so.
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(server, local_url), daemon=True).start()
    try:
        with _calm_stop_on_hangup(server, queue), _calm_stop_on_window_close(server, queue):
            server.run(sockets=[sock])
    except KeyboardInterrupt:  # Ctrl+C: uvicorn has already shut down gracefully, then passes it on
        pass
    try:
        print(f"{PRODUCT_NAME} has stopped.")
    except OSError:  # the window it ran in has been closed
        pass
    return 0


@contextmanager
def _calm_stop_on_hangup(server: Any, queue: Any) -> Iterator[None]:
    """Closing the Terminal window (SIGHUP) stops the app as calmly as Ctrl+C does, so a running job
    is put back in the queue for next time instead of looking like a crash. The jobs are saved at
    once (in a helper thread, as this runs between the web server's own steps), in case the Mac is
    logging out and doesn't wait for the web server to wind down."""
    hangup = getattr(signal, "SIGHUP", None)  # not on Windows: see _calm_stop_on_window_close
    if hangup is None or threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        threading.Thread(target=queue.stop, kwargs={"wait": _CLOSE_SAVE_SECONDS}, daemon=True).start()
        server.should_exit = True

    previous = signal.signal(hangup, stop)
    try:
        yield
    finally:
        signal.signal(hangup, signal.SIG_DFL if previous is None else previous)


@contextmanager
def _calm_stop_on_window_close(server: Any, queue: Any) -> Iterator[None]:
    """On Windows, closing the window (or signing out, or shutting down) saves the jobs as Ctrl+C
    would, as far as Windows allows: it ends a console program about 5 seconds after the window
    closes, and as soon as this handler returns. So the handler saves the running job as waiting
    first (JobQueue.stop does that before anything else), gives it up to _CLOSE_SAVE_SECONDS to wind
    down, and starts the web server's own shutdown alongside, which may not finish. A video then
    carries on next time from its last finished segment. Signing out and shutting down reach console
    programs only on some setups, so those are best effort. Elsewhere this does nothing."""

    def handle(event: int) -> bool:
        if event not in (_CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT):
            return False  # Ctrl+C and Ctrl+Break: Python and uvicorn handle those
        server.should_exit = True
        try:
            queue.stop(wait=_CLOSE_SAVE_SECONDS)
        except Exception:  # whatever happens, Windows is about to end the app
            log.exception("Couldn't save the jobs as the window closed")
        return True

    remove = _install_console_handler(handle)
    try:
        yield
    finally:
        if remove is not None:
            remove()


def _install_console_handler(handle: Callable[[int], bool]) -> Callable[[], None] | None:
    """Ask Windows to call `handle(event)` for console events (SetConsoleCtrlHandler). Returns how
    to take it away again, or None when not on Windows or Windows refused."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(lambda event: bool(handle(event)))
        if not kernel32.SetConsoleCtrlHandler(routine, True):
            return None
    except (AttributeError, OSError):
        log.warning("Couldn't watch for the window closing", exc_info=True)
        return None

    def remove() -> None:  # also keeps `routine` alive while Windows may call it
        kernel32.SetConsoleCtrlHandler(routine, False)

    return remove


def _wifi_socket(port: int) -> tuple[socket.socket, int] | None:
    """The socket for sharing on Wi-Fi and its port: `port`, or the next free one when another
    program has it. None (with a note) when none is free."""
    from upscaler.server import WIFI_HOST, share_ports

    for candidate in share_ports(port):
        try:
            return _listen_socket(WIFI_HOST, candidate), candidate
        except OSError:
            continue
    print(
        f"Sharing on Wi-Fi couldn't start because other programs are using port {port} and the ones after it. "
        f"{PRODUCT_NAME} still works on this computer."
    )
    return None


def _already_running(port: int, open_browser: bool, share: bool = False) -> int:
    local_url = f"http://127.0.0.1:{port}"
    print(f"{PRODUCT_NAME} is already running - open {local_url}")
    if share:  # the Share launcher: switch sharing on in the copy that is running
        _share_running_copy(port)
    if open_browser:
        webbrowser.open(local_url)
    return 0


def _share_running_copy(port: int) -> None:
    """Ask the copy running on `port` to share on Wi-Fi (as the Share switch in the app does), and
    print the addresses. It comes from this computer and no web page, so it counts as the host."""
    import httpx

    try:
        # Never through a proxy, which would send even 127.0.0.1 elsewhere (see _probe_running).
        response = httpx.post(
            f"http://127.0.0.1:{port}/api/share", json={"enabled": True}, timeout=_SHARE_ANSWER_SECONDS, trust_env=False
        )
        data = response.json()
    except (httpx.HTTPError, ValueError):
        response, data = None, None
    if response is not None and response.is_success and isinstance(data, dict) and data.get("enabled"):
        _print_wifi_addresses([url for url in data.get("urls") or [] if isinstance(url, str)])
        return
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, str) and detail:
        print(f"Sharing on Wi-Fi didn't start. {detail}")
    else:
        print("Sharing on Wi-Fi is off. To turn it on, click Share on Wi-Fi in the app.")


def _port_busy(port: int) -> int:
    print(f"Another program is using port {port}. Close it, or start {PRODUCT_NAME} with --port {port + 1}.")
    return 1


def _probe_running(port: int) -> bool | None:
    """True if this app already answers on this port, False if some other program does, None if
    nothing is listening."""
    import httpx

    try:  # a plain connection first: refused (or no answer at all) means the port is free
        socket.create_connection(("127.0.0.1", port), timeout=_PROBE_CONNECT_SECONDS).close()
    except OSError:
        return None
    try:
        # Never through a proxy: the computer's proxy settings would send even 127.0.0.1 to it, and
        # its error page would look like another program on the port.
        response = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=_PROBE_ANSWER_SECONDS, trust_env=False)
    except httpx.TimeoutException:
        return True  # listening but slow to answer: most likely this app, busy with a big picture
    except httpx.ConnectError:
        return None  # it stopped since we looked
    except httpx.HTTPError:
        return False  # something that isn't a web server
    try:
        data = response.json()
    except ValueError:
        return False
    return isinstance(data, dict) and "version" in data and "is_host" in data


def _listen_socket(host: str, port: int) -> socket.socket:
    from upscaler.server import listen_socket

    return listen_socket(host, port)


def _print_addresses(local_url: str, wifi_urls: list[str] | None) -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # show these at once, even when output goes to a log file
    except (AttributeError, ValueError):
        pass
    print(f"{PRODUCT_NAME} is running.")
    print(f"On this computer, open: {local_url}")
    if wifi_urls is not None:
        _print_wifi_addresses(wifi_urls)


def _print_wifi_addresses(wifi_urls: list[str]) -> None:
    if wifi_urls:
        print("On phones and other computers on the same Wi-Fi, open:")
        for url in wifi_urls:
            print(f"    {url}")
    else:
        print(
            "Sharing is on, but you don't seem to be connected to Wi-Fi or a network. Connect to your Wi-Fi, "
            "then open Share on Wi-Fi in the app for the address."
        )


def _open_when_ready(server: Any, url: str) -> None:
    deadline = time.monotonic() + _BROWSER_WAIT_SECONDS
    while not server.started:
        if server.should_exit or time.monotonic() > deadline:
            return
        time.sleep(0.1)
    webbrowser.open(url)


# --- image ------------------------------------------------------------------------------


def image_command(args: argparse.Namespace) -> int:
    from upscaler.hardware import detect_hardware
    from upscaler.pipeline_image import PHOTOREAL, ImageOptions, resolve_mode, upscale_image

    files = [Path(name).expanduser() for name in args.files]
    if missing := [path for path in files if not path.is_file()]:
        print(f"Can't find {missing[0]}")
        return 2
    settings = load_settings()
    options = ImageOptions(
        mode=args.mode or settings.default_image_mode,
        size=args.size,
        # --strength is the detail strength in every mode: Photo-real uses `resemblance` for it, the
        # others (and a Photo-real picture that falls back to Faithful) use `strength`.
        strength=args.strength / 100,
        resemblance=args.strength / 100,
        grain=args.grain or settings.grain,
        output_format=args.format or settings.output_format,
        dpi=args.dpi,
        photoreal_level=args.level,
    )
    try:
        mode, _ = resolve_mode(options.mode, detect_hardware())
    except ValueError as exc:
        print(exc)
        return 2
    out_dir = Path(args.out).expanduser() if args.out else config.host_outputs_dir()

    def upscale_one(path: Path, label: str) -> bool:
        progress = ProgressLine(label)
        try:
            result = run_cancellable(
                lambda cancel: upscale_image(path, options, out_dir, progress=progress, cancel=cancel)
            )
        except Cancelled:
            progress.end()
            raise
        except Exception as exc:  # reported, and the rest of the pictures still run
            progress.end()
            print(f"{path.name} wasn't upscaled: {_explain(exc)}")
            return False
        progress.end()
        print(
            f"Saved {result.output_path} ({result.width} × {result.height}, {result.mode_used}, {_duration(result.seconds)})"
        )
        for note in result.notes:
            print(f"    {note}")
        return True

    try:
        # Vector pictures are drawn at full size without any AI, so they never need the download.
        needs_photoreal = mode == PHOTOREAL and any(path.suffix.lower() not in _VECTOR_SUFFIXES for path in files)
        if needs_photoreal and not _prepare_photoreal():
            return 1
        with KeepAwake():
            done = [upscale_one(path, _label(path, n, len(files))) for n, path in enumerate(files, 1)]
    except Cancelled:
        print("Stopped.")
        return _EXIT_CANCELLED
    return 0 if all(done) else 1


def _label(path: Path, n: int, total: int) -> str:
    return path.name if total == 1 else f"[{n}/{total}] {path.name}"


def _prepare_photoreal() -> bool:
    """Download Photo-real the first time it's used. False if the download failed."""
    from upscaler.hardware import detect_hardware
    from upscaler.seedvr2 import get_seedvr2_engine

    engine = get_seedvr2_engine(detect_hardware())
    if engine is None or engine.weights_present():
        return True
    print("Photo-real needs a one-time 7 GB download (about 10 minutes on fast Wi-Fi). Downloading it now.")
    progress = ProgressLine("Photo-real")
    try:
        run_cancellable(lambda cancel: engine.download_weights(progress, cancel))
    except Cancelled:
        progress.end()
        raise
    except Exception as exc:  # reported in plain words
        progress.end()
        print(f"The Photo-real download didn't finish. {_explain(exc)}")
        return False
    progress.end()
    return True


# --- video ------------------------------------------------------------------------------


def video_command(args: argparse.Namespace) -> int:
    from upscaler.jobs import DISK_FULL_KEPT, video_work_worth_keeping
    from upscaler.video import pipeline

    src = Path(args.file).expanduser()
    if not src.is_file():
        print(f"Can't find {src}")
        return 2
    options = pipeline.VideoOptions(mode=args.mode, preview_seconds=args.preview_seconds, strength=args.strength / 100)
    out_dir = Path(args.out).expanduser() if args.out else config.host_outputs_dir()
    work = _video_work_folder(src, options)
    progress = ProgressLine(src.name)
    try:
        with KeepAwake():
            result = run_cancellable(
                lambda cancel: pipeline.upscale_video(src, out_dir, work, options, progress=progress, cancel=cancel)
            )
    except Cancelled:
        progress.end()
        print("Stopped. Run the same command again to carry on from where it stopped.")
        return _EXIT_CANCELLED
    except Exception as exc:  # reported in plain words
        progress.end()
        if video_work_worth_keeping(exc):  # out of disk space, or only the joining failed
            reason = DISK_FULL_KEPT if isinstance(exc, OSError) else _explain(exc)
            print(
                f"{src.name} wasn't upscaled: {reason} Then run the same command again to carry on from "
                "where it stopped: the parts already done are kept."
            )
        else:
            print(f"{src.name} wasn't upscaled: {_explain(exc)}")
            shutil.rmtree(work, ignore_errors=True)
        return 1
    progress.end()
    shutil.rmtree(work, ignore_errors=True)
    print(f"Saved {result.output_path} ({result.width} × {result.height}, {_duration(result.seconds)})")
    for note in result.notes:
        print(f"    {note}")
    return 0


def _video_work_folder(src: Path, options: Any) -> Path:
    """The same video with the same options always gets the same work folder, so running the command
    again resumes it. Other options make other segments, so they get a folder of their own."""
    key = json.dumps([str(src.resolve()), options.mode, options.preview_seconds, options.strength])
    return config.data_dir() / "cli-work" / hashlib.sha256(key.encode()).hexdigest()[:16]


# --- Helpers ----------------------------------------------------------------------------


def run_cancellable[T](work: Callable[[threading.Event], T]) -> T:
    """Run `work(cancel)` in a helper thread so Ctrl+C can stop it cleanly: the first Ctrl+C sets
    `cancel` and waits for the work to wind down (no half-written files); a second quits at once."""
    cancel = threading.Event()
    finished = threading.Event()
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = work(cancel)
        except BaseException as exc:  # noqa: BLE001  # re-raised in the caller's thread below
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=target, name="upscaler-work", daemon=True).start()
    # An Event, not Thread.join: a Ctrl+C that lands inside join() can make the thread look finished.
    while not finished.is_set():
        try:
            while not finished.wait(0.2):
                pass
        except KeyboardInterrupt:
            if cancel.is_set():
                raise
            print("\nStopping… (press Ctrl+C again to quit straight away)", flush=True)
            cancel.set()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class ProgressLine:
    """A progress line that updates in place on a terminal, or prints now and then into a log."""

    _LOG_EVERY_SECONDS = 5.0

    def __init__(self, label: str, stream: IO[str] | None = None):
        self.label = label
        self.stream = stream or sys.stdout
        self.terminal = self.stream.isatty()
        self._shown = False
        self._last_logged = 0.0

    def __call__(self, fraction: float, message: str) -> None:
        text = f"{self.label}: {int(min(max(fraction, 0.0), 1.0) * 100):3d}%  {message}"
        if self.terminal:
            width = max(20, shutil.get_terminal_size().columns - 1)
            self.stream.write("\r" + text[:width].ljust(width))
            self.stream.flush()
            self._shown = True
        elif time.monotonic() - self._last_logged >= self._LOG_EVERY_SECONDS:
            self._last_logged = time.monotonic()
            print(text, file=self.stream, flush=True)

    def end(self) -> None:
        if self._shown:
            self.stream.write("\n")
            self.stream.flush()
            self._shown = False


def _explain(exc: BaseException) -> str:
    """A plain-words message; the full details of anything unexpected go to the terminal too."""
    from upscaler.jobs import GENERIC_ERROR, friendly_error

    message = friendly_error(exc)
    if message == GENERIC_ERROR:
        log.error("Unexpected error", exc_info=exc)
    return message


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s" if seconds else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"
