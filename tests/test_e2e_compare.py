"""End-to-end stress tests for the compare viewers: a real Google Chrome against a real Re:Solution
server, with real Fast jobs (a 4K video with sound, a rotated portrait video, an 8K picture).

Opt-in (they take a few minutes and need Google Chrome): `pytest -m e2e`, or RESOLUTION_E2E=1.
RESOLUTION_E2E_CHROME names the browser if it isn't in the usual place, and RESOLUTION_E2E_REPORT
a JSON file to write the measurements to. RESOLUTION_E2E_HEADED=1 shows the window.

The browser is driven over the DevTools protocol with real mouse and keyboard input (so a control
that something invisible covers, or a click lost to a busy page, fails the test), and the page is
instrumented before it loads: every setTimeout, setInterval and requestAnimationFrame is counted,
long tasks and slow input (the Event Timing API) are recorded, and the time from the closing click
or key to the dialog's close event is measured. Window and document listeners are counted with
DOMDebugger.getEventListeners. The page's own view of what's running is window.__resolutionDebug
(js/lifecycle.js).

What is checked: open and close each viewer 25 times by every way of closing, in random states;
play/pause pressed 50 times in a row; scrubbing while playing; speed and frame steps; rapid switches
between Slider / Side by side and Fit / 100% / 200%; after every close the dialog is gone, nothing
is left running, the videos are stopped; no JavaScript errors; the two videos stay within one frame
of each other while playing; dropped frames and long tasks stay small.
"""

from __future__ import annotations

import base64
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
FRAME = 1 / FPS
CYCLES = 25
SPAM = 50
REPORT: dict = {}

CHROME_PLACES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

# Counts every timer and animation frame the page asks for, long tasks, slow input, errors, and the
# time from a closing click or key to the dialog's close event. Runs before the page's own scripts.
INSTRUMENT = r"""
(() => {
  const active = { timeouts: new Set(), intervals: new Set(), frames: new Set() };
  const st = window.setTimeout, ct = window.clearTimeout, si = window.setInterval, ci = window.clearInterval;
  const raf = window.requestAnimationFrame, caf = window.cancelAnimationFrame;
  window.setTimeout = function (fn, ms, ...rest) {
    let id = 0;
    id = st.call(window, function () { active.timeouts.delete(id); if (typeof fn === 'function') fn.apply(this, arguments); }, ms, ...rest);
    active.timeouts.add(id);
    return id;
  };
  window.clearTimeout = function (id) { active.timeouts.delete(id); return ct.call(window, id); };
  window.setInterval = function (fn, ms, ...rest) { const id = si.call(window, fn, ms, ...rest); active.intervals.add(id); return id; };
  window.clearInterval = function (id) { active.intervals.delete(id); return ci.call(window, id); };
  window.requestAnimationFrame = function (fn) {
    let id = 0;
    id = raf.call(window, (t) => { active.frames.delete(id); fn(t); });
    active.frames.add(id);
    return id;
  };
  window.cancelAnimationFrame = function (id) { active.frames.delete(id); return caf.call(window, id); };
  const e2e = window.__e2e = {
    active: () => ({ timeouts: active.timeouts.size, intervals: active.intervals.size, frames: active.frames.size }),
    longTasks: [], events: [], errors: [], closeLatency: [], inputAt: 0,
  };
  try {
    new PerformanceObserver((list) => { for (const e of list.getEntries()) e2e.longTasks.push({ at: e.startTime, ms: e.duration }); })
      .observe({ type: 'longtask', buffered: true });
  } catch (err) { /* no long task timing here */ }
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        if (['pointerdown', 'pointerup', 'click', 'keydown', 'mousedown', 'mouseup'].includes(e.name)) {
          e2e.events.push({ name: e.name, ms: e.duration, delay: e.processingStart - e.startTime });
        }
      }
    }).observe({ type: 'event', durationThreshold: 16, buffered: true });
  } catch (err) { /* no event timing here */ }
  window.addEventListener('error', (e) => e2e.errors.push(String(e.message || e)));
  window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    e2e.errors.push('unhandled rejection: ' + String((r && (r.stack || r.message)) || r));
  });
  document.addEventListener('click', (e) => { e2e.inputAt = e.timeStamp; }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') e2e.inputAt = e.timeStamp; }, true);
  document.addEventListener('DOMContentLoaded', () => {
    for (const d of document.querySelectorAll('dialog')) {
      d.addEventListener('close', () => {
        if (e2e.inputAt) e2e.closeLatency.push({ id: d.id, ms: performance.now() - e2e.inputAt });
        e2e.inputAt = 0;
      });
    }
  });
})();
"""


# --- A small DevTools-protocol driver ---------------------------------------------------------


def find_chrome() -> str | None:
    named = os.environ.get("RESOLUTION_E2E_CHROME")
    if named:
        return named if Path(named).exists() or shutil.which(named) else None
    for place in CHROME_PLACES:
        if Path(place).exists():
            return place
    for name in ("google-chrome", "google-chrome-stable", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None  # Chromium can't play H.264, which the comparison copies are, so it isn't used


class Chrome:
    KEYS = {
        "Escape": ("Escape", 27, ""),
        " ": ("Space", 32, " "),
        "Tab": ("Tab", 9, ""),
        "Enter": ("Enter", 13, "\r"),
        "ArrowLeft": ("ArrowLeft", 37, ""),
        "ArrowRight": ("ArrowRight", 39, ""),
        ",": ("Comma", 188, ","),
        ".": ("Period", 190, "."),
        "+": ("Equal", 187, "+"),
        "-": ("Minus", 189, "-"),
    }

    def __init__(self, binary: str, width: int = 1440, height: int = 900):
        from websockets.sync.client import connect

        self.profile = tempfile.mkdtemp(prefix="resolution-e2e-")
        args = [
            binary, f"--user-data-dir={self.profile}", "--remote-debugging-port=0", "--no-first-run",
            "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required", "--mute-audio",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", f"--window-size={width},{height}", "about:blank",
        ]  # fmt: skip
        if os.environ.get("RESOLUTION_E2E_HEADED") != "1":
            args.insert(1, "--headless=new")
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(self.profile) / "DevToolsActivePort"
        deadline = time.monotonic() + 30
        while not (port_file.exists() and port_file.read_text().strip()):
            if time.monotonic() > deadline or self.proc.poll() is not None:
                raise RuntimeError("Chrome didn't start")
            time.sleep(0.05)
        port = int(port_file.read_text().split()[0])
        page = None
        while page is None and time.monotonic() < deadline:
            try:
                targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5).read())
                page = next((t for t in targets if t["type"] == "page"), None)
            except OSError:
                pass
            if page is None:
                time.sleep(0.05)
        self.ws = connect(page["webSocketDebuggerUrl"], max_size=256 * 1024 * 1024, open_timeout=10).__enter__()
        self.seq = 0
        self.events: list[dict] = []
        for domain in ("Page", "Runtime", "Log", "Network"):
            self.send(f"{domain}.enable")
        self.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        self.send("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT)

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)

    def send(self, method: str, timeout: float = 30, **params):
        self.seq += 1
        mine = self.seq
        self.ws.send(json.dumps({"id": mine, "method": method, "params": params}))
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{method} got no answer in {timeout} s (is the page frozen?)")
            message = json.loads(self.ws.recv(timeout=left))
            if message.get("id") == mine:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            if "method" in message:
                self.events.append(message)

    def pump(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            try:
                message = json.loads(self.ws.recv(timeout=left))
            except TimeoutError:
                return
            if "method" in message:
                self.events.append(message)

    def js(self, expression: str, timeout: float = 30):
        result = self.send(
            "Runtime.evaluate", timeout=timeout, expression=expression, awaitPromise=True, returnByValue=True
        )
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError((details.get("exception") or {}).get("description") or details.get("text"))
        return result.get("result", {}).get("value")

    def wait(self, expression: str, timeout: float = 20, every: float = 0.05, what: str = ""):
        deadline = time.monotonic() + timeout
        probe = f"(() => {{ const v = ({expression}); return v instanceof Node ? true : v; }})()"  # a node isn't JSON
        while True:
            value = self.js(probe)
            if value:
                return value
            if time.monotonic() > deadline:
                page = self.js("(document.querySelector('#results') || document.body).innerText.slice(0, 600)")
                raise AssertionError(f"timed out waiting for {what or expression}; the page shows: {page!r}")
            self.pump(every)

    def goto(self, url: str) -> None:
        # Through a blank page, so going to the same address again (only the #part differs) still loads
        # a fresh page instead of keeping the last test's state.
        self.send("Page.navigate", url="about:blank")
        self.wait("location.href === 'about:blank' && document.readyState === 'complete'", 10, what="a blank page")
        self.send("Page.navigate", url=url)
        self.wait("document.readyState === 'complete' && Boolean(window.__resolutionDebug)", 30, what="the page")

    def box(self, selector: str):
        """The element's middle on screen (scrolled into view), and whether a click there reaches it."""
        found = self.js(
            "(() => {"
            f" const el = document.querySelector({json.dumps(selector)}); if (!el) return null;"
            " let r = el.getBoundingClientRect();"
            " if (r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) {"
            "  el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'}); r = el.getBoundingClientRect(); }"
            " const x = r.left + r.width / 2, y = r.top + r.height / 2; const hit = document.elementFromPoint(x, y);"
            " return {x, y, left: r.left, top: r.top, width: r.width, height: r.height, connected: el.isConnected,"
            "  reaches: Boolean(hit && (hit === el || el.contains(hit))), hit: hit ? hit.outerHTML.slice(0, 120) : null,"
            "  stack: document.elementsFromPoint(x, y).slice(0, 4).map(e => e.tagName + '.' + (e.className.baseVal ?? e.className))}; })()"
        )
        if not found:
            raise AssertionError(f"nothing matches {selector}")
        return found

    def mouse(self, kind: str, x: float, y: float, pressed: bool = False) -> None:
        self.send(
            "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1, buttons=1 if pressed else 0
        )

    def click_at(self, x: float, y: float, hold: float = 0.0) -> None:
        self.mouse("mouseMoved", x, y)
        self.mouse("mousePressed", x, y, pressed=True)
        if hold:
            self.pump(hold)
        self.mouse("mouseReleased", x, y)

    def click(self, selector: str, hold: float = 0.0, at: float | None = None):
        where = self.box(selector)
        assert where["reaches"], f"a click on {selector} lands on something else: {where}"
        x = where["left"] + where["width"] * at if at is not None else where["x"]
        self.click_at(x, where["y"], hold)
        return where

    def drag(self, selector: str, start: float, end: float, steps: int = 12, pause: float = 0.02) -> None:
        where = self.box(selector)
        y = where["y"]
        x0 = where["left"] + where["width"] * start
        x1 = where["left"] + where["width"] * end
        self.mouse("mouseMoved", x0, y)
        self.mouse("mousePressed", x0, y, pressed=True)
        for n in range(1, steps + 1):
            self.mouse("mouseMoved", x0 + (x1 - x0) * n / steps, y, pressed=True)
            self.pump(pause)
        self.mouse("mouseReleased", x1, y)

    def key(self, key: str) -> None:
        if key in self.KEYS:
            code, vk, text = self.KEYS[key]
        else:  # a letter or a digit
            code, vk, text = (f"Key{key.upper()}" if key.isalpha() else f"Digit{key}"), ord(key.upper()), key
        down = {"type": "keyDown", "key": key, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
        if text:
            down.update(text=text, unmodifiedText=text)
        self.send("Input.dispatchKeyEvent", **down)
        self.send(
            "Input.dispatchKeyEvent",
            type="keyUp",
            key=key,
            code=code,
            windowsVirtualKeyCode=vk,
            nativeVirtualKeyCode=vk,
        )

    def listeners(self) -> dict:
        """How many event listeners the window and the document have, by type."""
        counts = {}
        for name in ("window", "document"):
            handle = self.send("Runtime.evaluate", expression=name)["result"]["objectId"]
            found = self.send("DOMDebugger.getEventListeners", objectId=handle)["listeners"]
            for listener in found:
                key = f"{name}:{listener['type']}"
                counts[key] = counts.get(key, 0) + 1
            self.send("Runtime.releaseObject", objectId=handle)
        return counts

    def errors(self) -> list[str]:
        found = []
        for event in self.events:
            method, params = event["method"], event.get("params", {})
            if method == "Runtime.exceptionThrown":
                details = params["exceptionDetails"]
                found.append((details.get("exception") or {}).get("description") or details.get("text"))
            elif method == "Runtime.consoleAPICalled" and params.get("type") in ("error", "assert"):
                found.append(" ".join(str(a.get("value", a.get("description", ""))) for a in params.get("args", [])))
        return found + list(self.js("window.__e2e ? window.__e2e.errors : []") or [])

    def screenshot(self, path: Path) -> None:
        data = self.send("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))


# --- The server, its jobs and the browser -------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


@pytest.fixture(scope="module")
def chrome_binary():
    binary = find_chrome()
    if binary is None:
        pytest.skip("Google Chrome isn't installed (set RESOLUTION_E2E_CHROME to its path)")
    return binary


@pytest.fixture(scope="module")
def app(tmp_path_factory, chrome_binary):
    import platformdirs

    base = tmp_path_factory.mktemp("e2e")
    data, home = base / "data", base / "home"
    data.mkdir()
    (home / "Pictures").mkdir(parents=True)
    # The models already on this computer are used as they are (the Fast model is 4 MB if it isn't).
    models = Path(platformdirs.user_data_dir("Upscaler", appauthor=False)) / "models"
    if models.is_dir():
        (data / "models").symlink_to(models, target_is_directory=True)
    port = _free_port()
    env = {**os.environ, "UPSCALER_DATA_DIR": str(data), "HOME": str(home), "USERPROFILE": str(home)}
    log = (base / "server.log").open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "upscaler", "serve", "--no-browser", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while True:
        try:
            if _get(f"{url}/api/status").get("app", {}).get("name") == "Re:Solution":
                break
        except OSError:
            pass
        if time.monotonic() > deadline or proc.poll() is not None:
            proc.kill()
            pytest.fail(f"the server didn't start; see {base / 'server.log'}")
        time.sleep(0.25)
    yield SimpleNamespace(url=url, data=data, base=base)
    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def _clip(path: Path, *, size: str, seconds: int, sound: bool = False, rotate: int = 0) -> Path:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    plain = path.with_name(f"plain-{path.name}")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={FPS}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-shortest"] if sound else []
    subprocess.run([*cmd, str(plain)], check=True)
    if rotate:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", str(rotate),
                        "-i", str(plain), "-c", "copy", str(path)], check=True)  # fmt: skip
        plain.unlink()
    else:
        plain.replace(path)
    return path


@pytest.fixture(scope="module")
def jobs(app):
    """Real Fast jobs: a 4K video with sound, a rotated phone video (4K portrait), an 8K picture."""
    import httpx

    clips = app.base / "clips"
    clips.mkdir()
    landscape = _clip(clips / "testsrc2.mp4", size="640x360", seconds=8, sound=True)
    portrait = _clip(clips / "portrait.mp4", size="640x360", seconds=6, rotate=90)
    made = {}
    with httpx.Client(base_url=app.url, timeout=120) as client:
        for key, clip in (("video", landscape), ("portrait", portrait)):
            with clip.open("rb") as handle:
                answer = client.post("/api/videos", files={"file": (clip.name, handle, "video/mp4")},
                                     data={"options": json.dumps({"mode": "fast", "strength": 1})})  # fmt: skip
            answer.raise_for_status()
            made[key] = answer.json()["job"]["id"]
        answer = client.post(
            "/api/images/sample", json={"sample_id": "interior", "options": {"mode": "fast", "size": "8k"}}
        )
        answer.raise_for_status()
        made["image"] = answer.json()["job"]["id"]
        deadline = time.monotonic() + 900
        while True:
            listed = {job["id"]: job for job in client.get("/api/jobs").json()["jobs"]}
            states = {key: listed[job_id]["status"] for key, job_id in made.items()}
            if all(state == "done" for state in states.values()):
                break
            assert not any(state in ("failed", "cancelled") for state in states.values()), states
            assert time.monotonic() < deadline, f"the jobs took too long: {states}"
            time.sleep(1)
    return made


@pytest.fixture(scope="module")
def browser(chrome_binary, app, jobs):
    chrome = Chrome(chrome_binary)
    yield chrome
    chrome.close()
    if os.environ.get("RESOLUTION_E2E_REPORT"):
        Path(os.environ["RESOLUTION_E2E_REPORT"]).write_text(json.dumps(REPORT, indent=2))


def _settled(browser: Chrome) -> dict:
    """What the page has running when nothing is open: its own timers, frames and listeners."""
    browser.pump(1.2)  # let the page's own start-up timers run out
    return {"active": browser.js("window.__e2e.active()"), "listeners": browser.listeners()}


def _assert_nothing_left(browser: Chrome, baseline: dict, where: str) -> None:
    counts = browser.js("window.__resolutionDebug.counts()")
    assert counts == {key: 0 for key in counts}, f"{where}: still running after close: {counts}"
    active = browser.js("window.__e2e.active()")
    assert (
        active["intervals"] <= baseline["active"]["intervals"]
    ), f"{where}: intervals grew {baseline['active']} -> {active}"
    assert active["frames"] <= max(1, baseline["active"]["frames"]), f"{where}: animation frames left {active}"
    assert (
        active["timeouts"] <= baseline["active"]["timeouts"] + 3
    ), f"{where}: timeouts grew {baseline['active']} -> {active}"
    assert browser.listeners() == baseline["listeners"], f"{where}: window/document listeners changed"


def _close(browser: Chrome, dialog: str, how: str) -> None:
    if how == "button":
        browser.click(f"{dialog} [data-close]")
    elif how == "escape":
        browser.key("Escape")
    else:  # a click on the backdrop: give the full-screen dialog a margin so there is one to click
        browser.js(f"document.querySelector('{dialog}').style.cssText = 'inset: 6vh 6vw; width: 88vw; height: 88dvh;'")
        browser.pump(0.05)
        browser.click_at(8, 8)


def _closed(browser: Chrome, dialog: str) -> None:
    browser.wait(f"!document.querySelector('{dialog}').open", 2, what=f"{dialog} to close")
    browser.js(f"document.querySelector('{dialog}').style.cssText = ''")


def _error_check(browser: Chrome, where: str) -> None:
    errors = browser.errors()
    assert not errors, f"{where}: JavaScript errors: {errors[:5]}"


# --- The image viewer ---------------------------------------------------------------------------

VIEWER = "#viewer-dialog"


def test_image_viewer_opens_and_closes_25_times_every_way(browser, app, jobs):
    rng = random.Random(1018)
    browser.goto(f"{app.url}/#photos")
    browser.wait("document.querySelector('.result-tile')", 30, what="the finished picture")
    baseline = _settled(browser)
    states = ["just-opened", "fit", "100", "200", "side", "panned", "wheel"]
    hows = ["button", "escape", "backdrop"]
    for cycle in range(CYCLES):
        state, how = states[cycle % len(states)], hows[cycle % len(hows)]
        if cycle >= len(states) * 2:
            state, how = rng.choice(states), rng.choice(hows)
        browser.click(".result-tile")
        browser.wait(f"document.querySelector('{VIEWER}').open", 2, what="the viewer to open")
        if state != "just-opened":
            browser.wait("(window.__resolutionDebug.viewer() || {}).fit", 10, what="the picture's tiles")
        if state in ("100", "200"):
            browser.key("1" if state == "100" else "2")
        elif state == "side":
            browser.click(f"{VIEWER} .zv-modes label:nth-child(2)")
        elif state == "panned":
            browser.key("1")
            browser.pump(0.3)
            browser.drag(f"{VIEWER} .zv-stage", 0.6, 0.3, steps=8, pause=0.01)
        elif state == "wheel":
            where = browser.box(f"{VIEWER} .zv-stage")
            browser.send(
                "Input.dispatchMouseEvent", type="mouseWheel", x=where["x"], y=where["y"], deltaX=0, deltaY=-400
            )
        browser.pump(rng.uniform(0.0, 0.4))
        _close(browser, VIEWER, how)
        _closed(browser, VIEWER)
        assert not browser.js(f"document.querySelector('{VIEWER} .zv')"), "the picture is still in the closed viewer"
        browser.wait(
            "document.activeElement && document.activeElement !== document.body"
            f" && !document.querySelector('{VIEWER}').contains(document.activeElement)",
            1,
            what="focus to come back",
        )
        if state == "side":  # keep the next cycles starting from the slider
            browser.js("localStorage.removeItem('resolution.compareMode')")
        _error_check(browser, f"viewer cycle {cycle} ({state}, {how})")
    browser.pump(0.3)
    _assert_nothing_left(browser, baseline, "image viewer")
    latency = [e["ms"] for e in browser.js("window.__e2e.closeLatency.splice(0)") if e["id"] == "viewer-dialog"]
    REPORT["image_viewer_close_ms"] = {"n": len(latency), "median": statistics.median(latency), "max": max(latency)}
    assert len(latency) == CYCLES
    assert max(latency) < 100, f"closing took {max(latency):.0f} ms"


def test_image_viewer_controls_switch_rapidly(browser, app, jobs):
    browser.goto(f"{app.url}/#photos")
    browser.wait("document.querySelector('.result-tile')", 30)
    baseline = _settled(browser)
    browser.js("window.__e2e.longTasks.length = 0; window.__e2e.events.length = 0")
    browser.click(".result-tile")
    browser.wait("(window.__resolutionDebug.viewer() || {}).fit", 10)
    controls = [f"{VIEWER} .zv-modes label:nth-child(2)", f"{VIEWER} .zv-modes label:nth-child(1)",
                f"{VIEWER} [data-zoom='fit']", f"{VIEWER} [data-zoom='1']", f"{VIEWER} [data-zoom='2']"]  # fmt: skip
    for n in range(40):
        browser.click(controls[n % len(controls)])
        browser.pump(0.03)
    browser.click(f"{VIEWER} [data-zoom='2']")
    browser.pump(0.6)
    shown = browser.js("window.__resolutionDebug.viewer()")
    assert abs(shown["scale"] - 2) < 0.01 and not shown["fitted"]
    for key in ("0", "1", "2", "+", "-", "0"):
        browser.key(key)
        browser.pump(0.05)
    browser.pump(0.4)
    assert browser.js("window.__resolutionDebug.viewer().fitted")
    _close(browser, VIEWER, "escape")
    _closed(browser, VIEWER)
    _error_check(browser, "viewer controls")
    _assert_nothing_left(browser, baseline, "viewer controls")
    slow = browser.js("window.__e2e.events.filter(e => e.ms >= 100)")
    long_tasks = browser.js("window.__e2e.longTasks")
    REPORT["image_viewer_switching"] = {"slow_inputs": len(slow), "long_tasks": len(long_tasks),
                                        "longest_task_ms": max([t["ms"] for t in long_tasks], default=0)}  # fmt: skip
    assert not slow, f"controls that took 100 ms or more to react: {slow[:5]}"
    assert max([t["ms"] for t in long_tasks], default=0) < 200


# --- The video comparison ----------------------------------------------------------------------

VC = "#video-compare-dialog"
VIDEOS = f"document.querySelectorAll('{VC} video')"


def _compare_button(browser: Chrome, job_id: str) -> str:
    """The "Compare with the original" button of a job's card."""
    selector = browser.js(
        "(() => { for (const card of document.querySelectorAll('.job')) {"
        f" const download = card.querySelector('a[href*=\"/api/jobs/{job_id}/\"]');"
        " const button = card.querySelector('[data-focus=\"compare-0\"]');"
        " if (download && button) { button.id = button.id || 'e2e-compare-' + Math.random().toString(36).slice(2); return '#' + button.id; } }"
        " return null; })()"
    )
    assert selector, "no Compare button for the video"
    return selector


def _open_compare(browser: Chrome, button: str, ready: bool = True) -> None:
    browser.click(button)
    browser.wait(f"document.querySelector('{VC}').open", 2, what="the comparison to open")
    if ready:
        browser.wait(
            f"{VIDEOS}.length === 2 && Array.from({VIDEOS}).every(v => v.readyState >= 1)", 60, what="the two players"
        )


def _forget_copies(app, job_id: str) -> None:
    """Deletes the comparison's copies, so the next open prepares them again."""
    status_url = f"{app.url}/api/jobs/{job_id}/results/0/compare/status"
    deadline = time.monotonic() + 60
    while _get(status_url)["status"] in ("waiting", "preparing"):
        assert time.monotonic() < deadline, "the comparison copies never finished"
        time.sleep(0.2)
    shutil.rmtree(app.data / "jobs" / job_id / "compare" / "0", ignore_errors=True)


def _debug(browser: Chrome):
    return browser.js("window.__resolutionDebug.videoCompare()")


def _assert_videos_stopped(browser: Chrome, where: str) -> None:
    playing = browser.js("Array.from(document.querySelectorAll('video')).filter(v => !v.paused).length")
    assert playing == 0, f"{where}: {playing} video(s) still playing after close"
    assert browser.js(f"{VIDEOS}.length") == 0, f"{where}: players left in the closed comparison"


def test_video_compare_opens_and_closes_25_times_in_every_state(browser, app, jobs):
    rng = random.Random(2026)
    browser.goto(f"{app.url}/#videos")
    browser.wait("document.querySelector('[data-focus=\"compare-0\"]')", 30, what="the finished videos")
    baseline = _settled(browser)
    states = ["preparing", "paused", "playing", "seeking", "zoomed-playing", "loading", "stepping"]
    hows = ["button", "escape", "backdrop"]
    for cycle in range(CYCLES):
        state, how = states[cycle % len(states)], hows[cycle % len(hows)]
        if cycle >= len(states) * 2:
            state, how = rng.choice(states), rng.choice(hows)
        job = jobs["video"] if cycle % 2 == 0 else jobs["portrait"]
        if state == "preparing":
            _forget_copies(app, job)
        button = _compare_button(browser, job)
        if state == "loading":  # a slow connection: the players run out of data and wait
            browser.send("Network.setCacheDisabled", cacheDisabled=True)
            browser.send(
                "Network.emulateNetworkConditions",
                offline=False,
                latency=150,
                downloadThroughput=120_000,
                uploadThroughput=120_000,
            )
        try:
            _open_compare(browser, button, ready=state != "preparing")
            if state == "preparing":
                browser.wait(f"document.querySelector('{VC} .vc-prep')", 3, what="the preparing screen")
                browser.pump(rng.uniform(0.05, 0.8))
            elif state in ("playing", "zoomed-playing", "loading", "seeking"):
                if state == "zoomed-playing":
                    browser.click(f"{VC} .vc-zoom label:nth-child(3)")
                browser.click(f"{VC} .vc-play")
                browser.pump(rng.uniform(0.2, 1.0) if state != "loading" else 1.5)
                if state == "seeking":
                    browser.click(f"{VC} .vc-timeline", at=rng.uniform(0.1, 0.9))
                    browser.pump(rng.uniform(0.0, 0.1))
            elif state == "stepping":
                for _ in range(4):
                    browser.key(".")
            else:
                browser.pump(rng.uniform(0.0, 0.3))
            _close(browser, VC, how)
            _closed(browser, VC)
        finally:
            if state == "loading":
                browser.send(
                    "Network.emulateNetworkConditions",
                    offline=False,
                    latency=0,
                    downloadThroughput=-1,
                    uploadThroughput=-1,
                )
                browser.send("Network.setCacheDisabled", cacheDisabled=False)
        _assert_videos_stopped(browser, f"compare cycle {cycle} ({state}, {how})")
        browser.wait(
            "document.activeElement && document.activeElement !== document.body"
            f" && !document.querySelector('{VC}').contains(document.activeElement)",
            1,
            what="focus to come back",
        )
        _error_check(browser, f"compare cycle {cycle} ({state}, {how})")
    browser.pump(0.3)
    _assert_nothing_left(browser, baseline, "video comparison")
    closes = browser.js("window.__e2e.closeLatency.splice(0)")
    latency = [e["ms"] for e in closes if e["id"] == "video-compare-dialog"]
    REPORT["video_compare_close_ms"] = {"n": len(latency), "median": statistics.median(latency), "max": max(latency)}
    assert len(latency) == CYCLES, closes
    assert max(latency) < 100, f"closing took {max(latency):.0f} ms"


def _playing_together(browser: Chrome, timeout: float = 3) -> dict:
    return browser.wait(
        "(() => { const d = window.__resolutionDebug.videoCompare();"
        " return d && d.wantPlay && !d.paused[0] && !d.paused[1] && !d.seeking && !d.loading ? d : null; })()",
        timeout,
        what="both videos playing",
    )


def test_play_pause_pressed_50_times(browser, app, jobs):
    browser.goto(f"{app.url}/#videos")
    browser.wait("document.querySelector('[data-focus=\"compare-0\"]')", 30)
    _open_compare(browser, _compare_button(browser, jobs["video"]))
    browser.wait(f"Array.from({VIDEOS}).every(v => v.readyState >= 2)", 20)
    browser.click(f"{VC} .vc-loop")  # so the clip never ends half-way through the presses
    browser.js("window.__e2e.events.length = 0; window.__e2e.longTasks.length = 0")
    rng = random.Random(50)
    for press in range(SPAM):
        browser.click(f"{VC} .vc-play", hold=rng.choice([0.0, 0.0, 0.03, 0.08]))
        wanted = press % 2 == 0  # the first press plays
        if rng.random() < 0.4:  # a burst: the next press follows at once, before the videos have answered
            continue
        browser.pump(rng.uniform(0.02, 0.15))
        state = _debug(browser)
        assert state["wantPlay"] == wanted, f"press {press + 1} was lost (wanted {'play' if wanted else 'pause'})"
        # A moment later the videos do what was asked: an old 'pause' or 'play' event arriving late
        # (after more presses) mustn't turn it round.
        browser.pump(0.35)
        state = _debug(browser)
        assert state["wantPlay"] == wanted and state["paused"] == [not wanted, not wanted], (press + 1, state)
    # 50 presses: paused again. Both videos agree and are in step.
    browser.pump(0.5)
    state = _debug(browser)
    assert not state["wantPlay"] and state["paused"] == [True, True]
    assert abs(state["drift"]) <= FRAME / 2 + 1e-3
    # Pause and Play within one moment while the original is still catching up (so it has to seek
    # first): the late 'pause' event of the first press used to undo the second one.
    browser.click(f"{VC} .vc-play")
    _playing_together(browser)
    for attempt in range(5):
        browser.js(
            f"(() => {{ const [orig] = {VIDEOS}; orig.currentTime += 0.06;"
            f" const button = document.querySelector('{VC} .vc-play'); button.click(); button.click(); }})()"
        )
        browser.pump(0.6)
        assert _debug(browser)["wantPlay"], f"attempt {attempt}: the second press was undone"
        _playing_together(browser, 2)  # (a loop restart holds both still for a moment)
    browser.click(f"{VC} .vc-play")
    browser.pump(0.3)
    browser.click(f"{VC} .vc-play")
    _playing_together(browser)
    slow = browser.js("window.__e2e.events.filter(e => e.ms >= 100)")
    long_tasks = browser.js("window.__e2e.longTasks")
    REPORT["play_pause_spam"] = {"presses": SPAM, "slow_inputs": len(slow), "long_tasks": len(long_tasks),
                                 "longest_task_ms": max([t["ms"] for t in long_tasks], default=0)}  # fmt: skip
    assert not slow, f"presses that took 100 ms or more to react: {slow[:5]}"
    assert max([t["ms"] for t in long_tasks], default=0) < 200
    _close(browser, VC, "escape")
    _closed(browser, VC)
    _assert_videos_stopped(browser, "after the presses")
    _error_check(browser, "play/pause presses")


def test_scrubbing_speed_and_frame_steps(browser, app, jobs):
    browser.goto(f"{app.url}/#videos")
    browser.wait("document.querySelector('[data-focus=\"compare-0\"]')", 30)
    _open_compare(browser, _compare_button(browser, jobs["video"]))
    browser.click(f"{VC} .vc-loop")
    browser.click(f"{VC} .vc-play")
    _playing_together(browser, 10)
    # Scrub while playing: both hold still while the timeline is dragged and carry on after.
    where = browser.box(f"{VC} .vc-timeline")
    y = where["y"]
    x0, x1 = where["left"] + where["width"] * 0.15, where["left"] + where["width"] * 0.8
    browser.mouse("mouseMoved", x0, y)
    browser.mouse("mousePressed", x0, y, pressed=True)
    for n in range(1, 25):
        browser.mouse("mouseMoved", x0 + (x1 - x0) * n / 24, y, pressed=True)
        browser.pump(0.02)
        if n == 12:
            mid = _debug(browser)
            assert mid["scrubbing"] and mid["paused"] == [True, True], f"still playing while scrubbing: {mid}"
    browser.mouse("mouseReleased", x1, y)
    after = _playing_together(browser, 5)
    assert after["leader"] > 0.5 * after["frames"] / FPS, "the scrub didn't move the videos along"
    # Speed: both change together.
    browser.js(
        f"(() => {{ const s = document.querySelector('{VC} .vc-speed'); s.value = '2'; s.dispatchEvent(new Event('change', {{bubbles: true}})); }})()"
    )
    browser.pump(0.6)
    state = _debug(browser)
    assert all(abs(r - 2) <= 2 * 0.25 for r in state["rate"]) and abs(state["rate"][0] - 2) < 1e-6
    browser.js(
        f"(() => {{ const s = document.querySelector('{VC} .vc-speed'); s.value = '1'; s.dispatchEvent(new Event('change', {{bubbles: true}})); }})()"
    )
    # Frame steps: they pause, and each moves both videos by exactly one frame.
    browser.click(f"{VC} .vc-play")
    browser.wait("!window.__resolutionDebug.videoCompare().wantPlay", 2)
    browser.wait("!window.__resolutionDebug.videoCompare().seeking", 3)
    start = _debug(browser)["frame"]
    for _ in range(5):
        browser.click(f"{VC} [aria-label='Next frame (.)']")
        browser.pump(0.05)
    browser.wait("!window.__resolutionDebug.videoCompare().seeking", 3)
    browser.pump(0.2)
    state = _debug(browser)
    assert state["frame"] == min(start + 5, state["frames"] - 1), (start, state)
    assert state["paused"] == [True, True] and abs(state["drift"]) <= FRAME / 2 + 1e-3
    for _ in range(3):
        browser.key(",")
    browser.wait("!window.__resolutionDebug.videoCompare().seeking", 3)
    browser.pump(0.2)
    assert _debug(browser)["frame"] == state["frame"] - 3
    _close(browser, VC, "button")
    _closed(browser, VC)
    _assert_videos_stopped(browser, "after scrubbing")
    _error_check(browser, "scrubbing, speed and frame steps")


def test_video_zoom_switches_rapidly_and_playback_stays_in_step(browser, app, jobs):
    browser.goto(f"{app.url}/#videos")
    browser.wait("document.querySelector('[data-focus=\"compare-0\"]')", 30)
    baseline = _settled(browser)
    results = {}
    for name in ("video", "portrait"):
        _open_compare(browser, _compare_button(browser, jobs[name]))
        browser.click(f"{VC} .vc-loop")
        browser.click(f"{VC} .vc-play")
        _playing_together(browser, 10)
        browser.js("window.__e2e.events.length = 0; window.__e2e.longTasks.length = 0")
        for n in range(30):  # Fit / 100% / 200% as fast as they can be pressed, while playing
            browser.click(f"{VC} .vc-zoom label:nth-child({n % 3 + 1})")
            browser.pump(0.03)
        browser.click(f"{VC} .vc-zoom label:nth-child(1)")
        assert _debug(browser)["zoom"] == "fit"
        # Back to the start, then steady playback (no loop in the way): settle, then measure.
        browser.click(f"{VC} .vc-loop")
        browser.click(f"{VC} .vc-play")
        browser.wait("!window.__resolutionDebug.videoCompare().wantPlay", 2)
        browser.click(f"{VC} .vc-timeline", at=0.0)
        browser.wait(
            "(() => { const d = window.__resolutionDebug.videoCompare(); return !d.seeking && d.frame < 3; })()", 5
        )
        browser.click(f"{VC} .vc-play")
        _playing_together(browser, 5)
        browser.pump(1.0)  # a start settles within about half a second (the sound getting going)
        # Steady playback: sample both clocks (and the frames actually shown) for three seconds.
        sample = browser.js(
            "new Promise((done) => {"
            f" const [orig, up] = {VIDEOS}; const shown = {{orig: null, up: null}};"
            " const watch = (v, k) => { const f = (now, m) => { shown[k] = m.mediaTime; if (!stop) v.requestVideoFrameCallback(f); }; v.requestVideoFrameCallback(f); };"
            " let stop = false; watch(orig, 'orig'); watch(up, 'up');"
            " const q0 = [orig.getVideoPlaybackQuality(), up.getVideoPlaybackQuality()];"
            " const drift = [], frames = []; const t0 = performance.now();"
            " const tick = () => { drift.push(orig.currentTime - up.currentTime);"
            "  if (shown.orig != null && shown.up != null) frames.push(Math.round((shown.orig - shown.up) * 30));"
            "  if (performance.now() - t0 < 3000) requestAnimationFrame(tick);"
            "  else { stop = true; const q1 = [orig.getVideoPlaybackQuality(), up.getVideoPlaybackQuality()];"
            "   done({drift, frames, dropped: q1.map((q, i) => q.droppedVideoFrames - q0[i].droppedVideoFrames),"
            "         total: q1.map((q, i) => q.totalVideoFrames - q0[i].totalVideoFrames)}); } };"
            " requestAnimationFrame(tick); })",
            timeout=30,
        )
        duration = _debug(browser)["frames"] / FPS
        drift = [
            d - round(d / duration) * duration if abs(d) > duration / 2 else d for d in sample["drift"]
        ]  # a loop wrap
        long_tasks = browser.js("window.__e2e.longTasks")
        slow = browser.js("window.__e2e.events.filter(e => e.ms >= 100)")
        results[name] = {
            "max_drift_ms": round(max(abs(d) for d in drift) * 1000, 2),
            "mean_drift_ms": round(statistics.fmean(abs(d) for d in drift) * 1000, 2),
            "shown_frames_apart_max": max((abs(f) for f in sample["frames"]), default=0),
            "shown_frames_within_one": round(
                sum(abs(f) <= 1 for f in sample["frames"]) / max(1, len(sample["frames"])), 3
            ),
            "dropped": sample["dropped"],
            "total": sample["total"],
            "long_tasks": len(long_tasks),
            "longest_task_ms": max([t["ms"] for t in long_tasks], default=0),
            "slow_inputs": len(slow),
        }
        assert max(abs(d) for d in drift) <= FRAME + 0.002, f"{name}: drift over a frame: {results[name]}"
        # The frames on screen (requestVideoFrameCallback) are sampled once per animation frame, and the two
        # videos can present on neighbouring screen refreshes, so this is a looser check than the clocks above.
        assert results[name]["shown_frames_within_one"] >= 0.9, results[name]
        for dropped, total in zip(sample["dropped"], sample["total"], strict=True):
            assert total > 0 and dropped / total <= 0.05, f"{name}: dropped frames {results[name]}"
        assert results[name]["longest_task_ms"] < 200, results[name]
        assert not slow, f"{name}: zoom presses that took 100 ms or more: {slow[:5]}"
        _close(browser, VC, "escape")
        _closed(browser, VC)
        _assert_videos_stopped(browser, f"{name} after steady playback")
    REPORT["steady_playback"] = results
    _error_check(browser, "zoom switching and steady playback")
    _assert_nothing_left(browser, baseline, "steady playback")


def test_the_card_plays_the_browser_friendly_copy_one_at_a_time(browser, app, jobs):
    browser.goto(f"{app.url}/#videos")
    browser.wait("document.querySelectorAll('.result-video-play').length >= 2", 30, what="the video cards")
    # Nothing is loaded until Play is pressed: a still, not the 10-bit HEVC file.
    assert browser.js("document.querySelectorAll('.result video').length") == 0
    assert browser.js(
        "Array.from(document.querySelectorAll('.result-video img')).every(i => i.src.endsWith('/poster.jpg'))"
    )
    browser.js("document.querySelectorAll('.result-video-play').forEach((b, n) => { b.id = 'e2e-play-' + n; })")
    browser.click("#e2e-play-0")
    browser.wait(
        "(() => { const v = document.querySelector('.result video'); return v && !v.paused && v.currentTime > 0.2; })()",
        60,
        what="the card video",
    )
    source = browser.js("document.querySelector('.result video').currentSrc")
    assert (
        source.endswith("/compare/upscaled.mp4")
        and browser.js("document.querySelector('.result video').preload") == "metadata"
    )
    # A second card stops the first; opening the comparison stops any card.
    browser.click("#e2e-play-1")
    browser.wait("document.querySelectorAll('.result video').length === 1", 30, what="one card video")
    browser.wait("(() => { const v = document.querySelector('.result video'); return v && !v.paused; })()", 60)
    _open_compare(browser, _compare_button(browser, jobs["video"]))
    assert browser.js("document.querySelectorAll('.result video').length") == 0
    _close(browser, VC, "button")
    _closed(browser, VC)
    assert browser.js("window.__resolutionDebug.counts().videos") == 0
    # In the whole run, the page never asked for a result's own (10-bit HEVC) file.
    asked = [e["params"]["request"]["url"] for e in browser.events if e["method"] == "Network.requestWillBeSent"]
    assert asked and not [url for url in asked if url.endswith("/file")], "the page loaded a 10-bit HEVC result"
    _error_check(browser, "card videos")
