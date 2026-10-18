"""Share on Wi-Fi: the host's switch that lets phones and computers on the home Wi-Fi use the app.

The app itself listens only on 127.0.0.1. Sharing adds a second uvicorn listener for the same app on
every network (0.0.0.0) at port + 1, or the next free port, and removes it again, without a restart.
Whoever comes in through that listener is a guest.

Tests that run the real listener point it at 127.0.0.1 on a free port (`listen_here_only`), so the
computer's firewall never asks and a copy of the app that's running is never disturbed; the
production address is checked separately with stand-in sockets.
"""

import dataclasses
import queue as queue_module
import signal
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import keyring
import keyring.backend
import keyring.errors
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from upscaler import cli, config, jobs, server
from upscaler.hardware import Hardware
from upscaler.imageio_utils import render_vector
from upscaler.jobs import JobQueue, default_runners

MAC_16GB = Hardware("mps", "Apple M4", 16, "mac", True)
HOST_URL = "http://127.0.0.1:7860"
LAN_URL = "http://192.168.1.20:7861"
LAN_IP = "192.168.1.20"
NO_PASSWORD = "Anyone on your Wi-Fi can use Re:Solution. You can set a password in Settings."
NO_NETWORK = "You don't seem to be connected to Wi-Fi or a network"
MAC_NOTE = "If your Mac asks whether to allow incoming connections, click Allow."
SHARE_KEYS = {"enabled", "port", "urls", "qr_svg", "password_set", "can_toggle", "note", "warning"}


class MemoryKeyring(keyring.backend.KeyringBackend):
    priority = 1

    def __init__(self):
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError(username)
        del self.store[(service, username)]


class FakeEngine:
    name = "mlx"

    def weights_present(self):
        return True


class _NoKeepAwake:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class FakeSocket:
    """Stands in for a listening socket, so a test can check the production address without
    opening a real port on the network."""

    def __init__(self, host: str, port: int):
        self.address = (host, port)
        self.closed = False

    def getsockname(self):
        return self.address

    def close(self):
        self.closed = True


def _serve_until_stopped(wifi_server, sock):
    """Stands in for uvicorn serving the Wi-Fi listener: "runs" until asked to stop."""
    while not wifi_server.should_exit:
        time.sleep(0.005)
    sock.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(server, "detect_hardware", lambda: MAC_16GB)
    monkeypatch.setattr(server, "get_seedvr2_engine", lambda hw: FakeEngine())
    monkeypatch.setattr(jobs, "KeepAwake", _NoKeepAwake)
    addresses = [LAN_IP]
    monkeypatch.setattr(server, "lan_ipv4_addresses", lambda: list(addresses))
    previous = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    yield SimpleNamespace(tmp=tmp_path, store=tmp_path / "data" / "jobs", addresses=addresses)
    keyring.set_keyring(previous)


@pytest.fixture
def fake_listener(monkeypatch):
    """No real sockets: records where the Wi-Fi listener would bind. `taken` holds (host, port)
    pairs another program is using."""
    bound: list[FakeSocket] = []
    taken: set[tuple[str, int]] = set()

    def listen(host, port):
        if (host, port) in taken:
            raise OSError(48, "Address already in use")
        bound.append(FakeSocket(host, port))
        return bound[-1]

    monkeypatch.setattr(server, "listen_socket", listen)
    monkeypatch.setattr(server.WifiSharing, "_serve", staticmethod(_serve_until_stopped))
    return SimpleNamespace(bound=bound, taken=taken)


def make_app(env, share=False):
    queue = JobQueue(env.store, default_runners(env.store))
    return server.create_app(queue, share, port=7860), queue


def listen_here_only(app):
    """Test-only override: the real Wi-Fi listener, but on 127.0.0.1 and any free port."""
    sharing = app.state.upscaler.sharing
    sharing.host = "127.0.0.1"
    sharing.preferred_port = 0
    return sharing


def host_client(app) -> TestClient:
    return TestClient(app, base_url=HOST_URL, client=("127.0.0.1", 50000))


def guest_client(app, ip: str = "192.168.1.5") -> TestClient:
    return TestClient(app, base_url=LAN_URL, client=(ip, 50001))


def share_status(client) -> dict:
    return client.get("/api/status").json()["share"]


def wifi_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "upscaler-wifi" and thread.is_alive()]


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.02)


def answers(address: str) -> bool:
    try:
        return httpx.get(f"{address}/api/status", timeout=2, trust_env=False).status_code == 200
    except httpx.TransportError:
        return False


def decode_qr(svg: str, tmp_path: Path) -> str:
    """Reads a QR code back the way a phone's camera would."""
    path = tmp_path / "qr.svg"
    path.write_text(svg, encoding="utf-8")
    rgba = render_vector(path, (330, 330))
    grey = np.ascontiguousarray(rgba[..., :3].mean(axis=2).astype(np.uint8))
    return cv2.QRCodeDetector().detectAndDecode(grey)[0]


def drain(subscription) -> list[dict]:
    """The events a page's live updates would have received so far."""
    found = []
    while True:
        try:
            found.append(subscription.get_nowait())
        except queue_module.Empty:
            return found


def set_password(host: TestClient, password: str = "family") -> None:
    assert host.post("/api/settings", json={"access_password": password}).status_code == 200


@pytest.fixture(autouse=True)
def no_listener_left_behind():
    yield
    assert wifi_threads() == []


# --- Where it listens ----------------------------------------------------------------------------


def test_production_listener_is_on_every_network(env, fake_listener):
    assert server.WIFI_HOST == "0.0.0.0"
    app, _ = make_app(env)
    assert app.state.upscaler.sharing.host == "0.0.0.0"  # only tests point it elsewhere
    with host_client(app) as host:
        assert fake_listener.bound == []  # until the host switches it on, only this computer can reach the app
        assert host.post("/api/share", json={"enabled": True}).status_code == 200
        assert [s.address for s in fake_listener.bound] == [("0.0.0.0", 7861)]  # the app's port + 1
    assert fake_listener.bound[0].closed


def test_wifi_listener_never_trusts_forwarded_headers(env, fake_listener):
    app, _ = make_app(env)
    with host_client(app) as host:
        host.post("/api/share", json={"enabled": True})
        assert app.state.upscaler.sharing._server.config.proxy_headers is False


# --- The switch -----------------------------------------------------------------------------------


def test_switch_turns_the_real_listener_on_and_off(env):
    app, queue = make_app(env)
    sharing = listen_here_only(app)
    events = queue.subscribe()
    with host_client(app) as host:
        off = share_status(host)
        assert set(off) == SHARE_KEYS
        assert off["enabled"] is False and off["urls"] == [] and off["qr_svg"] is None and off["port"] is None
        assert off["can_toggle"] is True and off["password_set"] is False
        assert wifi_threads() == []

        response = host.post("/api/share", json={"enabled": True})
        assert response.status_code == 200
        on = response.json()
        port = on["port"]
        assert set(on) == SHARE_KEYS
        assert on["enabled"] is True and port == sharing.port and port > 0
        assert on["urls"] == [f"http://{LAN_IP}:{port}"]
        assert on["qr_svg"].startswith("<svg") and "<script" not in on["qr_svg"]
        assert on["password_set"] is False and on["can_toggle"] is True
        assert on["warning"] == NO_PASSWORD
        assert on["note"] == server.firewall_note()
        assert share_status(host) == on  # the status block says the same
        assert {"event": "status"} in drain(events)  # every open page shows "Sharing on"
        assert len(wifi_threads()) == 1

        address = f"http://127.0.0.1:{port}"
        wait_for(lambda: answers(address))
        through_wifi = httpx.get(f"{address}/api/status", timeout=5, trust_env=False).json()
        assert through_wifi["is_host"] is False  # a guest, even from this computer

        again = host.post("/api/share", json={"enabled": True})  # twice is fine: still one listener
        assert again.json()["port"] == port and len(wifi_threads()) == 1

        response = host.post("/api/share", json={"enabled": False})
        assert response.status_code == 200
        off = response.json()
        assert off["enabled"] is False and off["urls"] == [] and off["qr_svg"] is None
        assert off["port"] is None and off["note"] is None and off["warning"] is None
        assert {"event": "status"} in drain(events)
        assert wifi_threads() == []
        assert not answers(address)
        assert host.get("/api/status").json()["is_host"] is True  # the app itself carries on

        assert host.post("/api/share", json={"enabled": True}).json()["enabled"] is True  # and on again
        assert len(wifi_threads()) == 1
    assert wifi_threads() == []  # quitting the app stops sharing


def test_switching_to_the_same_state_sends_no_status_event(env, fake_listener):
    app, queue = make_app(env)
    with host_client(app) as host:
        events = queue.subscribe()
        host.post("/api/share", json={"enabled": False})  # already off
        assert {"event": "status"} not in drain(events)


def test_guests_cannot_switch_sharing(env, fake_listener):
    app, _ = make_app(env)
    with host_client(app) as host:
        guest = guest_client(app)
        for enabled in (True, False):
            response = guest.post("/api/share", json={"enabled": enabled})
            assert response.status_code == 403
            assert response.json()["detail"] == server.HOST_ONLY
        assert fake_listener.bound == [] and share_status(host)["enabled"] is False

        host.post("/api/share", json={"enabled": True})
        assert guest.post("/api/share", json={"enabled": False}).status_code == 403
        assert share_status(host)["enabled"] is True  # a guest can't switch it off either

        set_password(host)
        assert guest.post("/api/login", json={"password": "family"}).status_code == 200
        assert guest.post("/api/share", json={"enabled": False}).status_code == 403  # logged in is still a guest
        assert share_status(host)["enabled"] is True


def test_guests_see_sharing_but_no_switch(env, fake_listener):
    app, _ = make_app(env, share=True)
    with host_client(app):
        block = share_status(guest_client(app))
    assert block["enabled"] is True and block["urls"] == [LAN_URL]
    assert block["can_toggle"] is False
    assert block["note"] is None and block["warning"] is None  # those are for the person at the computer


def test_bad_switch_requests(env, fake_listener):
    app, _ = make_app(env)
    with host_client(app) as host:
        for body in ({}, {"enabled": "yes"}, {"enabled": 1}, {"enabled": None}):
            assert host.post("/api/share", json=body).status_code == 400
        assert fake_listener.bound == []


# --- Who is a guest -------------------------------------------------------------------------------

SPOOFED = {
    "X-Forwarded-For": "127.0.0.1",
    "X-Real-IP": "127.0.0.1",
    "Forwarded": "for=127.0.0.1;host=127.0.0.1:7860",
    "X-Forwarded-Host": "127.0.0.1:7860",
    "Client-IP": "127.0.0.1",
}


def test_lan_listener_requests_are_guests_even_with_spoofed_headers(env):
    app, _ = make_app(env)
    sharing = listen_here_only(app)
    with host_client(app) as host:
        host.post("/api/share", json={"enabled": True})
        address = f"http://127.0.0.1:{sharing.port}"
        wait_for(lambda: answers(address))
        with httpx.Client(base_url=address, timeout=5, trust_env=False) as lan:
            # The connection really comes from 127.0.0.1, and claims to be the host every way it can.
            headers = {**SPOOFED, "Host": "127.0.0.1:7860"}
            status = lan.get("/api/status", headers=headers).json()
            assert status["is_host"] is False and "settings" not in status
            assert status["share"]["can_toggle"] is False
            assert lan.get("/api/settings", headers=headers).status_code == 403
            assert lan.post("/api/share", json={"enabled": False}, headers=headers).status_code == 403
            assert lan.post("/api/open-folder", json={}, headers=headers).status_code == 403
        assert share_status(host)["enabled"] is True


def test_lan_clients_are_guests_whatever_their_headers_say(env, fake_listener):
    app, _ = make_app(env, share=True)
    with host_client(app):
        guest = guest_client(app)
        status = guest.get("/api/status", headers=SPOOFED).json()
        assert status["is_host"] is False
        assert guest.post("/api/share", json={"enabled": False}, headers=SPOOFED).status_code == 403
        # The app as the Wi-Fi listener serves it, reached from this very computer.
        via_wifi = TestClient(app.state.upscaler.sharing.app, base_url=HOST_URL, client=("127.0.0.1", 50002))
        assert via_wifi.get("/api/status", headers=SPOOFED).json()["is_host"] is False
        assert via_wifi.post("/api/share", json={"enabled": False}).status_code == 403
        assert via_wifi.get("/api/settings").status_code == 403


def test_main_listener_never_trusts_forwarded_headers(env, monkeypatch):
    configs = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: configs.append(self.config))
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: FakeSocket(host, port))
    assert cli.run(["--no-browser"]) == 0
    assert configs[0].proxy_headers is False


# --- The QR code ----------------------------------------------------------------------------------


def test_qr_code_only_while_sharing(env, fake_listener, tmp_path):
    app, _ = make_app(env)
    with host_client(app) as host:
        assert share_status(host)["qr_svg"] is None
        on = host.post("/api/share", json={"enabled": True}).json()
        assert on["qr_svg"].startswith("<svg") and "viewBox" in on["qr_svg"]  # scales to any size
        assert decode_qr(on["qr_svg"], tmp_path) == on["urls"][0] == LAN_URL
        assert share_status(host)["qr_svg"] == on["qr_svg"]
        assert host.post("/api/share", json={"enabled": False}).json()["qr_svg"] is None
        assert share_status(host)["qr_svg"] is None


def test_qr_code_is_for_the_first_address(env, fake_listener, tmp_path):
    env.addresses[:] = ["192.168.1.20", "10.0.0.5"]
    app, _ = make_app(env)
    with host_client(app) as host:
        on = host.post("/api/share", json={"enabled": True}).json()
    assert on["urls"] == ["http://192.168.1.20:7861", "http://10.0.0.5:7861"]
    assert decode_qr(on["qr_svg"], tmp_path) == "http://192.168.1.20:7861"


# --- Launching with --share -----------------------------------------------------------------------


def test_share_flag_starts_with_sharing_on(env):
    app, _ = make_app(env, share=True)  # as the "Start Re-Solution (Share on Wi-Fi)" launchers start it
    sharing = listen_here_only(app)
    assert not sharing.enabled  # nothing listens before the app starts
    with host_client(app) as host:
        block = share_status(host)
        assert block["enabled"] is True and block["port"] == sharing.port
        assert block["urls"] == [f"http://{LAN_IP}:{sharing.port}"] and block["qr_svg"]
        assert len(wifi_threads()) == 1
        wait_for(lambda: answers(f"http://127.0.0.1:{sharing.port}"))
    assert not sharing.enabled and wifi_threads() == []


def test_share_flag_passes_the_wifi_socket_to_the_app(env, monkeypatch, capsys):
    made, bound = [], []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", lambda host, port: bound.append((host, port)) or FakeSocket(host, port))
    real_create_app = server.create_app

    def create_app(queue, share, *, port, wifi_socket=None):
        made.append((share, port, wifi_socket))
        return real_create_app(queue, share, port=port, wifi_socket=wifi_socket)

    monkeypatch.setattr(server, "create_app", create_app)
    assert cli.run(["--share", "--no-browser"]) == 0
    assert bound == [("127.0.0.1", 7860), ("0.0.0.0", 7861)]
    assert [(share, port, wifi.address) for share, port, wifi in made] == [(True, 7860, ("0.0.0.0", 7861))]
    assert f"http://{LAN_IP}:7861" in capsys.readouterr().out


def test_sharing_is_not_remembered_across_restarts(env, fake_listener):
    app, _ = make_app(env)
    with host_client(app) as host:
        set_password(host)  # saves the settings while sharing is on
        host.post("/api/share", json={"enabled": True})
    assert not any("share" in field.name for field in dataclasses.fields(config.Settings))
    assert "share" not in (config.data_dir() / "settings.json").read_text(encoding="utf-8")
    restarted, _ = make_app(env)  # started again without --share
    with host_client(restarted) as host:
        assert share_status(host)["enabled"] is False
    assert len(fake_listener.bound) == 1


def test_quitting_the_app_stops_sharing_and_leaves_no_thread(env):
    app, _ = make_app(env)
    sharing = listen_here_only(app)
    with host_client(app) as host:
        host.post("/api/share", json={"enabled": True})
        port = sharing.port
        wait_for(lambda: answers(f"http://127.0.0.1:{port}"))
        assert len(wifi_threads()) == 1
    assert wifi_threads() == []
    assert not answers(f"http://127.0.0.1:{port}")


def test_wifi_listener_leaves_ctrl_c_to_the_app(env):
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    app, _ = make_app(env)
    sharing = listen_here_only(app)
    with host_client(app) as host:
        host.post("/api/share", json={"enabled": True})
        wait_for(lambda: answers(f"http://127.0.0.1:{sharing.port}"))
        assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


# --- Friendly problems ----------------------------------------------------------------------------


def test_busy_port_moves_to_the_next_free_one(env, fake_listener):
    fake_listener.taken.update({("0.0.0.0", 7861), ("0.0.0.0", 7862)})
    app, _ = make_app(env)
    with host_client(app) as host:
        on = host.post("/api/share", json={"enabled": True})
        assert on.status_code == 200
        assert on.json()["port"] == 7863 and on.json()["urls"] == [f"http://{LAN_IP}:7863"]
        host.post("/api/share", json={"enabled": False})

        fake_listener.taken.clear()  # the other program has quit: back to the usual port
        assert host.post("/api/share", json={"enabled": True}).json()["port"] == 7861


def test_busy_port_for_real(env):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    busy = blocker.getsockname()[1]
    try:
        app, _ = make_app(env)
        sharing = listen_here_only(app)
        sharing.preferred_port = busy
        with host_client(app) as host:
            on = host.post("/api/share", json={"enabled": True}).json()
            assert on["enabled"] is True
            assert busy < on["port"] < busy + server.SHARE_PORT_TRIES
            wait_for(lambda: answers(f"http://127.0.0.1:{on['port']}"))
    finally:
        blocker.close()


def test_every_port_busy_is_friendly(env, fake_listener):
    fake_listener.taken.update(("0.0.0.0", port) for port in range(7861, 7861 + server.SHARE_PORT_TRIES))
    app, _ = make_app(env)
    with host_client(app) as host:
        response = host.post("/api/share", json={"enabled": True})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "7861" in detail and "Re:Solution" in detail
        assert share_status(host)["enabled"] is False
    assert fake_listener.bound == []

    started, _ = make_app(env, share=True)  # a Share launcher: the app still starts, just not shared
    with host_client(started) as host:
        assert share_status(host)["enabled"] is False
        assert host.get("/api/status").json()["is_host"] is True


def test_no_network_is_friendly(env, fake_listener):
    env.addresses.clear()
    app, queue = make_app(env)
    with host_client(app) as host:
        events = queue.subscribe()
        response = host.post("/api/share", json={"enabled": True})
        assert response.status_code == 409
        assert response.json()["detail"].startswith(NO_NETWORK)
        assert fake_listener.bound == []  # nothing listens, so the firewall doesn't ask for nothing
        assert share_status(host)["enabled"] is False
        assert {"event": "status"} not in drain(events)

        env.addresses.append(LAN_IP)  # joined the Wi-Fi: it works straight away
        on = host.post("/api/share", json={"enabled": True}).json()
        assert on["enabled"] is True and on["urls"] == [LAN_URL]


def test_launched_sharing_without_a_network_says_so(env, fake_listener):
    env.addresses.clear()
    app, _ = make_app(env, share=True)
    with host_client(app) as host:
        block = share_status(host)
    assert block["enabled"] is True and block["urls"] == [] and block["qr_svg"] is None
    assert block["warning"].startswith(NO_NETWORK)


def test_password_warning_only_without_a_password(env, fake_listener):
    app, _ = make_app(env)
    with host_client(app) as host:
        on = host.post("/api/share", json={"enabled": True}).json()
        assert on["password_set"] is False and on["warning"] == NO_PASSWORD
        set_password(host)
        block = share_status(host)
        assert block["password_set"] is True and block["warning"] is None
        assert host.post("/api/share", json={"enabled": False}).json()["password_set"] is True


def test_firewall_note_per_computer():
    assert server.firewall_note("darwin") == MAC_NOTE
    assert "Windows" in server.firewall_note("win32") and "Allow" in server.firewall_note("win32")
    assert server.firewall_note("linux") is None


def test_note_shown_while_sharing(env, fake_listener, monkeypatch):
    monkeypatch.setattr(server, "firewall_note", lambda platform=None: MAC_NOTE)
    app, _ = make_app(env)
    with host_client(app) as host:
        assert share_status(host)["note"] is None
        assert host.post("/api/share", json={"enabled": True}).json()["note"] == MAC_NOTE


# --- The command line -----------------------------------------------------------------------------


def test_share_launcher_moves_to_the_next_free_port(env, monkeypatch, capsys):
    bound = []

    def listen(host, port):
        if (host, port) == ("0.0.0.0", 7861):
            raise OSError(48, "Address already in use")
        bound.append((host, port))
        return FakeSocket(host, port)

    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", listen)
    assert cli.run(["--share", "--no-browser"]) == 0
    assert bound == [("127.0.0.1", 7860), ("0.0.0.0", 7862)]
    out = capsys.readouterr().out
    assert f"http://{LAN_IP}:7862" in out and ":7861" not in out


def test_share_launcher_with_every_port_busy_still_starts(env, monkeypatch, capsys):
    def listen(host, port):
        if host == "0.0.0.0":
            raise OSError(48, "Address already in use")
        return FakeSocket(host, port)

    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)
    monkeypatch.setattr(cli, "_probe_running", lambda port: None)
    monkeypatch.setattr(cli, "_listen_socket", listen)
    assert cli.run(["--share", "--no-browser"]) == 0
    out = capsys.readouterr().out
    assert "port 7861" in out and "On this computer, open: http://127.0.0.1:7860" in out
