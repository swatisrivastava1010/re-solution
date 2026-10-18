"""The one-step installers (install.sh, install.ps1) and the Start Re-Solution launchers.

Nothing here touches the network, the real ~/Applications or the Desktop: every run gets a
scratch HOME, and `uname`, `sysctl`, `sw_vers`, `nvidia-smi` and `uv` are faked on PATH.
The installers look for a running Re:Solution on a free scratch port, never the real 7860.

install.sh and the .command launchers are tested on Macs and Linux; install.ps1 runs on Windows,
with pwsh (or powershell when pwsh isn't there). Set RE_SOLUTION_TEST_POWERSHELL to pwsh or
powershell to choose: CI runs this file once with each, so Windows PowerShell 5.1, which most
Windows PCs open, is tested as well as PowerShell 7.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import tomllib
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"
INSTALL_PS1 = ROOT / "install.ps1"
LAUNCHER = ROOT / "Start Re-Solution.command"
SHARE_LAUNCHER = ROOT / "Start Re-Solution (Share on Wi-Fi).command"
BAT = ROOT / "Start Re-Solution.bat"
SHARE_BAT = ROOT / "Start Re-Solution (Share on Wi-Fi).bat"
REPO = "github.com/swatisrivastava1010/re-solution"

ON_WINDOWS = sys.platform == "win32"
# macOS ships bash 3.2; testing with it keeps the scripts working for everyone. Not on Windows,
# which uses install.ps1 and the .bat launchers (a bash there is Git's or WSL's, not a Mac's).
BASH = None if ON_WINDOWS else ("/bin/bash" if Path("/bin/bash").exists() else shutil.which("bash"))
# Which PowerShell to test install.ps1 with: "pwsh" (PowerShell 7) or "powershell" (Windows
# PowerShell 5.1). Unset: pwsh, else powershell.
TEST_POWERSHELL = os.environ.get("RE_SOLUTION_TEST_POWERSHELL", "").strip()
if not ON_WINDOWS:
    PWSH = None  # install.ps1 is for Windows: no LOCALAPPDATA, robocopy or nvidia-smi.exe elsewhere
elif TEST_POWERSHELL:
    PWSH = shutil.which(TEST_POWERSHELL)  # missing: test_the_powershell_asked_for_is_the_one_tested fails
else:
    PWSH = shutil.which("pwsh") or shutil.which("powershell")
# Only reading the script, which any PowerShell does the same way on any system.
PARSER_PWSH = PWSH or (None if ON_WINDOWS else shutil.which("pwsh"))
ON_MAC = sys.platform == "darwin" and shutil.which("osacompile") is not None

needs_bash = pytest.mark.skipif(
    BASH is None, reason="install.sh and the .command launchers are for Macs and Linux, and need bash"
)
needs_mac = pytest.mark.skipif(not ON_MAC, reason="the Re-Solution app is built with macOS's osacompile")
needs_pwsh = pytest.mark.skipif(
    PWSH is None, reason="install.ps1 is tested on Windows, with PowerShell (pwsh or powershell)"
)
needs_windows = pytest.mark.skipif(
    not ON_WINDOWS or PWSH is None, reason="install.ps1 copies with robocopy, which only Windows has"
)


# --- fakes -----------------------------------------------------------------------------------


def _tool(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fake_computer(
    tmp_path: Path,
    *,
    system: str = "Darwin",
    machine: str = "arm64",
    apple_silicon: bool = True,
    macos: str = "26.0",
    nvidia_driver: str | None = None,
    compute_cap: str | None = "8.6",
) -> dict[str, str]:
    """An environment whose PATH pretends to be a given computer.

    nvidia_driver: None = no nvidia-smi, "broken" = nvidia-smi that fails, else the
    version it reports (for example "590.12").
    compute_cap: the card generation nvidia-smi reports (for example "8.6" for an RTX 3060,
    "6.1" for a GTX 1080); None = a driver too old to know the question, "broken" = it fails.
    """
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    _tool(bin_dir, "uname", f'case "$1" in -m) echo {machine} ;; *) echo {system} ;; esac\n')
    _tool(
        bin_dir,
        "sysctl",
        f'case "$*" in *hw.optional.arm64*) echo {1 if apple_silicon else 0} ;; *) exit 1 ;; esac\n',
    )
    _tool(bin_dir, "sw_vers", f"echo {macos}\n")
    if nvidia_driver == "broken":
        _tool(bin_dir, "nvidia-smi", "echo 'Failed to initialize NVML: Driver/library version mismatch'\nexit 9\n")
    elif nvidia_driver is not None:
        if compute_cap is None:
            cap_answer = "echo 'Field \"compute_cap\" is not a valid field to query.'\n    exit 2"
        elif compute_cap == "broken":
            cap_answer = "echo 'Unable to determine the device handle for GPU 0000:01:00.0: Unknown Error'\n    exit 15"
        else:
            cap_answer = f"echo {compute_cap}"
        _tool(
            bin_dir,
            "nvidia-smi",
            f'case "$*" in\n*compute_cap*)\n    {cap_answer} ;;\n*) echo {nvidia_driver} ;;\nesac\n',
        )
    # uv writes what it was asked to do into a log instead of installing anything.
    _tool(bin_dir, "uv", 'printf "%s\\n" "$*" >> "$FAKE_UV_LOG"\n')

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")}
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        HOME=str(home),
        NO_COLOR="1",
        FAKE_UV_LOG=str(tmp_path / "uv.log"),
        # Where the installers look for a running Re:Solution: nothing listens here.
        RE_SOLUTION_PORT=str(free_port()),
    )
    return env


# What a running Re:Solution answers at /api/status (see cli._probe_running).
STATUS_ANSWER = json.dumps({"version": "1.0.0", "is_host": True, "share": {"enabled": False}}).encode()


@contextmanager
def fake_web_server(body: bytes) -> Iterator[int]:
    """A web server on a free port that answers every GET with `body`; yields its port."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # the name http.server calls
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def uv_calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "uv.log"
    return log.read_text().splitlines() if log.exists() else []


def run_installer(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(INSTALL_SH), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def chosen_variant(output: str) -> str | None:
    match = re.search(r"PyTorch build: (\S+)", output)
    return match.group(1) if match else None


def make_source(tmp_path: Path, version: str = "1.1.0") -> Path:
    """A tiny stand-in for the app folder, with the kinds of local files a copy must skip."""
    src = tmp_path / "source"
    (src / "upscaler" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (src / "upscaler" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (src / "upscaler" / "__pycache__" / "cli.cpython-312.pyc").write_bytes(b"\0")
    (src / "pyproject.toml").write_text('[project]\nname = "upscaler"\n')
    (src / "uv.lock").write_text("version = 1\n")
    # The bundled public-domain samples: only the top-level samples/ folder is private.
    (src / "upscaler" / "web" / "samples").mkdir(parents=True, exist_ok=True)
    (src / "upscaler" / "web" / "samples" / "samples.json").write_text("[]\n")
    (src / "samples").mkdir(exist_ok=True)
    (src / "samples" / "README.md").write_text("Put your own test images here.\n")
    (src / "samples" / "family-photo.jpg").write_bytes(b"private")
    (src / ".venv" / "lib").mkdir(parents=True, exist_ok=True)
    (src / ".venv" / "lib" / "torch.so").write_bytes(b"big")
    (src / "model.safetensors.part").write_bytes(b"half")
    (src / ".torch-variant").write_text("cu130\n")
    # Stands in for the real launcher: records where it was started from.
    launcher = src / "Start Re-Solution.command"
    launcher.write_text('#!/bin/bash\ncd "$(dirname "$0")" && pwd -P > started.txt\n')
    launcher.chmod(0o644)  # a download can lose the executable bit; the installer restores it
    return src


# --- syntax ----------------------------------------------------------------------------------


@needs_bash
@pytest.mark.parametrize("script", [INSTALL_SH, LAUNCHER, SHARE_LAUNCHER], ids=lambda p: p.name)
def test_shell_scripts_parse(script):
    result = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_launchers_renamed():
    for path in (LAUNCHER, SHARE_LAUNCHER, BAT, SHARE_BAT):
        assert path.is_file(), path.name
    assert not list(ROOT.glob("Start Upscaler*")), "the old Start Upscaler launchers should be gone"
    if ON_WINDOWS:
        return  # Windows files have no executable bit (git still records it for Macs and Linux)
    for path in (LAUNCHER, SHARE_LAUNCHER, INSTALL_SH):
        assert path.stat().st_mode & stat.S_IXUSR, f"{path.name} must be executable"


def test_installers_point_at_the_github_repo():
    for path in (INSTALL_SH, INSTALL_PS1):
        text = path.read_text(encoding="utf-8")
        assert f"https://{REPO}" in text, path.name
    assert "Applications/Re-Solution" in INSTALL_SH.read_text(encoding="utf-8")
    assert "LOCALAPPDATA" in INSTALL_PS1.read_text(encoding="utf-8")


# --- install.sh: choosing the PyTorch build ---------------------------------------------------


@needs_bash
def test_dry_run_apple_silicon_mac_picks_cpu(tmp_path):
    env = fake_computer(tmp_path, system="Darwin", machine="arm64")
    target = tmp_path / "Re-Solution"
    result = run_installer(env, "--dry-run", "--dir", str(target))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cpu"
    assert "1/5 Checking your computer" in result.stdout
    # A dry run changes nothing.
    assert not target.exists()
    assert uv_calls(tmp_path) == []
    assert list((tmp_path / "home").iterdir()) == []


@needs_bash
def test_dry_run_rosetta_terminal_still_counts_as_apple_silicon(tmp_path):
    # A Terminal opened "using Rosetta" reports x86_64, but the Mac is still Apple Silicon.
    env = fake_computer(tmp_path, system="Darwin", machine="x86_64", apple_silicon=True)
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cpu"


@needs_bash
def test_dry_run_nvidia_driver_590_picks_cu130(tmp_path):
    env = fake_computer(tmp_path, system="Linux", machine="x86_64", nvidia_driver="590.12")
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cu130"


@needs_bash
def test_dry_run_nvidia_driver_470_picks_cpu(tmp_path):
    env = fake_computer(tmp_path, system="Linux", machine="x86_64", nvidia_driver="470.256.02", compute_cap=None)
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cpu"
    assert "CPU only: slow" in result.stdout
    assert "470" in result.stdout and "580" in result.stdout  # says why, and what to update to


@needs_bash
def test_dry_run_old_nvidia_card_with_a_new_driver_picks_cpu(tmp_path):
    # Driver 580 is the last one for GTX 10 cards, so an up-to-date one reports 580 or newer,
    # but the CUDA 13 build has no code for those cards: every job would fail on them.
    env = fake_computer(tmp_path, system="Linux", machine="x86_64", nvidia_driver="581.57", compute_cap="6.1")
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cpu"
    assert "too old" in result.stdout and "CPU only: slow" in result.stdout
    assert "RTX 20" in result.stdout  # what it would need
    assert "nvidia.com/drivers" not in result.stdout  # a newer driver wouldn't help
    assert "about 3 GB" not in result.stdout


@needs_bash
@pytest.mark.parametrize(
    ("compute_cap", "expected"),
    [
        ("7.5", "cu130"),  # GeForce RTX 20 and GTX 16 series (Turing): the oldest that works
        ("12.0", "cu130"),  # RTX 50 series
        ("7.0", "cpu"),  # Titan V (Volta)
        ("broken", "cpu"),  # can't tell, so the build that always works
    ],
)
def test_dry_run_checks_the_card_generation(tmp_path, compute_cap, expected):
    env = fake_computer(tmp_path, system="Linux", machine="x86_64", nvidia_driver="590.12", compute_cap=compute_cap)
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == expected


@needs_bash
def test_dry_run_broken_nvidia_driver_picks_cpu(tmp_path):
    env = fake_computer(tmp_path, system="Linux", machine="x86_64", nvidia_driver="broken")
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == "cpu"
    assert "CPU only: slow" in result.stdout


@needs_bash
def test_intel_mac_gets_a_plain_explanation(tmp_path):
    # PyTorch has no Intel Mac builds any more, so stop early with a clear reason.
    env = fake_computer(tmp_path, system="Darwin", machine="x86_64", apple_silicon=False)
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 1
    assert "Apple Silicon" in result.stderr
    assert "Traceback" not in result.stderr


@needs_bash
def test_old_macos_gets_a_plain_explanation(tmp_path):
    env = fake_computer(tmp_path, system="Darwin", macos="13.6.1")
    result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 1
    assert "macOS 14" in result.stderr and "13.6.1" in result.stderr


@needs_bash
def test_unknown_option_is_explained(tmp_path):
    result = run_installer(fake_computer(tmp_path), "--frobnicate")
    assert result.returncode == 1
    assert "--frobnicate" in result.stderr and "--help" in result.stderr


@needs_bash
def test_help(tmp_path):
    result = run_installer(fake_computer(tmp_path), "--help")
    assert result.returncode == 0
    for option in ("--dir", "--source", "--no-start", "--no-shortcuts", "--dry-run"):
        assert option in result.stdout


# --- install.sh: installing and updating ------------------------------------------------------


@needs_bash
def test_install_from_local_source(tmp_path):
    env = fake_computer(tmp_path, system="Darwin")
    src = make_source(tmp_path)
    target = tmp_path / "apps" / "Re-Solution"
    result = run_installer(env, "--source", str(src), "--dir", str(target), "--no-start", "--no-shortcuts")
    assert result.returncode == 0, result.stdout + result.stderr

    assert (target / "upscaler" / "__init__.py").read_text() == '__version__ = "1.1.0"\n'
    assert (target / "upscaler" / "web" / "samples" / "samples.json").exists()
    assert (target / "samples" / "README.md").exists()
    # Things that belong to the source computer are never copied.
    assert not (target / "samples" / "family-photo.jpg").exists()
    assert not (target / ".venv").exists()
    assert not (target / "upscaler" / "__pycache__").exists()
    assert not (target / "model.safetensors.part").exists()
    # The variant is this computer's, not the source's.
    assert (target / ".torch-variant").read_text().strip() == "cpu"
    assert os.access(target / "Start Re-Solution.command", os.X_OK)
    assert uv_calls(tmp_path) == [f"sync --frozen --no-dev --extra cpu --directory {target}"]
    for n in range(1, 6):
        assert f"{n}/5 " in result.stdout
    # --no-shortcuts: nothing outside the install folder.
    assert not (tmp_path / "home" / "Applications").exists()
    assert not (tmp_path / "home" / "Desktop").exists()
    if ON_MAC:
        assert (target / "Re-Solution.app" / "Contents" / "Info.plist").exists()


@needs_bash
def test_rerun_updates_in_place_and_keeps_local_files(tmp_path):
    env = fake_computer(tmp_path, system="Linux", nvidia_driver="590.12")
    src = make_source(tmp_path, version="1.1.0")
    target = tmp_path / "Re-Solution"
    args = ("--source", str(src), "--dir", str(target), "--no-start", "--no-shortcuts")
    assert run_installer(env, *args).returncode == 0

    # Local things a person (or the app) adds to the install folder.
    (target / ".venv").mkdir()
    (target / ".venv" / "keep.txt").write_text("installed packages")
    (target / "samples" / "my-test.jpg").write_bytes(b"mine")
    make_source(tmp_path, version="1.2.0")

    result = run_installer(env, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "upscaler" / "__init__.py").read_text() == '__version__ = "1.2.0"\n'
    assert (target / "upscaler" / "web" / "samples" / "samples.json").exists()
    assert (target / ".venv" / "keep.txt").exists()
    assert (target / "samples" / "my-test.jpg").exists()
    assert (target / ".torch-variant").read_text().strip() == "cu130"
    assert uv_calls(tmp_path)[-1] == f"sync --frozen --no-dev --extra cu130 --directory {target}"


@needs_bash
def test_refuses_to_install_into_a_folder_with_other_things(tmp_path):
    env = fake_computer(tmp_path)
    target = tmp_path / "Documents"
    target.mkdir()
    (target / "notes.txt").write_text("important")
    result = run_installer(env, "--source", str(make_source(tmp_path)), "--dir", str(target), "--no-start")
    assert result.returncode == 1
    assert "already has other" in result.stderr
    assert sorted(p.name for p in target.iterdir()) == ["notes.txt"]
    assert uv_calls(tmp_path) == []


@needs_bash
def test_refuses_to_install_inside_the_source(tmp_path):
    env = fake_computer(tmp_path)
    src = make_source(tmp_path)
    result = run_installer(env, "--source", str(src), "--dir", str(src / "Re-Solution"), "--no-start")
    assert result.returncode == 1
    assert not (src / "Re-Solution").exists()


@needs_bash
def test_stops_while_re_solution_is_open(tmp_path):
    # Updating under a running copy would swap its code while it runs, and the start at the
    # end would only reopen the old copy ("already running").
    env = fake_computer(tmp_path, system="Linux")
    src = make_source(tmp_path)
    target = tmp_path / "Re-Solution"
    with fake_web_server(STATUS_ANSWER) as port:
        env["RE_SOLUTION_PORT"] = str(port)
        result = run_installer(env, "--source", str(src), "--dir", str(target), "--no-start", "--no-shortcuts")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Re:Solution is open" in result.stderr
    assert "Close its window" in result.stderr and "Nothing was changed" in result.stderr
    assert "Traceback" not in result.stderr and "connected to the internet" not in result.stderr
    assert not target.exists()
    assert uv_calls(tmp_path) == []


@needs_bash
def test_dry_run_says_re_solution_is_open(tmp_path):
    env = fake_computer(tmp_path, system="Linux")
    with fake_web_server(STATUS_ANSWER) as port:
        env["RE_SOLUTION_PORT"] = str(port)
        result = run_installer(env, "--dry-run", "--dir", str(tmp_path / "Re-Solution"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Re:Solution is open" in result.stdout


@needs_bash
def test_another_program_on_the_port_is_not_mistaken_for_re_solution(tmp_path):
    env = fake_computer(tmp_path, system="Linux")
    target = tmp_path / "Re-Solution"
    with fake_web_server(b"<html>Some other program</html>") as port:
        env["RE_SOLUTION_PORT"] = str(port)
        result = run_installer(
            env, "--source", str(make_source(tmp_path)), "--dir", str(target), "--no-start", "--no-shortcuts"
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "pyproject.toml").exists()


def _git(*args: str, cwd: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def _publish_source_as_git_repo(src: Path) -> str:
    """Turns the stand-in app folder into a git repo on branch main; returns its URL."""
    (src / ".gitignore").write_text(".venv/\n__pycache__/\nsamples/*\n!samples/README.md\n*.part\n.torch-variant\n")
    _git("init", "--quiet", "--initial-branch", "main", cwd=src)
    _git("add", "-A", cwd=src)
    _git("commit", "--quiet", "-m", "app", cwd=src)
    return src.as_uri()


@needs_bash
@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_install_and_update_with_git(tmp_path):
    env = fake_computer(tmp_path, system="Linux")
    src = make_source(tmp_path, version="1.1.0")
    env["RE_SOLUTION_REPO_URL"] = _publish_source_as_git_repo(src)
    target = tmp_path / "Re-Solution"

    result = run_installer(env, "--dir", str(target), "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / ".git").is_dir()
    assert (target / "upscaler" / "__init__.py").read_text() == '__version__ = "1.1.0"\n'
    assert (target / "upscaler" / "web" / "samples" / "samples.json").exists()
    assert (target / ".torch-variant").read_text().strip() == "cpu"

    # A new version is published; the person has their own files in the install folder.
    (src / "upscaler" / "__init__.py").write_text('__version__ = "1.2.0"\n')
    _git("commit", "--quiet", "-am", "new version", cwd=src)
    (target / "samples" / "my-test.jpg").write_bytes(b"mine")
    (target / ".venv").mkdir()

    result = run_installer(env, "--dir", str(target), "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Updated to the latest version" in result.stdout
    assert (target / "upscaler" / "__init__.py").read_text() == '__version__ = "1.2.0"\n'
    assert (target / "samples" / "my-test.jpg").exists()
    assert (target / ".venv").is_dir()
    assert len(uv_calls(tmp_path)) == 2


def _zip_like_github(src: Path, archive: Path) -> str:
    """Zips the stand-in app folder laid out like GitHub's download; returns its URL."""
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file() and ".venv" not in path.parts:
                zf.write(path, f"re-solution-main/{path.relative_to(src)}")
    return archive.as_uri()


@needs_mac
@needs_bash
def test_mac_without_developer_tools_downloads_the_zip(tmp_path):
    # /usr/bin/git on a Mac without the developer tools only offers to install them.
    env = fake_computer(tmp_path)
    _tool(tmp_path / "fakebin", "xcode-select", "exit 2\n")
    env["PATH"] = f"{tmp_path / 'fakebin'}:/usr/bin:/bin:/usr/sbin:/sbin"
    src = make_source(tmp_path)
    env["RE_SOLUTION_ZIP_URL"] = _zip_like_github(src, tmp_path / "main.zip")
    target = tmp_path / "Re-Solution"

    result = run_installer(env, "--dir", str(target), "--no-start", "--no-shortcuts")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / ".git").exists()
    assert (target / "upscaler" / "__init__.py").exists()
    assert (target / "upscaler" / "web" / "samples" / "samples.json").exists()
    assert not (target / "samples" / "family-photo.jpg").exists()
    assert os.access(target / "Start Re-Solution.command", os.X_OK)
    assert uv_calls(tmp_path) == [f"sync --frozen --no-dev --extra cpu --directory {target}"]


@needs_bash
@pytest.mark.skipif(shutil.which("unzip") is None, reason="needs unzip")
def test_git_that_does_not_work_falls_back_to_the_zip(tmp_path):
    # For example the developer tools are there but Xcode's licence hasn't been accepted, or a
    # macOS upgrade broke them: git only complains, so every run would fail the same way.
    env = fake_computer(tmp_path)
    _tool(tmp_path / "fakebin", "xcode-select", "echo /Applications/Xcode.app/Contents/Developer\n")
    _tool(
        tmp_path / "fakebin",
        "git",
        "echo 'You have not agreed to the Xcode license agreements. Please run sudo xcodebuild -license' >&2\nexit 69\n",
    )
    env["RE_SOLUTION_ZIP_URL"] = _zip_like_github(make_source(tmp_path), tmp_path / "main.zip")
    target = tmp_path / "Re-Solution"

    result = run_installer(env, "--dir", str(target), "--no-start", "--no-shortcuts")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / ".git").exists()
    assert (target / "upscaler" / "__init__.py").exists()
    assert uv_calls(tmp_path) == [f"sync --frozen --no-dev --extra cpu --directory {target}"]


@needs_mac
@needs_bash
def test_shortcuts_go_to_home_applications_and_desktop(tmp_path):
    env = fake_computer(tmp_path)
    home = Path(env["HOME"])
    (home / "Desktop").mkdir()
    target = tmp_path / "Re-Solution"
    result = run_installer(env, "--source", str(make_source(tmp_path)), "--dir", str(target), "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    app = home / "Applications" / "Re-Solution.app"
    assert (app / "Contents" / "Info.plist").exists()
    alias = home / "Desktop" / "Re-Solution"
    assert alias.is_symlink() and alias.resolve() == app.resolve()
    assert not (target / "Re-Solution.app").exists()

    # Running it again replaces the app and alias instead of failing.
    result = run_installer(env, "--source", str(make_source(tmp_path)), "--dir", str(target), "--no-start")
    assert result.returncode == 0, result.stdout + result.stderr
    assert alias.resolve() == app.resolve()


def _applescript_launch_command(app: Path) -> str:
    source = subprocess.run(["osadecompile", str(app)], capture_output=True, text=True, check=True).stdout
    match = re.search(r'set launchCommand to "((?:[^"\\]|\\.)*)"', source)
    assert match, source
    return re.sub(r"\\(.)", r"\1", match.group(1))  # undo AppleScript string escapes


@needs_mac
@needs_bash
def test_mac_app_starts_the_launcher_from_an_awkward_folder_name(tmp_path):
    env = fake_computer(tmp_path)
    target = tmp_path / 'Bob\'s "Apps"' / "Re Solution"
    result = run_installer(
        env, "--source", str(make_source(tmp_path)), "--dir", str(target), "--no-start", "--no-shortcuts"
    )
    assert result.returncode == 0, result.stdout + result.stderr

    command = _applescript_launch_command(target / "Re-Solution.app")
    assert command.endswith("&& ./Start\\ Re-Solution.command")
    # Terminal runs the command in the person's login shell (zsh on current Macs).
    subprocess.run(["/bin/zsh", "-c", command], env=env, check=True, timeout=30)
    assert (target / "started.txt").read_text().strip() == str(target.resolve())


# --- the Mac launchers -----------------------------------------------------------------------


def _copy_launchers(tmp_path: Path) -> Path:
    folder = tmp_path / "Re-Solution (1)"
    folder.mkdir()
    for launcher in (LAUNCHER, SHARE_LAUNCHER):
        shutil.copy2(launcher, folder / launcher.name)
    (folder / "pyproject.toml").write_text('[project]\nname = "upscaler"\n')  # an app folder
    return folder


def _run_launcher(env: dict[str, str], launcher: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(launcher), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@needs_bash
def test_launcher_detects_and_remembers_the_variant(tmp_path):
    folder = _copy_launchers(tmp_path)
    env = fake_computer(tmp_path, system="Darwin")
    result = _run_launcher(env, folder / LAUNCHER.name, "--no-browser")
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve --no-browser"]
    assert (folder / ".torch-variant").read_text().strip() == "cpu"


@needs_bash
def test_launcher_keeps_a_saved_cuda_variant_when_nvidia_smi_does_not_answer(tmp_path):
    folder = _copy_launchers(tmp_path)
    (folder / ".torch-variant").write_text("cu130\r\n")  # written on Windows
    # Detection would say cpu here, so cu130 can only come from the saved file: a driver
    # hiccup shouldn't swap 3 GB of PyTorch for the processor version and back.
    env = fake_computer(tmp_path, system="Linux", nvidia_driver="broken")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cu130 python -m upscaler serve"]


@needs_bash
@pytest.mark.parametrize(
    ("nvidia_driver", "compute_cap"),
    [
        (None, None),  # no NVIDIA card: a folder zipped up on an NVIDIA PC and passed on
        ("470.1", None),  # a driver too old for the CUDA 13 build
        ("581.57", "6.1"),  # a GTX 1080, which the CUDA 13 build has nothing for
    ],
    ids=["no-nvidia", "old-driver", "old-card"],
)
def test_launcher_replaces_a_saved_cuda_variant_this_pc_cannot_use(tmp_path, nvidia_driver, compute_cap):
    folder = _copy_launchers(tmp_path)
    (folder / ".torch-variant").write_text("cu130\r\n")
    env = fake_computer(tmp_path, system="Linux", nvidia_driver=nvidia_driver, compute_cap=compute_cap)
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve"]
    assert "about 3 GB" not in result.stdout and "CPU only: slow" in result.stdout
    assert (folder / ".torch-variant").read_text().strip() == "cpu"


@needs_bash
def test_launcher_detects_cu130_with_a_new_nvidia_driver(tmp_path):
    folder = _copy_launchers(tmp_path)
    env = fake_computer(tmp_path, system="Linux", nvidia_driver="580.65.06")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cu130 python -m upscaler serve"]
    assert (folder / ".torch-variant").read_text().strip() == "cu130"


@needs_bash
def test_launcher_uses_the_processor_for_an_old_nvidia_card(tmp_path):
    folder = _copy_launchers(tmp_path)
    env = fake_computer(tmp_path, system="Linux", nvidia_driver="581.57", compute_cap="6.1")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve"]
    assert "too old" in result.stdout and "CPU only: slow" in result.stdout
    assert (folder / ".torch-variant").read_text().strip() == "cpu"


@needs_bash
def test_launcher_outside_the_app_folder_explains(tmp_path):
    # A lone copy of the launcher (copied out on its own, or opened from inside the zip)
    # would otherwise fail with "No module named upscaler".
    folder = tmp_path / "Downloads"
    folder.mkdir()
    shutil.copy2(LAUNCHER, folder / LAUNCHER.name)
    env = fake_computer(tmp_path, system="Darwin")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 1
    assert "folder is missing" in result.stdout and "zip" in result.stdout
    assert uv_calls(tmp_path) == []
    assert not (folder / ".torch-variant").exists()


@needs_bash
def test_launcher_checks_a_saved_cpu_variant_again_off_a_mac(tmp_path):
    # A folder zipped up on a Mac (which saves "cpu") and opened on a PC with a new NVIDIA
    # driver, or a PC whose driver was updated since: use the graphics card.
    folder = _copy_launchers(tmp_path)
    (folder / ".torch-variant").write_text("cpu\n")
    env = fake_computer(tmp_path, system="Linux", nvidia_driver="590.12")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cu130 python -m upscaler serve"]
    assert (folder / ".torch-variant").read_text().strip() == "cu130"


@needs_bash
def test_launcher_keeps_a_saved_cpu_variant_on_a_mac(tmp_path):
    folder = _copy_launchers(tmp_path)
    (folder / ".torch-variant").write_text("cpu\n")
    env = fake_computer(tmp_path, system="Darwin", nvidia_driver="590.12")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve"]


@needs_bash
def test_launcher_ignores_a_cuda_variant_on_a_mac(tmp_path):
    # A folder copied over from a PC: there is no CUDA build for Macs.
    folder = _copy_launchers(tmp_path)
    (folder / ".torch-variant").write_text("cu130\n")
    env = fake_computer(tmp_path, system="Darwin")
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve"]
    assert (folder / ".torch-variant").read_text().strip() == "cpu"


@needs_bash
def test_share_launcher_adds_share(tmp_path):
    folder = _copy_launchers(tmp_path)
    env = fake_computer(tmp_path, system="Darwin")
    result = _run_launcher(env, folder / SHARE_LAUNCHER.name, "--port", "7861")
    assert result.returncode == 0, result.stdout + result.stderr
    assert uv_calls(tmp_path) == ["run --frozen --no-dev --extra cpu python -m upscaler serve --share --port 7861"]
    assert "Wi-Fi" in result.stdout


@needs_bash
def test_launcher_on_an_intel_mac_explains(tmp_path):
    folder = _copy_launchers(tmp_path)
    env = fake_computer(tmp_path, system="Darwin", machine="x86_64", apple_silicon=False)
    result = _run_launcher(env, folder / LAUNCHER.name)
    assert result.returncode == 1
    assert "Apple Silicon" in result.stdout
    assert uv_calls(tmp_path) == []


# --- the Windows launchers (checked as text: there is no cmd.exe here) -----------------------


def test_bat_launchers():
    main = BAT.read_bytes()
    share = SHARE_BAT.read_bytes()
    for raw in (main, share):
        # cmd.exe can misread labels in LF-only batch files.
        assert raw.count(b"\n") == raw.count(b"\r\n")
        assert raw.isascii()
    text = main.decode()
    assert "--extra %VARIANT%" in text
    assert ".torch-variant" in text
    assert "nvidia-smi" in text and "580" in text
    assert "CPU only: slow" in text
    assert '"%~dp0Start Re-Solution.bat" --share %*' in share.decode()

    lines = text.splitlines()
    uv_run = next(i for i, line in enumerate(lines) if line.startswith("uv run "))
    detect = lines.index("call :detect_variant")
    assert detect < uv_run
    # A saved cu130 is only kept over what detection says when nvidia-smi is there but didn't
    # answer: a folder from another PC, or an older card or driver, gets detected again.
    keeps = [line for line in lines if 'set "VARIANT=cu130"' in line and "SAVED" in line]
    assert keeps and all('"%NVIDIA%"=="no_answer"' in line for line in keeps), keeps
    assert all(lines.index(line) > detect for line in keeps)
    # Windows on Arm is turned away with a reason before uv tries (and fails) to install.
    arm_checks = [i for i, line in enumerate(lines) if "ARM64" in line and "goto windows_on_arm" in line]
    assert arm_checks and max(arm_checks) < uv_run
    assert "Intel or AMD processor" in text
    # It runs in its own folder (pushd also works in a \\computer\share folder), and a lone
    # copy opened from inside the zip explains what to do instead of failing.
    pushd = lines.index('pushd "%~dp0"')
    folder_check = lines.index('if not exist "pyproject.toml" goto incomplete_folder')
    assert pushd < folder_check < detect
    assert "Extract All" in text
    assert 'if not exist "%~dp0Start Re-Solution.bat" goto incomplete_folder' in share.decode()
    assert "Extract All" in share.decode()


def test_bat_launcher_checks_the_card_generation():
    text = BAT.read_text(encoding="utf-8")
    detect = text[text.index("\n:detect_variant\n") :]
    # Driver 580 alone isn't enough: GTX 10 cards get that driver too, but the CUDA 13 build
    # has nothing for them. Their compute capability (6.1) is below the 7.5 it needs.
    assert "--query-gpu^=driver_version" in detect and "--query-gpu^=compute_cap" in detect
    assert "if %DRIVER% LSS 580 goto :eof" in detect
    assert "if %CAP% LSS 75 goto :eof" in detect
    assert detect.index("LSS 580") < detect.index("compute_cap") < detect.index('set "VARIANT=cu130"')
    assert "too old" in text


@pytest.mark.parametrize("bat", [BAT, SHARE_BAT], ids=lambda p: p.name)
def test_bat_labels_and_blocks(bat):
    lines = bat.read_text(encoding="utf-8").splitlines()
    labels = {line[1:].split()[0].lower() for line in lines if line.startswith(":") and not line.startswith("::")}
    targets = set()
    for line in lines:
        if line.lstrip().lower().startswith("rem "):
            continue
        targets.update(t.lower() for t in re.findall(r"\bgoto\s+:?(\w+)", line, flags=re.IGNORECASE))
        targets.update(t.lower() for t in re.findall(r"\bcall\s+:(\w+)", line, flags=re.IGNORECASE))
    targets.discard("eof")
    assert targets <= labels, f"goto/call to labels that don't exist: {targets - labels}"
    # goto instead of ( ) blocks, so a folder name with brackets can't break the script.
    assert not [line for line in lines if line.rstrip().endswith("(") or line.lstrip().startswith(")")]


# --- install.ps1 -----------------------------------------------------------------------------


def test_install_ps1_is_plain_ascii():
    # Windows PowerShell 5.1 reads a script without a BOM as the local code page.
    assert INSTALL_PS1.read_bytes().isascii()


def test_install_ps1_safety_checks_are_there():
    # Checked as text too, because PowerShell isn't installed everywhere these tests run.
    text = INSTALL_PS1.read_text(encoding="utf-8")
    assert "--query-gpu=$Query" in text and "'compute_cap'" in text and "'driver_version'" in text
    assert "$ReSolutionMinComputeCap = 75" in text
    assert "/api/status" in text and '"is_host"' in text and ".Proxy = $null" in text
    assert "WindowsBuiltInRole]::Administrator" in text and "Run as administrator" in text
    step_1, step_3 = text.index("Write-Step 1"), text.index("Write-Step 3")
    for call in ("(Test-AppRunning $ReSolutionPort)", "(Test-RunAsAdministrator)"):
        assert step_1 < text.index(call) < step_3, f"{call} must come before anything is changed"


@pytest.mark.skipif(
    not (TEST_POWERSHELL and ON_WINDOWS), reason="only when RE_SOLUTION_TEST_POWERSHELL names a Windows PowerShell"
)
def test_the_powershell_asked_for_is_the_one_tested():
    # CI asks for each PowerShell in turn; a missing one must fail here, not skip every test quietly.
    assert PWSH, f"RE_SOLUTION_TEST_POWERSHELL is {TEST_POWERSHELL!r}, but that isn't installed"
    result = subprocess.run(
        [PWSH, "-NoProfile", "-Command", "$PSVersionTable.PSEdition; $PSVersionTable.PSVersion.Major"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    edition, major = result.stdout.split()[:2]
    if Path(TEST_POWERSHELL).stem.lower() == "powershell":  # Windows PowerShell, built into Windows
        assert (edition, major) == ("Desktop", "5"), result.stdout
    else:
        assert edition == "Core" and int(major) >= 7, result.stdout


@pytest.mark.skipif(PARSER_PWSH is None, reason="PowerShell (pwsh) isn't installed on this computer")
def test_install_ps1_parses():
    script = (
        "$errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile($env:INSTALL_PS1, [ref]$null, [ref]$errors); "
        "if ($errors) { $errors | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    result = subprocess.run(
        [PARSER_PWSH, "-NoProfile", "-Command", script],
        env={**os.environ, "INSTALL_PS1": str(INSTALL_PS1)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_install_ps1(tmp_path: Path, *args: str, port: int | None = None, **env: str):
    # -ExecutionPolicy Bypass: a Windows PC that has never run a script refuses -File otherwise.
    # (The one-line install pipes the script into iex, which that setting doesn't block.)
    return subprocess.run(
        [PWSH, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALL_PS1), *args]
        + ["-Dir", str(tmp_path / "R")],
        env={**os.environ, "RE_SOLUTION_PORT": str(port or free_port()), **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@needs_pwsh
@pytest.mark.parametrize(
    ("fake_gpu", "expected"),
    [
        ("590.12,8.6", "cu130"),
        ("580.65, 7.5", "cu130"),  # as nvidia-smi writes it: GTX 16 / RTX 20, the oldest that works
        ("581.57,6.1", "cpu"),  # a GTX 1080 with an up-to-date driver
        ("590.12", "cpu"),  # the card generation couldn't be read
        ("470.5", "cpu"),
        ("none", "cpu"),
    ],
)
def test_install_ps1_dry_run_picks_variant(tmp_path, fake_gpu, expected):
    result = _run_install_ps1(tmp_path, "-DryRun", "-FakeGpu", fake_gpu)
    assert result.returncode == 0, result.stdout + result.stderr
    assert chosen_variant(result.stdout) == expected
    assert not (tmp_path / "R").exists()


@needs_pwsh
def test_install_ps1_dry_run_explains_an_old_card(tmp_path):
    result = _run_install_ps1(tmp_path, "-DryRun", "-FakeGpu", "581.57,6.1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "too old" in result.stdout and "RTX 20" in result.stdout
    assert "nvidia.com/drivers" not in result.stdout


@needs_pwsh
def test_install_ps1_dry_run_says_re_solution_is_open(tmp_path):
    with fake_web_server(STATUS_ANSWER) as port:
        result = _run_install_ps1(tmp_path, "-DryRun", "-FakeGpu", "none", port=port)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Re:Solution is open" in result.stdout


@needs_pwsh
def test_install_ps1_explains_windows_on_arm(tmp_path):
    # PyTorch's vision package has no Windows on Arm version, so uv sync would fail later
    # with a message about the internet connection.
    result = _run_install_ps1(tmp_path, "-DryRun", "-FakeGpu", "none", PROCESSOR_ARCHITECTURE="ARM64")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Intel or AMD processor" in result.stdout
    assert chosen_variant(result.stdout) is None  # stopped in step 1


@needs_windows
def test_install_ps1_copy_keeps_the_bundled_samples(tmp_path):
    src = make_source(tmp_path)
    target = tmp_path / "Re-Solution"
    # Loads just the copy function (and Stop-Install, which it calls) and runs it.
    script = (
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:INSTALL_PS1, [ref]$null, [ref]$null); "
        "foreach ($name in 'Stop-Install', 'Copy-AppFiles') { "
        "$function = $ast.Find({ param($node) "
        "$node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true); "
        ". ([scriptblock]::Create($function.Extent.Text)) }; "
        "Copy-AppFiles $env:COPY_FROM $env:COPY_TO"
    )
    env = {**os.environ, "INSTALL_PS1": str(INSTALL_PS1), "COPY_FROM": str(src), "COPY_TO": str(target)}
    result = subprocess.run(
        [PWSH, "-NoProfile", "-Command", script], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert (target / "upscaler" / "__init__.py").exists()
    assert (target / "upscaler" / "web" / "samples" / "samples.json").exists()
    assert (target / "samples" / "README.md").exists()
    assert not (target / "samples" / "family-photo.jpg").exists()
    assert not (target / ".venv").exists()
    assert not (target / ".torch-variant").exists()
    assert not (target / "upscaler" / "__pycache__").exists()
    assert not (target / "model.safetensors.part").exists()


# --- project files ---------------------------------------------------------------------------


def test_pyproject_torch_variants():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = project["project"]["dependencies"]
    assert not [d for d in base if re.match(r"torch(vision)?\b", d)], "torch belongs in the cpu/cu130 extras"
    extras = project["project"]["optional-dependencies"]
    for name in ("cpu", "cu130"):
        assert sorted(d.split(">")[0] for d in extras[name]) == ["torch", "torchvision"]
    uv = project["tool"]["uv"]
    assert uv["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu130"}]]
    for package in ("torch", "torchvision"):
        assert {(s["extra"], s["index"]) for s in uv["sources"][package]} == {
            ("cpu", "pytorch-cpu"),
            ("cu130", "pytorch-cu130"),
        }
    indexes = {i["name"]: i for i in uv["index"]}
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cu130"]["url"] == "https://download.pytorch.org/whl/cu130"
    assert all(i.get("explicit") for i in indexes.values())

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert '{ package = "upscaler", extra = "cpu" }' in lock
    assert 'version = "2.14.1+cu130"' in lock or re.search(r'version = "[\d.]+\+cu130"', lock)
    assert 'source = { registry = "https://download.pytorch.org/whl/cpu" }' in lock
    assert 'name = "mflux"' in lock


def test_license_and_gitignore():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "Copyright (c) 2026 Swati Srivastava" in license_text
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for entry in (".venv/", "__pycache__/", "samples/*", "!samples/README.md", "*.part", ".torch-variant", ".DS_Store"):
        assert entry in ignored, entry
