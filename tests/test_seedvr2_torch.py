import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import respx
from PIL import Image

from upscaler.config import Cancelled
from upscaler.seedvr2 import SeedVR2Unavailable, torch_engine
from upscaler.seedvr2.base import soften
from upscaler.seedvr2.torch_engine import TorchSeedVR2Engine, find_uv, runner_command


def _no_progress(fraction: float, message: str) -> None:
    pass


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def _picture(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], axis=-1).astype(np.uint8)


FAKE_CLI = """
import argparse, sys, time
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output")
parser.add_argument("--resolution", type=int)
args, _ = parser.parse_known_args()
behaviour = {behaviour!r}
if behaviour == "fail":
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    sys.exit(1)
if behaviour == "hang":
    time.sleep(60)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
for path in sorted(Path(args.input).iterdir()):
    image = Image.open(path)
    scale = args.resolution / min(image.size)
    image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS).save(out / path.name)
"""


class FakeRunnerEngine(TorchSeedVR2Engine):
    """The real engine, pointed at a stand-in CLI that runs under the test's own Python."""

    def __init__(self, root: Path, behaviour: str = "ok", vram_gb: float = 24):
        super().__init__(root, tile=64, overlap=16, vram_gb=vram_gb)
        root.mkdir(parents=True, exist_ok=True)
        self._cli = root / "fake_cli.py"
        self._cli.write_text(FAKE_CLI.format(behaviour=behaviour))
        self.commands: list[list[str]] = []

    def _run_runner(self, cmd, **kwargs):
        self.commands.append(cmd)
        return super()._run_runner(cmd, **kwargs)

    @property
    def venv_python(self) -> Path:
        return Path(sys.executable)

    @property
    def cli_path(self) -> Path:
        return self._cli

    def runner_installed(self) -> bool:
        return True


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    models = tmp_path / "hf-snapshot"
    models.mkdir()
    monkeypatch.setattr(torch_engine, "cached_weights_dir", lambda: models)
    return models


def test_runner_command_uses_the_cli_argument_names():
    cmd = runner_command(Path("py"), Path("cli.py"), Path("in"), Path("out"), Path("models"), 2048)

    assert cmd[:3] == ["py", "cli.py", "in"]
    assert "--blocks_to_swap" not in cmd
    assert _flag(cmd, "--output") == "out"
    assert _flag(cmd, "--output_format") == "png"
    assert _flag(cmd, "--model_dir") == "models"
    assert _flag(cmd, "--dit_model") == "seedvr2_ema_3b_fp16.safetensors"
    assert _flag(cmd, "--resolution") == "2048"
    assert _flag(cmd, "--seed") == "42"
    assert _flag(cmd, "--dit_offload_device") == "cpu"
    assert _flag(cmd, "--vae_offload_device") == "cpu"
    assert {"--cache_dit", "--cache_vae", "--vae_encode_tiled", "--vae_decode_tiled"} <= set(cmd)


def test_runner_command_can_start_through_the_launcher_and_swap_blocks():
    cmd = runner_command(
        Path("py"), Path("cli.py"), Path("in"), Path("out"), Path("models"), 2048,
        launcher=Path("launch.py"), blocks_to_swap=16,
    )  # fmt: skip

    assert cmd[:4] == ["py", "launch.py", "cli.py", "in"]
    assert _flag(cmd, "--blocks_to_swap") == "16"
    assert _flag(cmd, "--dit_offload_device") == "cpu"  # BlockSwap needs it


@pytest.mark.parametrize("vram_gb, swaps", [(11.99, True), (16, True), (24, False), (48, False)])
def test_cards_with_less_than_20gb_swap_model_blocks_to_system_memory(tmp_path, weights_dir, vram_gb, swaps):
    engine = FakeRunnerEngine(tmp_path, vram_gb=vram_gb)

    engine.upscale(_picture(80, 80), 2, progress=_no_progress, cancel=threading.Event())

    (cmd,) = engine.commands
    assert ("--blocks_to_swap" in cmd) is swaps
    if swaps:
        assert int(_flag(cmd, "--blocks_to_swap")) == torch_engine.BLOCKS_TO_SWAP


def test_find_uv_falls_back_to_local_bin(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_engine.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert find_uv() is None

    uv = tmp_path / ".local" / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")
    uv.parent.mkdir(parents=True)
    uv.write_text("")
    assert find_uv() == uv


def test_upscale_runs_the_cli_once_and_blends(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path)
    rgb = _picture(150, 97)

    out = engine.upscale(rgb, 4, progress=_no_progress, cancel=threading.Event())

    assert out.shape == (97 * 4, 150 * 4, 3) and out.dtype == np.uint8
    lanczos = np.asarray(Image.fromarray(rgb).resize((600, 388), Image.LANCZOS))
    assert np.abs(out.astype(int) - lanczos.astype(int)).mean() < 1


def _detail(rgb: np.ndarray) -> float:
    """Mean size of the jumps between neighbouring pixels: higher means more fine detail."""
    return float(np.abs(np.diff(rgb.astype(int), axis=1)).mean())


def test_soften_matches_seedvr2_softness():
    rgb = _picture(96, 64)
    rgb[::2] = 255 - rgb[::2]  # plenty of fine detail to lose
    assert soften(rgb, 2, 0.0) is rgb
    assert soften(rgb, 4, 0.25) is rgb  # mflux shrinks 4x / 2.75: still bigger than the original
    softened = soften(rgb, 2, 0.5)  # 2x / 4.5: below the original size, so detail goes
    assert softened.shape == rgb.shape and softened.dtype == np.uint8
    assert _detail(softened) < _detail(rgb) / 2


def test_upscale_works_in_a_folder_with_non_english_letters(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path / "José नमस्ते 写真")
    rgb = _picture(150, 97)

    out = engine.upscale(rgb, 2, progress=_no_progress, cancel=threading.Event())

    assert out.shape == (97 * 2, 150 * 2, 3)
    (cmd,) = engine.commands
    assert cmd[1] == str(engine.runner_dir / torch_engine.LAUNCHER_NAME)  # OpenCV gets Python's file access
    assert cmd[2] == str(engine.cli_path)


def test_upscale_clears_work_left_by_a_closed_window(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path)
    work = tmp_path / "work"
    left_behind, in_use = work / "tmpold", work / "tmpnew"
    for folder in (left_behind, in_use):
        (folder / "in").mkdir(parents=True)
        (folder / "in" / "tile_0000.png").write_bytes(b"png")
    hours_ago = time.time() - 3 * 3600
    for path in (left_behind / "in" / "tile_0000.png", left_behind / "in", left_behind):
        os.utime(path, (hours_ago, hours_ago))

    engine.upscale(_picture(80, 80), 2, progress=_no_progress, cancel=threading.Event())

    assert not left_behind.exists()
    assert in_use.exists()  # recent: it may belong to a Photo-real run in another window
    assert sorted(p.name for p in work.iterdir()) == ["tmpnew"]


def test_upscale_with_softness_gives_the_runner_a_softened_picture(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path)
    rgb = _picture(150, 97)
    rgb[::2] = 255 - rgb[::2]

    plain = engine.upscale(rgb, 2, progress=_no_progress, cancel=threading.Event())
    soft = engine.upscale(rgb, 2, progress=_no_progress, cancel=threading.Event(), softness=0.5)

    assert soft.shape == plain.shape == (97 * 2, 150 * 2, 3)
    assert _detail(soft) < _detail(plain) / 2


def test_runner_reads_links_to_the_cached_weights_not_the_cache_itself(tmp_path, weights_dir):
    for file in torch_engine.SEEDVR2_FILES:
        (weights_dir / file.name).write_bytes(b"weights")
    engine = FakeRunnerEngine(tmp_path)

    engine.upscale(_picture(80, 80), 4, progress=_no_progress, cancel=threading.Event())

    for file in torch_engine.SEEDVR2_FILES:
        link = tmp_path / "models" / file.name
        assert link.resolve() == (weights_dir / file.name).resolve()


def test_cli_failure_is_unavailable_with_a_plain_reason(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path, behaviour="fail")

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        engine.upscale(_picture(80, 80), 4, progress=_no_progress, cancel=threading.Event())
    assert "memory" in excinfo.value.reason.lower()


def test_cancel_stops_the_cli(tmp_path, weights_dir):
    engine = FakeRunnerEngine(tmp_path, behaviour="hang")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()

    with pytest.raises(Cancelled):
        engine.upscale(_picture(80, 80), 4, progress=_no_progress, cancel=cancel)


def test_upscale_before_setup_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_engine, "cached_weights_dir", lambda: None)
    engine = TorchSeedVR2Engine(tmp_path)

    assert not engine.weights_present()
    with pytest.raises(SeedVR2Unavailable):
        engine.upscale(_picture(80, 80), 4, progress=_no_progress, cancel=threading.Event())


# --- installing the runner ---------------------------------------------------------------

FAKE_UV = """
import json, sys
from pathlib import Path

args = sys.argv[1:]
with open({log!r}, "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "venv":
    python = Path(args[-1]) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("")
"""


RUNNER_FILES = {
    "inference_cli.py": "print('hello')\n",
    "requirements.txt": "einops\n",
    "src/utils/constants.py": "__version__ = '2.5.24'\n",
}


def _runner_zip(files: dict[str, str] = RUNNER_FILES, *, compression=zipfile.ZIP_STORED, date=(2026, 1, 1, 0, 0, 0)):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        top = "ComfyUI-SeedVR2_VideoUpscaler-abc123/"
        zf.writestr(zipfile.ZipInfo(top, date), "")
        for name, text in files.items():
            zf.writestr(zipfile.ZipInfo(top + name, date), text, compress_type=compression)
    return buffer.getvalue()


def _files_digest(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return torch_engine.runner_files_digest(zf)


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    log = tmp_path / "uv-calls.jsonl"
    script = tmp_path / "fake_uv.py"
    script.write_text(FAKE_UV.format(log=str(log)))
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    # A shell wrapper, because a shebang line can't hold the spaces in this project's path.
    uv.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(torch_engine, "find_uv", lambda: uv)
    return log


@pytest.fixture
def runner_download(monkeypatch):
    payload = _runner_zip()
    url = "https://example.test/runner.zip"
    monkeypatch.setattr(torch_engine, "RUNNER_URL", url)
    monkeypatch.setattr(torch_engine, "RUNNER_FILES_SHA256", _files_digest(payload))
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get(url).mock(return_value=httpx.Response(200, content=payload))
        yield route


@pytest.mark.skipif(sys.platform == "win32", reason="the fake uv is a POSIX script")
def test_install_runner_uses_an_isolated_venv(tmp_path, fake_uv, runner_download):
    root = tmp_path / "seedvr2-torch"
    engine = TorchSeedVR2Engine(root)

    engine.install_runner(_no_progress, threading.Event())

    calls = [json.loads(line) for line in fake_uv.read_text().splitlines()]
    venv = str(root / ".venv")
    assert calls[0][0] == "venv" and calls[0][-1] == venv
    pip_calls = [c for c in calls if c[:2] == ["pip", "install"]]
    assert len(pip_calls) == 2
    for call in pip_calls:
        assert _flag(call, "--python") == str(engine.venv_python)  # never the app's own environment
    assert "torch" in " ".join(pip_calls[0]) and _flag(pip_calls[0], "--index-url") == torch_engine.CUDA_TORCH_INDEX
    assert _flag(pip_calls[1], "-r") == str(engine.runner_dir / "requirements.txt")
    # The runner's own packages come as they were on the day it was pinned, never newer releases.
    # torch is pinned exactly instead: PyTorch's index has no upload dates, so a cutoff would hide it.
    assert _flag(pip_calls[1], "--exclude-newer") == torch_engine.RUNNER_PACKAGES_CUTOFF
    assert "--exclude-newer" not in pip_calls[0]
    assert engine.cli_path.read_text() == "print('hello')\n"
    assert engine.runner_installed()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake uv is a POSIX script")
def test_install_runner_is_skipped_when_already_installed(tmp_path, fake_uv, runner_download):
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")
    engine.install_runner(_no_progress, threading.Event())
    calls_before = fake_uv.read_text()

    engine.install_runner(_no_progress, threading.Event())

    assert fake_uv.read_text() == calls_before
    assert runner_download.call_count == 1


def test_install_runner_with_a_damaged_download_is_unavailable(tmp_path, fake_uv, runner_download, monkeypatch):
    monkeypatch.setattr(torch_engine, "RUNNER_FILES_SHA256", "0" * 64)
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")

    with pytest.raises(SeedVR2Unavailable):
        engine.install_runner(_no_progress, threading.Event())
    assert not engine.runner_dir.exists()
    assert not list((tmp_path / "seedvr2-torch").glob("*.part"))
    assert not list((tmp_path / "seedvr2-torch").glob("runner.*"))
    assert not fake_uv.exists()  # nothing was installed


@pytest.mark.skipif(sys.platform == "win32", reason="the fake uv is a POSIX script")
def test_install_runner_accepts_the_same_files_zipped_differently(tmp_path, fake_uv, runner_download):
    # GitHub builds these archives on request and may change how it compresses them.
    rezipped = _runner_zip(compression=zipfile.ZIP_DEFLATED, date=(2027, 5, 6, 7, 8, 10))
    runner_download.mock(return_value=httpx.Response(200, content=rezipped))
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")

    engine.install_runner(_no_progress, threading.Event())

    assert engine.runner_installed()


@pytest.mark.parametrize(
    "payload",
    [
        _runner_zip({**RUNNER_FILES, "inference_cli.py": "print('changed')\n"}),
        _runner_zip({**RUNNER_FILES, "extra.py": "import os\n"}),
        _runner_zip({k: v for k, v in RUNNER_FILES.items() if k != "requirements.txt"}),
        b"not a zip file at all",
    ],
    ids=["changed file", "added file", "missing file", "not a zip"],
)
def test_install_runner_refuses_different_files(tmp_path, fake_uv, runner_download, payload):
    runner_download.mock(return_value=httpx.Response(200, content=payload))
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        engine.install_runner(_no_progress, threading.Event())
    assert "damaged" in excinfo.value.reason
    assert not engine.runner_dir.exists()
    assert not fake_uv.exists()


def test_runner_files_digest_names_every_file_and_its_contents():
    digest = _files_digest(_runner_zip())

    assert digest == _files_digest(_runner_zip(compression=zipfile.ZIP_DEFLATED))
    moved = {name: text for name, text in RUNNER_FILES.items() if name != "src/utils/constants.py"}
    moved["src/utils/other.py"] = RUNNER_FILES["src/utils/constants.py"]
    assert digest != _files_digest(_runner_zip(moved))  # the same contents under another name
    assert len(digest) == 64


@pytest.mark.network
def test_pinned_runner_archive_matches_its_files_digest():
    response = httpx.get(torch_engine.RUNNER_URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    assert _files_digest(response.content) == torch_engine.RUNNER_FILES_SHA256


def test_download_weights_needs_room_for_the_add_on_and_the_weights(tmp_path, fake_uv, runner_download, monkeypatch):
    import huggingface_hub.constants  # loaded lazily otherwise: the attribute only exists once imported

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(torch_engine, "missing_download_bytes", lambda: 7_300_000_000)
    monkeypatch.setattr(torch_engine.shutil, "disk_usage", lambda path: SimpleNamespace(free=10_000_000_000))
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")

    with pytest.raises(SeedVR2Unavailable) as excinfo:  # 10 GB free: room for either, not both
        engine.download_weights(_no_progress, threading.Event())
    assert "disk space" in excinfo.value.reason and "14 GB" in excinfo.value.reason
    assert runner_download.call_count == 0 and not fake_uv.exists()  # nothing was started


# --- the launcher that gives OpenCV Python's file access -----------------------------------

LAUNCHER_PROBE_CLI = """
import sys
import cv2
import numpy as np
from pathlib import Path

folder = Path(sys.argv[1])
assert sys.argv[0].endswith("probe_cli.py"), sys.argv
assert __name__ == "__main__"
assert getattr(cv2.imread, "__code__", None) is not None, "OpenCV's own imread was left in place"
picture = np.zeros((6, 8, 3), np.uint8)
picture[..., 2] = 200
assert cv2.imwrite(str(folder / "tile_0001.png"), picture)
back = cv2.imread(str(folder / "tile_0001.png"), cv2.IMREAD_UNCHANGED)
assert back is not None and back.shape == (6, 8, 3) and int(back[0, 0, 2]) == 200
assert cv2.imread(str(folder / "missing.png"), cv2.IMREAD_UNCHANGED) is None
print("probe ok", sys.argv[2:])
"""


def test_launcher_runs_the_cli_with_opencv_using_python_file_access(tmp_path):
    pytest.importorskip("cv2")
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")
    launcher = engine._write_launcher()
    folder = tmp_path / "José नमस्ते"
    folder.mkdir()
    cli = tmp_path / "probe_cli.py"
    cli.write_text(LAUNCHER_PROBE_CLI)

    result = subprocess.run(
        [sys.executable, str(launcher), str(cli), str(folder), "--flag", "x"],
        capture_output=True, text=True, encoding="utf-8", timeout=120, env=torch_engine._clean_env(),
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert "probe ok ['--flag', 'x']" in result.stdout
    assert (folder / "tile_0001.png").is_file()


def test_install_runner_without_uv_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_engine, "find_uv", lambda: None)
    engine = TorchSeedVR2Engine(tmp_path / "seedvr2-torch")

    with pytest.raises(SeedVR2Unavailable) as excinfo:
        engine.install_runner(_no_progress, threading.Event())
    assert "uv" in excinfo.value.reason
    assert "Start Re-Solution launcher" in excinfo.value.reason
