import sys

import pytest
import torch

from upscaler import hardware
from upscaler.hardware import Hardware, detect_hardware, photoreal_support


def test_photoreal_support():
    assert photoreal_support(Hardware("mps", "Apple M4", 16, "mac", True)) == (True, "")

    ok, reason = photoreal_support(Hardware("mps", "Apple M1", 8, "mac", True))
    assert ok is False
    assert "16 GB" in reason

    assert photoreal_support(Hardware("cuda", "RTX 3060", 12, "windows", False)) == (True, "")

    ok, reason = photoreal_support(Hardware("cpu", "Intel", 32, "windows", False))
    assert ok is False
    assert reason


def test_photoreal_support_allows_cards_sold_as_12gb():
    # PyTorch reports an RTX 3060 "12 GB" as 11.99 GiB.
    assert photoreal_support(Hardware("cuda", "RTX 3060", 11.99, "windows", False))[0] is True
    ok, reason = photoreal_support(Hardware("cuda", "RTX 2080 Ti", 10.7, "windows", False))
    assert ok is False
    assert "12 GB" in reason


def test_photoreal_needs_apple_silicon_on_a_mac():
    ok, reason = photoreal_support(Hardware("mps", "Intel Core i9", 32, "mac", False))
    assert ok is False
    assert reason


@pytest.fixture
def fresh_detection():
    detect_hardware.cache_clear()
    yield
    detect_hardware.cache_clear()


def test_detect_hardware_on_apple_silicon(monkeypatch, fresh_detection):
    sysctl = {"hw.memsize": "17179869184", "machdep.cpu.brand_string": "Apple M4"}
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_sysctl", sysctl.get)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert detect_hardware() == Hardware("mps", "Apple M4", 16.0, "mac", True)


def test_detect_hardware_reads_memory_on_windows_cpu(monkeypatch, fresh_detection):
    # Windows has no os.sysconf, so its memory comes from GlobalMemoryStatusEx instead.
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "Intel64 Family 6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_windows_ram_bytes", lambda: 32 * 2**30)

    hw = detect_hardware()
    assert (hw.device, hw.memory_gb, hw.os, hw.apple_silicon) == ("cpu", 32.0, "windows", False)


def test_unknown_memory_is_left_out_not_made_up(monkeypatch, fresh_detection):
    # The app then says "an Intel processor" without inventing "with 8 GB of memory".
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_windows_ram_bytes", lambda: None)

    hw = detect_hardware()
    assert hw.device == "cpu" and hw.memory_gb is None
    assert photoreal_support(hw)[0] is False


@pytest.mark.skipif(sys.platform == "win32", reason="not Windows")
def test_windows_memory_reader_is_quiet_elsewhere():
    assert hardware._windows_ram_bytes() is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_memory_reader_on_this_pc():
    assert hardware._windows_ram_bytes() >= 2**30


def test_detect_hardware_is_cached(fresh_detection):
    assert detect_hardware() is detect_hardware()


@pytest.mark.skipif(sys.platform != "darwin", reason="Mac only")
def test_detect_hardware_on_this_mac(fresh_detection):
    # What this Mac really has: GitHub's Mac runners are Apple Silicon virtual machines with 7 GB of
    # memory and no Apple GPU for PyTorch (MPS), so they use the processor.
    import torch

    hw = detect_hardware()
    assert hw.os == "mac"
    assert hw.memory_gb and hw.memory_gb > 0
    assert hw.device_name
    if hw.apple_silicon:
        assert hw.device == ("mps" if torch.backends.mps.is_available() else "cpu")


class _FakeCardProperties:
    name = "NVIDIA GeForce RTX 4070"
    total_memory = 12 * 2**30


def _nvidia(monkeypatch, capability, arch_list):
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "Intel64 Family 6")
    monkeypatch.setattr(hardware, "_windows_ram_bytes", lambda: 32 * 2**30)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _FakeCardProperties())
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))
    if isinstance(capability, Exception):

        def broken(index):
            raise capability

        monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    else:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: capability)


CU130_ARCHES = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def test_a_supported_nvidia_card_is_used(monkeypatch, fresh_detection):
    _nvidia(monkeypatch, (8, 9), CU130_ARCHES)
    hw = detect_hardware()
    assert (hw.device, hw.device_name, hw.memory_gb, hw.note) == ("cuda", "NVIDIA GeForce RTX 4070", 12.0, "")


@pytest.mark.parametrize(
    ("capability", "arch_list"),
    [
        ((6, 1), CU130_ARCHES),  # a GTX 1060: older than anything this PyTorch build has code for
        ((7, 0), ["sm_70", "sm_75", "sm_80"]),  # a build that still lists Volta, but below 7.5 is too old
        ((7, 0), []),  # a build that won't say: 7.5 is the floor
    ],
)
def test_an_nvidia_card_too_old_for_the_gpu_build_uses_the_processor(
    monkeypatch, fresh_detection, capability, arch_list
):
    _nvidia(monkeypatch, capability, arch_list)
    hw = detect_hardware()
    assert hw.device == "cpu"
    assert hw.note == (
        "Your NVIDIA graphics card is too old for the graphics software Re:Solution uses, "
        "so it uses the processor instead (slow)"
    )
    assert (hw.device_name, hw.memory_gb, hw.os) == ("Intel64 Family 6", 32.0, "windows")  # the processor's
    assert photoreal_support(hw)[0] is False


def test_the_newest_cards_are_never_called_too_old(monkeypatch, fresh_detection):
    _nvidia(monkeypatch, (12, 0), CU130_ARCHES)
    assert detect_hardware().device == "cuda"
    hardware.detect_hardware.cache_clear()
    _nvidia(monkeypatch, (7, 5), CU130_ARCHES)  # Turing (RTX 20 series, GTX 16 series): the oldest that works
    assert detect_hardware().device == "cuda"


def test_an_nvidia_card_that_cannot_be_asked_uses_the_processor(monkeypatch, fresh_detection):
    _nvidia(monkeypatch, RuntimeError("CUDA error: no kernel image is available"), CU130_ARCHES)
    hw = detect_hardware()
    assert hw.device == "cpu" and "NVIDIA" in hw.note and "processor" in hw.note


def test_cards_are_compared_as_numbers_not_text():
    assert hardware._lowest_arch(["sm_100", "sm_75", "sm_90", "compute_120"]) == (7, 5)
    assert hardware._lowest_arch(["sm_120"]) == (12, 0)
    assert hardware._lowest_arch(["compute_90", "nonsense"]) is None
    assert hardware._lowest_arch([]) is None


def test_hardware_has_no_note_by_default():
    assert Hardware("mps", "Apple M4", 16, "mac", True).note == ""
