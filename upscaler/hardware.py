"""Which GPU this computer has, how much memory, and whether Photo-real can run on it."""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

PHOTOREAL_MIN_MAC_GB = 16
PHOTOREAL_MIN_CUDA_GB = 12

# The oldest NVIDIA cards the GPU build runs on: compute capability 7.5 (Turing: RTX 20 and GTX 16
# series), or the oldest the installed PyTorch has code for, whichever is newer.
MIN_CUDA_CAPABILITY = (7, 5)
OLD_NVIDIA_NOTE = (  # the installer's wording, which the README quotes
    "Your NVIDIA graphics card is too old for the graphics software Re:Solution uses, "
    "so it uses the processor instead (slow)"
)
NVIDIA_UNUSABLE_NOTE = "Re:Solution couldn't use your NVIDIA card; using the processor (slow)"
_ARCH = re.compile(r"sm_(\d+)(\d)")  # "sm_75" is 7.5, "sm_120" is 12.0

# Memory is reported in GiB, so a card sold as "12 GB" can show up as 11.99.
_MEMORY_SLACK_GB = 0.25
_FALLBACK_RAM_BYTES = 8 * 2**30  # a Mac that won't say (Photo-real's check needs a number)


@dataclass(frozen=True)
class Hardware:
    device: str  # "mps" | "cuda" | "cpu"
    device_name: str  # e.g. "Apple M4"
    memory_gb: float | None  # unified RAM (mac), VRAM (cuda), RAM (cpu); None when the system won't say
    os: str  # "mac" | "windows" | "linux"
    apple_silicon: bool
    note: str = ""  # for people, when the app isn't using what they might expect (an NVIDIA card too old)


def _os_name() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _system_ram_bytes() -> int | None:
    if sys.platform == "win32":  # Windows has no os.sysconf
        return _windows_ram_bytes()
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


class _MemoryStatus(ctypes.Structure):
    """Windows' MEMORYSTATUSEX."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_ram_bytes() -> int | None:
    """The PC's memory from GlobalMemoryStatusEx; None anywhere else, or if Windows won't say."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (AttributeError, OSError):  # not Windows: there is no ctypes.windll
        return None
    return status.ullTotalPhys or None


def _gb(num_bytes: int) -> float:
    return round(num_bytes / 2**30, 1)


def _mac_hardware() -> Hardware:
    memsize = _sysctl("hw.memsize")
    ram = int(memsize) if memsize and memsize.isdigit() else (_system_ram_bytes() or _FALLBACK_RAM_BYTES)
    return Hardware(
        device="mps" if torch.backends.mps.is_available() else "cpu",
        device_name=_sysctl("machdep.cpu.brand_string") or "Mac",
        memory_gb=_gb(ram),
        os="mac",
        apple_silicon=platform.machine() == "arm64",
    )


@functools.cache
def detect_hardware() -> Hardware:
    os_name = _os_name()
    note = ""
    if torch.cuda.is_available():
        note = _nvidia_problem()
        if not note:
            props = torch.cuda.get_device_properties(0)
            return Hardware("cuda", props.name, _gb(props.total_memory), os_name, False)
    if os_name == "mac":
        return _mac_hardware()
    ram = _system_ram_bytes()
    # Unknown memory stays unknown, so the app leaves it out instead of showing a made-up number.
    return Hardware("cpu", platform.processor() or "CPU", _gb(ram) if ram else None, os_name, False, note)


def _nvidia_problem() -> str:
    """Why the NVIDIA card can't be used (a note for people), or "" when it can. PyTorch still says
    CUDA is available on a card older than its GPU build supports, then fails on the first picture."""
    try:
        capability = tuple(torch.cuda.get_device_capability(0))
        lowest = _lowest_arch(torch.cuda.get_arch_list())
    except Exception:  # a driver or card problem: the processor still works
        log.warning("Couldn't ask the NVIDIA card what it supports; using the processor", exc_info=True)
        return NVIDIA_UNUSABLE_NOTE
    if capability < max(lowest or MIN_CUDA_CAPABILITY, MIN_CUDA_CAPABILITY):
        log.warning("NVIDIA card with compute capability %s is too old for this PyTorch build", capability)
        return OLD_NVIDIA_NOTE
    return ""


def _lowest_arch(arch_list: list[str]) -> tuple[int, int] | None:
    """The oldest card PyTorch has code for, from names like "sm_75" (None if it names none)."""
    found = [(int(match[1]), int(match[2])) for name in arch_list if (match := _ARCH.fullmatch(name))]
    return min(found) if found else None


def photoreal_support(hw: Hardware) -> tuple[bool, str]:
    """Whether Photo-real (SeedVR2) can run here.

    The reason is a short phrase with no final full stop, so callers can extend it,
    e.g. "Photo-real needs 16 GB of memory; used Faithful instead."
    """
    if hw.device == "mps" and hw.apple_silicon:
        if hw.memory_gb + _MEMORY_SLACK_GB >= PHOTOREAL_MIN_MAC_GB:
            return True, ""
        return False, f"Photo-real needs {PHOTOREAL_MIN_MAC_GB} GB of memory"
    if hw.device == "cuda":
        if hw.memory_gb + _MEMORY_SLACK_GB >= PHOTOREAL_MIN_CUDA_GB:
            return True, ""
        return False, f"Photo-real needs a graphics card with {PHOTOREAL_MIN_CUDA_GB} GB of memory"
    return False, "Photo-real needs an Apple Silicon Mac or an NVIDIA graphics card"
