"""Keep the computer from going to sleep while a job runs. The screen may still turn off.

Mac: `caffeinate -i -w <our pid>`, so the assertion also ends if the app itself quits or crashes.
Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), cleared with ES_CONTINUOUS.
The Windows state belongs to the calling thread, so enter and leave on the same thread.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Self

log = logging.getLogger(__name__)

# https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windll() -> Any:
    import ctypes

    return ctypes.windll  # type: ignore[attr-defined]  # only exists on Windows


class KeepAwake:
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._windows = False

    def __enter__(self) -> Self:
        try:
            self._start()
        except Exception:  # staying awake is a nicety; it must never stop a job
            log.warning("Couldn't keep the computer awake", exc_info=True)
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._stop()
        except Exception:
            log.warning("Couldn't release the keep-awake request", exc_info=True)

    def _start(self) -> None:
        if sys.platform == "darwin":
            self._process = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "win32":
            if not _windll().kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                raise OSError("SetThreadExecutionState failed")
            self._windows = True

    def _stop(self) -> None:
        if self._process is not None:
            process, self._process = self._process, None
            process.terminate()
            process.wait(timeout=5)
        if self._windows:
            self._windows = False
            _windll().kernel32.SetThreadExecutionState(ES_CONTINUOUS)
