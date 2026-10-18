"""App-wide paths, saved settings, and the small types every module shares."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import platformdirs

log = logging.getLogger(__name__)

APP_NAME = "Upscaler"  # the data folder and keychain entries; kept, so updating loses nothing
PRODUCT_NAME = "Re:Solution"  # what people see ("Re-Solution" in file names)
DEFAULT_PORT = 7860

# Moves all app data somewhere else; used by tests and handy for portable installs.
DATA_DIR_ENV = "UPSCALER_DATA_DIR"

# Free space the app always leaves on a disk it writes to: uploads stop short of it, and jobs won't
# start a picture or a video segment that would eat into it, so the computer's other apps (and the
# job records) still have room to save their work.
SPACE_RESERVE_BYTES = 2 * 1024**3
SPACE_RESERVE_TEXT = "2 GB"  # SPACE_RESERVE_BYTES, as messages say it

ProgressFn = Callable[[float, str], None]
"""Progress callback: fraction done (0..1) and a short message for people."""


class Cancelled(Exception):
    """The user cancelled the job."""


class UserError(ValueError):
    """A problem explained in plain words for the person using the app, who sees the message as it
    is. A ValueError, so code that checks input with `except ValueError` still catches it."""


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return _ensure_dir(Path(override))
    return _ensure_dir(Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)))


def models_dir() -> Path:
    return _ensure_dir(data_dir() / "models")


def jobs_dir() -> Path:
    return _ensure_dir(data_dir() / "jobs")


def host_outputs_dir() -> Path:
    """Where the host's own results go: Upscaled in the Pictures folder people see, which on Windows
    may have been moved (to OneDrive\\Pictures, say); ~/Pictures when the system can't say."""
    return _ensure_dir(_pictures_dir() / "Upscaled")


def _pictures_dir() -> Path:
    try:
        pictures = Path(platformdirs.user_pictures_dir())
    except Exception:  # a system that can't say where Pictures is shouldn't stop the app
        log.warning("Couldn't look up the Pictures folder", exc_info=True)
        pictures = None
    if pictures is None or not pictures.is_absolute():
        return Path.home() / "Pictures"
    return pictures


def guest_outputs_dir() -> Path:
    return _ensure_dir(data_dir() / "guest")


# "webp" holds pictures up to 16,383 px on a side; "original" saves in the picture's own format when
# it can be written (imageio_utils.resolve_format), otherwise PNG.
OUTPUT_FORMATS = ("jpeg", "png", "tiff", "webp", "original")
GRAIN_CHOICES = ("off", "subtle", "medium")


@dataclass
class Settings:
    access_password_hash: str | None = None  # hashlib.scrypt hex; None = no password
    allow_guests_cloud: bool = False
    default_image_mode: str = "photoreal"
    output_format: str = "jpeg"
    grain: str = "off"

    def __post_init__(self) -> None:
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, not {self.output_format!r}")
        if self.grain not in GRAIN_CHOICES:
            raise ValueError(f"grain must be one of {GRAIN_CHOICES}, not {self.grain!r}")


def _is_valid(name: str, value: object) -> bool:
    match name:
        case "access_password_hash":
            return value is None or isinstance(value, str)
        case "allow_guests_cloud":
            return isinstance(value, bool)
        case "default_image_mode":
            return isinstance(value, str) and bool(value)
        case "output_format":
            return value in OUTPUT_FORMATS
        case "grain":
            return value in GRAIN_CHOICES
    return False


def _settings_path() -> Path:
    return data_dir() / "settings.json"


LOCKED_PASSWORD_HASH = "locked"
"""Stands in for the password when settings.json is damaged and the real one is unknown: no password
matches it, so guests stay out until the host sets a new password (or removes it) in Settings."""

# One reader or writer of settings.json at a time: every request reads it, and two changes at once
# would otherwise undo each other (or, on Windows, collide while the file is being replaced).
_settings_lock = threading.RLock()
_last_good: dict[Path, Settings] = {}  # settings.json -> what it held when last read or saved


def load_settings() -> Settings:
    """Read the saved settings. A missing or partly invalid file never stops the app: anything
    unusable falls back to its default. A damaged file never quietly drops the access password:
    the last good settings are kept (or guests are locked out if there are none), and the damaged
    file is set aside as settings.json.damaged."""
    path = _settings_path()
    with _settings_lock:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _remember(path, Settings())
        except OSError:  # there, but unreadable for now (another program has it open, say)
            log.warning("Couldn't read %s; using the last good settings", path, exc_info=True)
            return replace(_last_good.get(path) or Settings(access_password_hash=LOCKED_PASSWORD_HASH))
        try:
            stored = json.loads(text)
        except ValueError:
            stored = None
        password_ok = isinstance(stored, dict) and _is_valid("access_password_hash", stored.get("access_password_hash"))
        if not password_ok:
            return _recover(path)
        usable = {name: stored[name] for name in (f.name for f in fields(Settings)) if name in stored}
        return _remember(path, Settings(**{name: value for name, value in usable.items() if _is_valid(name, value)}))


def _remember(path: Path, settings: Settings) -> Settings:
    _last_good[path] = replace(settings)
    return settings


def _recover(path: Path) -> Settings:
    """settings.json is damaged: keep the last good settings, or lock guests out when there are none,
    and save them in its place so the app carries on."""
    recovered = replace(_last_good.get(path) or Settings(access_password_hash=LOCKED_PASSWORD_HASH))
    log.error(
        "%s was damaged, so it was set aside as %s.damaged. %s",
        path,
        path.name,
        "The last good settings were kept."
        if path in _last_good
        else "Guests now need a new password: set one (or remove it) in Settings.",
    )
    try:
        os.replace(path, path.with_name(path.name + ".damaged"))
        save_settings(recovered)
    except OSError:
        log.exception("Couldn't replace the damaged %s", path)
        return recovered
    return replace(recovered)


def save_settings(s: Settings) -> None:
    path = _settings_path()
    with _settings_lock:
        # A file of its own each time, written in full, then swapped in: nobody ever sees half a file.
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(asdict(s), indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        _remember(path, s)


@contextmanager
def changing_settings() -> Iterator[Settings]:
    """Load, change and save the settings as one step, so two changes at once never undo each other.
    They are saved when the block ends, and not at all if it raises."""
    with _settings_lock:
        settings = load_settings()
        yield settings
        save_settings(settings)
