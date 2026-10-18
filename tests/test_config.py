import json
import threading
import time

import pytest

from upscaler import config
from upscaler.config import Cancelled, Settings, load_settings, save_settings


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


def test_constants():
    assert config.APP_NAME == "Upscaler"  # data folder and keychain name: kept, so nothing is lost
    assert config.PRODUCT_NAME == "Re:Solution"  # what people see
    assert config.DEFAULT_PORT == 7860
    assert config.OUTPUT_FORMATS == ("jpeg", "png", "tiff", "webp", "original")


@pytest.mark.parametrize("fmt", ["tiff", "webp", "original"])
def test_every_format_can_be_the_saved_format(data_dir, fmt):
    save_settings(Settings(output_format=fmt))
    assert load_settings().output_format == fmt


def test_settings_roundtrip(data_dir):
    settings = Settings(
        access_password_hash="ab12cd",
        allow_guests_cloud=True,
        default_image_mode="faithful",
        output_format="png",
        grain="subtle",
    )
    save_settings(settings)
    assert (data_dir / "settings.json").is_file()
    assert load_settings() == settings


def test_settings_default_to_photoreal_when_missing(data_dir):
    settings = load_settings()
    assert settings == Settings()
    assert settings.default_image_mode == "photoreal"
    assert settings.output_format == "jpeg"
    assert settings.grain == "off"
    assert settings.access_password_hash is None
    assert settings.allow_guests_cloud is False


def test_damaged_settings_file_keeps_the_last_good_settings(data_dir):
    good = Settings(access_password_hash="ab12cd", allow_guests_cloud=True, output_format="png")
    save_settings(good)
    assert load_settings() == good
    (data_dir / "settings.json").write_text('{"grain": "off"\n}f"\n}', encoding="utf-8")  # a torn write
    assert load_settings() == good  # the password is never quietly dropped
    assert (data_dir / "settings.json.damaged").read_text(encoding="utf-8").startswith('{"grain"')
    assert json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))["access_password_hash"] == "ab12cd"


@pytest.mark.parametrize("damage", ["{not json", "[]", '{"access_password_hash": 12}'])
def test_damaged_settings_file_with_nothing_to_fall_back_on_locks_guests_out(data_dir, damage):
    (data_dir / "settings.json").write_text(damage, encoding="utf-8")
    settings = load_settings()
    # Guests need a password nobody has, until the host sets a new one (or removes it) in Settings.
    assert settings.access_password_hash == config.LOCKED_PASSWORD_HASH
    assert settings.allow_guests_cloud is False
    assert load_settings() == settings  # remembered: the damaged file was set aside and replaced


def test_saves_at_the_same_moment_never_damage_the_file(data_dir):
    save_settings(Settings(access_password_hash="ab12cd"))
    problems = []

    def save(settings, start):
        start.wait()
        try:
            save_settings(settings)
        except Exception as exc:  # noqa: BLE001  # collected and checked below
            problems.append(exc)

    for _ in range(30):
        start = threading.Barrier(6)
        choices = [Settings(access_password_hash="ab12cd", output_format=fmt) for fmt in config.OUTPUT_FORMATS]
        threads = [threading.Thread(target=save, args=(s, start)) for s in [*choices, Settings(grain="medium")][:6]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))  # always a whole file
    assert problems == []
    assert not list(data_dir.glob("*.tmp"))


def test_changes_at_the_same_moment_both_count(data_dir):
    save_settings(Settings(access_password_hash="ab12cd"))
    inside = threading.Event()

    def slow_change():
        with config.changing_settings() as settings:
            inside.set()
            time.sleep(0.2)  # the other change arrives meanwhile
            settings.output_format = "png"

    def quick_change():
        inside.wait(5)
        with config.changing_settings() as settings:
            settings.grain = "subtle"

    threads = [threading.Thread(target=slow_change), threading.Thread(target=quick_change)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert load_settings() == Settings(access_password_hash="ab12cd", output_format="png", grain="subtle")


def test_a_change_that_fails_saves_nothing(data_dir):
    save_settings(Settings(output_format="png"))
    with pytest.raises(RuntimeError):
        with config.changing_settings() as settings:
            settings.output_format = "tiff"
            raise RuntimeError("refused")
    assert load_settings().output_format == "png"


def test_unknown_and_invalid_settings_are_ignored(data_dir):
    stored = {"grain": "extreme", "output_format": "png", "allow_guests_cloud": "yes", "added_later": 1}
    (data_dir / "settings.json").write_text(json.dumps(stored), encoding="utf-8")
    assert load_settings() == Settings(output_format="png")


def test_invalid_settings_cannot_be_created():
    with pytest.raises(ValueError):
        Settings(output_format="gif")
    with pytest.raises(ValueError):
        Settings(grain="heavy")


def test_app_dirs_live_under_data_dir(data_dir):
    assert config.data_dir() == data_dir
    for path, name in [
        (config.models_dir(), "models"),
        (config.jobs_dir(), "jobs"),
        (config.guest_outputs_dir(), "guest"),
    ]:
        assert path == data_dir / name
        assert path.is_dir()


def test_host_outputs_dir_is_pictures_upscaled(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("WIN_PD_OVERRIDE_MYPICTURES", str(tmp_path / "Pictures"))  # Windows asks the system
    path = config.host_outputs_dir()
    assert path == tmp_path / "Pictures" / "Upscaled"
    assert path.is_dir()


def test_host_outputs_follow_a_moved_pictures_folder(tmp_path, monkeypatch):
    # e.g. Windows with OneDrive backing up Pictures: the folder people see is OneDrive\Pictures.
    moved = tmp_path / "OneDrive" / "Pictures"
    monkeypatch.setattr(config.platformdirs, "user_pictures_dir", lambda: str(moved))
    assert config.host_outputs_dir() == moved / "Upscaled"
    assert (moved / "Upscaled").is_dir()


@pytest.mark.parametrize("answer", ["", "relative/Pictures", OSError("no known folder")])
def test_host_outputs_fall_back_to_home_pictures(tmp_path, monkeypatch, answer):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def pictures():
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(config.platformdirs, "user_pictures_dir", pictures)
    assert config.host_outputs_dir() == tmp_path / "Pictures" / "Upscaled"


def test_cancelled_is_an_exception():
    with pytest.raises(Cancelled):
        raise Cancelled()


def test_user_error_is_a_value_error():
    # Code that checks input with `except ValueError` still catches the plain-words kind.
    assert issubclass(config.UserError, ValueError)
