"""README.md, THIRD_PARTY_LICENSES.md and docs/DESIGN.md stay true to the code: links and pictures resolve,
the install commands, launcher names and addresses are the real ones, every model, library and sample
picture is credited, and no public page points at anyone's private pictures. The limits, waits and
graphics cards the README names are the app's own, and its testing badge points at a workflow that
really tests Windows and the Mac. Every screenshot is shown, with words that describe the picture as it
is now, and the README, DESIGN and the plan's API table describe what shipped.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
import unicodedata
from pathlib import Path

import pytest

from upscaler.config import DEFAULT_PORT
from upscaler.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LICENCES = ROOT / "THIRD_PARTY_LICENSES.md"
DESIGN = ROOT / "docs" / "DESIGN.md"
APP_JS = ROOT / "upscaler" / "web" / "app.js"
SAMPLES_JSON = ROOT / "upscaler" / "web" / "samples" / "samples.json"
SEEDVR2_REPO = "numz/SeedVR2_comfyUI"

MAC_INSTALL = "curl -fsSL https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.sh | bash"
WINDOWS_INSTALL = "irm https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.ps1 | iex"

# [text](target), ![alt](target) and <img src="target">; a target may carry a "title".
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_LINK = re.compile(r"""<(?:img|a|source)\b[^>]*?\b(?:src|href|srcset)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_REMOTE = ("http://", "https://", "mailto:")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _links(text: str) -> list[str]:
    return _MARKDOWN_LINK.findall(text) + _HTML_LINK.findall(text)


def _github_anchor(heading: str) -> str:
    """The id GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    plain = re.sub(r"<[^>]+>", "", heading)  # inline HTML
    plain = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", plain)  # links keep their words
    plain = plain.replace("`", "").replace("*", "").strip().lower()
    kept = "".join(ch for ch in plain if ch in " -_" or unicodedata.category(ch)[0] in "LN")
    return kept.replace(" ", "-")


def _normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())


@pytest.mark.parametrize("doc", [README, LICENCES], ids=lambda p: p.name)
def test_local_links_and_pictures_exist(doc: Path):
    missing = []
    for target in _links(_text(doc)):
        if target.startswith(_REMOTE) or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        if not (doc.parent / path).exists():
            missing.append(target)
    assert not missing, f"{doc.name} links to files that don't exist: {missing}"


@pytest.mark.parametrize("doc", [README, LICENCES], ids=lambda p: p.name)
def test_section_links_point_at_headings(doc: Path):
    text = _text(doc)
    headings = {_github_anchor(h) for h in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.MULTILINE)}
    broken = [t for t in _links(text) if t.startswith("#") and t[1:] not in headings]
    assert not broken, f"{doc.name} links to sections that don't exist: {broken}"


def test_readme_shows_the_real_one_line_installs():
    readme = _text(README)
    assert MAC_INSTALL in readme
    assert WINDOWS_INSTALL in readme
    # The same commands the installers themselves print and document.
    assert MAC_INSTALL in _text(ROOT / "install.sh")
    assert WINDOWS_INSTALL in _text(ROOT / "install.ps1")


def test_readme_names_only_real_launchers():
    named = set(re.findall(r"Start Re-Solution(?: \(Share on Wi-Fi\))?\.(?:command|bat)", _text(README)))
    assert {"Start Re-Solution.command", "Start Re-Solution.bat"} <= named
    assert all((ROOT / name).is_file() for name in named), named


def test_readme_screenshots_come_from_docs_images():
    local = [t for t in _links(_text(README)) if not t.startswith(_REMOTE)]
    pictures = [t for t in local if t.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))]
    assert pictures, "the README should show the app"
    assert all(t.startswith("docs/images/") for t in pictures), pictures


def test_every_model_is_credited():
    readme, licences = _text(README), _text(LICENCES)
    for spec in MODELS.values():
        assert spec.file in licences, f"{spec.file} isn't in THIRD_PARTY_LICENSES.md"
        assert spec.source_url in licences, f"{spec.source_url} isn't in THIRD_PARTY_LICENSES.md"
        assert spec.source_url in readme, f"{spec.source_url} isn't in README.md"
        assert spec.license in licences and spec.license in readme, spec.license
    for doc in (readme, licences):  # Photo-real lives outside MODELS
        assert "huggingface.co/numz/SeedVR2_comfyUI" in doc
        assert "github.com/ByteDance-Seed/SeedVR" in doc


def test_every_dependency_is_listed_in_the_licences():
    project = tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]
    specs = list(project["dependencies"]) + [s for extra in project["optional-dependencies"].values() for s in extra]
    names = {re.match(r"[A-Za-z0-9._-]+", spec).group(0) for spec in specs}
    licences = _normalised(_text(LICENCES))

    def listed(name: str) -> bool:  # as a whole word: "av" mustn't match inside "java"
        return re.search(rf"(?<![a-z0-9-]){re.escape(_normalised(name))}(?![a-z0-9-])", licences) is not None

    missing = [name for name in sorted(names) if not listed(name)]
    assert not missing, f"THIRD_PARTY_LICENSES.md doesn't list: {missing}"


def test_every_sample_picture_is_credited():
    readme, licences = _text(README), _text(LICENCES)
    for sample in json.loads(_text(SAMPLES_JSON)):
        assert sample["file"] in readme and sample["file"] in licences, sample["file"]
        assert sample["source_url"] in licences, sample["source_url"]


def test_readme_credits_the_designer_and_the_licence():
    readme = _text(README)
    assert "Designed by Swati Srivastava" in readme
    assert "https://swatisrivastava.in" in readme
    assert "https://www.linkedin.com/in/swatisrivastava1010" in readme
    assert "](LICENSE)" in readme


def _section(text: str, heading: str) -> str:
    """The text under a Markdown heading, up to the next heading of the same or a higher level."""
    match = re.search(rf"^(#+)\s+{re.escape(heading)}\s*$", text, re.MULTILINE)
    assert match, f"no '{heading}' heading"
    level = len(match.group(1))
    rest = text[match.end():]
    end = re.search(rf"^#{{1,{level}}}\s", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def _table_row(text: str, first_cell: str) -> str:
    rows = [line for line in text.splitlines() if line.startswith(f"| {first_cell} |")]
    assert len(rows) == 1, f"expected one '{first_cell}' row, found {len(rows)}"
    return rows[0]


def _public_markdown() -> list[Path]:
    """Every Markdown page the public repo ships (not .venv, which is never published)."""
    pages = list(ROOT.glob("*.md"))
    for folder in ("docs", "research", "samples", "upscaler"):
        pages += (ROOT / folder).rglob("*.md")
    return sorted(pages)


# A path into someone's Downloads folder, or any absolute home-folder path: these name private files.
_PRIVATE_PATH = re.compile(r"(?:~|<Downloads>|/Users/[^/\s`]+|[A-Za-z]:\\Users\\[^\\\s`]+)[/\\]Downloads\b|<Downloads>|/Users/[^/\s`<]+/")


@pytest.mark.parametrize("doc", _public_markdown(), ids=lambda p: str(p.relative_to(ROOT)))
def test_public_docs_name_no_private_picture_paths(doc: Path):
    hits = [m.group(0) for m in _PRIVATE_PATH.finditer(_text(doc))]
    assert not hits, f"{doc.relative_to(ROOT)} points at private files on someone's computer: {hits}"


@pytest.mark.parametrize("doc", [README, DESIGN], ids=lambda p: p.name)
def test_docs_name_only_real_launchers(doc: Path):
    named = set(re.findall(r"Start [A-Za-z:() -]+?\.(?:command|bat)\b", _text(doc)))
    missing = sorted(name for name in named if not (ROOT / name).is_file())
    assert not missing, f"{doc.name} names launchers that don't exist: {missing}"


@pytest.mark.parametrize("doc", [README, DESIGN], ids=lambda p: p.name)
def test_wifi_addresses_use_the_share_port(doc: Path):
    offset = re.search(r"^SHARE_PORT_OFFSET\s*=\s*(\d+)", _text(ROOT / "upscaler" / "server.py"), re.MULTILINE)
    share_port = DEFAULT_PORT + int(offset.group(1))
    ports = re.findall(r"http://(?:192\.168|10)\.\d+\.\d+(?:\.\d+)?:(\d+)", _text(doc))
    assert all(int(port) == share_port for port in ports), (doc.name, ports, share_port)


def test_design_is_not_still_a_draft():
    assert "draft for approval" not in _text(DESIGN).lower()


def test_design_stay_close_default_matches_the_app():
    default = int(re.search(r"^\s*stayClose:\s*(\d+)", _text(APP_JS), re.MULTILINE).group(1))
    lines = [line for line in _text(DESIGN).splitlines() if "Stay close to the original" in line and "default" in line]
    assert lines, "DESIGN.md should give the Stay close slider's default"
    for line in lines:
        stated = re.search(r"default (\d+)\s*%", line)
        assert stated and int(stated.group(1)) == default, line


def test_design_shows_the_real_command_line():
    design = _text(DESIGN)
    assert "scripts" not in tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]  # no `upscale` command exists
    assert not re.search(r"`upscale\s", design), "there's no `upscale` command; it's `python -m upscaler image`"
    assert "python -m upscaler image" in design


def test_design_says_where_models_really_live():
    design = _text(DESIGN)
    assert "cached in the app folder" not in design  # they live in the data folder and the Hugging Face cache
    assert "data folder" in design and "Hugging Face cache" in design


def test_design_names_the_real_video_tools():
    design = _text(DESIGN)
    assert "imageio-ffmpeg" in design  # its ffmpeg program decodes, encodes and joins; PyAV only reads details
    assert not re.search(r"PyAV[^.\n]*for video decode and encode", design)


def test_readme_warns_windows_about_hevc_playback():
    encoders = _text(ROOT / "upscaler" / "video" / "ffmpeg.py")
    if not re.search(r"\b(?:hevc_\w+|libx265)\b", encoders):
        pytest.skip("videos aren't HEVC any more")
    video = _section(_text(README), "Video to 4K")
    assert "HEVC Video Extensions" in video and "VLC" in video
    assert "HEVC Video Extensions" in _section(_text(README), "Troubleshooting")


def test_zip_instructions_steer_clear_of_synced_folders():
    zip_steps = _section(_text(README), "Another way: download the zip")
    assert not re.search(r"such as your (?:Documents|Desktop)", zip_steps)
    assert "iCloud" in zip_steps and "OneDrive" in zip_steps


def test_uninstall_removes_photoreal_with_the_cache_tool_before_the_app_goes():
    steps = _section(_text(README), "Uninstalling")
    command = f"cache rm model/{SEEDVR2_REPO}"
    assert command in steps, "a plain folder delete can leave the 7 GB behind in Hugging Face's shared blobs"
    assert ".venv/bin/hf " + command in steps and r".venv\Scripts\hf.exe" in steps
    # The command uses the app's own copy of the tool, so it has to come before the app is deleted.
    assert steps.index(command) < steps.index("Delete the app")


CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CI_PAGE = "https://github.com/swatisrivastava1010/re-solution/actions/workflows/ci.yml"


def test_readme_ci_badge_matches_the_workflow():
    yaml = pytest.importorskip("yaml")
    readme = _text(README)
    assert f"[![Tests]({CI_PAGE}/badge.svg)]({CI_PAGE})" in readme, "the Continuous testing badge is missing"
    assert CI_PAGE.split("/actions/")[0] in _text(ROOT / "install.sh")  # the same repository the installers use

    workflow = yaml.safe_load(_text(CI_WORKFLOW))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as true
    assert "main" in triggers["push"]["branches"] and "pull_request" in triggers, "every change is tested"
    (job,) = workflow["jobs"].values()
    computers = {entry["os"] for entry in job["strategy"]["matrix"]["include"]}
    assert {"windows-latest", "macos-15", "ubuntu-latest"} <= computers  # Windows, Apple Silicon Mac, Linux
    steps = job["steps"]
    commands = [step.get("run", "") for step in steps]
    assert any("uv sync --frozen --extra cpu" in run for run in commands)
    assert any(re.search(r"uv run .*--extra cpu pytest -q\s*$", run) for run in commands)
    # On Windows, the installer tests under both PowerShells, each one asked for by name.
    for shell in ("pwsh", "powershell"):
        matching = [
            step
            for step in steps
            if step.get("shell") == shell
            and step.get("env", {}).get("RE_SOLUTION_TEST_POWERSHELL") == shell
            and "tests/test_install_scripts.py" in step.get("run", "")
        ]
        assert matching and all(step.get("if") == "runner.os == 'Windows'" for step in matching), shell
    smoke = "\n".join(run for run in commands if "/api/status" in run)
    assert "python -m upscaler --version" in smoke
    assert all(word in smoke for word in ("'serve'", "'--no-browser'", "'--port'", "Stop-Process"))


def test_readme_limits_and_waits_match_the_app():
    from upscaler import server

    readme = _text(README)
    pictures = _section(readme, "Every kind of picture")
    assert f"up to {server.MAX_IMAGE_BYTES // 2**20} MB" in pictures
    assert f"jobs of up to {server.MAX_IMAGES_PER_BATCH}" in pictures
    page_limit = re.search(r"MAX_PICTURES_PER_JOB\s*=\s*(\d+)", _text(APP_JS))
    if page_limit:  # the page splits a big batch into jobs the server accepts
        assert int(page_limit.group(1)) == server.MAX_IMAGES_PER_BATCH

    wifi = _section(readme, "Share on Wi-Fi")
    assert f"at least {server.MIN_PASSWORD_LENGTH} characters" in wifi
    assert f"After {server.LOGIN_FREE_TRIES} wrong tries in a row" in wifi
    assert f"after {server.LOGIN_FREE_TRIES_EVERYONE} wrong tries from all guests together" in wifi
    assert f"wait {server.LOGIN_WAIT_SECONDS} seconds" in wifi
    assert f"up to {server.LOGIN_MAX_WAIT_SECONDS // 60} minutes" in wifi

    privacy = _section(readme, "Privacy")
    assert f"after {server.SESSION_KEY_IDLE_SECONDS // 3600} hours without use" in privacy
    assert "Forget my key" in privacy and "close their browser" in privacy


def test_readme_names_the_nvidia_cards_the_installers_accept():
    readme = _text(README)
    # Compute capability 7.5 and up: what install.sh, install.ps1 and the launchers check for.
    for installer in ("install.sh", "install.ps1", "Start Re-Solution.bat"):
        assert "GeForce RTX 20 or GTX 16 series" in _text(ROOT / installer), installer
    assert "GeForce RTX 20 or GTX 16 series" in _section(readme, "What you need")
    # A newer driver can't help a card that's too old, so that advice never says to update one.
    help_text = _section(readme, "Troubleshooting")
    too_old = next(line for line in help_text.splitlines() if "too old for the graphics software" in line)
    assert "GTX 10" in too_old and "nvidia.com/drivers" not in too_old


def _imageio_ffmpeg_licence() -> str:
    import imageio_ffmpeg

    conf = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-buildconf"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    ).stdout
    version = "3.0" if "--enable-version3" in conf else ("2.0" if "--enable-gpl" in conf else "2.1")
    return f"{'GPL' if '--enable-gpl' in conf else 'LGPL'}-{version}-or-later"


def test_ffmpeg_licence_matches_the_bundled_builds():
    row = _table_row(_text(LICENCES), "FFmpeg")
    # Checked against the locked imageio-ffmpeg 0.6.0 wheels: the Mac ffmpeg is --enable-gpl (GPLv2+),
    # the Windows one (a gyan.dev build) adds --enable-version3 (GPLv3+).
    assert "GPL-2.0-or-later on a Mac" in row and "GPL-3.0-or-later on Windows" in row, row
    where = {"darwin": "on a Mac", "win32": "on Windows"}.get(sys.platform)
    if where:  # and the build on this computer agrees
        assert f"{_imageio_ffmpeg_licence()} {where}" in row, row
    try:
        from av._core import library_meta
    except ImportError:
        return
    pyav = library_meta["libavcodec"]["license"]  # what PyAV's FFmpeg says it is, e.g. "LGPL version 3 or later"
    spdx = {
        "LGPL version 2.1 or later": "LGPL-2.1-or-later",
        "LGPL version 3 or later": "LGPL-3.0-or-later",
        "GPL version 2 or later": "GPL-2.0-or-later",
        "GPL version 3 or later": "GPL-3.0-or-later",
    }
    assert spdx.get(pyav, pyav) in row, (pyav, row)


# --- Screenshots ---------------------------------------------------------------------------------------

SCREENSHOTS = ROOT / "docs" / "images"
# The old inline before-and-after card, replaced by the full-screen viewer. It's no longer shown, and it
# goes from docs/images when the screenshots are next tidied.
RETIRED_SCREENSHOTS = {"compare.png"}
# Each screenshot as it was when its alt text (and caption) in README.md was last checked against it. A
# retaken screenshot fails here: look at it, make its alt text and caption describe what it shows now,
# then put its new SHA-256 here.
SCREENSHOT_FINGERPRINTS = {
    "batch.png": "d2124586962c9cead21b724a54700618c3ceb39845901cd4169a081e242aa8cd",
    "hero.png": "6f17bb4842c847287a9f9446db291bafe7253e2e1486ae23b51c4a0b91e0e8b0",
    "phone.png": "a016da90def2b77814a5dbc27446b98b2f6e89c7e2159b38907706b1cd557de7",
    "preview-styles.png": "24957dfee6e865d6cfbea604854cf858fefaa4c4165d5f0241748846069041f3",
    "share.png": "9ba7d92c50e6e556a53433ef5b5512150cbacc35c54721ff8e199f3126fed3cd",
    "video-compare.png": "78e53c673c5cbe3f7c3d3ed1566c3e51e3929515be5b87ddd3c8afab4981d71e",
    "viewer-zoom.png": "db979ea6d537e1f5b2ed2df47a307fe378a276250e9f8c958389f04f370aa1bf",
    "viewer.png": "8212fcba1e06d44f2a5748349fb7ebd842c89d02a765eefa824f649ceb525f27",
    "wizard.png": "4a7f627e2b96590863a28fea15cdb7993996a551b1527636c89ccbce43cb6796",
    "workspace.png": "fbd4b03c7bbf16199028944371b223d1505de7f4f2a29d4a7b3ed4f444b9cbe7",
}


def _screenshot_alts(text: str) -> dict[str, str]:
    """Each screenshot in the page, as ![alt](…) or <img> (not the small icons): its file name and alt text."""
    found = dict((name, alt) for alt, name in re.findall(r"!\[([^\]]*)\]\(docs/images/([^)/\s]+)\)", text))
    for tag in re.findall(r"<img\b[^>]*>", text):
        src = re.search(r'\bsrc="docs/images/([^"/]+)"', tag)
        if src:
            alt = re.search(r'\balt="([^"]*)"', tag)
            found[src.group(1)] = alt.group(1) if alt else ""
    return found


def test_readme_shows_every_screenshot_and_no_retired_one():
    shown = set(_screenshot_alts(_text(README)))
    screenshots = {path.name for path in SCREENSHOTS.glob("*.png")}
    assert not shown & RETIRED_SCREENSHOTS, f"the README still shows retired screenshots: {shown & RETIRED_SCREENSHOTS}"
    unused = screenshots - shown - RETIRED_SCREENSHOTS
    assert not unused, f"screenshots in docs/images that the README never shows: {sorted(unused)}"


def test_screenshot_alt_text_was_checked_against_the_picture_as_it_is():
    alts = _screenshot_alts(_text(README))
    for name, alt in alts.items():
        assert len(alt) >= 60, f"{name} needs alt text that says what the screenshot shows: {alt!r}"
    assert set(SCREENSHOT_FINGERPRINTS) == set(alts), "every screenshot shown needs a fingerprint here, and only those"
    retaken = [
        name
        for name, fingerprint in SCREENSHOT_FINGERPRINTS.items()
        if hashlib.sha256((SCREENSHOTS / name).read_bytes()).hexdigest() != fingerprint
    ]
    assert not retaken, (
        f"{retaken} changed since their README alt text was checked: make the alt text and caption describe "
        "the new picture, then update SCREENSHOT_FINGERPRINTS"
    )


def test_wifi_screenshot_is_described_as_a_guests_view():
    # The phone screenshot is a guest's: no Share on Wi-Fi button and no Settings tab.
    wifi = _section(_text(README), "Share on Wi-Fi")
    alt = _screenshot_alts(wifi)["phone.png"]
    assert "guest" in alt.lower() and "main computer's own view" not in wifi


def test_readme_screenshots_say_where_their_pictures_come_from():
    credits = _section(_text(README), "Credits")
    assert "use only these sample pictures" in credits
    assert "test pattern" in credits  # the video comparison is a generated clip, not a sample picture


# --- What the README says the app does ----------------------------------------------------------------


def test_readme_says_interrupted_cloud_jobs_arent_sent_again():
    from upscaler import jobs

    readme = _text(README)
    assert "charged twice" in jobs.CLOUD_INTERRUPTED  # the app's own words
    updating = _section(readme, "Updating")
    assert "cloud job" in updating and "charged twice" in updating, "not every unfinished job carries on"
    assert "charged twice" in _section(readme, "Cloud quality (optional)")
    # A guest's own key lives only in memory, so a restart stops their cloud jobs (jobs.OWN_KEY_FORGOTTEN).
    assert "restart" in jobs.OWN_KEY_FORGOTTEN
    assert "when Re:Solution closes or restarts" in _section(readme, "Privacy")


def test_readme_says_how_long_originals_are_kept():
    from upscaler import jobs

    assert set(jobs.ORIGINALS_KEPT_FOR) == {"image", "video"}
    using = _section(_text(README), "Using it")
    days, hours = jobs.HOST_JOB_RECORD_SECONDS // 86400, jobs.GUEST_RESULTS_SECONDS // 3600
    assert f"for {days} days" in using and f"{hours} hours for guests" in using
    assert "original" in using and "data folder" in using
    assert "originals" in _section(_text(README), "Uninstalling")


def test_readme_explains_the_viewer():
    using = _section(_text(README), "Using it")
    app, zoom = _text(APP_JS), _text(ROOT / "upscaler" / "web" / "js" / "deepzoom.js")
    assert "key === '['" in app and "key === '0'" in zoom  # the viewer's own keys
    for words in ("**Fit**", "**100%**", "**200%**", "slider", "side by side", "**Esc**"):
        assert words in using, words
    for key in ("**+**", "**−**", "**0**", "**1**", "**2**", "**[**", "**]**"):
        assert key in using, key
    assert "except the newest" in using  # finished batches fold, apart from the newest one
    assert "one zip for each quality" in using  # a batch sends one job per quality, and each job is one zip


def test_readme_video_section_covers_compare_and_retry():
    video = _section(_text(README), "Video to 4K")
    app = _text(APP_JS)
    assert "'Compare with the original'" in app and "**Compare with the original**" in video
    assert "'Retry'" in app and "**Retry**" in video
    assert "docs/images/video-compare.png" in video


def test_readme_doesnt_overstate_the_windows_installer_tests():
    calls = re.findall(r"= _run_install_ps1\(([^\n]*)\)", _text(ROOT / "tests" / "test_install_scripts.py"))
    if not calls or not all('"-DryRun"' in call for call in calls):
        pytest.skip("install.ps1 is now tested beyond a dry run")
    note = next(line for line in _text(README).splitlines() if "**Continuous testing**" in line)
    assert "dry run" in note and "A full install on Windows hasn't been run yet" in note, note
    design = next(line for line in _text(DESIGN).splitlines() if "GitHub Actions" in line)
    assert "dry run" in design and "full install" in design, design


# --- DESIGN and the plan record what shipped ------------------------------------------------------------

PLAN = ROOT / "docs" / "superpowers" / "plans" / "2026-10-17-upscaler.md"


def test_design_and_plan_record_the_portfolio_monogram():
    assert "brand/swati-mark.png" in _text(ROOT / "upscaler" / "web" / "app.css")
    for doc in (DESIGN, PLAN):
        globe = [line for line in _text(doc).splitlines() if re.search(r"gradient[- ]globe", line)]
        assert all("monogram" in line for line in globe), f"{doc.name}: {globe}"
        assert "swati-mark.png" in _text(doc), doc.name


def test_design_records_the_shipped_tray_and_batch_choices():
    app, design = _text(APP_JS), _text(DESIGN)
    batch = _section(design, "Batch-friendly layout (user request, 2026-10-18)")
    assert "'Show less'" in app and "Show all / Show less" in batch
    folding = [line for line in batch.splitlines() if "auto-collapse" in line]
    assert folding and all("newest" in line for line in folding), folding
    for label in ("Clear all finished", "Clear from list", "Vector"):
        assert label in _text(ROOT / "upscaler" / "web" / "index.html") + app and label in batch, label


def test_design_gives_the_real_video_sync_rule():
    player = _text(ROOT / "upscaler" / "web" / "js" / "videocompare.js")
    assert "correcting ? frameTime / 4 : frameTime / 2" in player
    sync = next(line for line in _text(DESIGN).splitlines() if line.startswith("- **Sync:**"))
    assert "half a frame" in sync and "quarter of a frame" in sync and "over one frame" not in sync, sync


_ROUTE = re.compile(r'^@router\.(get|post|delete|put|patch)\("([^"]+)"', re.MULTILINE)
_LISTED_ROUTE = re.compile(r"((?:GET|POST|DELETE|PUT|PATCH)(?:/(?:GET|POST|DELETE|PUT|PATCH))*) (/api/\S+)")


def _route(method: str, path: str) -> tuple[str, str]:
    return method.upper(), re.sub(r"\{[^}]*\}", "{}", path.split("?", 1)[0])


def test_plan_api_table_lists_every_endpoint_as_it_ships():
    server = _text(ROOT / "upscaler" / "server.py")
    prefix = re.search(r'^router = APIRouter\(prefix="([^"]*)"\)', server, re.MULTILINE).group(1)
    served = {_route(method, prefix + path) for method, path in _ROUTE.findall(server)}
    plan = _text(PLAN)
    table = plan[plan.index("**API contract**") : plan.index("**Errors and limits")]
    listed = set()
    for span in re.findall(r"`([^`]+)`", table):
        match = _LISTED_ROUTE.fullmatch(span.strip())
        if match:
            listed |= {_route(method, match.group(2)) for method in match.group(1).split("/")}
    assert not served - listed, f"the plan's API table leaves out: {sorted(served - listed)}"
    assert not listed - served, f"the plan's API table lists endpoints that don't exist: {sorted(listed - served)}"
    for field in ("retryable", "own_key", "source_width", "source_height"):
        assert field in table, field


def test_plan_task_15_is_ticked_off():
    plan = _text(PLAN)
    task = plan[plan.index("### Task 15:") : plan.index("### Task 16:")]
    assert "- [ ]" not in task and task.count("- [x]") == 4
