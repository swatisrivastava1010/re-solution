"""Static checks for the web UI in upscaler/web (Task 7, and Task 11 for v1.1).

The UI is plain HTML/CSS/JS with no build step, so these tests guard the
things that are easy to break silently: no requests to the internet, only
endpoints from the API contract, labelled controls, icons that exist,
the agreed copy, and the small pure helpers in app.js (run through Node when
it is installed).
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "upscaler" / "web"
SCRIPTS = [
    "wizard.js", "js/lifecycle.js", "js/compare.js", "js/deepzoom.js", "js/styles.js", "js/viewer.js",
    "js/videocompare.js", "app.js",
]  # in the order index.html loads them (lifecycle.js before the viewers that use it)
FILES = ["index.html", "app.css", "icons.svg", *SCRIPTS]

# Every endpoint the frontend may call: the Task 6 API contract plus the v1.1 additions.
CONTRACT = [
    "/api/inspect",
    "/api/staging/{x}/thumb.jpg",
    "/api/images/sample",
    "/api/estimate",
    "/api/preview",
    "/api/status",
    "/api/login",
    "/api/images",
    "/api/videos",
    "/api/jobs",
    "/api/jobs/{x}",
    "/api/jobs/{x}/cancel",
    "/api/jobs/{x}/retry",  # a failed job that kept its work (a video out of disk space) carries on
    "/api/jobs/{x}/results/{x}/file",
    "/api/jobs/{x}/results/{x}/preview.jpg",
    "/api/jobs/{x}/results/{x}/before.jpg",
    "/api/jobs/{x}/results/{x}/after.jpg",
    # The deep-zoom compare viewer (DESIGN "Deep-zoom compare viewer"): the tile pyramid, its tiles, the
    # original's tiles made on request, and the whole picture to open in a new tab.
    "/api/jobs/{x}/results/{x}/viewer.json",
    "/api/jobs/{x}/results/{x}/tiles/{x}/{x}_{x}.jpg",
    "/api/jobs/{x}/results/{x}/before/{x}/{x}_{x}.jpg",
    "/api/jobs/{x}/results/{x}/view",
    # Side-by-side video compare: browser-friendly copies, made on request (the .mp4 addresses come from it).
    "/api/jobs/{x}/results/{x}/compare",
    "/api/jobs/{x}/results/{x}/compare/status",
    "/api/jobs/{x}/results/{x}/compare/cancel",
    # A still from a finished video for its card (the card plays the comparison's H.264 copy, never the HEVC file).
    "/api/jobs/{x}/results/{x}/poster.jpg",
    "/api/jobs/{x}/zip",
    "/api/events",
    "/api/photoreal/download",
    "/api/providers/{x}/guide",
    "/api/providers/{x}/key",
    "/api/providers/{x}/session-key",
    "/api/settings",
    "/api/open-folder",
    # Share on Wi-Fi without a restart, and the QR code for phones (DESIGN "Header controls").
    "/api/share",
    "/api/share/qr.svg?url={x}",
]


def read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def js_sources() -> str:
    return "\n".join(read(name) for name in SCRIPTS)


class _Tags(HTMLParser):
    """Collects start tags with their attributes and enclosing <label> state."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None], bool]] = []
        self.label_for: set[str] = set()
        self._label_depth = 0
        self._button_stack: list[dict] = []
        self.buttons: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "label":
            self._label_depth += 1
            if a.get("for"):
                self.label_for.add(a["for"])
        if tag == "button":
            entry = {"attrs": a, "text": ""}
            self._button_stack.append(entry)
            self.buttons.append(entry)
        self.tags.append((tag, a, self._label_depth > 0))

    def handle_endtag(self, tag):
        if tag == "label" and self._label_depth:
            self._label_depth -= 1
        if tag == "button" and self._button_stack:
            self._button_stack.pop()

    def handle_data(self, data):
        for entry in self._button_stack:
            entry["text"] += data


def parse_index() -> _Tags:
    parser = _Tags()
    parser.feed(read("index.html"))
    return parser


# --- files and network -------------------------------------------------------


def test_web_files_exist():
    for name in FILES:
        assert (WEB / name).is_file(), f"missing upscaler/web/{name}"


def test_scripts_are_local_and_load_in_order():
    loaded = [attrs.get("src") for tag, attrs, _ in parse_index().tags if tag == "script" and attrs.get("src")]
    assert loaded == SCRIPTS
    for tag, attrs, _ in parse_index().tags:
        if tag == "script" and attrs.get("src"):
            assert "defer" in attrs, f"{attrs['src']} should be deferred"


def test_index_loads_only_local_assets():
    for tag, attrs, _ in parse_index().tags:
        for attr in ("src", "srcset", "poster", "data"):
            value = attrs.get(attr) or ""
            assert not re.match(r"^(https?:)?//", value), f"<{tag} {attr}={value}> is remote"
        if tag == "link":
            href = attrs.get("href") or ""
            assert not re.match(r"^(https?:)?//", href), f"<link href={href}> is remote"


def test_outbound_links_open_safely():
    for tag, attrs, _ in parse_index().tags:
        if tag == "a" and re.match(r"^https?://", attrs.get("href") or ""):
            assert attrs.get("target") == "_blank", attrs
            assert "noopener" in (attrs.get("rel") or ""), attrs


def test_css_has_no_remote_resources():
    css = read("app.css")
    assert "@import" not in css
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", css)


def test_js_makes_no_requests_to_fixed_remote_urls():
    # Links that come from the server (guide steps) are fine; hard-coded remote
    # fetches, scripts or fonts are not.
    assert not re.search(r"""(fetch|EventSource|open)\(\s*['"`]https?://""", js_sources())
    assert "fonts.googleapis" not in read("index.html") + read("app.css")


def test_js_only_calls_contract_endpoints():
    used = set(re.findall(r"""['"`](/api/[^'"`\s]*)['"`]""", js_sources()))
    assert used, "no /api paths found in the JS"
    allowed = set(CONTRACT)
    for path in used:
        normalised = re.sub(r"\$\{[^}]*\}", "{x}", path)
        assert normalised in allowed, f"{path} is not in the Task 6 API contract"


# --- accessibility -----------------------------------------------------------


def test_every_form_control_has_a_label():
    page = parse_index()
    for tag, attrs, inside_label in page.tags:
        if tag not in ("input", "select", "textarea") or attrs.get("type") == "hidden":
            continue
        labelled = (
            inside_label
            or attrs.get("aria-label")
            or attrs.get("aria-labelledby")
            or (attrs.get("id") and attrs["id"] in page.label_for)
        )
        assert labelled, f"<{tag} {attrs}> has no label"


def test_every_button_has_a_name():
    for button in parse_index().buttons:
        name = button["text"].strip() or button["attrs"].get("aria-label")
        assert name, f"button {button['attrs']} has no accessible name"


def test_page_basics():
    html = read("index.html")
    assert re.search(r"<html[^>]*\blang=", html)
    assert 'name="viewport"' in html
    assert "<title>" in html


# --- colours (DESIGN "Visual identity: Earth & Light") ------------------------------------------------

DESIGN = WEB.parents[1] / "docs" / "DESIGN.md"
Colour = tuple[float, float, float]  # sRGB, 0..255


def _declarations(block: str) -> dict[str, str]:
    """The custom properties set in a block of CSS: name (without --) -> its value as written."""
    return {m[1]: m[2].strip() for m in re.finditer(r"--([\w-]+):\s*([^;]+);", block)}


def _split_args(text: str) -> list[str]:
    """Split "a, f(b, c), d" at the commas that aren't inside brackets."""
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        depth += {"(": 1, ")": -1}.get(char, 0)
        current += char
    return [*parts, current.strip()]


def _oklch_to_srgb(lightness: float, chroma: float, hue: float) -> Colour:
    a, b = chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )
    encode = lambda v: 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055  # noqa: E731
    return tuple(255 * encode(min(1.0, max(0.0, v))) for v in linear)


def _colour(tokens: dict[str, str], value: str, depth: int = 0) -> Colour:
    """An sRGB colour from a token value: #RRGGBB, oklch(), var(--x), or color-mix(in srgb, a p%, b)."""
    assert depth < 10, f"tokens refer to each other in a loop: {value}"
    value = value.strip()
    if m := re.fullmatch(r"#([0-9a-fA-F]{6})", value):
        return tuple(float(int(m[1][i:i + 2], 16)) for i in (0, 2, 4))
    if m := re.fullmatch(r"var\(--([\w-]+)\)", value):
        assert m[1] in tokens, f"--{m[1]} isn't defined"
        return _colour(tokens, tokens[m[1]], depth + 1)
    if m := re.fullmatch(r"oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)", value):
        return _oklch_to_srgb(float(m[1]), float(m[2]), float(m[3]))
    if m := re.fullmatch(r"color-mix\(in srgb,\s*(.*)\)", value):
        first, second = _split_args(m[1])
        first, _, share = first.rpartition(" ")
        assert share.endswith("%"), value
        p = float(share[:-1]) / 100
        a, b = _colour(tokens, first, depth + 1), _colour(tokens, second, depth + 1)
        return tuple(p * x + (1 - p) * y for x, y in zip(a, b))
    raise AssertionError(f"can't read the colour {value!r}")


def _luminance(rgb: Colour) -> float:
    """WCAG relative luminance of an sRGB colour."""
    def channel(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: Colour, bg: Colour) -> float:
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _light_block() -> str:
    css = read("app.css")
    return css[css.index(":root {"):css.index("@media (prefers-color-scheme: dark)")]


def _dark_blocks() -> tuple[str, str]:
    """The dark tokens used when the computer is dark (Auto), and when the theme button picked Dark."""
    css = read("app.css")
    auto_at = css.index("@media (prefers-color-scheme: dark)")
    forced_at = css.index(':root[data-color-scheme="dark"]')
    return css[auto_at:forced_at], css[forced_at:css.index("/* ---------- Base")]


def _themes() -> dict[str, dict[str, str]]:
    light = _declarations(_light_block())
    auto_dark, forced_dark = (_declarations(block) for block in _dark_blocks())
    # Light picked with the theme button uses the base tokens (the dark block is skipped for it). Tokens
    # made from others (var(), color-mix()) are worked out with each theme's own palette, as browsers do.
    return {
        "light": light,
        "dark (follows the computer)": {**light, **auto_dark},
        "dark (theme button)": {**light, **forced_dark},
    }


def _design_tokens() -> dict[str, tuple[str, str]]:
    """The Earth & Light table in docs/DESIGN.md: token name -> (light, dark) hex values."""
    text = DESIGN.read_text(encoding="utf-8")
    section = text[text.index('### Visual identity: "Earth & Light"'):]
    named_rows = {"Mode Photo-real": "photoreal", "Mode Faithful": "faithful", "Mode Fast": "fast",
                  "Mode Illustration": "illustration", "Mode Cloud": "cloud", "Mode Video": "video",
                  "Status queued": "queued"}
    tokens: dict[str, tuple[str, str]] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not re.search(r"#[0-9A-Fa-f]{6}", cells[1]):
            continue
        names = re.findall(r"`--([\w-]+)`", cells[0])
        if names and re.search(r"/\s*tint", cells[0]):
            names.append(f"{names[0]}-tint")
        if not names:
            base = next(token for label, token in named_rows.items() if cells[0].startswith(label))
            names = [base, f"{base}-tint"]
        light, dark = (re.findall(r"#[0-9A-Fa-f]{6}", cell) for cell in cells[1:3])
        assert len(names) == len(light) == len(dark), line
        for name, lv, dv in zip(names, light, dark):
            tokens[name] = (lv.upper(), dv.upper())
    return tokens


# Text colour on the background it is drawn on, as used in app.css.
TEXT_PAIRS = [
    ("text", "page"), ("text", "surface"), ("text", "surface-raised"), ("text", "surface-2"), ("text", "selected"),
    ("text", "accent-tint"), ("text", "info-tint"), ("text", "danger-tint"), ("text", "success-tint"),
    ("text-2", "page"), ("text-2", "surface"), ("text-2", "surface-raised"), ("text-2", "selected"),
    ("text-muted", "page"), ("text-muted", "surface"), ("text-muted", "surface-raised"), ("text-muted", "surface-2"),
    ("primary", "page"), ("primary", "surface"), ("primary", "surface-raised"), ("primary", "selected"),
    ("on-primary", "primary"), ("on-primary", "primary-hover"),
    ("warning", "warning-tint"), ("warning", "page"), ("warning", "surface"),
    ("danger", "page"), ("danger", "surface"), ("danger", "danger-tint"),
    ("on-danger", "danger"), ("on-danger", "danger-hover"),
    ("info", "info-tint"), ("info", "page"), ("success", "success-tint"), ("success", "page"),
    # Job states: stone (queued), lake (working), moss (done), brick (problem).
    ("queued", "queued-tint"),
]

# The colour of each kind of work (DESIGN "UX direction"), always shown with an icon and a label.
MODE_COLOURS = ["photoreal", "faithful", "fast", "illustration", "cloud", "video"]
for _mode in MODE_COLOURS:
    TEXT_PAIRS += [
        (_mode, f"{_mode}-tint"), (_mode, "page"), (_mode, "surface"), (_mode, "surface-raised"),
        # A chosen card's words: muted text switches to --text-2 there (see test_chosen_things_use_stronger_muted_text).
        ("text", f"{_mode}-tint"), ("text-2", f"{_mode}-tint"),
    ]


def _contrast_problems(pairs, minimum):
    problems = []
    for theme, tokens in _themes().items():
        for fg, bg in pairs:
            if fg not in tokens or bg not in tokens:
                problems.append(f"{theme}: --{fg} or --{bg} isn't defined")
                continue
            ratio = _contrast(_colour(tokens, tokens[fg]), _colour(tokens, tokens[bg]))
            if ratio < minimum:
                problems.append(f"{theme}: {fg} on {bg} is {ratio:.2f}:1")
    return problems


def test_colour_reader_matches_known_values():
    # The helpers above, on values whose answers are known: WCAG's black on white is 21:1.
    assert _contrast(_colour({}, "#000000"), _colour({}, "#FFFFFF")) == pytest.approx(21.0)
    assert _colour({"a": "#000000", "b": "#FFFFFF"}, "color-mix(in srgb, var(--a) 25%, var(--b))") == (
        pytest.approx(191.25), pytest.approx(191.25), pytest.approx(191.25))
    assert _colour({}, "oklch(1 0 0)") == (pytest.approx(255), pytest.approx(255), pytest.approx(255))


def test_every_design_token_is_set_in_both_themes():
    design = _design_tokens()
    assert len(design) >= 36, f"read only {len(design)} tokens from the DESIGN table"
    light = _declarations(_light_block())
    for block, which in ((light, 1), *((_declarations(b), 2) for b in _dark_blocks())):
        for name, values in design.items():
            want = values[which - 1]
            assert name in block, f"--{name} isn't set in the {'light' if which == 1 else 'dark'} colours"
            assert block[name].upper() == want, f"--{name} is {block[name]}, DESIGN says {want}"


def test_text_colours_meet_contrast_minimum():
    problems = _contrast_problems(TEXT_PAIRS, 4.5)
    assert not problems, problems


def test_every_text_colour_reads_on_the_page_and_on_cards():
    # WCAG AA (4.5:1) for each colour used for words, on the page and on cards, in both themes.
    words = ["text", "text-2", "text-muted", "primary", "info", "success", "warning", "danger", "queued", *MODE_COLOURS]
    problems = _contrast_problems([(fg, bg) for fg in words for bg in ("page", "surface", "surface-raised")], 4.5)
    assert not problems, problems


def test_mode_colours_are_visible_as_borders_and_icons():
    # Selection rings, ticks and icons are not text, so WCAG asks for 3:1 against what is next to them.
    pairs = [(mode, "page") for mode in MODE_COLOURS] + [("on-primary", mode) for mode in MODE_COLOURS]
    problems = _contrast_problems(pairs, 3.0)
    assert not problems, problems


def test_honey_focus_ring_and_progress_show_on_every_background():
    # The focus ring and the progress bar are honey (--accent); WCAG 1.4.11 asks for 3:1.
    css = read("app.css")
    assert "--focus: var(--accent);" in _light_block()
    assert re.search(r"\.progress > span \{[^}]*background: var\(--accent\)", css)
    problems = _contrast_problems([("focus", bg) for bg in ("page", "surface", "surface-raised", "surface-2")], 3.0)
    assert not problems, problems


def test_earth_and_light_roles():
    css = read("app.css")
    rule = lambda selector: re.search(rf"\n{selector} \{{([^}}]*)\}}", css)[1]  # noqa: E731
    # Main buttons: Mother Earth with cream words (cream with Mother Earth words in dark).
    assert "background: var(--primary)" in rule(r"\.btn-primary") and "color: var(--on-primary)" in rule(r"\.btn-primary")
    # "Recommended": The Bay; chosen chips: Walking Vinnie.
    assert "background: var(--accent-tint)" in rule(r"\.badge")
    assert "background: var(--selected)" in rule(r"\.chip:has\(input:checked\)")
    assert "background: var(--selected)" in rule(r'\.filter\[aria-pressed="true"\]')
    # Job states: stone waiting, lake working, moss done, brick problem.
    assert "var(--queued-tint)" in rule(r"\.status") and "var(--queued)" in rule(r"\.status")
    assert "var(--info)" in rule(r'\.status\[data-status="running"\]')
    assert "var(--success)" in rule(r'\.status\[data-status="done"\]')
    assert "var(--danger)" in rule(r'\.status\[data-status="failed"\]')
    for mode in MODE_COLOURS:
        assert f'[data-theme="{mode}"] {{ --m: var(--{mode}); --m-tint: var(--{mode}-tint); }}' in css


def test_chosen_things_use_stronger_muted_text():
    # Muted text is too faint on Walking Vinnie and on the tints, so chosen things swap in --text-2.
    css = read("app.css")
    rule = re.search(r"([^{}]*)\{\s*--text-muted: var\(--text-2\);\s*\}", css)
    assert rule, "no rule swaps --text-muted for --text-2"
    for selector in (".chip:has(input:checked)", '.filter[aria-pressed="true"]', ".mode:has(input:checked)",
                     ".level:has(input:checked)", ".choice:has(input:checked)", ".style-card.is-chosen",
                     '.tray[data-view="list"] .item.is-selected', ".provider:has(input:checked)"):
        assert selector in rule[1], selector
    assert _contrast_problems([("text-muted", "selected")], 4.5), "if muted text now reads on --selected, drop the swap"


def test_a_chosen_cloud_service_card_reads_well_on_its_lighter_wash():
    # The chosen service card is big, so it takes a lighter wash of Walking Vinnie than a chip does.
    rule = re.search(r"\n\.provider:has\(input:checked\) \{([^}]*)\}", read("app.css"))[1]
    wash = re.search(r"background: ([^;]+);", rule)[1]
    assert "var(--selected)" in wash, wash
    for theme, tokens in _themes().items():
        background = _colour(tokens, wash)
        for fg in ("text", "text-2", "primary"):
            ratio = _contrast(_colour(tokens, tokens[fg]), background)
            assert ratio >= 4.5, f"{theme}: {fg} on the chosen service card is {ratio:.2f}:1"


# --- fonts ------------------------------------------------------------------------------------------

# DESIGN "Visual identity": headlines in Noto Serif, everything else in Inter (the user's choice), both
# SIL OFL 1.1 from Google Fonts, self-hosted as the latin variable-weight WOFF2 files.
FONTS = {
    "Noto Serif": {"file": "noto-serif-latin-wght-normal.woff2", "licence": "noto-serif-OFL.txt",
                   "authors": "The Noto Project Authors", "token": "font-head", "fallback": "Georgia, serif"},
    "Inter": {"file": "inter-latin-wght-normal.woff2", "licence": "inter-OFL.txt",
              "authors": "The Inter Project Authors", "token": "font-body", "fallback": "system-ui"},
}
# The first proposal (Fraunces and Figtree) was replaced by the user's choice; nothing of it may remain.
REPLACED_FONTS = ("Fraunces", "Figtree")

# WOFF2 names its tables by an index into this list (the WOFF2 spec, "Known Table Tags"); 63 means the
# four-letter tag follows. Enough to see the variable-font tables without unpacking the font.
_WOFF2_TAGS = (
    "cmap", "head", "hhea", "hmtx", "maxp", "name", "OS/2", "post", "cvt ", "fpgm", "glyf", "loca", "prep", "CFF ",
    "VORG", "EBDT", "EBLC", "gasp", "hdmx", "kern", "LTSH", "PCLT", "VDMX", "vhea", "vmtx", "BASE", "GDEF", "GPOS",
    "GSUB", "EBSC", "JSTF", "MATH", "CBDT", "CBLC", "COLR", "CPAL", "SVG ", "sbix", "acnt", "avar", "bdat", "bloc",
    "bsln", "cvar", "fdsc", "feat", "fmtx", "fvar", "gvar", "hsty", "just", "lcar", "mort", "morx", "opbd", "prop",
    "trak", "Zapf", "Silf", "Glat", "Gloc", "Feat", "Sill",
)


def _woff2_tables(data: bytes) -> set[str]:
    """The table tags in a WOFF2 file's directory (header 48 bytes; then flags, tag?, UIntBase128 lengths)."""
    assert data[:4] == b"wOF2"
    count, at, tags = int.from_bytes(data[12:14], "big"), 48, set()

    def base128() -> None:
        nonlocal at
        while data[at] & 0x80:
            at += 1
        at += 1

    for _ in range(count):
        flags = data[at]
        at += 1
        index = flags & 0x3F
        if index == 63:
            tag = data[at:at + 4].decode("latin-1")
            at += 4
        else:
            tag = _WOFF2_TAGS[index]
        base128()  # origLength
        transform = (flags >> 6) & 0x03
        if (tag in ("glyf", "loca") and transform == 0) or (tag not in ("glyf", "loca") and transform != 0):
            base128()  # transformLength
        tags.add(tag)
    return tags


def test_woff2_table_reader_knows_the_tag_list():
    assert len(_WOFF2_TAGS) == 63 and all(len(tag) == 4 for tag in _WOFF2_TAGS)
    assert (_WOFF2_TAGS[5], _WOFF2_TAGS[10], _WOFF2_TAGS[28], _WOFF2_TAGS[47], _WOFF2_TAGS[48]) == (
        "name", "glyf", "GSUB", "fvar", "gvar")


def test_fonts_are_bundled_and_loaded_only_from_the_fonts_folder():
    css = read("app.css")
    faces = re.findall(r"@font-face\s*\{([^}]*)\}", css)
    assert len(faces) == len(FONTS), "one @font-face per bundled font"
    for family, font in FONTS.items():
        face = next((f for f in faces if f'font-family: "{family}";' in f), None)
        assert face, f"no @font-face for {family}"
        assert "font-display: swap" in face
        assert "font-style: normal" in face
        assert re.search(r"font-weight: 100 900;", face), "the variable font's whole weight range"
        urls = re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", face)
        assert urls == [f"fonts/{font['file']}"], urls
        data = (WEB / "fonts" / font["file"]).read_bytes()
        assert data[:4] == b"wOF2", f"{font['file']} isn't a WOFF2 file"
        assert 20_000 < len(data) < 200_000, f"{font['file']} is {len(data)} bytes"
        tables = _woff2_tables(data)
        assert {"fvar", "gvar", "glyf", "name", "GSUB"} <= tables, f"{font['file']} isn't a variable font: {tables}"
    # Every url() in the stylesheet is local: the fonts folder, the brand mark, or a data: URL.
    for url in re.findall(r"url\(\s*['\"]?([^'\")]+)", css):
        assert url.startswith(("fonts/", "brand/", "data:")), f"app.css loads {url}"
        if url.startswith("brand/"):
            assert (WEB / url).is_file(), f"app.css loads {url}, which isn't in upscaler/web"
    for page in ("index.html", "app.css", *SCRIPTS):
        text = read(page)
        assert not re.search(r"fonts\.(googleapis|gstatic)\.com|jsdelivr|unpkg|cdnjs|use\.typekit", text), page
        assert not re.search(r"<link[^>]+(\.woff2?|font)", text), f"{page} loads a font outside app.css"
    # Fallbacks: Georgia for the headlines, the computer's own sans for the rest.
    root = _declarations(_light_block())
    for family, font in FONTS.items():
        assert root[font["token"]].startswith(f'"{family}",'), root[font["token"]]
        assert font["fallback"] in root[font["token"]]
    assert root["font-head"].endswith("Georgia, serif")


def test_only_the_chosen_fonts_and_their_licences_are_in_the_fonts_folder():
    wanted = {name for font in FONTS.values() for name in (font["file"], font["licence"])}
    present = {p.name for p in (WEB / "fonts").iterdir() if not p.name.startswith(".")}
    assert present == wanted, f"unexpected or missing files in upscaler/web/fonts: {sorted(present ^ wanted)}"


def test_the_replaced_fonts_are_gone_everywhere():
    root = WEB.parents[1]
    places = [WEB / name for name in FILES] + [root / "README.md", root / "THIRD_PARTY_LICENSES.md"]
    for path in places:
        text = path.read_text(encoding="utf-8")
        for old in REPLACED_FONTS:
            assert old.lower() not in text.lower(), f"{path.name} still mentions {old}"
    for path in WEB.rglob("*"):
        assert not any(old.lower() in path.name.lower() for old in REPLACED_FONTS), path


def test_font_licences_come_with_the_fonts():
    for family, font in FONTS.items():
        licence = (WEB / "fonts" / font["licence"]).read_text(encoding="utf-8")
        assert "SIL OPEN FONT LICENSE Version 1.1" in licence
        assert f"{font['authors']}" in licence, family
        assert "PERMISSION & CONDITIONS" in licence and "TERMINATION" in licence, "the whole licence text"


def test_headlines_use_noto_serif_and_everything_else_inter():
    css = read("app.css")
    assert re.search(r"\nbody \{[^}]*font-family: var\(--font-body\)", css)
    # Headlines, the brand wordmark, step titles and numbers, and the big result sizes.
    for selector in (r"h1, h2, h3", r"\.brand", r"\.step-num", r"\.result-dims", r"\.dz-title"):
        assert re.search(rf"\n{selector} \{{[^}}]*font-family: var\(--font-head\)", css), selector
    # Form controls follow the page's Inter instead of the browser's own font.
    assert "button, input, select, textarea { font: inherit; color: inherit; }" in css
    # The fonts are used from 400 to 600 (DESIGN), and <strong> stays inside that range.
    rules = re.sub(r"@font-face\s*\{[^}]*\}", "", css)  # the faces declare each font's whole range
    weights = {int(w) for w in re.findall(r"font-weight: (\d+)", rules)}
    assert weights and max(weights) <= 600 and min(weights) >= 400, sorted(weights)
    assert re.search(r"\nstrong, b \{ font-weight: 600; \}", css)
    # The bundled Inter has no character variants or stylistic sets (only calt, tnum, frac...), so the
    # stylesheet doesn't ask for any: they would silently do nothing.
    assert "font-feature-settings" not in css or not re.search(r"['\"](cv\d\d|ss\d\d)['\"]", css)
    # Times and sizes line up: tabular figures (both fonts have them).
    for selector in (r"\.estimate-facts dd", r"\.action-bar-totals", r"\.tray-summary", r"\.job-percent",
                     r"\.mode-meta", r"\.level-eta", r"\.chip-sub", r"\.batch-progress-text"):
        assert re.search(rf"\n{selector} \{{[^}}]*font-variant-numeric: tabular-nums", css), selector


def test_about_credits_the_fonts():
    html = read("index.html")
    about = html[html.index('data-view="about"'):html.index("</main>")]
    for words in ("Noto Serif", "Google, the Noto project", "Inter", "Rasmus Andersson", "OFL-1.1",
                  "https://fonts.google.com/specimen/Noto+Serif", "https://fonts.google.com/specimen/Inter"):
        assert words in about, words


def test_readme_and_licences_credit_the_fonts():
    root = WEB.parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    credits = readme[readme.index("## Credits"):]
    licences = (root / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    for doc, text in (("README.md credits", credits), ("THIRD_PARTY_LICENSES.md", licences)):
        for words in ("Noto Serif", "Inter", "Rasmus Andersson", "Noto project", "Open Font License",
                      "https://fonts.google.com/specimen/Noto+Serif", "https://fonts.google.com/specimen/Inter"):
            assert words in text, f"{doc} is missing {words}"
    for font in FONTS.values():
        assert font["file"] in licences and font["licence"] in licences


def test_every_mode_has_its_colour_and_icon():
    css = read("app.css")
    for mode in MODE_COLOURS:
        assert re.search(rf'\[data-theme="{mode}"\]', css), f"no colour rule for {mode}"
    symbols = set(re.findall(r'<symbol[^>]*\bid="i-([a-z0-9-]+)"', read("icons.svg")))
    # Photo-real sparkles, Faithful shield-check, Fast bolt, Illustration brush, Cloud cloud, Video film.
    assert {"sparkle", "shield-check", "bolt", "brush", "cloud", "film"} <= symbols


def test_danger_buttons_use_their_own_text_colour():
    rule = re.search(r"\.btn-danger\s*\{([^}]*)\}", read("app.css"))
    assert rule and "var(--on-danger)" in rule[1]


def test_css_supports_dark_mode_and_reduced_motion():
    css = read("app.css")
    assert "prefers-color-scheme: dark" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css


def test_referenced_icons_exist():
    symbols = set(re.findall(r'<symbol[^>]*\bid="i-([a-z0-9-]+)"', read("icons.svg")))
    used = set(re.findall(r'href="#i-([a-z0-9-]+)"', read("index.html")))
    used |= set(re.findall(r"""icon\(\s*['"]([a-z0-9-]+)['"]""", js_sources()))
    for block in re.findall(r"ICONS\s*=\s*\{(.*?)\}", js_sources(), flags=re.S):
        used |= set(re.findall(r""":\s*['"]([a-z0-9-]+)['"]""", block))
    missing = used - symbols
    assert not missing, f"icons used but not in icons.svg: {sorted(missing)}"


# --- agreed copy -------------------------------------------------------------


def test_photoreal_warning_copy():
    text = read("index.html") + js_sources()
    assert "Photo-real invents fine detail. Check faces, logos and text, or switch to Faithful." in text


def test_first_run_download_modal_copy():
    html = read("index.html")
    assert "Photo-real needs a one-time 7 GB download (about 10 minutes on fast Wi-Fi)" in html
    assert "Download now" in html
    assert "Use Faithful this time" in html


def test_about_page_promise_and_credits():
    html = read("index.html")
    assert "Your images stay on this computer unless you choose a cloud mode." in html
    for credit in ("4xBHI_dat2_real", "4xNomos2_hq_dat2", "4x-ClearRealityV1",
                   "RealESRGAN_x4plus_anime_6B", "realesr-general-x4v3", "SeedVR2"):
        assert credit in html, f"About page is missing credit for {credit}"
    assert "CC-BY-4.0" in html


def test_guest_key_copy_in_wizard():
    assert "Use my own key for this visit" in read("wizard.js")
    assert "Test connection" in read("wizard.js")


class _Text(HTMLParser):
    """The words a person sees or hears on the page (text and accessible-name attributes)."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        for name, value in attrs:
            if name in ("aria-label", "title", "alt", "placeholder") and value:
                self.parts.append(value)

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def test_product_name_is_resolution_everywhere():
    page = _Text()
    page.feed(read("index.html"))
    # "SeedVR2 Video Upscaler" is the name of numz's project in the credits, not ours.
    text = " ".join(page.parts).replace("SeedVR2 Video Upscaler", "")
    assert re.search(r"<title>Re:Solution</title>", read("index.html"))
    assert re.search(r'class="brand-name"[^>]*>[^<]*Re', read("index.html"))
    assert "Upscaler" not in text, "the page still says 'Upscaler' somewhere people can see it"
    # Strings in the scripts that people see (identifiers like UpscalerWizard are fine).
    literals = re.findall(r"""(['"`])((?:(?!\1).)*?)\1""", js_sources())
    shown = [value for _, value in literals if re.search(r"\bUpscaler\b", value)]
    assert not shown, shown
    assert "Start Re-Solution (Share on Wi-Fi)" in js_sources()


def test_designer_credit_in_footer_and_about():
    html = read("index.html")
    assert html.count("Designed by Swati Srivastava") >= 2  # footer and About
    links = [attrs for tag, attrs, _ in parse_index().tags if tag == "a"]
    for url in ("https://swatisrivastava.in", "https://www.linkedin.com/in/swatisrivastava1010"):
        found = [a for a in links if a.get("href") == url]
        assert len(found) >= 2, f"{url} should be linked from the footer and from About"
        for attrs in found:
            assert attrs.get("target") == "_blank" and "noopener" in (attrs.get("rel") or "")
    symbols = read("icons.svg")
    assert 'id="i-linkedin"' in symbols and "#0a66c2" in symbols  # LinkedIn keeps its blue "in"
    assert 'id="i-globe"' not in symbols  # replaced by Swati's own "S" monogram
    # The portfolio link shows her "S" mark (brand/swati-mark.png) as a CSS mask in the theme's brand
    # colour, hidden from screen readers, and the link itself says whose portfolio it is.
    portfolio = [a for a in links if a.get("href") == "https://swatisrivastava.in"]
    for attrs in portfolio:
        assert attrs.get("aria-label", "").startswith("Swati Srivastava's portfolio")
    html_links = re.findall(r'<a class="credit-link" href="https://swatisrivastava\.in"[^>]*>(.*?)</a>', html)
    assert len(html_links) >= 2
    for inner in html_links:
        assert '<span class="credit-mark" aria-hidden="true"></span>' in inner and "Portfolio" in inner
    css = read("app.css")
    rule = re.search(r"\.credit-mark \{([^}]*)\}", css).group(1)
    assert "background-color: currentColor" in rule and "var(--primary)" in rule
    assert 'mask: url("brand/swati-mark.png")' in rule and '-webkit-mask: url("brand/swati-mark.png")' in rule
    mark = WEB / "brand" / "swati-mark.png"
    assert mark.is_file() and mark.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_step_titles_and_groups_read_right():
    html = read("index.html")
    assert "Choose how it's saved" in html and "Choose the file" not in html  # step 4 isn't "pick a file"
    # The quality cards share their box with the Photo-real panel (levels and sliders), so it's a group.
    assert re.search(r'id="photo-modes" role="group"', html)


def test_reassurance_pills_and_plain_promises():
    html = read("index.html")
    for pill in ("Runs on your computer", "No watermark", "No sign-up", "Up to 16K"):
        assert pill in html


def test_v11_controls_are_on_the_page():
    ids = {attrs.get("id") for _, attrs, _ in parse_index().tags}
    for needed in (
        "sample-row",        # Try a sample
        "preset-chips",      # use-case presets
        "photoreal-panel",   # detail level and the creative sliders
        "creativity",
        "stay-close",
        "format-chips",
        "dpi-chips",
        "estimate",          # live size / print / time / file readout
        "styles-dialog",     # Preview styles
        "photo-tray",        # the picture tray (DESIGN "Batch-friendly layout")
        "photo-drawer",      # one picture's details
        "viewer-dialog",     # the full-screen before/after viewer
        "photo-action-bar",  # the sticky bar with the totals and the Upscale button
        "drop-overlay",      # drop anywhere
    ):
        assert needed in ids, f"#{needed} is missing from index.html"
    text = read("index.html") + js_sources()
    for words in ("Creativity", "Stay close to the original", "Try a sample", "Preview styles",
                  "Download all", "Same as original", "No change needed", "Suggested", "Use this",
                  "Use for all"):
        assert words in text, f"missing the words {words!r}"


def test_photoreal_level_time_hints():
    text = js_sources()
    for hint in ("About 3–6 minutes", "About 10–22 minutes", "About 3–5 minutes"):
        assert hint in text


# --- pure helpers in app.js (via Node) ----------------------------------------


NODE = shutil.which("node")


def run_app_js(expression: str):
    """Evaluate `expression` with app.js's exports bound to `m`; return the JSON result."""
    script = (
        f"const m = require({json.dumps(str(WEB / 'app.js'))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_js_files_parse():
    for name in SCRIPTS:
        out = subprocess.run([NODE, "--check", str(WEB / name)], capture_output=True, text=True, encoding="utf-8")
        assert out.returncode == 0, out.stderr


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_target_size_matches_engine_values():
    # Same values as Task 1's sizing tests, so the preview in the UI agrees with the engine.
    cases = [
        ([2048, 1539, "8k"], [8192, 6156]),
        ([941, 1672, "8k"], [4610, 8192]),
        ([2048, 1539, "6k"], [6144, 4617]),
        ([941, 1672, "8k_tv"], [4320, 7676]),
        ([1000, 500, "3x"], [3000, 1500]),
        ([9000, 6000, "8k"], [9000, 6000]),
        ([512, 385, "4k"], [3840, 2888]),
        # 16K presets, which the real /api/status now lists (values from upscaler/sizing.py).
        ([2048, 1539, "16k"], [16384, 12312]),
        ([2048, 1539, "16k_tv"], [11498, 8640]),
        ([941, 1672, "16k_tv"], [8640, 15352]),
    ]
    calls = ",".join(f"m.targetSize({a},{b},{json.dumps(c)})" for (a, b, c), _ in cases)
    results = run_app_js(f"[{calls}]")
    assert results == [expected for _, expected in cases]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_video_output_size_is_4k_and_even():
    results = run_app_js(
        "[m.videoOutputSize(1920,1080), m.videoOutputSize(1080,1081), m.videoOutputSize(4096,2160)]"
    )
    assert results[0] == [3840, 2160]
    w, h = results[1]
    assert max(w, h) == 3840 and w % 2 == 0 and h % 2 == 0
    assert results[2] == [4096, 2160]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_format_eta_reads_naturally():
    results = run_app_js(
        "[m.formatEta(20), m.formatEta(70), m.formatEta(600), m.formatEta(4800), m.formatEta(null)]"
    )
    assert results == [
        "Less than a minute left",
        "About 1 minute left",
        "About 10 minutes left",
        "About 1 hour 20 minutes left",
        "",
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_photoreal_is_the_default_image_mode():
    modes = [
        {"id": "photoreal", "available": True},
        {"id": "faithful", "available": True},
        {"id": "fast", "available": True},
    ]
    unsupported = [dict(m, available=m["id"] != "photoreal") for m in modes]
    results = run_app_js(
        "["
        f"m.DEFAULTS.imageMode, m.DEFAULTS.size, m.DEFAULTS.videoMode,"
        f"m.pickDefaultMode({{image_modes:{json.dumps(modes)}}}),"
        f"m.pickDefaultMode({{image_modes:{json.dumps(unsupported)}}}),"
        f"m.pickDefaultMode({{image_modes:{json.dumps(modes)}, settings:{{default_image_mode:'fast'}}}})"
        "]"
    )
    assert results == ["photoreal", "8k", "fast", "photoreal", "faithful", "fast"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_split_size_label():
    results = run_app_js("[m.splitSizeLabel('8K (8192 px)'), m.splitSizeLabel('2x')]")
    assert results == [["8K", "8192 px"], ["2x", ""]]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_format_duration_rounds_to_whole_minutes():
    results = run_app_js(
        "[m.formatDuration(45), m.formatDuration(90), m.formatDuration(3660),"
        " m.formatDuration(5400), m.formatDuration(7170), m.formatDuration(7199)]"
    )
    assert results == ["45 s", "1 min 30 s", "1 h 1 min", "1 h 30 min", "2 h", "2 h"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_only_guests_are_sent_back_to_the_password_screen():
    # (HTTP status, path, viewer is the host) -> show the password screen?
    cases = [
        ([401, "/api/jobs", False], True),
        ([401, "/api/images", False], True),
        ([401, "/api/jobs", True], False),
        ([401, "/api/login", False], False),
        ([401, "/api/providers/fal/key", True], False),
        ([401, "/api/providers/fal/session-key", False], False),
        ([403, "/api/settings", False], False),
        ([200, "/api/jobs", False], False),
    ]
    calls = ",".join(f"m.asksForPassword({json.dumps(a)[1:-1]})" for a, _ in cases)
    assert run_app_js(f"[{calls}]") == [expected for _, expected in cases]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_job_updates_keep_this_viewers_mine_flag():
    # Server-sent job events are built for each viewer (server.job_view), so their "mine" flag is
    # used as it comes; an update without one keeps the flag this viewer already had.
    results = run_app_js(
        "["
        "m.mergeJob({id:'a', mine:true}, {id:'a', status:'done'}),"
        "m.mergeJob({id:'a', mine:true}, {id:'a', mine:false}),"
        "m.mergeJob({id:'a', mine:false}, {id:'a', mine:true}),"
        "m.mergeJob(undefined, {id:'a'}),"
        "typeof m.fromBroadcast"
        "]"
    )
    assert results == [
        {"id": "a", "status": "done", "mine": True},
        {"id": "a", "mine": False},
        {"id": "a", "mine": True},
        {"id": "a"},
        "undefined",
    ]
    source = read("app.js")
    assert "upsertJob(job.job || job)" in source and "upsertJob(data.job)" in source


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_trying_a_cloud_service_starts_on_its_seedvr2_model():
    modes = [
        {"id": "cloud:fal:topaz_precision", "providerId": "fal"},
        {"id": "cloud:fal:seedvr2", "providerId": "fal"},
        {"id": "cloud:replicate:topaz_hf", "providerId": "replicate"},
    ]
    results = run_app_js(
        f"[m.preferredCloudMode({json.dumps(modes)}, 'fal'),"
        f" m.preferredCloudMode({json.dumps(modes)}, 'replicate'),"
        f" m.preferredCloudMode({json.dumps(modes)}, 'magnific')]"
    )
    assert [r and r["id"] for r in results] == ["cloud:fal:seedvr2", "cloud:replicate:topaz_hf", None]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_settings_shows_the_default_this_computer_really_uses():
    modes = [
        {"id": "photoreal", "label": "Photo-real", "available": True},
        {"id": "faithful", "label": "Faithful", "available": True},
        {"id": "fast", "label": "Fast", "available": True},
        {"id": "cloud:fal:seedvr2", "label": "SeedVR2 (cloud)", "available": True},
    ]
    unsupported = [dict(m, available=m["id"] != "photoreal") for m in modes]
    saved_photoreal = {"default_image_mode": "photoreal"}
    results = run_app_js(
        "["
        f"m.defaultModeChoices({{image_modes:{json.dumps(modes)}}}, {{}}),"
        f"m.defaultModeChoices({{image_modes:{json.dumps(unsupported)}}}, {json.dumps(saved_photoreal)}),"
        f"m.defaultModeChoices({{image_modes:{json.dumps(modes)}}}, {{default_image_mode:'fast'}})"
        "]"
    )
    shown = [(r["value"], [o["id"] for o in r["options"]]) for r in results]
    assert shown == [
        ("photoreal", ["photoreal", "faithful", "fast"]),
        ("faithful", ["faithful", "fast"]),
        ("fast", ["photoreal", "faithful", "fast"]),
    ]


# --- v1.1 helpers (Task 11) ---------------------------------------------------------


def js(value) -> str:
    return json.dumps(value)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_print_size_matches_engine_values():
    # Same numbers as sizing.print_size: 16384×12312 at 300 DPI.
    results = run_app_js("[m.printSize(16384, 12312, 300), m.formatPrint(m.printSize(16384, 12312, 300))]")
    assert results[0] == {"inches": [54.6, 41.0], "cm": [138.7, 104.2], "dpi": 300}
    assert results[1] == "54.6 × 41.0 in (138.7 × 104.2 cm) at 300 DPI"


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_local_estimate_when_the_server_cannot_estimate():
    # The readout falls back to this when /api/estimate isn't there; sizes and bytes per pixel match sizing.py.
    results = run_app_js(
        "["
        "m.localEstimate({width:2048, height:1539, size:'16k', format:'jpeg', dpi:300}),"
        "m.localEstimate({width:2048, height:1539, size:'16k', format:'png', dpi:300}).file_mb,"
        "m.localEstimate({width:2048, height:1539, size:'16k', format:'tiff', dpi:300}).file_mb,"
        "m.localEstimate({width:2048, height:1539, size:'8k', format:'jpeg', dpi:300})"
        "]"
    )
    big, png_mb, tiff_mb, eight = results
    assert big["out"] == [16384, 12312]
    assert big["megapixels"] == 201.7
    assert big["file_mb"] == 40.3
    assert big["print"] == {"inches": [54.6, 41.0], "cm": [138.7, 104.2], "dpi": 300}
    assert big["seconds"] is None
    assert any(w.startswith("16K files are very large") for w in big["warnings"])
    assert (png_mb, tiff_mb) == (322.8, 363.1)
    assert eight["out"] == [8192, 6156] and eight["megapixels"] == 50.4 and eight["file_mb"] == 10.1
    assert eight["warnings"] == []


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_estimate_format_for_same_as_original():
    results = run_app_js(
        "[m.estimateFormat('original','PNG'), m.estimateFormat('original','HEIC'), m.estimateFormat('original','tiff'),"
        " m.estimateFormat('original', undefined), m.estimateFormat('webp','JPEG'), m.estimateFormat('original','JPEG')]"
    )
    assert results == ["png", "jpeg", "tiff", "png", "webp", "jpeg"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_webp_is_only_offered_up_to_its_limit():
    formats = [{"id": "jpeg"}, {"id": "png"}, {"id": "webp", "max_long_edge": 16383}]
    results = run_app_js(
        "["
        f"m.formatAllowed('webp', [[8192, 6156]], {js(formats)}),"
        f"m.formatAllowed('webp', [[8192, 6156], [16384, 12312]], {js(formats)}),"
        f"m.formatAllowed('jpeg', [[16384, 12312]], {js(formats)}),"
        "m.formatAllowed('webp', [[16384, 9000]], undefined),"
        "m.formatAllowed('webp', [], undefined)"
        "]"
    )
    assert results == [True, False, True, False, True]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_webp_limit_without_max_long_edge_from_the_server():
    # The server may list {id: "webp"} on its own; the 16,383 px limit still applies (and is shown).
    bare = [{"id": "jpeg"}, {"id": "webp"}]
    results = run_app_js(
        "["
        f"m.formatLimit('webp', {js(bare)}), m.formatLimit('jpeg', {js(bare)}),"
        f"m.formatLimit('webp', {js([{'id': 'webp', 'max_long_edge': 9000}])}),"
        f"m.formatAllowed('webp', [[16384, 12312]], {js(bare)})"
        "]"
    )
    assert results == [16383, 65535, 9000, False]  # JPEG stops at 65,535 px, even when the server doesn't say


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_file_type_blocked_by_the_size_comes_back_when_it_fits():
    formats = [{"id": "jpeg"}, {"id": "png"}, {"id": "webp", "max_long_edge": 16383}]
    eight, sixteen = [[8192, 6156]], [[16384, 12312]]
    results = run_app_js(
        "["
        f"m.fitFormat('webp', null, {js(eight)}, {js(formats)}),"     # fits: kept
        f"m.fitFormat('webp', null, {js(sixteen)}, {js(formats)}),"   # 16K: JPEG for now
        f"m.fitFormat('jpeg', 'webp', {js(sixteen)}, {js(formats)}),"  # still 16K: still JPEG
        f"m.fitFormat('jpeg', 'webp', {js(eight)}, {js(formats)}),"    # back to 8K: WebP again
        f"m.fitFormat('png', null, {js(sixteen)}, {js(formats)}),"     # PNG has no limit
        f"m.fitFormat('jpeg', 'webp', {js(eight)}, {js(formats[:2])})"  # WebP no longer offered
        "]"
    )
    assert results == [
        {"format": "webp", "blocked": None},
        {"format": "jpeg", "blocked": "webp"},
        {"format": "jpeg", "blocked": "webp"},
        {"format": "webp", "blocked": None},
        {"format": "png", "blocked": None},
        {"format": "jpeg", "blocked": None},
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_auto_suggest_preselects_but_the_users_choice_wins():
    selectable = ["photoreal", "faithful", "fast", "illustration"]
    batch = {"mode": "photoreal", "level": "balanced", "explicit": False, "levelExplicit": False}
    cases = [
        ({"suggestion": {"mode": "faithful", "photoreal_level": None}}, batch),
        ({"suggestion": {"mode": "photoreal", "photoreal_level": "balanced"}}, batch),
        ({"suggestion": {"mode": "faithful"}}, {**batch, "mode": "fast", "explicit": True}),
        ({"suggestion": {"mode": "faithful"}, "choice": {"mode": "photoreal", "level": "rich"}},
         {**batch, "mode": "fast", "explicit": True}),
        ({"suggestion": {"mode": "vector", "photoreal_level": None}}, batch),
        ({"suggestion": {"mode": "photoreal", "photoreal_level": "balanced"}},
         {**batch, "level": "gentle", "levelExplicit": True}),
        ({}, {**batch, "mode": "faithful"}),
    ]
    calls = ",".join(f"m.choiceFor({js(item)}, {js(b)}, {js(selectable)})" for item, b in cases)
    assert run_app_js(f"[{calls}]") == [
        {"mode": "faithful", "level": None, "source": "suggested"},
        {"mode": "photoreal", "level": "balanced", "source": "suggested"},
        {"mode": "fast", "level": None, "source": "chosen"},
        {"mode": "photoreal", "level": "rich", "source": "chosen"},
        {"mode": "photoreal", "level": "balanced", "source": "default"},
        {"mode": "photoreal", "level": "gentle", "source": "suggested"},
        {"mode": "faithful", "level": None, "source": "default"},
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_preview_routes_map_to_choices():
    results = run_app_js(
        "[m.routeChoice('photoreal:rich'), m.routeChoice('photoreal:balanced'), m.routeChoice('faithful'),"
        " m.routeChoice('fast'), m.routeChoice('original'), m.DEFAULT_PREVIEW_ROUTES]"
    )
    assert results[:5] == [
        {"mode": "photoreal", "level": "rich"},
        {"mode": "photoreal", "level": "balanced"},
        {"mode": "faithful", "level": None},
        {"mode": "fast", "level": None},
        None,
    ]
    assert results[5] == ["photoreal:balanced", "photoreal:rich", "photoreal:gentle", "faithful", "fast"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_preview_styles_include_the_suggested_style():
    # A cartoon suggested as Illustration (or a sharp photo as faithful_sharp) can be previewed and
    # picked too; only styles preview.LABELS knows are asked for, and only ones this computer offers.
    local = ["photoreal", "faithful", "faithful_sharp", "fast", "illustration"]
    results = run_app_js(
        "["
        f"m.previewRoutes(true, ['illustration', 'photoreal'], {js(local)}),"
        f"m.previewRoutes(false, ['faithful_sharp', 'faithful_sharp'], {js(local)}),"
        f"m.previewRoutes(true, ['fast', 'cloud:fal:topaz_precision', 'vector', undefined], {js(local)}),"
        f"m.previewRoutes(true, ['illustration'], {js(['photoreal', 'faithful', 'fast'])})"
        "]"
    )
    five = ["photoreal:balanced", "photoreal:rich", "photoreal:gentle", "faithful", "fast"]
    assert results == [
        [*five, "illustration"],
        ["faithful", "fast", "faithful_sharp"],
        five,
        five,
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_preview_results_keep_the_reason_a_style_failed():
    # preview.run_preview gives a style it couldn't make no crop and a friendly `error`: no picture
    # (so no broken image, time or "Use this"), and the reason is kept to show instead.
    job = {
        "id": "j1",
        "results": [
            {"route": "original", "label": "Original (enlarged)", "seconds": 0.1},
            {"route": "faithful", "label": "Faithful", "seconds": 4.2, "est_full_seconds": 240},
            {"route": "fast", "label": "Fast", "seconds": 1.0, "est_full_seconds": 60,
             "error": "Re:Solution couldn't download the Fast model."},
            {"seconds": 3.0},
        ],
    }
    results = run_app_js(
        f"m.previewResults({js(job)}, ['faithful', 'fast', 'illustration'],"
        " (id, n) => `/api/jobs/${id}/results/${n}/file`)"
    )
    assert [r["route"] for r in results] == ["original", "faithful", "fast", "illustration"]
    assert results[1]["url"] == "/api/jobs/j1/results/1/file" and results[1]["error"] is None
    assert results[1]["estFull"] == 240
    assert results[2]["url"] is None
    assert results[2]["error"] == "Re:Solution couldn't download the Fast model."
    assert results[3]["label"] == "illustration" and results[3]["url"] == "/api/jobs/j1/results/3/file"


def test_preview_styles_hides_failed_styles_and_sends_the_size():
    source = read("js/styles.js")
    assert "usable(result)" in source and "state.results.filter(usable)" in source  # A/B offers made styles only
    assert "...deps.previewOptions()" in source
    app = read("app.js")
    assert "previewOptions: () => ({ size: state.photo.size, creativity: round2(state.photo.creativity / 100) })" in app


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_missing_endpoints_are_told_apart_from_real_errors():
    # An endpoint that hasn't landed yet answers like the static file server does.
    cases = [
        ([404, "Not Found"], True),
        ([405, "Method Not Allowed"], True),
        ([404, "That picture isn't here any more."], False),
        ([400, "Not Found"], False),
        ([500, "Internal Server Error"], False),
    ]
    calls = ",".join(f"m.missingEndpoint({js(a)[1:-1]})" for a, _ in cases)
    assert run_app_js(f"[{calls}]") == [expected for _, expected in cases]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_presets_set_only_choices_this_app_offers():
    status = {
        "sizes": [{"id": "8k"}, {"id": "16k"}],
        "formats": [{"id": "jpeg"}, {"id": "tiff"}],
        "dpi_choices": [72, 150, 300, 600],
        "image_modes": [{"id": "photoreal", "available": True}, {"id": "faithful", "available": True}],
    }
    good = {"id": "print", "mode": "photoreal", "photoreal_level": "balanced", "size": "16k",
            "output_format": "tiff", "dpi": 300, "grain": "off"}
    odd = {"id": "odd", "mode": "faithful", "size": "huge", "output_format": "gif", "dpi": 96, "grain": "lots"}
    results = run_app_js(f"[m.presetChoices({js(good)}, {js(status)}), m.presetChoices({js(odd)}, {js(status)})]")
    assert results == [
        {"size": "16k", "mode": "photoreal", "level": "balanced", "format": "tiff", "dpi": 300, "grain": "off"},
        {"mode": "faithful", "level": None},
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_batches_with_different_qualities_become_one_job_each():
    choices = [
        {"id": "a", "mode": "photoreal", "level": "balanced"},
        {"id": "b", "mode": "faithful", "level": None},
        {"id": "c", "mode": "photoreal", "level": "balanced"},
        {"id": "d", "mode": "photoreal", "level": "rich"},
    ]
    assert run_app_js(f"m.groupBySettings({js(choices)})") == [
        {"mode": "photoreal", "level": "balanced", "ids": ["a", "c"]},
        {"mode": "faithful", "level": None, "ids": ["b"]},
        {"mode": "photoreal", "level": "rich", "ids": ["d"]},
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_region_picker_turns_a_click_into_a_256_px_spot():
    results = run_app_js(
        "[m.regionAt(100, 50, 200, 150, 2048, 1536), m.regionAt(0, 0, 200, 150, 2048, 1536),"
        " m.regionAt(200, 150, 200, 150, 2048, 1536), m.regionAt(10, 10, 200, 100, 200, 100)]"
    )
    assert results == [[896, 384, 256, 256], [0, 0, 256, 256], [1792, 1280, 256, 256], [0, 0, 200, 100]]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_image_options_carry_every_v11_field():
    batch = {"size": "8k", "level": "balanced", "strength": 80, "creativity": 30, "stayClose": 25,
             "grain": "subtle", "format": "tiff", "dpi": 600}
    results = run_app_js(
        f"[m.imageOptionsFor({js(batch)}, {{mode:'photoreal', level:'rich'}}),"
        f" m.imageOptionsFor({js(batch)}, {{mode:'faithful', level:null}})]"
    )
    assert results[0] == {
        "mode": "photoreal", "photoreal_level": "rich", "size": "8k", "strength": 0.75, "creativity": 0.3,
        "resemblance": 0.75, "grain": "subtle", "output_format": "tiff", "dpi": 600,
    }
    assert results[1] == {
        "mode": "faithful", "photoreal_level": "balanced", "size": "8k", "strength": 0.8, "creativity": 0,
        "resemblance": 1, "grain": "subtle", "output_format": "tiff", "dpi": 600,
    }


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_batch_summary_counts_photos_not_jobs():
    jobs = [
        {"status": "done", "inputs": ["a", "b"], "results": [{}, {}]},
        {"status": "running", "inputs": ["c", "d"], "progress": 0.5, "results": []},
        {"status": "failed", "inputs": ["e"], "results": []},
    ]
    assert run_app_js(f"m.batchSummary({js(jobs)})") == {
        "photos": 5, "done": 2, "failed": 1, "stopped": 0, "working": 2, "fraction": 0.8, "finished": False,
    }
    finished = [dict(jobs[0]), {"status": "done", "inputs": ["c"], "results": [{}]}]
    summary = run_app_js(f"m.batchSummary({js(finished)})")
    assert summary["finished"] is True and summary["done"] == 3 and summary["fraction"] == 1


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_mode_colours_and_labels():
    status = {"image_modes": [{"id": "photoreal", "label": "Photo-real"}, {"id": "faithful", "label": "Faithful"}]}
    results = run_app_js(
        "["
        "m.modeTheme('photoreal'), m.modeTheme('faithful'), m.modeTheme('faithful_sharp'), m.modeTheme('fast'),"
        "m.modeTheme('illustration'), m.modeTheme('cloud:fal:seedvr2'), m.modeTheme('fast', 'video'), m.modeTheme('x'),"
        f"m.modeLabelFor('original', 'image', {js(status)}),"
        f"m.modeLabelFor('photoreal', 'image', {js(status)}, 'rich'),"
        f"m.modeLabelFor('faithful', 'image', {js(status)}, 'rich'),"
        f"m.modeLabelFor('vector', 'image', {js(status)})"
        "]"
    )
    assert results == [
        "photoreal", "faithful", "faithful", "fast", "illustration", "cloud", "video", "neutral",
        "No change needed", "Photo-real · Rich", "Faithful", "Drawn directly (vector)",
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_photoreal_levels_default_and_from_status():
    from_status = {"image_modes": [{"id": "photoreal", "levels": [{"id": "balanced", "label": "Balanced",
                                                                   "description": "d", "eta_hint": "e"}]}]}
    results = run_app_js(f"[m.photorealLevels({{}}), m.photorealLevels({js(from_status)})]")
    assert [level["id"] for level in results[0]] == ["balanced", "rich", "gentle"]
    assert all(level["label"] and level["description"] and level["eta_hint"] for level in results[0])
    assert results[1] == from_status["image_modes"][0]["levels"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_file_picker_accepts_what_the_server_reads():
    accept = {"images": [".jpg", ".heic", ".cr3"], "videos": [".mov", ".mts"]}
    results = run_app_js(
        "["
        f"m.acceptAttribute({js(accept)}, 'images'), m.acceptAttribute({js(accept)}, 'videos'),"
        "m.acceptAttribute(undefined, 'images').includes('.heic'),"
        f"m.fileKind({{name:'IMG_1.CR3', type:''}}, {js(accept)}),"
        f"m.fileKind({{name:'clip.MTS', type:''}}, {js(accept)}),"
        f"m.fileKind({{name:'notes.docx', type:'application/msword'}}, {js(accept)}),"
        f"m.fileKind({{name:'paste.png', type:'image/png'}}, {js(accept)}),"
        "m.fileKind({name:'old.nef', type:''}, undefined)"
        "]"
    )
    assert results == ["image/*,.jpg,.heic,.cr3", "video/*,.mov,.mts", True, "image", "video", None, "image", "image"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_approximate_time_reads_naturally():
    results = run_app_js(
        "[m.formatApproxTime(20), m.formatApproxTime(250), m.formatApproxTime(4800), m.formatApproxTime(null),"
        " m.formatFileSize(40.3), m.formatFileSize(1234.5), m.formatFileSize(0.4)]"
    )
    assert results == [
        "Less than a minute", "About 4 minutes", "About 1 hour 20 minutes", "",
        "about 40 MB", "about 1.2 GB", "about 1 MB",
    ]


# --- UI-VERIFY fixes (found while using the real server) ------------------------------


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_job_eta_line_never_repeats_the_message():
    # The video engine already says "…, less than a minute left" in its message before the queue has
    # a time of its own; the time line under it must not then say "Working out how long this takes…".
    jobs = [
        {"status": "running", "kind": "video", "message": "Upscaling frame 10 of 50, less than a minute left",
         "eta_seconds": None},
        {"status": "running", "kind": "video", "message": "Upscaling frame 30 of 50, about 2 minutes left",
         "eta_seconds": 120},
        {"status": "running", "kind": "video", "message": "Upscaling frame 1 of 50", "eta_seconds": None},
        {"status": "running", "kind": "image", "message": "Upscaling", "eta_seconds": 600},
        {"status": "running", "kind": "image", "message": "Upscaling", "eta_seconds": None},
        {"status": "queued", "kind": "video", "message": "Waiting", "eta_seconds": None},
    ]
    assert run_app_js(f"{js(jobs)}.map(m.jobEtaText)") == [
        "", "", "Working out how long this takes…", "About 10 minutes left", "", "",
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_dropzone_words_fit_the_device():
    # Phones can't drag files or paste with a keyboard shortcut, so they aren't told to.
    results = run_app_js(
        "[m.dropzoneCopy({touch:false, count:0, pasteKeys:'⌘V'}), m.dropzoneCopy({touch:false, count:2, pasteKeys:'Ctrl+V'}),"
        " m.dropzoneCopy({touch:true, count:0, pasteKeys:'⌘V'}), m.dropzoneCopy({touch:true, count:1, pasteKeys:'⌘V'})]"
    )
    desktop, desktop_more, phone, phone_more = results
    assert desktop["title"] == "Drag pictures here" and desktop["button"] == "Choose pictures"
    assert desktop["hint"].startswith("Or paste one (⌘V).") and "300 MB" in desktop["hint"]
    assert desktop_more["title"] == "Drop more pictures here" and desktop_more["button"] == "Add more"
    for words in (phone, phone_more):
        text = " ".join(words.values()).lower()
        assert "drag" not in text and "drop" not in text and "paste" not in text and "⌘" not in text
        assert "300 MB" in words["hint"]
    assert phone["button"] == "Choose pictures" and phone_more["button"] == "Add more"


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_video_preview_is_offered_only_when_the_clip_is_longer():
    results = run_app_js(
        "[m.offerVideoPreview(2), m.offerVideoPreview(3.4), m.offerVideoPreview(10), m.offerVideoPreview(0),"
        " m.offerVideoPreview(NaN), m.offerVideoPreview(undefined)]"
    )
    assert results == [False, False, True, True, True, True]  # unknown length: still offer it


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_settings_offers_every_file_type_the_server_saves():
    status = {"formats": [{"id": "jpeg"}, {"id": "png"}, {"id": "tiff"}, {"id": "webp", "max_long_edge": 16383},
                          {"id": "original"}]}
    results = run_app_js(f"[m.formatChoices({js(status)}), m.formatChoices({{}})]")
    expected = [["jpeg", "JPEG"], ["png", "PNG"], ["tiff", "TIFF"], ["webp", "WebP"], ["original", "Same as original"]]
    assert [[f["id"], f["label"]] for f in results[0]] == expected
    assert [[f["id"], f["label"]] for f in results[1]] == expected
    html = read("index.html")
    assert 'id="default-format-options"' in html
    assert 'name="default-format" value="jpeg"' not in html  # drawn from the server's list, not hard-coded


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_sent_batch_lets_new_pictures_get_their_own_suggestion():
    # Picking a quality for one batch shouldn't hide the suggestions of pictures added after it was sent.
    results = run_app_js(
        "[m.choicesAfterSending({mode:'fast', explicit:true, levelExplicit:true, size:'4k'}, 0),"
        " m.choicesAfterSending({mode:'fast', explicit:true, levelExplicit:false, size:'4k'}, 2)]"
    )
    assert results[0] == {"mode": "fast", "explicit": False, "levelExplicit": False, "size": "4k"}
    assert results[1] == {"mode": "fast", "explicit": True, "levelExplicit": False, "size": "4k"}


def test_scale_sizes_start_their_own_row():
    rule = re.search(r"\.chip-gap\s*\{([^}]*)\}", read("app.css"))
    assert rule and "flex-basis: 100%" in rule[1]


def test_style_cards_show_pixels_at_full_size_and_name_their_buttons():
    styles = read("js/styles.js")
    # "Use this" says which style, for screen readers, and every card pans together at 100%.
    assert re.search(r"'aria-label':\s*`Use \$\{result\.label\}", styles)
    assert "--shot-x" in styles and "--shot-y" in styles
    css = read("app.css")
    rule = re.search(r"\.style-shot img\s*\{([^}]*)\}", css)
    assert rule and "object-fit: none" in rule[1] and "var(--shot-x" in rule[1]


def test_overridden_suggestion_can_be_used_in_one_click():
    app = read("app.js")
    assert "suggestion-use" in app and "Use the suggestion" in app


# --- Review fixes: theme button, Share on Wi-Fi, honest estimates ---------------------------


def _header() -> str:
    html = read("index.html")
    return html[html.index('<header class="topbar">'):html.index("</header>")]


def test_theme_button_is_in_the_header_for_everyone():
    header = _header()
    button = re.search(r'<button[^>]*id="theme-button"[^>]*>', header)
    assert button, "the header has no theme button"
    tag = button[0]
    assert "data-host-only" not in tag and " hidden" not in tag  # hosts and guests both get it
    assert re.search(r'aria-label="Theme: [^"]+"', tag) and re.search(r'title="Theme: [^"]+"', tag)
    symbols = set(re.findall(r'<symbol[^>]*\bid="i-([a-z0-9-]+)"', read("icons.svg")))
    assert {"sun", "moon", "monitor"} <= symbols


def test_theme_choice_is_applied_before_the_first_paint():
    html = read("index.html")
    head = html[:html.index("</head>")]
    inline = re.search(r"<script>(.*?)</script>", head, flags=re.S)
    assert inline, "no inline theme script in <head>"
    assert inline.start() < head.index('<link rel="stylesheet" href="app.css">')
    code = inline[1]
    assert "localStorage.getItem('resolution.theme')" in code and "dataset.colorScheme" in code
    assert code.strip().startswith("try {")  # storage can throw in private windows
    assert "THEME_KEY = 'resolution.theme'" in read("app.js")  # the button stores under the same key


def test_dark_colours_follow_the_computer_and_the_theme_button():
    css = read("app.css")
    auto, forced = _dark_blocks()
    assert ':root:not([data-color-scheme="light"])' in auto  # a Light pick wins over a dark computer
    assert "color-scheme: dark" in forced
    assert re.search(r':root\[data-color-scheme="light"\]\s*\{\s*color-scheme: light;\s*\}', css)
    # [data-theme] already means a kind of work (Photo-real violet...), so the theme uses its own attribute.
    assert "data-theme=\"light\"" not in css and "data-theme=\"dark\"" not in css
    auto_tokens, forced_tokens = _declarations(auto), _declarations(forced)
    assert auto_tokens and auto_tokens == forced_tokens, "the two dark blocks have drifted apart"
    for name in ("backdrop", "shadow-1", "shadow-2", "shadow-3"):
        assert name in auto_tokens, name


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_theme_button_cycles_light_dark_auto():
    results = run_app_js(
        "[m.nextTheme('light'), m.nextTheme('dark'), m.nextTheme('auto'), m.nextTheme(null), m.themeFrom('blue'),"
        " m.themeLabel('auto'), m.themeLabel('light'), m.themeLabel('dark')]"
    )
    assert results == [
        "dark", "auto", "light", "light", "auto",
        "Theme: Auto (follows your computer). Switch to Light",
        "Theme: Light. Switch to Dark",
        "Theme: Dark. Switch to Auto",
    ]
    # The page's starting label is the one for Auto.
    assert 'aria-label="Theme: Auto (follows your computer). Switch to Light"' in _header()


def test_share_button_is_host_only_and_shows_when_sharing_is_on():
    button = re.search(r'<button[^>]*id="share-button"[^>]*>(.*?)</button>', _header(), flags=re.S)
    assert button, "the header has no Share on Wi-Fi button"
    tag, inside = button[0], button[1]
    assert "data-host-only" in tag and " hidden" in tag  # shown for the host by renderHostBits
    # It opens a panel, so it's not a toggle button (no aria-pressed): its words say the state.
    assert 'data-sharing="off"' in tag and "aria-pressed" not in tag
    assert 'aria-haspopup="dialog"' in tag and 'aria-controls="share-dialog"' in tag
    assert "#i-wifi" in inside and "Share on Wi-Fi" in inside  # icon and words
    app = read("app.js")
    assert "el.shareButton.dataset.sharing = on ? 'on' : 'off';" in app
    assert "'Sharing on'" in app
    css = read("app.css")
    assert re.search(r'\.share-button\[data-sharing="on"\]\s*\{[^}]*var\(--success-tint\)', css)  # moss green "on"
    assert "share-pill" not in read("index.html") + app  # the button replaces the old status pill


def test_share_dialog_has_the_switch_address_qr_and_warnings():
    page = parse_index()
    ids = {attrs.get("id") for _, attrs, _ in page.tags}
    assert {"share-dialog", "share-switch", "share-details", "share-feedback"} <= ids
    switch = next(attrs for tag, attrs, _ in page.tags if attrs.get("id") == "share-switch")
    assert switch.get("role") == "switch" and "share-switch" in page.label_for
    app = read("app.js")
    assert "api(API.share, { method: 'POST', body: { enabled: on } })" in app
    assert "API.shareQr(urls[0])" in app and "copyButton(url, code)" in app
    assert "shareNote(share, navigator.platform" in app
    assert "Set a password in Settings" in app and "sharePasswordSet(share)" in app
    # Without the endpoint (an older server), the restart instructions come back.
    assert "missingEndpoint(err.status, err.detail)" in app and "state.endpoints.share = false" in app
    assert "openShare" in app[app.index("function renderSettings"):]  # Settings opens the same dialog


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_firewall_note_fits_the_hosts_computer():
    results = run_app_js("[m.firewallNote('MacIntel'), m.firewallNote('Win32'), m.firewallNote('Linux x86_64')]")
    assert results == [
        "If your Mac asks whether to allow incoming connections, click Allow.",
        "If Windows asks about the firewall, click Allow access.",
        "",
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_saved_format_mirrors_the_pipeline():
    # pipeline_image._output_format: see-through pictures become PNG unless the type keeps them;
    # imageio_utils.resolve_format: "Same as original" falls back when the source type can't hold the size.
    sixteen, eight = [16384, 12288], [8192, 6144]
    results = run_app_js(
        "["
        f"m.savedFormat('jpeg', 'PNG', {js(sixteen)}, true), m.savedFormat('tiff', 'PNG', {js(sixteen)}, true),"
        f"m.savedFormat('original', 'WEBP', {js(sixteen)}, false), m.savedFormat('original', 'WEBP', {js(sixteen)}, true),"
        f"m.savedFormat('original', 'WEBP', {js(eight)}, false),"
        f"m.savedFormat('original', 'HEIF', {js(sixteen)}, false), m.savedFormat('original', 'HEIC', {js(sixteen)}, true),"
        f"m.savedFormat('original', 'HEIC', {js(eight)}, false), m.savedFormat('original', 'SVG', {js(eight)}, true),"
        f"m.estimateFormat('jpeg', 'PNG', {js(sixteen)}, true), m.estimateFormat('original', 'WEBP', {js(sixteen)}, false),"
        f"m.estimateFormat('original', 'HEIC', {js(eight)}, false), m.estimateFormat('original', 'HEIC', {js(eight)}, true)"
        "]"
    )
    assert results == [
        "png", "tiff", "jpeg", "png", "webp", "jpeg", "png", "heic", "png",
        "png", "jpeg", "jpeg", "jpeg",
    ]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_estimates_count_see_through_areas():
    # An 800×600 transparent PNG with JPEG chosen, at 16K: really a PNG of 16384×12288 × 1.6 × 4/3 bytes.
    local, server = run_app_js(
        "[m.localEstimate({width:800, height:600, size:'16k', format:m.estimateFormat('jpeg','PNG',[16384,12288],true),"
        " dpi:300, alpha:true}),"
        " [m.withAlpha({file_mb: 322.1, out: [16384, 12288]}, true).file_mb, m.withAlpha({file_mb: 322.1}, false).file_mb]]"
    )
    assert local["out"] == [16384, 12288] and local["file_mb"] == 429.5
    assert any("about 429 MB as PNG" in w for w in local["warnings"])
    assert server == [429.5, 322.1]  # /api/estimate doesn't know about see-through areas, so the UI scales it


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_batch_warnings_come_once_per_kind():
    big = "16K files are very large: about {} as JPEG. They mainly help big prints; on a screen they look the same as 8K."
    space = ("Not enough free space: this picture needs about 120 MB free to save safely, and the disk has 100 MB. "
             "Free up some space first.")
    items = [
        {"warnings": [big.format("42 MB"), "This will take about 12 minutes."], "file_mb": 41.9, "seconds": 720,
         "format_label": "JPEG"},
        {"warnings": [big.format("39 MB").replace("as JPEG", "as PNG"), "This will take about 11 minutes.", space],
         "file_mb": 38.4, "seconds": 660, "format_label": "PNG"},
    ]
    single = [{"warnings": [big.format("322 MB").replace("JPEG", "PNG"), space], "file_mb": 429.5, "format_label": "PNG"}]
    no_times = [dict(item, seconds=None) for item in items]
    batch, one, untimed = run_app_js(
        f"[m.mergeWarnings({js(items)}), m.mergeWarnings({js(single)}), m.mergeWarnings({js(no_times)})]"
    )
    assert batch == [
        big.format("80 MB in total").replace(" as JPEG", ""),
        "This will take about 23 minutes for all of them.",
        space,
    ]
    assert one == [big.format("430 MB").replace("JPEG", "PNG"), space]  # the readout's own size, with alpha
    assert untimed[1] == "This will take about 12 minutes."


def test_vector_results_say_how_they_were_made():
    app = read("app.js")
    assert "Drawn at full size: it's a vector picture, so no AI was needed." in app
    assert "modeUsed !== 'original' && modeUsed !== VECTOR" in app  # no before/after for a drawn picture
    assert "used.every((mode) => mode === VECTOR)" in app  # the job's tag says "Drawn directly (vector)"
    assert "modeUsed === VECTOR && /vector/i.test(note)" in app  # the server's matching note isn't repeated


def test_about_gives_resvgs_real_licence():
    html = read("index.html")
    row = re.search(r"<tr><td><a [^>]*>resvg-py</a>.*?</tr>", html)
    assert row, "the About page has no resvg row"
    text = re.sub(r"<[^>]+>", "", row[0])
    assert "MIT; Apache-2.0 or MIT" in text and "MPL" not in text


# --- UI verification fixes: header controls, share panel, layout -----------------------------


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_share_panel_shows_the_servers_qr_code_as_a_picture():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 33 33"><path d="M4 4h7"/></svg>\n'
    url = "http://192.168.1.7:7865"
    drawn, older, odd, nothing = run_app_js(
        f"[m.shareQrSource({{qr_svg: {js(svg)}}}, [{js(url)}]), m.shareQrSource({{enabled: true}}, [{js(url)}]),"
        f" m.shareQrSource({{qr_svg: '<script>alert(1)</script>'}}, [{js(url)}]), m.shareQrSource({{}}, [])]"
    )
    # status.share.qr_svg goes in an <img> as a data: URL, so nothing inside it can run.
    assert drawn.startswith("data:image/svg+xml;charset=utf-8,")
    assert urllib.parse.unquote(drawn.split(",", 1)[1]) == svg
    # An older server without qr_svg (or anything that isn't an SVG) draws it at /api/share/qr.svg.
    assert older == odd == "/api/share/qr.svg?url=http%3A%2F%2F192.168.1.7%3A7865"
    assert nothing is None
    app = read("app.js")
    assert "src: qrSource" in app and "innerHTML" not in app


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_share_note_comes_from_the_server_when_it_says():
    mac = "If your Mac asks whether to allow incoming connections, click Allow."
    results = run_app_js(
        f"[m.shareNote({{note: {js(mac)}}}, 'Win32'), m.shareNote({{note: null}}, 'MacIntel'),"
        " m.shareNote({enabled: true}, 'MacIntel'), m.shareNote(null, 'Win32')]"
    )
    assert results == [mac, "", mac, "If Windows asks about the firewall, click Allow access."]


def test_share_switch_keeps_the_whole_share_status():
    app = read("app.js")
    setter = app[app.index("async function setSharing"):app.index("/* ---------- Settings (host only)")]
    # Not just {enabled, urls}: the QR code, note, warning and password state come back too.
    assert "share: { ...data, urls:" in setter
    assert "typeof share.password_set === 'boolean' ? share.password_set : hasPassword(currentSettings())" in app


def test_sharing_on_shows_colour_icon_and_words():
    button = re.search(r'<button[^>]*id="share-button"[^>]*>(.*?)</button>', _header(), flags=re.S)[1]
    assert 'class="share-live"' in button  # the live dot on the Wi-Fi icon
    css = read("app.css")
    assert re.search(r'\.share-button\[data-sharing="on"\] \.share-live \{[^}]*display: block', css)
    assert re.search(r"\.share-live \{[^}]*display: none", css)  # only while sharing
    assert re.search(r"\.share\.is-on \.share-art \{[^}]*var\(--success-tint\)", css)
    assert "el.shareDialog.classList.toggle('is-on', on)" in read("app.js")


def test_header_moves_the_tabs_to_their_own_row_before_it_overflows():
    # Name + four tabs + "Share on Wi-Fi" + theme need about 820 px; below 900 px the tabs wrap.
    css = read("app.css")
    block = css[css.index("@media (max-width: 900px)"):]
    block = block[:block.index("\n}\n")]
    assert ".topbar-inner { flex-wrap: wrap;" in block
    assert re.search(r"\.tabs \{ order: 3; width: 100%;", block)
    assert ".banner { top: 0; position: relative; }" in block  # the taller header no longer covers it


def test_picking_a_size_keeps_the_size_chips_in_step():
    # Without the redraw, pressing a preset that sets the size back to the earlier one left the
    # chip the person clicked still looking chosen.
    app = read("app.js")
    handler = app[app.index("el.sizeChips.addEventListener('change'"):]
    handler = handler[:handler.index("});")]
    assert "state.photo.size = event.target.value;" in handler and "renderSizes();" in handler


def test_the_first_thing_to_do_stands_out():
    css = read("app.css")
    todo = re.search(r'\.step\[data-state="todo"\] \.step-num \{([^}]*)\}', css)
    ready = re.search(r'\.step\[data-state="ready"\] \.step-num \{([^}]*)\}', css)
    assert todo and "background: var(--primary)" in todo[1]
    assert ready and "background" not in ready[1] and "color: var(--primary)" in ready[1]


def test_about_command_line_uses_the_apps_own_python():
    html = read("index.html")
    cli = html[html.index("Use it from the command line"):html.index('id="about-version"')]
    assert ".venv/bin/python -m upscaler image" in cli and ".venv/bin/python -m upscaler video" in cli
    assert ".venv\\Scripts\\python" in cli  # Windows
    assert "uv run" not in cli  # needs uv on PATH and the right --extra, which installs don't set up



# --- Review fixes: big batches, guest keys, focus, honest results -----------------------------


def _function(source: str, name: str) -> str:
    """The text of `function name(...) {...}` in a script (up to the next top-level function)."""
    start = re.search(rf"\n  (?:async )?function {name}\(", source)
    assert start, f"no function {name}"
    rest = source[start.end():]
    end = re.search(r"\n  (?:async )?function |\n  /\* -{5,}", rest)
    return rest[: end.start() if end else len(rest)]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_big_batches_are_sent_in_jobs_the_server_takes():
    # The server takes at most 100 pictures in one job, so a group of 230 goes as 100 + 100 + 30.
    now = 1_000_000_000_000
    day = 24 * 3600 * 1000
    staged = [{"id": f"s{n}", "token": f"t{n}", "file": {}, "checkedAt": now - 1000} for n in range(230)]
    old = [{"id": "old", "token": "gone", "file": {}, "checkedAt": now - day}]  # the server has tidied it away
    files = [{"id": f"f{n}", "token": None, "file": {}} for n in range(3)]
    samples = [{"id": "sample", "token": None, "file": None, "sample": {"id": "food"}}]
    plan = run_app_js(
        f"m.sendingPlan({js(staged + old + files + samples)}, {now})"
        ".map((part) => [part.by, part.items.length, part.items[0].id])"
    )
    assert plan == [
        ["tokens", 100, "s0"], ["tokens", 100, "s100"], ["tokens", 30, "s200"],
        ["files", 4, "old"],
        ["sample", 1, "sample"],
    ]
    assert run_app_js("m.MAX_PICTURES_PER_JOB") == 100
    assert "sendingPlan(" in _function(read("app.js"), "submitPhotos")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_pictures_the_server_no_longer_has_are_sent_again_as_files():
    gone = "That picture isn't here any more. Pictures are kept for 24 hours after you add them, so please add it again."
    results = run_app_js(
        f"[m.stagedGone(400, {js(gone)}), m.stagedGone(400, 'Send at most 100 pictures at once.'),"
        f" m.stagedGone(500, {js(gone)})]"
    )
    assert results == [True, False, False]
    app = read("app.js")
    assert "sendStaged(" in _function(app, "submitPhotos")
    resend = _function(app, "sendStaged")
    assert "stagedGone(" in resend and "uploadFiles(" in resend


def test_the_picture_list_can_be_emptied_in_one_go():
    buttons = [b for b in parse_index().buttons if b["attrs"].get("id") == "photo-clear"]
    assert buttons and "Remove all" in buttons[0]["text"]
    app = read("app.js")
    assert "photoClear: '#photo-clear'" in app
    assert "Remove all" in app and "Keep them" in app  # asks first, like Cancel does


def test_a_guests_own_key_can_be_forgotten_and_the_wizard_says_how_long_it_lasts():
    wizard = read("wizard.js")
    # It isn't true that closing the page forgets it: the server keeps it until it's forgotten or restarts.
    assert "until you close this page" not in wizard
    assert "only kept for this visit" not in wizard
    assert "Forget my key" in wizard
    app = read("app.js")
    assert "sessionKey: (id) =>" in app and "method: 'DELETE'" in _function(app, "forgetOwnKey")
    assert 'id="own-keys"' in read("index.html")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_guests_see_forget_only_for_keys_they_added():
    providers = [
        {"id": "fal", "connected": True},                       # this browser added a key
        {"id": "replicate", "connected": True},                 # the host shares theirs
        {"id": "magnific", "connected": False},                 # added, but gone (Re:Solution restarted)
        {"id": "other", "connected": True, "own_key": True},    # the server says it's the guest's own
        {"id": "shared", "connected": True, "own_key": False},  # the server says it isn't
    ]
    assert run_app_js(
        f"m.ownKeyProviders({js(providers)}, ['fal', 'magnific', 'shared']).map((p) => p.id)"
    ) == ["fal", "other"]


def test_an_open_page_catches_up_after_the_connection_comes_back():
    # After Re:Solution restarts, sharing and a guest's own key are gone: the status is fetched again.
    app = read("app.js")
    connected = _function(app, "setConnected")
    assert "queueStatusRefresh()" in connected and "connectionLost" in connected
    events = _function(app, "connectEvents")
    assert "state.connectionLost = true" in events


def test_redrawing_the_picture_list_keeps_keyboard_focus():
    app = read("app.js")
    render = _function(app, "renderTrayItems")
    assert "focusedControl(" in render and "restoreFocus(" in render
    item = _function(app, "trayItem")
    for spot in ("check", "open"):
        assert f"'data-focus': '{spot}'" in item, spot
    # The details drawer is redrawn as its picture changes; focus stays on the same control.
    details = _function(app, "renderDrawer") + _function(app, "drawerContent") + _function(app, "suggestionBlock")
    assert "data-focus" in _function(app, "renderDrawer")
    for spot in ("remove", "suggestion", "styles"):
        assert f"'data-focus': '{spot}'" in details, spot
    # Preview styles replaces its opener while it's open, so focus goes back to the new one on close.
    assert "el.stylesDialog.addEventListener('close'" in app


def test_other_actions_keep_keyboard_focus():
    app = read("app.js")
    sharing = _function(app, "setSharing")
    assert "el.shareSwitch.focus()" in sharing
    assert "start({ moveFocus: true })" in _function(app, "submitLogin")
    assert "focus" in _function(app, "setPhotorealPart")
    cancel = _function(app, "cancelJob")
    assert "aria-disabled" in cancel and "button.disabled = true" not in cancel
    assert "hadFocus" in _function(app, "buildJobCard")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_before_size_follows_the_picture_each_result_came_from():
    note = "fail-first.png wasn't upscaled: That file isn't a picture."
    failed_first = {"status": "done", "inputs": ["fail-first.png", "good.png"], "results": [{"notes": [note]}]}
    all_good = {"status": "done", "inputs": ["a.png", "b.png"], "results": [{}, {}]}
    same_names = {"status": "done", "inputs": ["a.png", "a.png"], "results": [{"notes": ["a.png wasn't upscaled: x"]}]}
    unknown = {"status": "done", "inputs": ["a.png", "b.png", "c.png"], "results": [{}]}
    stopped = {"status": "cancelled", "inputs": ["a.png", "b.png", "c.png"], "results": [{}]}
    results = run_app_js(
        f"[m.resultInputIndex({js(failed_first)}, 0), m.resultInputIndex({js(all_good)}, 1),"
        f" m.resultInputIndex({js(same_names)}, 0), m.resultInputIndex({js(unknown)}, 0),"
        f" m.resultInputIndex({js(stopped)}, 0)]"
    )
    assert results == [1, 1, None, None, 0]
    assert "resultInputIndex(" in _function(read("app.js"), "sourceSize")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_stopped_batch_isnt_counted_as_failed():
    jobs = [
        {"status": "done", "inputs": ["a"], "results": [{}]},
        {"status": "cancelled", "inputs": ["b", "c", "d"], "results": [{}]},
    ]
    summary = run_app_js(f"m.batchSummary({js(jobs)})")
    assert (summary["done"], summary["failed"], summary["stopped"], summary["finished"]) == (2, 0, 2, True)
    app = read("app.js")
    assert "batchProgressText(" in _function(app, "updateBatchLive")
    assert "stopped" in _function(app, "batchProgressText")
    # Pictures finished before the stop were saved, so the card doesn't say nothing was.
    body = _function(app, "jobBody")
    assert "Nothing was saved" not in body or "inputs" in body
    assert "finished before you stopped" in app


def test_interactive_edges_meet_three_to_one():
    # WCAG 1.4.11: the off switch, text boxes, chips and quality cards need a 3:1 edge.
    pairs = [("edge", bg) for bg in ("page", "surface", "surface-2", "surface-raised")]
    problems = _contrast_problems(pairs, 3.0)
    assert not problems, problems
    css = read("app.css")
    for selector, prop in ((r"\.switch", "background"), (r"\.input", "border"), (r"\.chip", "border"),
                           (r"\.mode", "border")):
        rule = re.search(rf"\n{selector} \{{([^}}]*)\}}", css)
        assert rule and re.search(rf"{prop}:[^;]*var\(--edge\)", rule[1]), selector


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_jpeg_gives_way_when_a_picture_is_too_wide_for_it():
    formats = [{"id": "jpeg"}, {"id": "png"}, {"id": "tiff"}, {"id": "webp", "max_long_edge": 16383}]
    panorama = [[80000, 4000]]  # 4x a 20,000 px panorama
    results = run_app_js(
        "["
        f"m.formatAllowed('jpeg', {js(panorama)}, {js(formats)}),"
        f"m.fitFormat('jpeg', null, {js(panorama)}, {js(formats)}),"
        f"m.fitFormat('webp', null, {js(panorama)}, {js(formats)}),"
        f"m.fitFormat('png', 'jpeg', [[8000, 400]], {js(formats)})"
        "]"
    )
    assert results == [
        False,
        {"format": "png", "blocked": "jpeg"},
        {"format": "png", "blocked": "webp"},
        {"format": "jpeg", "blocked": None},  # fits again: JPEG comes back
    ]


def test_style_crops_open_in_a_tab_instead_of_downloading():
    styles = read("js/styles.js")
    assert "URL.createObjectURL" in styles and "URL.revokeObjectURL" in styles


# --- Task 15: batch-friendly layout (DESIGN "Batch-friendly layout") ------------------------------


def _steps() -> list[dict]:
    """Each <section class="step">: its id, and the attributes of its fold button and summary."""
    html = read("index.html")
    found = []
    for match in re.finditer(r'<section class="step" id="([a-z-]+)"(.*?)</section>', html, flags=re.S):
        step_id, body = match[1], match[2]
        toggle = re.search(r"<button[^>]*data-step-toggle[^>]*>", body)
        found.append({
            "id": step_id,
            "body": body,
            "toggle": toggle[0] if toggle else "",
            "summary": re.search(rf'<p class="step-summary" id="{step_id}-summary" hidden></p>', body) is not None,
            "has_body": f'class="step-body" id="{step_id}-body"' in body,
        })
    return found


def test_every_step_folds_into_a_one_line_summary():
    steps = _steps()
    assert {s["id"] for s in steps} == {
        "photo-step-files", "photo-step-size", "photo-step-mode", "photo-step-file", "video-step-file", "video-step-mode",
    }
    for step in steps:
        assert step["summary"], f"{step['id']} has no one-line summary"
        assert step["has_body"], f"{step['id']} has no body for its button to fold"
        toggle = step["toggle"]
        assert 'aria-expanded="true"' in toggle and f'aria-controls="{step["id"]}-body"' in toggle, step["id"]
    app = read("app.js")
    folded = _function(app, "setStepFolded")
    assert "'Change' : 'Hide'" in folded and "aria-expanded" in folded and ".hidden = folded" in folded
    # Remembered per browser, and read back when the page opens.
    assert "STEPS_KEY = 'resolution.steps'" in app
    assert "localStorage.setItem(STEPS_KEY" in _function(app, "saveSteps")
    assert "parseSteps(window.localStorage.getItem(STEPS_KEY))" in _function(app, "loadSteps")
    assert "applySteps();" in _function(app, "boot")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_folded_steps_are_remembered_and_settle_after_the_first_batch():
    results = run_app_js(
        "[m.parseSteps('{\"photo-step-size\": true, \"photo-step-mode\": false, \"odd\": \"yes\"}'),"
        " m.parseSteps('not json'), m.parseSteps('[1, 2]'), m.parseSteps(null),"
        " m.foldAfterSending({}), m.foldAfterSending({'photo-step-mode': false, 'photo-step-files': true})]"
    )
    assert results[0] == {"photo-step-size": True, "photo-step-mode": False}
    assert results[1] == results[2] == results[3] == {}
    # After the first batch, the settings steps fold, unless the person opened or closed one by hand.
    assert results[4] == {"photo-step-size": True, "photo-step-mode": True, "photo-step-file": True}
    assert results[5] == {"photo-step-mode": False, "photo-step-files": True, "photo-step-size": True, "photo-step-file": True}
    assert "foldAfterSending(state.steps)" in _function(read("app.js"), "submitPhotos")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_folded_step_summaries_read_naturally():
    results = run_app_js(
        "[m.qualitySummary(['Photo-real · Balanced', 'Photo-real · Balanced']),"
        " m.qualitySummary(['Faithful', 'Photo-real · Balanced', 'Photo-real · Balanced', 'Illustration', 'Fast', 'Faithful', 'Photo-real · Balanced']),"
        " m.qualitySummary([]),"
        " m.savedSummary({format: 'JPEG', dpi: 300}),"
        " m.savedSummary({format: 'TIFF', dpi: 600, grain: 'subtle', strength: 80, showStrength: true}),"
        " m.savedSummary({format: 'PNG', dpi: 72, strength: 80, showStrength: false})]"
    )
    assert results == [
        "Photo-real · Balanced",
        "Photo-real · Balanced (3), Faithful (2), +2 more",
        "",
        "JPEG, 300 DPI",
        "TIFF, 600 DPI, subtle grain, 80% detail",
        "PNG, 72 DPI",
    ]


def test_tray_switches_between_grid_and_list_and_folds_to_a_strip():
    html = read("index.html")
    tray = html[html.index('id="photo-tray"'):html.index('id="photo-error"')]
    assert 'role="radiogroup"' in tray and 'name="tray-view" value="grid" checked' in tray and 'name="tray-view" value="list"' in tray
    assert 'id="tray-fold"' in tray and 'aria-controls="tray-body"' in tray and 'id="tray-strip"' in tray
    css = read("app.css")
    assert '.tray[data-view="list"] .item-open' in css and '.tray[data-view="grid"] .item-check' in css
    assert re.search(r"\.tray-strip \{[^}]*overflow-x: auto", css)  # one horizontal strip
    app = read("app.js")
    assert "el.tray.dataset.view = state.tray.view;" in _function(app, "renderTray")
    assert "localStorage.setItem(TRAY_KEY" in _function(app, "saveTray")
    assert "'Show all' : 'Show less'" in _function(app, "renderTray")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_tray_view_is_remembered_and_odd_values_fall_back():
    results = run_app_js(
        "[m.parseTray('{\"view\":\"list\",\"folded\":true}'), m.parseTray('{\"view\":\"cards\",\"folded\":\"yes\"}'),"
        " m.parseTray('}{'), m.parseTray(null)]"
    )
    assert results[0] == {"view": "list", "folded": True}
    assert results[1] == results[2] == results[3] == {"view": "grid", "folded": False}


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_no_more_than_48_pictures_are_drawn_before_show_more():
    results = run_app_js(
        "(() => { const list = Array.from({length: 200}, (_, i) => i);"
        " const first = m.trayPage(list); const second = m.trayPage(list, 96); const all = m.trayPage(list, 240);"
        " const few = m.trayPage(list.slice(0, 30));"
        " return [m.TRAY_PAGE, first.items.length, first.rest, first.items[47], second.items.length, all.rest,"
        "  few.items.length, few.rest, m.moreLabel(152), m.moreLabel(12), m.moreLabel(1), m.moreLabel(0)]; })()"
    )
    assert results == [48, 48, 152, 47, 96, 0, 30, 0, "Show 48 more (152 not shown)", "Show the other 12", "Show the last one", ""]
    app = read("app.js")
    assert "trayPage(list, state.tray.shown)" in _function(app, "renderTrayItems")  # the grid and list
    assert "trayPage(state.photos, TRAY_PAGE)" in _function(app, "renderStrip")  # the folded strip
    assert "trayPage(items, shownCount)" in _function(app, "batchBody")  # a batch's finished pictures
    # Thumbnails load as they scroll into view, and only come from the server's small copies while checking.
    assert "loading: 'lazy'" in _function(app, "trayItem")
    assert "item.status === 'checking') return null" in _function(app, "thumbSource")


def test_drawer_is_a_modal_panel_with_a_title_close_and_neighbours():
    html = read("index.html")
    drawer = html[html.index('<dialog class="drawer" id="photo-drawer"'):]
    drawer = drawer[:drawer.index("</dialog>")]
    assert 'aria-labelledby="drawer-title"' in drawer and 'id="drawer-title" tabindex="-1"' in drawer
    assert "data-close" in drawer and 'aria-label="Close"' in drawer
    assert 'id="drawer-prev"' in drawer and 'id="drawer-next"' in drawer
    app = read("app.js")
    opened = _function(app, "openDrawer")
    assert "el.drawer.showModal()" in opened and "el.drawerTitle.focus(" in opened
    bound = _function(app, "bindDrawer")
    assert "event.key !== 'Tab'" in bound and "trapFocus(" in bound  # Tab wraps around inside it
    assert "trayOpener(id)" in bound  # closing gives focus back to the picture it was opened for
    # Nothing expands inline: a picture's suggestion and why, its own quality, Preview styles and remove.
    content = _function(app, "drawerContent") + _function(app, "suggestionBlock") + _function(app, "qualityBlock")
    for words in ("Suggested", "class: 'why'", "Quality for this picture", "Preview styles", "Remove this picture",
                  "Same as the other pictures", "Use the suggestion: "):
        assert words in content, words
    css = read("app.css")
    assert re.search(r"\.drawer\[open\] \{[^}]*display: flex", css)  # never shown while closed
    assert "@keyframes drawer-in" in css and "@keyframes sheet-in" in css  # side panel, bottom sheet on phones


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_tab_wraps_around_inside_the_drawer():
    # (controls, focused index, Shift held) -> the control to move to, or -1 to let the browser move on.
    cases = [
        ([5, 4, False], 0), ([5, 0, True], 4), ([5, 2, False], -1), ([5, 2, True], -1),
        ([5, -1, False], 0), ([5, -1, True], 4), ([0, -1, False], -1), ([1, 0, False], 0),
    ]
    calls = ",".join(f"m.trapFocus({json.dumps(a)[1:-1]})" for a, _ in cases)
    assert run_app_js(f"[{calls}]") == [expected for _, expected in cases]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_shift_click_selects_a_range():
    order = ["a", "b", "c", "d", "e"]
    results = run_app_js(
        "["
        f"m.rangeSelect([], {js(order)}, null, 'b', true),"
        f"m.rangeSelect(['b'], {js(order)}, 'b', 'e', true),"
        f"m.rangeSelect(['a', 'b', 'c', 'd', 'e'], {js(order)}, 'b', 'd', false),"
        f"m.rangeSelect(['e'], {js(order)}, 'e', 'c', true),"
        f"m.rangeSelect(['x', 'a'], {js(order)}, 'a', 'b', true),"
        f"m.rangeSelect(['a'], {js(order)}, 'gone', 'c', true),"
        f"m.rangeSelect(['a', 'c'], {js(order)}, null, 'c', false)"
        "]"
    )
    assert results == [
        ["b"],
        ["b", "c", "d", "e"],
        ["a", "e"],  # unticking with Shift unticks the range
        ["c", "d", "e"],  # backwards works too
        ["a", "b", "x"],  # a picture a filter hides stays selected
        ["a", "c"],  # the earlier click's picture was removed: just this one
        ["a"],
    ]
    app = read("app.js")
    bound = _function(app, "bindTray")
    assert "event.shiftKey ? state.tray.anchor : null" in bound and "rangeSelect(" in bound
    for words in ("Use suggestions for all", "Remove selected", "Select all"):
        assert words in read("index.html") + app, words
    assert "useSuggestionsInScope" in bound and "removeSelected" in bound


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_filter_chips_count_and_show_the_right_pictures():
    items = [
        {"id": "broken", "status": "problem", "warnings": []},
        {"id": "warned", "status": "ready", "warnings": ["Animated GIF: only the first frame"], "suggestion": {"mode": "photoreal"}},
        {"id": "real", "status": "ready", "warnings": [], "suggestion": {"mode": "photoreal"}},
        {"id": "faces", "status": "ready", "warnings": [], "suggestion": {
            "mode": "faithful", "reasons": ["Faces fill a good part of this picture. Faithful keeps them exactly as they are."]}},
        {"id": "sharp", "status": "ready", "warnings": [], "suggestion": {"mode": "faithful_sharp", "faces": 0, "reasons": []}},
        {"id": "group", "status": "ready", "warnings": [], "suggestion": {"mode": "photoreal", "faces": 3}},
        {"id": "checking", "status": "checking", "warnings": []},
    ]
    choice = "(item) => ({mode: (item.suggestion || {}).mode || 'photoreal'})"
    counts, attention, people, faithful, everything = run_app_js(
        f"[m.filterCounts({js(items)}, {choice}),"
        f" m.filterItems({js(items)}, 'attention', {choice}).map((i) => i.id),"
        f" m.filterItems({js(items)}, 'people', {choice}).map((i) => i.id),"
        f" m.filterItems({js(items)}, 'faithful', {choice}).map((i) => i.id),"
        f" m.filterItems({js(items)}, 'all', {choice}).length]"
    )
    assert counts == {"all": 7, "attention": 2, "photoreal": 3, "faithful": 2, "people": 2}
    assert attention == ["broken", "warned"]
    assert people == ["faces", "group"]  # the server's face count when it sends one, else the reasons
    assert faithful == ["faces", "sharp"]
    assert everything == 7
    for label in ("All", "Needs attention", "Photo-real", "Faithful", "Has people"):
        assert f"'{label}'" in read("app.js"), label


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_sticky_bar_and_tray_add_up_the_batch():
    rows = [
        {"status": "ready", "attention": False, "seconds": 120, "file_mb": 40},
        {"status": "ready", "attention": True, "seconds": 60, "file_mb": 20.5},
        {"status": "problem", "attention": True, "seconds": None, "file_mb": None},
    ]
    cloud = [
        {"status": "ready", "attention": False, "seconds": 30, "file_mb": 10, "cloud": True, "cost": 0.05},
        {"status": "ready", "attention": False, "seconds": 30, "file_mb": 10, "cloud": True, "cost": 0.05},
    ]
    checking = [*rows[:2], {"status": "checking", "attention": False, "seconds": None, "file_mb": None}]
    totals, summary, bar, cloud_totals, cloud_bar, checking_bar = run_app_js(
        f"[m.trayTotals({js(rows)}), m.traySummaryText(m.trayTotals({js(rows)})), m.actionBarTotals(m.trayTotals({js(rows)})),"
        f" m.trayTotals({js(cloud)}), m.actionBarTotals(m.trayTotals({js(cloud)})), m.actionBarTotals(m.trayTotals({js(checking)}))]"
    )
    assert totals == {"count": 2, "problems": 1, "attention": 2, "checking": 0, "seconds": 180, "mb": 60.5, "cloud": 0, "cost": None}
    assert summary == "3 pictures · 2 need attention · about 3 minutes · about 61 MB"
    assert bar == "About 3 minutes · about 61 MB. 1 picture can't be used and is left out."
    assert cloud_totals["cost"] == 0.1
    assert cloud_bar == "About 1 minute · about 20 MB · about $0.10 for 2 cloud pictures"
    assert checking_bar == "Still checking 1 · about 61 MB so far"  # no time until every picture is measured
    css = read("app.css")
    bar_rule = re.search(r"\n\.action-bar \{([^}]*)\}", css)[1]
    assert "position: sticky" in bar_rule and "env(safe-area-inset-bottom)" in bar_rule
    phone = css[css.index("@media (max-width: 640px)"):]
    assert re.search(r"\.action-bar \{[^}]*max\(10px, env\(safe-area-inset-bottom\)\)", phone)
    html = read("index.html")
    bar_html = html[html.index('id="photo-action-bar"'):]
    bar_html = bar_html[:bar_html.index("</form>")]
    assert 'id="photo-count"' in bar_html and 'id="photo-eta"' in bar_html and 'id="photo-start"' in bar_html
    assert "actionBarTotals(trayTotals(trayRows()))" in _function(read("app.js"), "updatePhotoStart")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_results_are_grouped_by_the_batch_they_were_sent_in():
    jobs = [
        {"id": "v1", "kind": "video", "created": 50},
        {"id": "a", "kind": "image", "created": 10},
        {"id": "b", "kind": "image", "created": 11},
        {"id": "c", "kind": "image", "created": 30},
        {"id": "old", "kind": "image", "created": 5},
    ]
    batches = [{"id": "c", "jobIds": ["c"]}, {"id": "a", "jobIds": ["a", "b", "gone"]}]
    groups = run_app_js(f"m.groupJobs({js(jobs)}, {js(batches)}).map((g) => [g.id, g.jobs.map((j) => j.id)])")
    assert groups == [
        ["job-v1", ["v1"]],
        ["batch-c", ["c"]],
        ["batch-a", ["a", "b"]],  # both jobs of one batch (two qualities) share one card
        ["job-old", ["old"]],  # sent before this browser kept batches: a card of its own
    ]
    app = read("app.js")
    assert "groupJobs(visibleJobs(), state.mine.batches)" in _function(app, "renderJobs")
    built = _function(app, "buildBatchCard") + _function(app, "batchBody") + _function(app, "batchActions")
    for words in ("result-grid", "downloadAllControl(", "Clear finished", "batchMosaic("):
        assert words in built, words
    # Finished batches fold, except the newest one.
    assert "!summary.finished || group.id === state.newestGroup" in _function(app, "batchIsOpen")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_batch_progress_counts_pictures_and_time_left():
    running = [
        {"id": "a", "status": "done", "inputs": ["1", "2"], "results": [{}, {}]},
        {"id": "b", "status": "running", "inputs": ["3", "4", "5", "6"], "results": [{}], "progress": 0.5, "eta_seconds": 300},
        {"id": "c", "status": "queued", "inputs": ["7"], "results": []},
    ]
    summary, left, unknown, text = run_app_js(
        f"[m.batchSummary({js(running)}), m.batchSecondsLeft({js(running)}, {{c: 240}}),"
        f" m.batchSecondsLeft({js(running)}, {{}}),"
        f" m.batchProgressText(m.batchSummary({js(running)}), m.batchSecondsLeft({js(running)}, {{c: 240}}))]"
    )
    # A running job's finished pictures count as done straight away.
    assert (summary["photos"], summary["done"], summary["working"], summary["finished"]) == (7, 3, 4, False)
    assert left == 540 and unknown is None
    assert text == "3 of 7 pictures done · about 9 minutes left"
    finished = [{"status": "done", "inputs": ["1"], "results": [{}]}, {"status": "cancelled", "inputs": ["2", "3"], "results": []},
                {"status": "failed", "inputs": ["4"], "results": []}]
    texts = run_app_js(
        f"[m.batchProgressText(m.batchSummary({js(finished)}), null),"
        " m.batchProgressText(m.batchSummary([{status: 'done', inputs: ['1', '2'], results: [{}, {}]}]), null),"
        " m.batchProgressText(m.batchSummary([{status: 'done', inputs: ['1'], results: [{}]}]), null)]"
    )
    assert texts == ["Finished: 1 of 4 pictures done, 1 didn't work, 2 stopped.", "All 2 pictures are done.", "Your picture is done."]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_moves_with_the_arrow_keys_and_closes_with_escape():
    results = run_app_js(
        "[m.viewerTarget(3, 10, 'ArrowRight'), m.viewerTarget(3, 10, 'ArrowLeft'), m.viewerTarget(0, 10, 'ArrowLeft'),"
        " m.viewerTarget(9, 10, 'ArrowRight'), m.viewerTarget(4, 10, 'Home'), m.viewerTarget(4, 10, 'End'),"
        " m.viewerTarget(4, 10, 'Enter'), m.viewerTarget(0, 0, 'ArrowRight')]"
    )
    assert results == [4, 2, 0, 9, 0, 9, None, None]
    # [ and ] always move on (the arrows are for looking around while zoomed in).
    assert run_app_js("[m.viewerTarget(3, 10, '['), m.viewerTarget(3, 10, ']'), m.viewerTarget(0, 10, '['), m.viewerTarget(9, 10, ']')]") == [2, 4, 0, 9]
    html = read("index.html")
    viewer = html[html.index('<dialog class="viewer" id="viewer-dialog"'):]
    viewer = viewer[:viewer.index("</dialog>")]
    assert 'aria-labelledby="viewer-title"' in viewer and "data-close" in viewer  # a <dialog>: Esc closes it
    assert 'aria-label="Previous picture"' in viewer and 'aria-label="Next picture"' in viewer
    module = read("js/viewer.js")
    assert "dialog.addEventListener('keydown'" in module and "target(index, items.length, event.key)" in module
    # The compare slider and a zoomed-in picture keep their own arrow keys.
    assert ".compare-range" in module and ".compare.is-zoomed" in module
    assert "compare.create(" in module  # the close-ups (js/compare.js), when there are no zoomable tiles
    assert "deepzoom.create(" in module  # the whole picture, every pixel (js/deepzoom.js)
    app = read("app.js")
    assert "viewerTarget" in _function(app, "boot") and "viewer.open(viewerItems(group), index" in _function(app, "openViewer")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_pictures_are_checked_three_at_a_time():
    script = (
        f"const m = require({json.dumps(str(WEB / 'app.js'))});"
        "const run = m.limiter(m.INSPECT_AT_ONCE); let now = 0, most = 0; const order = [];"
        "const task = (n) => () => new Promise((resolve) => { now += 1; most = Math.max(most, now);"
        "  setTimeout(() => { now -= 1; order.push(n); resolve(n * 2); }, 5 + (n % 3)); });"
        "Promise.all(Array.from({length: 10}, (_, n) => run(task(n)))).then((values) =>"
        "  process.stdout.write(JSON.stringify({most, values, done: order.length, active: run.active(), waiting: run.waiting()})));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["most"] == 3 and result["done"] == 10 and result["active"] == 0 and result["waiting"] == 0
    assert result["values"] == [n * 2 for n in range(10)]
    assert run_app_js("[m.INSPECT_AT_ONCE, m.checkingText(36, 120), m.checkingText(0, 1), m.checkingText(119, 120), m.checkingText(0, 0)]") == [
        3, "Checking 37 of 120…", "Checking your picture…", "Checking 120 of 120…", "",
    ]
    app = read("app.js")
    assert "inspectPool(" in _function(app, "queueInspect")
    assert "const inspectPool = limiter(INSPECT_AT_ONCE);" in app


def test_pictures_that_cant_be_used_become_needs_attention_items():
    app = read("app.js")
    inspect = _function(app, "inspectPhoto")
    assert "status: 'problem'" in inspect and "dropPhoto(" not in inspect  # stays in the tray, saying why
    assert "status: 'problem'" in _function(app, "addPhotos")
    assert "sendablePhotos()" in _function(app, "submitPhotos")  # and isn't sent
    assert "Checking 37 of 120" in app or "checkingText(done, total)" in _function(app, "renderChecking")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_needs_attention_means_a_problem_a_warning_or_no_preview():
    results = run_app_js(
        "[m.needsAttention({status: 'problem'}), m.needsAttention({status: 'ready', warnings: ['x']}),"
        " m.needsAttention({status: 'ready', warnings: [], previewBroken: true, width: 0}),"
        " m.needsAttention({status: 'ready', warnings: [], previewBroken: true, width: 900}),"
        " m.needsAttention({status: 'checking', warnings: []}), m.needsAttention(null)]"
    )
    assert results == [True, True, True, False, False, False]


# --- Web follow-ups from the whole-system review ------------------------------------------------


def test_guest_password_needs_six_characters_and_says_so():
    page = parse_index()
    field = next(attrs for tag, attrs, _ in page.tags if attrs.get("id") == "new-password")
    assert field.get("minlength") == "6" and field.get("aria-describedby") == "new-password-hint"
    assert re.search(r'id="new-password-hint">Use at least 6 characters', read("index.html"))
    app = read("app.js")
    assert "const MIN_PASSWORD = 6;" in app
    assert "password.length < MIN_PASSWORD" in app and "Use at least ${MIN_PASSWORD} characters." in app


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_jobs_the_server_no_longer_lists_are_dropped():
    known = [{"id": "kept", "seenAt": 100}, {"id": "gone", "seenAt": 100}, {"id": "new", "seenAt": 300}]
    assert run_app_js(f"m.jobsToDrop({js(known)}, ['kept'], 200)") == ["gone"]  # "new" arrived after asking
    refresh = _function(read("app.js"), "refreshJobs")
    assert "jobsToDrop(" in refresh and "state.jobs.delete(id)" in refresh


def test_about_links_ffmpegs_own_licence_page():
    html = read("index.html")
    row = re.search(r"<tr><td><a [^>]*>FFmpeg</a>.*?</tr>", html)
    assert row, "the About page has no FFmpeg row"
    assert '<a href="https://ffmpeg.org/legal.html" target="_blank" rel="noopener noreferrer">GPL (bundled FFmpeg build)</a>' in row[0]


def test_video_results_say_how_to_play_them_on_windows():
    app = read("app.js")
    note = "If Windows can't play it, install the free HEVC Video Extensions from the Microsoft Store."
    assert f'WINDOWS_HEVC_NOTE = "{note}"' in app
    assert "isVideo ? h('p', { class: 'result-hint' }, icon('info'), WINDOWS_HEVC_NOTE)" in _function(app, "resultItem")


def test_estimates_tell_the_server_about_vector_pictures():
    request = _function(read("app.js"), "estimateRequest")
    assert "is_vector: Boolean(photo.isVector)" in request


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_cloud_cost_is_shown_per_picture_and_for_the_batch():
    results = run_app_js(
        "[m.cloudCost({cost_usd: 0.05}, [8192, 6144], 0.16), m.cloudCost(null, [8192, 6144], 0.16),"
        " m.cloudCost({}, [4096, 3072], 0.16), m.cloudCost(null, [4096, 3072], null), m.cloudCost(null, null, 0.16),"
        " m.formatUsd(0.05), m.formatUsd(1.2), m.formatUsd(0.008), m.formatUsd(NaN)]"
    )
    assert results == [0.05, 0.16, 0.04, None, None, "$0.05", "$1.20", "$0.008", ""]
    app = read("app.js")
    fact = _function(app, "cloudCostFact")
    assert "cloudCost(row.est, row.est.out, mode && mode.cost)" in fact and "pictures × about" in fact
    assert "cloudCostFact(rows)" in _function(app, "renderEstimate")


def test_a_cancelled_batch_shows_what_was_finished():
    app = read("app.js")
    # Stopped jobs keep the pictures finished before the stop: shown in the batch card and downloadable.
    assert "job.status === 'cancelled'" in _function(app, "batchBody") and "stoppedText(job)" in _function(app, "batchBody")
    assert "(job.status === 'done' || job.status === 'cancelled') && (job.results || []).length" in _function(app, "batchActions")
    assert "finished before you stopped" in _function(app, "stoppedText")
    # The Before size comes from the server's source_width/height when it sends them.
    assert "result.source_width" in _function(app, "sourceSize")


# --- found in the 60-picture browser check (Task 15) ---------------------------


def test_tray_summary_never_prints_null():
    # replaceChildren writes a null out as the word "null": "60 picturesnull" while every picture was
    # still being checked (nothing but the count to show).
    tray = _function(read("app.js"), "renderTray")
    assert "...(rest.length ? [` · ${rest.join(' · ')}`] : [])" in tray
    assert ": null);" not in tray


def test_retry_button_for_failed_jobs_that_kept_their_work():
    app = read("app.js")
    assert "retry: (id) => `/api/jobs/${enc(id)}/retry`," in app
    can = _function(app, "canRetry")
    assert "job.retryable === true" in can and "job.status === 'failed'" in can and "canCancel(job)" in can
    controls = _function(app, "retryControls")
    assert "api(API.retry(job.id), { method: 'POST' })" in controls and "upsertJob(" in controls
    # 403 and 409 come back with the server's own words, shown as they are.
    assert "h('p', { class: 'field-error', role: 'alert' }, err.message)" in controls
    assert "canRetry(job) ? retryControls(job) : null" in _function(app, "jobBody")
    assert "if (canRetry(job)) body.append(retryControls(job));" in _function(app, "batchBody")


def test_cancelled_jobs_show_the_servers_reason_first():
    app = read("app.js")
    stopped = _function(app, "stoppedText")
    # A cloud job interrupted by a restart (never sent again) or a guest's forgotten key says why.
    assert stopped.index("if (job.error) return job.error;") < stopped.index("finished before you stopped")
    # A finished batch that made nothing stays open, so that reason isn't folded away.
    assert "(!summary.done && group.jobs.some((job) => Boolean(job.error)))" in _function(app, "batchIsOpen")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_cloud_pictures_paid_from_plan_credits_say_so():
    results = run_app_js(
        "[m.cloudUsesCredits({cost_usd: null}, 0.16), m.cloudUsesCredits({cost_usd: 0.05}, 0), m.cloudUsesCredits(null, 0),"
        " m.cloudUsesCredits(null, 0.16), m.cloudUsesCredits({out: [1, 1]}, 0),"
        " m.cloudCostText(null, true), m.cloudCostText(0, false), m.cloudCostText(0.05, false), m.cloudCostText(null, false)]"
    )
    assert results == [True, False, True, False, True,
                       "Uses your plan's credits", "Nothing to pay", "about $0.05", "Price not published"]
    app = read("app.js")
    assert 'if (usd === 0) return "Uses your plan\'s credits";' in _function(app, "formatCost")
    fact = _function(app, "cloudCostFact")
    assert "entries.every((e) => e.credits)" in fact and "Uses your plan's credits" in fact
    assert "cloudCostText(cost, cloudUsesCredits(est, mode && mode.cost))" in _function(app, "drawerContent")


def test_a_batch_stopped_part_way_says_stopped_not_done():
    app = read("app.js")
    status = _function(app, "batchStatus")
    assert "if (summary.stopped) return 'cancelled';" in status
    assert status.index("summary.stopped) return 'cancelled'") < status.index("return 'done'")
    assert "statusChipFor(status, status === 'cancelled' ? 'Stopped' : null)" in _function(app, "buildBatchCard")
    assert "label || STATUS_TEXT[status] || status" in _function(app, "statusChipFor")


def test_sticky_action_bar_slides_under_the_top_bar():
    css = read("app.css")
    topbar = re.search(r"\n\.topbar \{([^}]*)\}", css)[1]
    bar = re.search(r"\n\.action-bar \{([^}]*)\}", css)[1]
    assert "z-index: var(--z-sticky);" in topbar
    # Same layer, later on the page, would paint the bar over the header once it settles and scrolls up.
    assert "z-index: calc(var(--z-sticky) - 1);" in bar


def test_viewer_keeps_a_lone_picture_in_the_middle():
    # A batch of one hides the arrows; the picture must keep the middle column instead of sliding left.
    css = read("app.css")
    assert ".viewer-main > .viewer-stage { grid-column: 2; }" in css
    # On a phone the arrows float over the picture and the stage is the only column.
    assert ("  .viewer-main { grid-template-columns: minmax(0, 1fr); padding: 10px; }\n"
            "  .viewer-main > .viewer-stage { grid-column: 1; }") in css


def test_viewer_fits_a_whole_picture_on_screen():
    css = read("app.css")
    whole = re.search(r"\n\.viewer-whole \{([^}]*)\}", css)[1]
    img = re.search(r"\n\.viewer-whole img \{([^}]*)\}", css)[1]
    # A grid row grows to the picture's height, so a tall vector picture ran under the info bar.
    assert "display: flex" in whole and "height: calc(100dvh - 230px)" in whole
    assert "max-height: 100%" in img and "max-width: 100%" in img and "object-fit: contain" in img


def test_preview_styles_starts_on_its_heading():
    styles = read("js/styles.js")
    opening = styles[styles.index("    function open(item) {"):styles.index("    function routeFor(choice) {")]
    assert re.search(r"dialog\.showModal\(\);\s*//[^\n]*\n\s*focusHeading\(\);", opening)
    assert ".styles-body h3:focus { outline: none; }" in read("app.css")



# --- The deep-zoom compare viewer (DESIGN "Deep-zoom compare viewer") ----------------------------


def run_module(name: str, expression: str):
    """Evaluate `expression` with web/<name>'s exported helpers bound to `m`; return the JSON result."""
    script = (
        f"const m = require({json.dumps(str(WEB / name))});"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_fit_shows_the_whole_picture_and_100_percent_is_actual_pixels():
    fit, tall, level_fit, level_100, level_200, level_mid, label = run_module("js/deepzoom.js", (
        "[m.fitScale(8192, 6144, 2400, 1140), m.fitScale(1000, 4000, 2400, 1140),"
        " m.levelFor(0.186, 4), m.levelFor(1, 4), m.levelFor(2, 4), m.levelFor(0.7, 4),"
        " [m.zoomLabel(0.186, true), m.zoomLabel(1, false), m.zoomLabel(2, false), m.zoomLabel(0.374, false)]]"
    ))
    assert fit == pytest.approx(1140 / 6144) and tall == pytest.approx(1140 / 4000)  # the whole picture, letterboxed
    # The level drawn keeps at least one of its pixels per screen pixel, so every zoom is sharp.
    assert (level_fit, level_100, level_200, level_mid) == (2, 0, 0, 0)
    assert label == ["Fit · 19%", "100% · actual pixels", "200%", "37%"]
    deepzoom = read("js/deepzoom.js")
    # Scale is screen pixels per picture pixel: the canvas has devicePixelRatio times as many pixels as CSS pixels.
    assert "window.devicePixelRatio" in deepzoom and "pane.w = Math.max(1, Math.round(pane.el.clientWidth * state.dpr))" in deepzoom
    # Sized from the layout, not getBoundingClientRect: the viewer opens with a brief scale(0.985), and a
    # canvas sized from that shrunken box never filled its pane (Fit was off-centre, with a gap right and below).
    measure = _function_js(deepzoom, "measure")
    assert ".getBoundingClientRect()" not in measure
    assert "const box = { width: stage.clientWidth, height: stage.clientHeight };" in measure
    assert "pane.h = Math.max(1, Math.round(pane.el.clientHeight * state.dpr))" in measure
    assert "pane.canvas.style.width = `${pane.w / state.dpr}px`" in deepzoom
    assert "Math.abs(s - Math.round(s)) < 1e-6" in deepzoom  # 100%, 200%: picture pixels on whole screen pixels


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_zoom_keeps_the_point_under_the_cursor_and_the_picture_on_screen():
    zoomed, spot_before, spot_after, clamped, centred = run_module("js/deepzoom.js", (
        "(() => { const v = {scale: 0.25, cx: 4096, cy: 3072};"
        " const z = m.zoomAround(v, 1, 300, 200, 2000, 1000);"
        " return [z, m.pictureAt(v, 300, 200, 2000, 1000), m.pictureAt(z, 300, 200, 2000, 1000),"
        "  m.clampView({scale: 1, cx: -500, cy: 99999}, 8192, 6144, 2000, 1000),"
        "  m.clampView({scale: 0.1, cx: 10, cy: 10}, 8192, 6144, 2000, 1000)]; })()"
    ))
    assert zoomed["scale"] == 1
    assert spot_after["x"] == pytest.approx(spot_before["x"]) and spot_after["y"] == pytest.approx(spot_before["y"])
    assert clamped == {"scale": 1, "cx": 1000, "cy": 6144 - 500}  # edge to edge, never past the picture
    assert centred == {"scale": 0.1, "cx": 4096, "cy": 3072}  # smaller than the view: kept in the middle


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_only_the_tiles_in_view_are_loaded_middle_first():
    level0 = {"level": 0, "width": 8192, "height": 6144, "cols": 16, "rows": 12}
    level3 = {"level": 3, "width": 1024, "height": 768, "cols": 2, "rows": 2}
    near, whole, corner = run_module("js/deepzoom.js", (
        f"[m.visibleTiles({{scale: 1, cx: 4096, cy: 3072}}, {json.dumps(level0)}, 8192, 6144, 512, 1024, 1024),"
        f" m.visibleTiles({{scale: 0.125, cx: 4096, cy: 3072}}, {json.dumps(level3)}, 8192, 6144, 512, 1024, 768),"
        f" m.visibleTiles({{scale: 1, cx: 300, cy: 300}}, {json.dumps(level0)}, 8192, 6144, 512, 600, 600)]"
    ))
    assert sorted((t["col"], t["row"]) for t in near) == [(7, 5), (7, 6), (8, 5), (8, 6)]
    assert len(whole) == 4
    assert corner[0] == {"col": 0, "row": 0} and len(corner) == 4  # nearest the middle of the view first
    deepzoom = read("js/deepzoom.js")
    # Tiles that leave the view before they arrive are cancelled; a soft preview (the smallest level) is underneath.
    assert "new AbortController()" in deepzoom and "entry.ctrl.abort()" in deepzoom
    assert "for (let lv = state.levels.length - 1; lv >= target; lv -= 1)" in deepzoom
    assert "IN_FLIGHT" in deepzoom and "evict(side)" in deepzoom


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_compare_mode_is_remembered_and_side_by_side_picks_its_layout():
    result = run_module("js/deepzoom.js", (
        "(() => { const store = {}; const s = {getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; }};"
        " const broken = {getItem() { throw new Error('private'); }, setItem() { throw new Error('private'); }};"
        " const first = m.readMode(s); m.saveMode(s, 'side'); m.saveMode(broken, 'side');"
        " return [first, m.readMode(s), m.readMode(broken), store[m.MODE_KEY], m.MODE_KEY,"
        "  m.sideLayout(4000, 3000, 1200, 600), m.sideLayout(4000, 1000, 400, 800)]; })()"
    ))
    assert result == ["slider", "side", "slider", "side", "resolution.compareMode", "columns", "rows"]


def test_deep_zoom_controls_and_gestures():
    deepzoom = read("js/deepzoom.js")
    # Fit, 100% and 200%, free zoom with the wheel, a trackpad pinch (ctrl+wheel, or Safari's gestures) and touch.
    for words in ("'Fit'", "'100%'", "'200%'", "'wheel'", "event.ctrlKey", "'gesturechange'", "kind: 'pinch'",
                  "'dblclick'", "DOUBLE_TAP_MS", "toggleAt(", "{ passive: false }"):
        assert words in deepzoom, words
    # Keys: + − 0 1 2, the arrows to look around; js/viewer.js asks the zoom viewer first.
    for key in ("'+'", "'-'", "'0'", "'1'", "'2'", "key.startsWith('Arrow')"):
        assert key in deepzoom, key
    assert "current.handleKey(event)" in read("js/viewer.js")
    # The divider is a keyboard-accessible slider; a click or tap moves it; side by side shares one view.
    for words in ("role: 'slider'", "'aria-valuenow'", "aria-valuetext", "'Home'", "'End'", "setPos(posFromClient(event.clientX)); // click or tap"):
        assert words in deepzoom, words
    assert "['slider', 'arrows-h', 'Slider'], ['side', 'columns', 'Side by side']" in deepzoom
    # The overview map hides at Fit and can be dragged.
    assert "pannable() && !state.fitted" in deepzoom and "moveToMap(event)" in deepzoom
    css = read("app.css")
    assert re.search(r"\.zv-stage \{[^}]*touch-action: none", css)  # pinch and two-finger pan reach the viewer
    assert re.search(r"\.zv-stage\.is-pannable \{ cursor: grab; \}", css) and ".zv-stage.is-panning { cursor: grabbing; }" in css
    assert ".zv-minimap.is-shown" in css


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_labels_say_the_sizes():
    labels = run_app_js("[m.compareLabels([1024, 795], [3840, 2981]), m.compareLabels(null, [3840, 2981])]")
    assert labels[0] == {"before": "Before · 1,024 × 795, simply enlarged", "after": "After · 3,840 × 2,981", "single": "Upscaled · 3,840 × 2,981"}
    assert labels[1]["before"] == "Before · simply enlarged"
    items = _function(read("app.js"), "viewerItems")
    assert "manifest: API.viewerInfo(job.id, n)" in items and "API.beforeTile : API.tile" in items
    assert "open: { href: API.view(job.id, n), label: 'Open in new tab' }" in items
    assert "download: { href: API.file(job.id, n), name }" in items


def test_viewer_falls_back_to_the_close_ups_when_there_are_no_tiles():
    viewer = read("js/viewer.js")
    assert "onUnavailable: () =>" in viewer and "fallbackView(item)" in viewer
    deepzoom = read("js/deepzoom.js")
    assert "manifest.unavailable" in deepzoom and "options.onUnavailable()" in deepzoom
    # Without the original (results made before originals were kept), only the upscaled side is shown, with a word why.
    assert "manifest.before === false" in deepzoom and "only the upscaled picture is shown" in deepzoom


def test_viewer_keeps_focus_inside_and_moves_it_with_the_picture():
    viewer = read("js/viewer.js")
    assert "if (event.key === 'Tab')" in viewer and "trapTab(event)" in viewer
    assert "current.focus()" in viewer  # [ and ] replace the focused picture: the new one gets focus
    assert "trap: trapFocus" in _function(read("app.js"), "boot")


def test_viewers_respect_reduced_motion():
    css = read("app.css")
    motion = css[css.index("@media (prefers-reduced-motion: reduce)"):]
    assert ".zv-loading-dot, .vc-buffering { animation: none; }" in motion
    assert ".vc-notice { animation: none; }" in motion
    assert "reducedMotion()" in read("js/deepzoom.js")
    # The video zoom changes at once (animating the size of two 4K videos costs a layout every frame).
    assert "is-animating" not in read("js/videocompare.js") and ".vc-stage.is-animating" not in css


def test_viewers_fit_a_phone():
    css = read("app.css")
    phone = css[css.index("/* The compare viewers on phones and narrow windows. */"):]
    phone = phone[:phone.index("/* ---------- Motion preferences")]
    assert "@media (max-width: 600px)" in phone
    assert ".zv-modes .zv-mode-word" in phone  # the mode buttons keep their icons; the words stay for screen readers
    assert ".vc-hint { display: none; }" in phone


# --- Side-by-side video compare (DESIGN "Side-by-side video compare") ----------------------------


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_video_timing_helpers():
    rates, frames, clocks, drifts, layouts, sizes = run_module("js/videocompare.js", (
        "[[m.parseRate('30000/1001'), m.parseRate('25/1'), m.parseRate(24), m.parseRate('nope')],"
        " [m.frameAt(m.timeOfFrame(0, 25), 25), m.frameAt(m.timeOfFrame(41, 25), 25), m.frameAt(1.0, 25), m.frameAt(0.999, 25)],"
        " [m.formatClock(3.24), m.formatClock(65), m.formatClock(3723.5), m.formatClock(NaN)],"
        " [m.driftAction(0.02, 30, 1), m.driftAction(0.1, 30, 1), m.driftAction(-0.1, 30, 2), m.driftAction(0.9, 30, 1)],"
        " [m.videoLayout(3840, 2160, 1200, 600), m.videoLayout(3840, 2160, 390, 700), m.videoLayout(2160, 3840, 1200, 700)],"
        " [m.displaySize('1', {width: 3840, height: 2160}, {width: 600, height: 400}, 2),"
        "  m.displaySize('2', {width: 3840, height: 2160}, {width: 600, height: 400}, 2),"
        "  m.displaySize('fit', {width: 3840, height: 2160}, {width: 600, height: 400}, 2)]]"
    ))
    assert rates[0] == pytest.approx(29.97, abs=0.01) and rates[1:] == [25, 24, 0]
    assert frames == [0, 41, 25, 24]
    assert clocks == ["0:03.2", "1:05.0", "1:02:03.5", "0:00.0"]
    within, ahead, behind, far = drifts
    assert within == {"seek": False, "rate": 1}  # within a frame: leave it
    assert not ahead["seek"] and ahead["rate"] < 1  # the original is ahead: it slows down a little
    assert not behind["seek"] and behind["rate"] > 2  # behind: it speeds up a little
    assert far["seek"] is True  # more than half a second out: it jumps
    assert layouts == ["columns", "rows", "columns"]
    one, two, fit = sizes
    assert one == {"width": 1920, "height": 1080} and two == {"width": 3840, "height": 2160}  # 1 and 2 video pixels per screen pixel
    assert fit["width"] == pytest.approx(600) and fit["height"] == pytest.approx(337.5)


def test_video_compare_is_two_frames_with_one_control_bar_and_no_slider():
    html = read("index.html")
    dialog = html[html.index('<dialog class="viewer viewer-video" id="video-compare-dialog"'):]
    dialog = dialog[:dialog.index("</dialog>")]
    assert 'aria-labelledby="video-compare-title"' in dialog and "data-close" in dialog
    assert 'data-vc="stage"' in dialog and 'data-vc="actions"' in dialog
    module = read("js/videocompare.js")
    assert "ReCompare" not in module and "compare-range" not in module  # no before/after slider for videos
    for words in ("label('Original', sizes.original)", "label('Upscaled', sizes.upscaled)", "'Play (Space)'",
                  "'Previous frame (,)'", "'Next frame (.)'", "'Timeline'", "'Playback speed'", "'Loop'",
                  "'Mute (M)'", "'This video has no sound'", "{ id: 'fit', label: 'Fit' }", "{ id: '1', label: '100%' }",
                  "{ id: '2', label: '200%' }", "'Preparing the comparison… '"):
        assert words in module, words
    assert run_module("js/videocompare.js", "m.SPEEDS") == [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
    # Kept in step: the upscaled player leads, drift over half a frame is corrected at most every 250 ms (so
    # the two show the same frame, not 55 next to 56), both wait while either loads.
    assert "const leader = upscaled;" in module and "const follower = original;" in module
    assert "const SYNC_MS = 250;" in module and "ps.interval(sync, SYNC_MS)" in module
    assert "driftAction(drift, fps, rate, correcting ? frameTime / 4 : frameTime / 2)" in module
    assert "ps.listen(node, 'waiting'" in module and "startLoading()" in module
    assert "follower.muted = true; // the original never plays sound" in module
    # One drag pans both; the keys: Space, , and . (and J, K, L).
    assert "for (const node of [original, upscaled])" in module and "pan.x -=" in module
    for key in ("key === ' '", "key === ','", "key === '.'", "key === 'j'", "key === 'k'", "key === 'l'"):
        assert key in module, key
    app = read("app.js")
    assert "Compare with the original" in _function(app, "videoCompareButton")
    assert "isVideo && videoCompare ? videoCompareButton(job, result, n, name) : null" in _function(app, "resultItem")
    opener = _function(app, "openVideoCompare")
    assert "startUrl: API.compare(job.id, n)" in opener and "cancelUrl: API.compareCancel(job.id, n)" in opener
    css = read("app.css")
    assert '.vc-stage[data-layout="rows"]' in css  # one above the other when that shows them bigger (phones)


# --- Responsive, freeze-free viewers (user report: "video playback freezes, close button does not work at times") ---


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_lifecycle_scope_stops_everything_it_started_once():
    script = (
        f"const life = require({json.dumps(str(WEB / 'js' / 'lifecycle.js'))});"
        "const scope = life.createScope('test'); const fired = []; const target = new EventTarget();"
        "scope.timeout(() => fired.push('timeout'), 5); scope.interval(() => fired.push('interval'), 5);"
        "scope.frame(() => fired.push('frame'));"
        "scope.listen(target, 'ping', () => fired.push('listener'));"
        "const observer = {disconnected: 0, disconnect() { this.disconnected += 1; }}; scope.observe(observer);"
        "const ctrl = scope.controller();"
        "const video = {log: [], paused: false, pause() { this.log.push('pause'); }, removeAttribute(n) { this.log.push('remove ' + n); },"
        " querySelectorAll() { return []; }, load() { this.log.push('load'); },"
        " requestVideoFrameCallback() { return 7; }, cancelVideoFrameCallback(id) { this.log.push('cancel ' + id); }};"
        "scope.video(video); scope.videoFrame(video, () => fired.push('videoFrame'));"
        "scope.add(() => { throw new Error('one clean-up fails'); }); scope.add(() => fired.push('cleanup'));"
        "const during = life.counts(); const warn = console.warn; console.warn = () => {};"
        "scope.dispose(); scope.dispose(); console.warn = warn;"
        "target.dispatchEvent(new Event('ping')); const late = scope.timeout(() => fired.push('late'), 1);"
        "setTimeout(() => process.stdout.write(JSON.stringify({during, after: life.counts(), fired, video: video.log,"
        " aborted: ctrl.signal.aborted, disconnected: observer.disconnected, late, disposed: scope.disposed,"
        " abort: [life.isAbort({name: 'AbortError'}), life.isAbort(new Error('x'))]})), 40);"
    )
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["during"] == {"scopes": 1, "timeouts": 1, "intervals": 1, "frames": 1, "videoFrames": 1,
                                "listeners": 1, "observers": 1, "requests": 1, "videos": 1}
    assert result["after"] == {key: 0 for key in result["during"]}  # everything stopped, counted once
    assert result["fired"] == ["cleanup"]  # no timer, frame or listener ran after dispose; a failing clean-up didn't stop the rest
    assert result["video"] == ["cancel 7", "pause", "remove src", "load"]  # paused and let go of its file
    assert result["aborted"] and result["disconnected"] == 1 and result["late"] == 0 and result["disposed"]
    assert result["abort"] == [True, False]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_video_sync_helpers_for_play_refusals_loops_and_settling():
    failures, wraps, settle = run_module("js/videocompare.js", (
        "[[m.playFailure({name: 'AbortError'}), m.playFailure({name: 'NotAllowedError'}), m.playFailure(new Error('x'))],"
        " [m.loopDrift(4.9, 5, true), m.loopDrift(-4.9, 5, true), m.loopDrift(0.02, 5, true), m.loopDrift(4.9, 5, false)],"
        " [m.driftAction(0.02, 30, 1), m.driftAction(0.02, 30, 1, 1 / 120), m.SYNC_MS, m.LOADING_SLOW_MS]]"
    ))
    # Our own pause interrupting play() isn't a failure (it used to flip Play back to paused).
    assert failures == ["interrupted", "blocked", "failed"]
    assert wraps[0] == pytest.approx(-0.1) and wraps[1] == pytest.approx(0.1) and wraps[2:] == [0.02, 4.9]
    assert settle[0] == {"seek": False, "rate": 1}  # within a frame: left alone...
    assert settle[1]["rate"] < 1  # ...unless it is already being brought back, which goes on to a quarter frame
    assert settle[2] == 250 and settle[3] >= 5000


def test_closing_a_viewer_is_one_teardown_from_any_state():
    for name in ("js/viewer.js", "js/videocompare.js"):
        module = read(name)
        # Close, Esc and a click on the backdrop all close straight away; the 'close' event tears down too.
        assert "if (event.key === 'Escape') {\n        event.preventDefault();\n        close();" in module, name
        assert "dialog.querySelectorAll('[data-close]')) button.addEventListener('click', close)" in module, name
        assert "dialog.addEventListener('cancel'," in module, name
        assert "event.clientX < box.left || event.clientX > box.right" in module, name  # outside the box only
        assert "back.focus({ preventScroll: true })" in module, name  # focus goes back to what opened it
    viewer = read("js/viewer.js")
    assert "function clearStage()" in viewer and "for (const shown of [current, closeUps])" in viewer
    video = read("js/videocompare.js")
    assert "function teardown()" in video and "old.destroy();" in video
    # Destroying is idempotent and ends everything: one scope per comparison, one per player.
    assert "if (scope.disposed) return;" in video and "destroy: () => ps.dispose()" in video
    assert "life.createScope('video-compare')" in video and "life.createScope('video-player')" in video
    deepzoom = read("js/deepzoom.js")
    assert "life.createScope('deep-zoom')" in deepzoom and "scope.dispose();" in _function_js(deepzoom, "destroy")
    assert "if (state.destroyed) return;" in _function_js(deepzoom, "destroy")
    # Nothing the viewers start bypasses their scope.
    for name in ("js/videocompare.js", "js/deepzoom.js"):
        module = read(name)
        for raw in ("window.setTimeout(", "window.setInterval(", "window.requestAnimationFrame(", "window.addEventListener("):
            assert raw not in module, f"{name} uses {raw} outside its lifecycle scope"


def _function_js(source: str, name: str) -> str:
    """The body of a nested `function name(...) {` in a viewer module (to its closing brace)."""
    start = re.search(rf"\n([ \t]*)function {name}\(", source)
    assert start, f"no function {name}"
    indent = start.group(1)
    rest = source[start.end():]
    end = rest.index(f"\n{indent}}}")
    return rest[:end]


def test_video_player_redraws_once_per_frame_and_never_loses_a_click():
    video = read("js/videocompare.js")
    # One redraw loop: it can't be started twice, and each frame schedules exactly one more (the old
    # loop scheduled two per frame, so after a second of playback thousands ran every frame).
    start = _function_js(video, "startLoop")
    assert "if (loop || ps.disposed) return;" in start and "ps.videoFrame(leader, onFrame)" in start
    frame = _function_js(video, "onFrame")
    assert frame.split("{", 1)[1].strip().startswith("loop = 0;") and frame.count("ps.videoFrame(") == 1
    # Only what changed is redrawn; numbers use one Intl.NumberFormat (toLocaleString built a new one per call).
    assert "if (n === lastFrame && !force) return;" in _function_js(video, "draw")
    assert "const numberFormat = new Intl.NumberFormat();" in video and ".toLocaleString(" not in video.replace("toLocaleString() builds", "")
    # The play button keeps its icon element (swapping it mid-click lost the click) and icons never take clicks.
    assert "playIcon.firstChild.setAttribute('href'" in video and "playButton.replaceChildren" not in video
    assert "button .icon, a .icon, label .icon { pointer-events: none; }" in read("app.css")


def test_video_seeks_are_coalesced_and_play_refusals_handled():
    video = read("js/videocompare.js")
    pump = _function_js(video, "pumpSeek")
    assert "if (seekTarget == null || seeking()) return;" in pump  # one seek in flight; only the newest target waits
    assert "SEEK_WATCHDOG_MS" in pump
    assert "playFailure(err) === 'interrupted'" in _function_js(video, "start")
    # Never a drift correction while seeking, loading, dragging or starting.
    sync = _function_js(video, "sync")
    assert "if (!wantPlay || held || scrub || broken || startWait || seekTarget != null || seeking()) return;" in sync
    # Waiting for data: both pause, "Loading…" shows, and "Try again" after a while; errors offer it too.
    for words in ("const LOADING = 'Loading…';", "const SLOW = 'This is taking longer than usual.';", "showNotice(SLOW, true)",
                  "ps.listen(node, 'stalled'", "ps.listen(node, 'error', fail)", "showNotice(BROKEN, true)"):
        assert words in video, words
    # Scrubbing holds the pointer, pauses while dragging and carries on after.
    for words in ("timeline.setPointerCapture(event.pointerId)", "['pointerup', 'pointercancel', 'lostpointercapture']"):
        assert words in video, words
    assert "follower.playbackRate = rate;" in video  # the speed is mirrored
    # The page's own play/pause events are counted, so one arriving late (after more presses) isn't
    # mistaken for a media key and doesn't undo the newest press.
    assert "if (!leader.paused) ownPauses += 1;" in _function_js(video, "stopBoth")
    assert "if (node === leader && leader.paused) ownPlays += 1;" in _function_js(video, "start")
    # Both set off together (the original once the upscaled one's clock moves), and Loop restarts both.
    assert "startTogether()" in _function_js(video, "resume") and "leader.loop" not in video
    assert "if (looping && wantPlay) {" in video
    css = read("app.css")
    assert ".vc-notice {" in css and ".vc-notice.has-action" in css
    # On a short window (a phone on its side) the picture and the videos keep some room; the viewer scrolls.
    assert "grid-template-rows: auto minmax(240px, 1fr) auto; overflow-y: auto;" in css
    assert "grid-template-rows: minmax(180px, 1fr) auto auto;" in css


def test_finished_video_cards_play_the_browser_friendly_copy_one_at_a_time():
    app = read("app.js")
    preview = _function(app, "videoPreview")
    assert "API.poster(job.id, n)" in preview and "API.file" not in preview  # a still, not the 10-bit HEVC file
    player = _function(app, "playCardVideo")
    assert "preload: 'metadata'" in player and "attach(st.upscaled_url)" in player and "scope.video(video)" in player
    assert "stopCardVideo();" in player  # one card at a time
    for name in ("openVideoCompare", "openViewer", "showView"):
        assert "stopCardVideo()" in _function(app, name), name
    assert "releaseCardVideo(card)" in _function(app, "buildJobCard")
    assert "if (signal) init.signal = signal;" in _function(app, "api")  # requests a viewer started can be stopped
    assert "window.__resolutionDebug = Object.assign(" in _function(app, "boot")
    assert "window.__resolutionDebug = Object.assign(window.__resolutionDebug || {}, { counts });" in read("js/lifecycle.js")
    css = read("app.css")
    assert ".result-video { position: relative; display: grid; grid-template: minmax(0, 1fr) / minmax(0, 1fr);" in css


# --- Fixes from the 60-picture check in the browser (Task 15 verification) ---------------------------


def test_finished_videos_can_be_cleared_from_the_list_too():
    app = read("app.js")
    # "Clear all finished" takes every finished card, videos included, and shows once there are two.
    bind = _function(app, "bindResults")
    assert "filter((g) => !g.jobs.some(isActive))" in bind and "groupKind(g) === 'batch'" not in bind
    render = _function(app, "renderJobs")
    assert "const finished = groups.filter((g) => !g.jobs.some(isActive));" in render
    # Each finished video (or stopped or failed job) has its own "Clear from list", per browser like a batch's.
    clear = _function(app, "clearJobButton")
    assert "'Clear from list'" in clear and "clearGroups([{ id: `job-${job.id}`, jobs: [job] }], button)" in clear
    assert "clearJobButton(job)" in _function(app, "resultsBlock")
    assert "clearJobButton(job)" in _function(app, "stoppedBlock")
    assert "clearJobButton(job)" in _function(app, "jobBody")  # a failed one


def test_stopping_a_batch_says_them_and_that_finished_pictures_stay():
    app = read("app.js")
    stop = _function(app, "batchStopControls")
    assert "several ? 'Stop them' : 'Stop it'" in stop
    assert "'Stop the pictures not done yet? Finished ones are kept.'" in stop
    # A stopped batch doesn't repeat its progress line ("3 of 10 done, 7 stopped") in a second, bigger note:
    # only the server's own reason, or a hint when nothing finished is listed.
    body = _function(app, "batchBody")
    assert "job.status === 'cancelled' && (job.error || !(job.results || []).length)" in body
    assert "class: 'muted batch-note'" in body
    assert ".batch-where, .batch-note { font-size: var(--text-s); }" in read("app.css")


def test_a_vector_picture_says_it_is_drawn_directly_and_offers_no_ai_styles():
    app = read("app.js")
    item = _function(app, "trayItem")
    assert "const vector = photo.isVector && !problem && !checking;" in item
    assert "h('span', { class: 'tag-name' }, 'Vector')" in item and "vector ? VECTOR_LABEL" in item
    content = _function(app, "drawerContent")
    assert "photo.isVector ? VECTOR_LABEL : modeLabel(choice.mode, 'image', choice.level)" in content
    assert content.index("photo.isVector) {\n      parts.push(vectorBlock(photo));") < content.index("qualityBlock(photo, choice)")
    block = _function(app, "vectorBlock")
    assert "So there\\'s no quality to pick for this one." in block and "modeTag(VECTOR, 'image')" in block
    assert "const VECTOR_NOTE = \"It's a vector drawing, so it's drawn sharp at the new size. No AI is needed.\";" in app


def test_drawer_preview_shows_the_whole_picture():
    css = read("app.css")
    # The 4:3 box is capped at 250 px; the picture covers the box, contained, instead of a taller grid item.
    assert ".drawer-preview img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }" in css
    assert re.search(r"\.drawer-preview \{[^}]*position: relative;", css)


def test_tab_never_leaves_a_picture_under_the_sticky_header_or_action_bar():
    app = read("app.js")
    room = _function(app, "bindStickyRoom")
    assert "watch(topbar, '--topbar-h');" in room and "watch(el.actionBar, '--actionbar-h');" in room
    assert "new ResizeObserver(" in room
    assert "bindStickyRoom();" in _function(app, "boot")
    css = read("app.css")
    assert "scroll-padding-top: calc(var(--topbar-h, 64px) + 8px);" in css
    assert "scroll-padding-bottom: calc(var(--actionbar-h, 0px) + 24px);" in css


def test_result_titles_focused_after_a_redraw_show_no_ring():
    css = read("app.css")
    assert 'h1[tabindex="-1"]:focus, h2[tabindex="-1"]:focus, .job-name[tabindex="-1"]:focus { outline: none; }' in css


def test_video_compare_keys_work_after_dragging_the_timeline_and_show_as_keys():
    video = read("js/videocompare.js")
    # The timeline (a range) keeps only its own keys: arrows, Home, End, Page Up/Down. Space , . J L K M 0 1 2
    # still drive the players while it has focus, which it does after a drag.
    assert "if (from.closest('select, textarea, input:not([type=\"radio\"]):not([type=\"range\"])')) return;" in video
    assert "from.closest('input[type=\"radio\"], input[type=\"range\"]')) return;" in video
    assert "key('Space')" in video and "key(',', 'Comma')" in video and "key('.', 'Full stop')" in video
    assert "h('kbd', label ? { title: label } : null, name)" in video
    assert ".vc-hint kbd {" in read("app.css")
    # Laid out from layout sizes, like the picture viewer (the dialog opens with a brief scale).
    layout = _function_js(video, "layout")
    assert ".getBoundingClientRect()" not in layout
    assert "const box = { width: left.view.clientWidth, height: left.view.clientHeight };" in layout


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_video_sync_starts_catching_up_past_half_a_frame():
    # Measured in the browser: the original ran a steady 25 ms (0.6 of a frame at 24 fps) ahead, so the two
    # often showed different frames (55 next to 56). Catching up now starts past half a frame.
    left_alone, caught = run_module("js/videocompare.js", "[m.driftAction(0.025, 24, 1), m.driftAction(0.025, 24, 1, 1 / 48)]")
    assert left_alone == {"seek": False, "rate": 1}  # the old one-frame window left it
    assert not caught["seek"] and caught["rate"] < 1  # half a frame: the original slows a touch until it's back
    assert "correcting ? frameTime / 4 : frameTime / 2" in _function_js(read("js/videocompare.js"), "sync")


def test_preview_styles_shows_a_preview_that_is_already_done():
    styles = read("js/styles.js")
    start = styles[styles.index("async function startPreview()"):styles.index("function showRunning(")]
    # A preview whose last update arrived before the answer to "start" would otherwise wait forever.
    assert "if (job.status === 'done') showResults(job);\n        else showRunning(job);" in start


# --- Web fixes from the final review ------------------------------------------------------------


def _clear_groups_script(host: bool) -> str:
    """Runs app.js's real clearGroups against a small fake page: a results header with the
    "Clear all finished" button, three finished jobs, and the parts of renderJobs it relies on."""
    source = "function clearGroups(" + _function(read("app.js"), "clearGroups")
    return (
        "class N {"
        " constructor(tag, attrs) { this.tag = tag; this.attrs = attrs || {}; this.children = []; this.parentNode = null;"
        "  this.listeners = {}; this.hidden = false; this.disabled = false; this.text = ''; }"
        " append(...kids) { for (let k of kids) { if (k == null) continue; if (typeof k === 'string') { const t = new N('#text'); t.text = k; k = t; }"
        "  if (k.parentNode) k.remove(); k.parentNode = this; this.children.push(k); } }"
        " remove() { if (!this.parentNode) return; const p = this.parentNode; p.children.splice(p.children.indexOf(this), 1); this.parentNode = null; }"
        " replaceWith(other) { const p = this.parentNode; if (!p) return; if (other.parentNode) other.remove();"
        "  p.children[p.children.indexOf(this)] = other; other.parentNode = p; this.parentNode = null; }"
        " get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n === root; }"
        " addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }"
        " click() { if (this.disabled) return; for (const f of this.listeners.click || []) f({ target: this }); }"
        " focus() { focused = this; }"
        " querySelector() { return null; }"
        " get textContent() { return this.tag === '#text' ? this.text : this.children.map((c) => c.textContent).join(''); }"
        "}"
        "let focused = null; const root = new N('html'); const head = new N('div'); root.append(head);"
        "const title = new N('h2'); title.append('Your results');"
        "const clearButton = new N('button'); clearButton.append('Clear all finished'); head.append(title, clearButton);"
        "const jobList = new N('ol'); root.append(jobList);"
        "const h = (tag, attrs, ...kids) => { const n = new N(tag, attrs); n.append(...kids); return n; };"
        "const el = { resultsClear: clearButton, resultsTitle: title, jobList };"
        "const state = { cleared: new Set(), batchOpen: new Set(), jobs: [] };"
        "const isActive = (job) => job.status === 'running' || job.status === 'queued';"
        f"const isHost = () => {'true' if host else 'false'};"
        "const saveCleared = () => {}; const toasts = []; const toast = (m) => toasts.push(m); let renders = 0;"
        "function renderJobs() { renders += 1;"
        " const finished = state.jobs.filter((j) => !state.cleared.has(j.id) && !isActive(j));"
        " el.resultsClear.hidden = finished.length < 2; }"
        f"{source}\n"
        "state.jobs = [{ id: 'a', status: 'done' }, { id: 'b', status: 'done' }, { id: 'c', status: 'done' }];"
        "const groups = state.jobs.map((j) => ({ id: 'job-' + j.id, jobs: [j] })); renderJobs();"
        "const seen = {};"
    )


def _run_node(script: str):
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_guests_clear_all_confirm_goes_away_and_the_button_comes_back():
    # Reported: after "Clear them" the question stayed above the emptied list, and "Clear all finished"
    # was gone from the page, so it never came back for new results.
    result = _run_node(_clear_groups_script(host=False) + (
        "clearGroups(groups, el.resultsClear);"
        "seen.asking = head.textContent; const ask = head.children[1]; const yes = ask.children[1];"
        "yes.click();"
        "seen.after = head.textContent; seen.connected = clearButton.isConnected; seen.cleared = [...state.cleared];"
        "seen.toasts = toasts.length; seen.renders = renders;"
        "yes.click();"  # a second press does nothing more
        "seen.toastsAgain = toasts.length; seen.rendersAgain = renders;"
        "state.jobs.push({ id: 'd', status: 'done' }, { id: 'e', status: 'done' }); renderJobs();"
        "seen.newHidden = clearButton.hidden; seen.newConnected = clearButton.isConnected; seen.finalHead = head.textContent;"
        "process.stdout.write(JSON.stringify(seen));"
    ))
    assert "Clear these?" in result["asking"] and "Clear all finished" not in result["asking"]
    assert result["after"] == "Your resultsClear all finished"  # the question is gone, the button is back
    assert result["connected"] is True and result["cleared"] == ["a", "b", "c"] and result["toasts"] == 1
    assert result["toastsAgain"] == 1 and result["rendersAgain"] == result["renders"]
    assert result["newHidden"] is False and result["newConnected"] is True  # shown again for new results
    assert "Clear these?" not in result["finalHead"]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_keep_them_puts_the_button_back_and_the_host_is_not_asked():
    kept = _run_node(_clear_groups_script(host=False) + (
        "clearGroups(groups, el.resultsClear); head.children[1].children[2].click();"
        "process.stdout.write(JSON.stringify({ head: head.textContent, cleared: [...state.cleared], toasts: toasts.length,"
        " focused: focused === clearButton }));"
    ))
    assert kept == {"head": "Your resultsClear all finished", "cleared": [], "toasts": 0, "focused": True}
    host = _run_node(_clear_groups_script(host=True) + (
        "clearGroups(groups, el.resultsClear);"
        "process.stdout.write(JSON.stringify({ head: head.textContent, cleared: [...state.cleared], toasts }));"
    ))
    assert host["head"] == "Your resultsClear all finished" and host["cleared"] == ["a", "b", "c"]
    assert host["toasts"] == ["Cleared from the list. Your pictures are still in Pictures › Upscaled."]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_the_cloud_notice_names_every_service_a_batch_uses():
    counts = run_app_js(
        "[m.cloudProviderCounts(['cloud:fal:topaz', 'photoreal', 'cloud:replicate:topaz', 'cloud:fal:seedvr2', null]),"
        " m.cloudProviderCounts(['faithful', 'fast']), m.cloudProviderCounts(['cloud:fal:topaz'])]"
    )
    assert counts == [[{"id": "fal", "count": 2}, {"id": "replicate", "count": 1}], [], [{"id": "fal", "count": 1}]]
    fal_note = "Your picture is sent to fal.ai to be upscaled. fal.ai deletes it after an hour."
    rep_note = "Your picture is sent to Replicate. Replicate deletes it after one hour."
    one, one_known, two, two_known, three = run_app_js(
        "[m.cloudNoticeLines([{id: 'fal', name: 'fal.ai', count: 3}], {}),"
        f" m.cloudNoticeLines([{{id: 'fal', name: 'fal.ai', count: 3}}], {{fal: {js(fal_note)}}}),"
        " m.cloudNoticeLines([{id: 'fal', name: 'fal.ai', count: 2}, {id: 'replicate', name: 'Replicate', count: 1}], {}),"
        " m.cloudNoticeLines([{id: 'fal', name: 'fal.ai', count: 1}, {id: 'replicate', name: 'Replicate', count: 1}],"
        f"  {{fal: {js(fal_note)}, replicate: {js(rep_note)}}}),"
        " m.cloudNoticeLines([{id: 'fal', name: 'fal.ai', count: 2}, {id: 'replicate', name: 'Replicate', count: 1},"
        "  {id: 'magnific', name: 'Magnific', count: 4}], {})]"
    )
    assert one == ["Your pictures will be sent to fal.ai to be upscaled. fal.ai charges for each picture."]
    assert one_known == [f"{fal_note} fal.ai charges for each picture."]
    assert two == ["2 pictures will be sent to fal.ai and 1 to Replicate to be upscaled. Each service charges for each picture."]
    assert two_known == [
        "1 picture will be sent to fal.ai and 1 to Replicate to be upscaled. Each service charges for each picture.",
        fal_note, rep_note,  # each service's own word on what it keeps
    ]
    assert three[0].startswith("2 pictures will be sent to fal.ai, 1 to Replicate and 4 to Magnific to be upscaled.")
    app = read("app.js")
    start = _function(app, "updatePhotoStart")
    assert "cloudProviderCounts(" in start and ".find((c) => isCloudId(c.mode))" not in start
    notice = _function(app, "renderCloudNotice")
    assert "cloudNoticeLines(" in notice and "getGuide(" in notice


def test_closing_a_dialog_ignores_a_close_event_that_arrives_after_it_was_opened_again():
    # The 'close' event comes a frame later; a dialog closed and opened again in between kept its new
    # picture, comparison or wizard step, instead of being emptied by the late event.
    app = read("app.js")
    drawer = _function(app, "bindDrawer")
    assert re.search(r"el\.drawer\.addEventListener\('close', \(\) => \{\n(\s+//[^\n]*\n)?\s+if \(el\.drawer\.open\) return;", drawer)
    assert "el.stylesDialog.addEventListener('close', () => {\n        if (el.stylesDialog.open) return;" in _function(app, "boot")
    for name in ("js/viewer.js", "js/videocompare.js", "js/styles.js", "wizard.js"):
        module = read(name)
        assert re.search(r"dialog\.addEventListener\('close', \(\) => \{\n\s+if \(dialog\.open\) return;", module), name


def test_a_click_on_the_dimmed_page_beside_the_drawer_closes_it():
    drawer = _function(read("app.js"), "bindDrawer")
    assert "el.drawer.addEventListener('pointerdown'" in drawer and "el.drawer.addEventListener('click'" in drawer
    assert "event.target === el.drawer" in drawer
    # Only a press that starts and ends outside the panel: a drag out of it (selecting text) doesn't close it.
    assert "pressedOutside" in drawer and "outsideBox(el.drawer, event)" in drawer
    assert "event.clientX < box.left || event.clientX > box.right" in _function(read("app.js"), "outsideBox")


def test_the_viewers_close_button_stays_top_right_on_phones():
    css = read("app.css")
    assert ".viewer-head .dialog-close { position: static; margin-left: auto; }" in css
    phone = css[css.index("@media (max-width: 640px)"):]
    assert re.search(r"\.viewer-head \{[^}]*position: relative;[^}]*padding-right: 56px;", phone)
    assert re.search(r"\.viewer-head \.dialog-close \{[^}]*position: absolute;[^}]*top: max\(8px, env\(safe-area-inset-top\)\);[^}]*right: 8px;", phone)


def test_folding_the_tray_to_the_strip_shows_every_picture_and_drops_the_filter():
    fold = _function(read("app.js"), "setTrayFolded")
    # The strip draws every picture, so the drawer steps through all of them too ("3 of 60", not "in Photo-real").
    assert "if (folded) state.tray.filter = 'all';" in fold


def test_stopping_the_video_comparison_is_a_calm_note_not_an_error():
    video = read("js/videocompare.js")
    stopped = _function_js(video, "showStopped")
    assert "role: 'status'" in stopped and "alert-danger" not in stopped and "Try again" in stopped
    state = _function_js(video, "onState")
    assert "if (state.status === 'cancelled') {\n        showStopped(state.message || 'Stopped.');" in state
    assert "showProblem(state.error || FAILED);" in state  # a real failure is still a problem
    assert ".vc-stopped-text {" in read("app.css")


def test_the_drawer_shows_one_time_for_a_picture():
    quality = _function(read("app.js"), "qualityBlock")
    # The Time fact above says how long this picture takes; the level only says what it does.
    assert "level.eta_hint" not in quality and "level && level.description ? h('p', { class: 'hint' }, level.description)" in quality


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_download_all_says_what_it_saves():
    titles = run_app_js(
        "[m.downloadAllTitle([{results: [1, 2, 3, 4, 5]}, {results: [1]}]),"
        " m.downloadAllTitle([{results: [1, 2]}, {results: [1, 2]}]),"
        " m.downloadAllTitle([{results: [1]}, {results: [1]}, {results: [1]}]),"
        " m.downloadAllTitle([{results: [1, 2]}, {results: [1, 2]}, {results: [1]}, {results: [1]}])]"
    )
    assert titles == ["Saves 1 .zip file and 1 picture", "Saves 2 .zip files", "Saves 3 pictures", "Saves 2 .zip files and 2 pictures"]
    assert "title: downloadAllTitle(jobs)" in _function(read("app.js"), "downloadAllControl")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_select_alls_name_starts_with_what_it_shows():
    labels = run_app_js(
        "[m.selectAllLabel(0, 'all', 60), m.selectAllLabel(0, 'photoreal', 35), m.selectAllLabel(3, 'all', 60),"
        " m.selectAllLabel(1200, 'photoreal', 35)]"
    )
    assert labels == ["Select all pictures", "Select all 35 pictures shown", "3 selected. Select all pictures",
                      "1,200 selected. Select all 35 pictures shown"]
    actions = _function(read("app.js"), "renderTrayActions")
    assert "selectAllLabel(selected, state.tray.filter, list.length)" in actions


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_a_pictures_name_for_screen_readers_has_one_full_stop():
    problem = "It couldn't be opened as a picture. It may be damaged, or saved in a way Re:Solution can't read."
    labels = run_app_js(
        f"[m.openLabel(['broken-file.jpg', {js(problem)}, null]), m.openLabel(['a.jpg', 'Photo-real · Balanced', 'your pick', null]),"
        " m.openLabel(['b.png', 'Faithful', '', 'needs a look  '])]"
    )
    assert labels == [
        f"broken-file.jpg, {problem} Show details",
        "a.jpg, Photo-real · Balanced, your pick. Show details",
        "b.png, Faithful, needs a look. Show details",
    ]
    assert "'aria-label': openLabel(said)" in _function(read("app.js"), "trayItem")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_the_picture_list_is_one_tab_stop_with_arrow_keys():
    # 10 pictures, two controls each (tick box, then the picture), 4 to a row in the grid.
    cases = [
        ((20, 4, 1, "ArrowRight", 2), 2),  # the picture -> the next picture's tick box
        ((20, 4, 2, "ArrowLeft", 2), 1),
        ((20, 4, 0, "ArrowLeft", 2), 0),  # nothing before the first: stays (and the page doesn't scroll)
        ((20, 4, 1, "ArrowDown", 2), 9),  # a row down, the same kind of control
        ((20, 4, 13, "ArrowDown", 2), 19),  # no picture right below: the last one
        ((20, 4, 17, "ArrowDown", 2), 17),  # already on the last row
        ((20, 4, 9, "ArrowUp", 2), 1),
        ((20, 4, 3, "ArrowUp", 2), 3),
        ((20, 4, 7, "Home", 2), 1),
        ((20, 4, 2, "End", 2), 18),
        ((20, 1, 3, "ArrowDown", 2), 5),  # the list: one picture to a row
        ((20, 4, 3, "Enter", 2), None),  # Enter and Space keep their own jobs
        ((20, 4, 3, " ", 2), None),
        ((49, 49, 5, "ArrowRight", 1), 6),  # the folded strip: one control each, one row
        ((49, 49, 5, "ArrowDown", 1), 5),
        ((0, 4, 0, "ArrowRight", 2), None),
    ]
    calls = ",".join(f"m.rovingTarget({json.dumps(list(a))[1:-1]})" for a, _ in cases)
    assert run_app_js(f"[{calls}]") == [expected for _, expected in cases]
    app = read("app.js")
    assert "bindRoving(el.photoList, TRAY_CONTROLS, 2," in _function(app, "bindTray")
    assert "bindRoving(el.trayStrip, 'button', 1," in _function(app, "bindTray")
    assert "refreshRoving(el.photoList, TRAY_CONTROLS" in _function(app, "renderTrayItems")
    assert "refreshRoving(el.trayStrip, 'button'" in _function(app, "renderStrip")
    assert "control.setAttribute('tabindex', want)" in _function(app, "setRovingStop")  # one 0, the rest -1
    assert "setRovingStop(list, selector, control)" in _function(app, "bindRoving")  # focus moves the stop
    html = read("index.html")
    assert 'id="tray-keys"' in html and "arrow keys" in html
    assert ".tray:has(.tray-items :focus-visible, .tray-strip :focus-visible) .tray-keys {" in read("app.css")
