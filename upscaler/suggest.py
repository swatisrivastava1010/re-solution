"""Auto-suggest: which mode suits a picture, with short plain-English reasons.

analyze() measures a few cheap signals on the processor; suggest() turns them into a
recommendation with the rules from docs/DESIGN.md ("Photo-real detail levels, auto-suggest, and
Preview styles"), in this order:

1. a vector drawing -> drawn directly at full size (no AI)
2. an illustration -> Illustration
3. large faces (more than 3% of the picture) -> Faithful, with a tip to preview Photo-real
4. text or a logo -> Faithful
5. the AI "mesh" texture, or a small picture (long edge under 1200 px) -> Photo-real Balanced
6. a sharp, clean, large photo (long edge 3000 px or more) -> Faithful for sharp photos
7. anything else -> Photo-real Balanced

The thresholds were calibrated on real pictures; see docs/RESEARCH.md §2b. The suggestion is only
preselected: the person's own choice always wins.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from upscaler.config import models_dir
from upscaler.imageio_utils import ImageInfo
from upscaler.models import MODELS, ensure_model

log = logging.getLogger(__name__)

# --- Thresholds (docs/RESEARCH.md §2b) ---------------------------------------------------
LARGE_FACES = 0.03  # share of the picture covered by faces
ILLUSTRATION_LIKELY = 0.5  # illustration_score, 0..1
TEXT_LIKELY = 0.004  # share of the picture covered by lines of text
MESH_LIKELY = 9.0  # mesh_score: how far the strongest fine repeating pattern stands out (robust z-score)
SMALL_PICTURE = 1200  # long edge, px
LARGE_PICTURE = 3000  # long edge, px
SHARP = 150.0  # sharpness: variance of the Laplacian in the picture's most detailed places
BLOCKY = 1.2  # blockiness: 1.0 = no JPEG blocks; well-saved JPEGs stay under about 1.1

PHOTOREAL, BALANCED = "photoreal", "balanced"

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_WORKING_EDGE = 768  # the illustration and face signals look at a copy this size


@dataclass
class Signals:
    faces: int  # faces found
    face_area: float  # share of the picture they cover, 0..1
    mesh_score: float  # natural detail scores about 3-6, no real texture 0; MESH_LIKELY and up = the AI mesh
    illustration_score: float  # 0..1
    text_score: float  # share of the picture covered by lines of text, 0..1
    sharpness: float
    blockiness: float
    long_edge: int  # of the picture analyzed, px
    is_vector: bool  # analyze() sees pixels only, so always False there; suggest() also reads ImageInfo


@dataclass
class Suggestion:
    mode: str  # an image mode id ("photoreal", "faithful", "faithful_sharp", "fast", "illustration")
    photoreal_level: str | None  # "balanced" when mode is Photo-real, else None
    reasons: list[str]  # one to three short sentences for "Why?"
    signals: Signals


# --- The rules ---------------------------------------------------------------------------

_VECTOR = "This is a vector drawing, so it's drawn directly at full size. No AI is needed."
_ILLUSTRATION = "It looks like a drawing or illustration, with flat colours and clean outlines."
_FACES = "Faces fill a good part of this picture. Faithful keeps them exactly as they are."
_FACES_TIP = "Tip: use Preview styles to see Photo-real Balanced, which adds more skin detail."
_TEXT = "It has text or a logo. Faithful keeps letters crisp and exact; Photo-real can change them."
_MESH = (
    "It has a faint grid-like texture, common in AI-made and resized pictures. Photo-real smooths it away naturally."
)
_SMALL = "It's quite small, so Photo-real has the most real-looking detail to add."
_SHARP_PHOTO = "It's a large, sharp photo already. Faithful for sharp photos keeps its real detail."
_DEFAULT = "Photo-real Balanced gives the most natural, photo-like result for most pictures."
_BLOCKS = "It shows JPEG compression blocks, which Photo-real cleans up."


def suggest(info: ImageInfo, signals: Signals) -> Suggestion:
    """The recommended mode for a picture: the first rule that applies (see the module docstring)."""

    def chosen(mode: str, *reasons: str) -> Suggestion:
        return Suggestion(mode, BALANCED if mode == PHOTOREAL else None, list(reasons), signals)

    long_edge = max(info.width, info.height)
    if info.is_vector or signals.is_vector:
        return chosen("fast", _VECTOR)  # vectors are drawn directly whatever the mode; Fast keeps estimates honest
    if signals.illustration_score >= ILLUSTRATION_LIKELY:
        return chosen("illustration", _ILLUSTRATION)
    if signals.face_area > LARGE_FACES:
        return chosen("faithful", _FACES, _FACES_TIP)
    if signals.text_score >= TEXT_LIKELY:
        return chosen("faithful", _TEXT)
    blocky = signals.blockiness >= BLOCKY
    reasons = [_MESH] if signals.mesh_score >= MESH_LIKELY else []
    if long_edge < SMALL_PICTURE:
        reasons.append(_SMALL)
    if reasons:
        return chosen(PHOTOREAL, *reasons, *([_BLOCKS] if blocky else []))
    if long_edge >= LARGE_PICTURE and signals.sharpness >= SHARP and not blocky:
        return chosen("faithful_sharp", _SHARP_PHOTO)
    return chosen(PHOTOREAL, _DEFAULT, *([_BLOCKS] if blocky else []))


# --- Measuring -------------------------------------------------------------------------


def analyze(rgb: np.ndarray) -> Signals:
    """Measure the signals for an (H, W, 3) uint8 picture; an alpha channel is ignored.

    Takes about 0.15 s for a 2K photo on an M4's processor and 0.4 s for a 2K page full of text,
    and stays well under a second for big pictures (the face finder's very first use may add up to
    FACE_MODEL_WAIT_SECONDS while its 230 KB model downloads).
    """
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w = rgb.shape[:2]
    small = _shrink(rgb, _WORKING_EDGE)
    boxes = detect_faces(small)
    tiles = _detailed_tiles(rgb, _MESH_TILE)
    mesh_rgb = _shrink(rgb, _MESH_EDGE)
    mesh_tiles = tiles if mesh_rgb is rgb else _detailed_tiles(mesh_rgb, _MESH_TILE)
    return Signals(
        faces=len(boxes),
        face_area=round(min(1.0, sum((bw * bh for _, _, bw, bh in boxes), 0.0)), 4),
        mesh_score=round(_mesh_score(mesh_rgb, mesh_tiles), 3),
        illustration_score=round(_illustration_score(small), 3),
        text_score=round(_text_score(rgb), 4),
        sharpness=round(_sharpness(rgb, tiles), 1),
        blockiness=round(_blockiness(rgb), 3),
        long_edge=max(h, w),
        is_vector=False,
    )


def _shrink(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _ramp(value: float, low: float, high: float) -> float:
    """0 at `low`, 1 at `high`, straight in between and flat beyond (falling when low > high)."""
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


# --- Faces (OpenCV's YuNet) --------------------------------------------------------------

FACE_MODEL_WAIT_SECONDS = 3.0  # how long a picture waits for the face model's first download
_FACE_RETRY_SECONDS = 600.0  # after a failed download or load, try again this much later
_FACE_SCORE = 0.85  # YuNet confidence; lower lets in faces seen in patterns
_FACE_SCALES = (640, 256)  # YuNet finds faces of about 10-300 px, so big close-ups need the small copy
_SAME_FACE_IOU = 0.4


class _FaceFinder:
    """YuNet, loaded on first use.

    While its model isn't on this computer yet, it downloads in the background and pictures are
    analysed without faces, so a slow or missing internet connection never holds up a suggestion:
    only the picture that starts the download waits for it, and for FACE_MODEL_WAIT_SECONDS at most.
    After a failed download or load, the next try is _FACE_RETRY_SECONDS later.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # one detector, used by one picture at a time
        self._detector = None
        self._download: threading.Thread | None = None
        self._retry_at = 0.0

    def detector(self):
        """The cv2.FaceDetectorYN, or None while its model isn't available."""
        spec = MODELS["face_yunet"]
        path = models_dir() / spec.file
        with self._lock:
            if self._detector is not None or time.monotonic() < self._retry_at:
                return self._detector  # None until the next try after a failure
            if _is_complete(path, spec.size_bytes):
                return self._load(path)
            if self._download is not None and self._download.is_alive():
                return None  # only the picture that started the download waits for it
            download = self._download = threading.Thread(target=self._fetch, name="face-model-download", daemon=True)
            download.start()
        download.join(FACE_MODEL_WAIT_SECONDS)
        with self._lock:
            if self._detector is None and time.monotonic() >= self._retry_at and _is_complete(path, spec.size_bytes):
                return self._load(path)
            return self._detector

    def _fetch(self) -> None:
        try:
            ensure_model(MODELS["face_yunet"])
        except Exception:  # offline, a firewall, a full disk: suggestions just go without faces
            log.warning("The face finder model couldn't be downloaded; suggesting without faces", exc_info=True)
            with self._lock:
                self._retry_at = time.monotonic() + _FACE_RETRY_SECONDS

    def _load(self, path: Path):
        """The detector from the model at `path`, or None (holding the lock).

        It's read from memory, because OpenCV can't open a file whose path has letters outside
        English, as under a Windows profile folder such as C:\\Users\\Zoë.
        """
        try:
            model = np.fromfile(path, np.uint8)
            self._detector = cv2.FaceDetectorYN.create(
                "onnx", model, np.empty(0, np.uint8), (320, 320), _FACE_SCORE, 0.3, 50
            )
        except (cv2.error, OSError):
            log.warning("The face finder model couldn't be loaded; suggesting without faces", exc_info=True)
            self._retry_at = time.monotonic() + _FACE_RETRY_SECONDS
        return self._detector

    def detect(self, detector, bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            _, faces = detector.detect(bgr)
        return np.zeros((0, 15), np.float32) if faces is None else faces


def _is_complete(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


_faces = _FaceFinder()


def detect_faces(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Faces as (x, y, width, height) fractions of the picture; [] if the face model isn't available."""
    detector = _faces.detector()
    h, w = rgb.shape[:2]
    if detector is None or min(h, w) < 16:
        return []
    found: list[tuple[float, float, float, float]] = []
    for edge in _FACE_SCALES:
        copy = _shrink(rgb, edge)
        ch, cw = copy.shape[:2]
        for face in _faces.detect(detector, cv2.cvtColor(copy, cv2.COLOR_RGB2BGR)):
            x, y, fw, fh = (float(v) for v in face[:4])
            x0, y0 = max(0.0, x / cw), max(0.0, y / ch)
            box = (x0, y0, min(1.0, (x + fw) / cw) - x0, min(1.0, (y + fh) / ch) - y0)
            if box[2] > 0 and box[3] > 0 and all(_iou(box, other) < _SAME_FACE_IOU for other in found):
                found.append(box)
    return found


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# --- AI mesh: a repeating fine pattern in the luminance spectrum -------------------------
# The spectrum of the more detailed tiles, averaged, minus its own smooth background: natural
# detail leaves a smooth slope, while the AI mesh stands out as sharp peaks at periods of about
# 2-5 px. The score is the strongest peak in units of the leftover's typical spread (a robust
# z-score), so it doesn't depend on how busy the picture is.
# - It's measured on a copy no longer than 2048 px. AI pictures are made at about 1-2K, so up to
#   there the mesh is in single pixels. A big camera photo can carry a fine pixel-level pattern of
#   its own (probably from the camera's processing), which scores like the mesh at full size but
#   averages away in the smaller copy; an AI picture that was enlarged gets its mesh back in it.
# - Flat tiles are left out, because in sky and plain walls the faint regular ripples of an earlier
#   resize show up as peaks too; a picture with no real texture at all scores 0, because the
#   8-bit steps of a smooth gradient repeat as regularly as the mesh does.

_MESH_EDGE = 2048  # px: the long edge of the copy the mesh is measured on
_MESH_TILE = 128
_MESH_TILES = 64  # spread evenly over the more detailed half of the tiles
_MESH_MIN_FREQUENCY = 0.2  # cycles per pixel: periods up to 5 px
_MESH_BACKGROUND = 9  # median window (spectrum bins) for the smooth background
_MESH_MIN_SHARPNESS = 20.0  # less than this (see _sharpness) is a gradient or a blur, not texture
_MESH_MIN_SPREAD = 0.1  # floor for the leftover's spread; real pictures leave 0.14-0.30
_MAX_TILES = 600  # spread over the picture; enough for any size


def _tile_origins(h: int, w: int, size: int) -> list[tuple[int, int]]:
    rows = np.arange(0, h - size + 1, size)
    cols = np.arange(0, w - size + 1, size)
    step = max(1, int(np.ceil(np.sqrt(len(rows) * len(cols) / _MAX_TILES))))
    return [(int(r), int(c)) for r in rows[::step] for c in cols[::step]]


def _grey(pixels: np.ndarray) -> np.ndarray:
    return pixels.astype(np.float32) @ _LUMA


def _laplacian_variance(grey: np.ndarray) -> float:
    lap = 4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    return float(lap.var())


def _detailed_tiles(rgb: np.ndarray, size: int) -> list[tuple[float, int, int]]:
    """(Laplacian variance, row, column) of tiles over the picture, most detailed first."""
    tiles = [
        (_laplacian_variance(_grey(rgb[r : r + size, c : c + size])), r, c)
        for r, c in _tile_origins(*rgb.shape[:2], size)
    ]
    return sorted(tiles, reverse=True)


_spectrum_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _spectrum_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The Hann window and each bin's frequency (cycles per pixel) for n x n tiles."""
    if n not in _spectrum_cache:
        window = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
        f = (np.arange(n) - n // 2) / n
        _spectrum_cache[n] = window, np.hypot(*np.meshgrid(f, f))
    return _spectrum_cache[n]


def _mesh_score(rgb: np.ndarray, tiles: list[tuple[float, int, int]]) -> float:
    """`rgb`: the copy to measure (see _MESH_EDGE); `tiles`: _detailed_tiles(rgb, _MESH_TILE)."""
    detailed = tiles[: max(2, len(tiles) // 2)]
    if len(detailed) < 2 or _sharpness(rgb, tiles) < _MESH_MIN_SHARPNESS:
        return 0.0
    chosen = [detailed[i] for i in np.unique(np.linspace(0, len(detailed) - 1, _MESH_TILES).astype(int))]
    n = _MESH_TILE
    window, radius = _spectrum_grid(n)
    log_power = np.zeros((n, n), np.float32)
    for _, r, c in chosen:
        grey = _grey(rgb[r : r + n, c : c + n])
        spectrum = np.fft.fftshift(np.fft.fft2((grey - grey.mean()) * window))
        log_power += np.log(np.abs(spectrum) ** 2 + 1e-2)
    log_power /= len(chosen)

    mid = n // 2
    candidates = radius >= _MESH_MIN_FREQUENCY
    # Bins beside the axes carry the shoulders of edge streaks; the axes themselves are compared
    # along the axis (see _smooth_background), so a repeating pattern on them still counts.
    candidates[[mid - 1, mid + 1], :] = False
    candidates[:, [mid - 1, mid + 1]] = False
    peaks = (log_power - _smooth_background(log_power))[candidates]
    centre = float(np.median(peaks))
    spread = 1.4826 * float(np.median(np.abs(peaks - centre)))  # the standard deviation, if it were Gaussian
    return max(0.0, (float(peaks.max()) - centre) / max(spread, _MESH_MIN_SPREAD))


def _smooth_background(log_power: np.ndarray) -> np.ndarray:
    """A median-filtered copy of the spectrum. On the two axes the median runs along the axis,
    because straight edges in the picture put a whole line of energy there."""
    k = _MESH_BACKGROUND
    low, high = float(log_power.min()), float(log_power.max())
    levels = np.round((log_power - low) / (high - low + 1e-6) * 255).astype(np.uint8)  # medianBlur(9) needs 8 bits
    background = cv2.medianBlur(levels, k).astype(np.float32) / 255 * (high - low) + low
    mid = log_power.shape[0] // 2
    for axis in (0, 1):
        line = log_power[mid, :] if axis == 0 else log_power[:, mid]
        along = np.median(sliding_window_view(np.pad(line, k // 2, mode="wrap"), k), axis=-1)
        if axis == 0:
            background[mid, :] = along
        else:
            background[:, mid] = along
    return background


# --- Sharpness and JPEG blocks -----------------------------------------------------------

_SHARP_PERCENTILE = 90
_BLOCK = 8  # JPEG's block size
_BLOCK_LINES = 256  # rows and columns sampled
_STEP_FLOOR = 0.5  # grey levels, so a fine JPEG's tiny block steps on a plain white wall don't stand out


def _sharpness(rgb: np.ndarray, tiles: list[tuple[float, int, int]]) -> float:
    """Laplacian variance of the detailed tiles (90th percentile), at full resolution."""
    if not tiles:
        return _laplacian_variance(_grey(rgb)) if min(rgb.shape[:2]) >= 3 else 0.0
    return float(np.percentile([v for v, _, _ in tiles], _SHARP_PERCENTILE))


def _blockiness(rgb: np.ndarray) -> float:
    """How much stronger the steps between neighbouring pixels are on one of the 8 possible JPEG
    block grids than elsewhere, across and down. 1.0 = no blocks.

    Medians keep a few strong edges from looking like a grid, and comparing with the position half
    a block away keeps the regular ripples of an earlier resize (every 2 or 4 px) out.
    """
    h, w = rgb.shape[:2]
    ratios = []
    for axis, length, across in ((1, w, h), (0, h, w)):
        if length < 4 * _BLOCK or across < 1:
            continue
        lines = np.unique(np.linspace(0, across - 1, min(across, _BLOCK_LINES)).astype(int))
        sample = _grey(rgb[lines] if axis == 1 else rgb[:, lines].swapaxes(0, 1))
        steps = np.abs(np.diff(sample, axis=1)).mean(axis=0)
        usable = len(steps) // _BLOCK * _BLOCK
        by_phase = np.median(steps[:usable].reshape(-1, _BLOCK), axis=0) + _STEP_FLOOR
        grid = int(by_phase.argmax())
        ratios.append(float(min(by_phase[grid] / np.median(by_phase), by_phase[grid] / by_phase[(grid + 4) % 8])))
    return float(np.mean(ratios)) if ratios else 1.0


# --- Text: lines of letter-shaped regions (MSER) -----------------------------------------

_TEXT_EDGE = 1600
_LETTER_HEIGHT = (8, 0.15)  # px, and share of the picture's height
_LETTER_ASPECT = (0.12, 1.3)  # width / height of one letter
_LETTER_FILL = (0.2, 0.85)  # share of its box a letter covers
_LINE_MIN_LETTERS = 3
_LINE_CONTRAST = 40  # grey levels between letters and their background
_LINE_SEPARATION = 0.6  # Otsu: share of a line's variance explained by letters vs background
_LINE_QUIET = 0.35  # the band above or below a line is this calm, relative to the line


def _text_score(rgb: np.ndarray) -> float:
    grey = cv2.cvtColor(_shrink(rgb, _TEXT_EDGE), cv2.COLOR_RGB2GRAY)
    h, w = grey.shape
    if min(h, w) < 2 * _LETTER_HEIGHT[0]:
        return 0.0
    mser = cv2.MSER_create(5, 20, int(0.004 * h * w) + 100)
    regions, boxes = mser.detectRegions(grey)
    letters = []
    for (x, y, bw, bh), points in zip(boxes, regions):
        if not _LETTER_HEIGHT[0] <= bh <= _LETTER_HEIGHT[1] * h:
            continue
        shaped = _LETTER_ASPECT[0] <= bw / bh <= _LETTER_ASPECT[1]
        if shaped and _LETTER_FILL[0] <= len(points) / (bw * bh) <= _LETTER_FILL[1]:
            letters.append((x, y, bw, bh))
    covered = sum(_line_area(grey, line) for line in _lines(_without_nested(letters)))
    return covered / (h * w)


def _without_nested(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """MSER finds a letter several times over (and its holes); keep the biggest of each.

    Biggest first, a box is dropped when a box already kept covers more than half of it. Kept boxes
    are filed in a coarse grid, so a page full of letters doesn't compare every box with every other.
    """
    if not boxes:
        return np.zeros((0, 4), np.float32)
    b = np.array(boxes, np.float32)
    b = b[np.argsort(-(b[:, 2] * b[:, 3]), kind="stable")]
    cell = max(8, int(np.median(b[:, 3])))
    grid: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    keep = []
    for i, (x, y, w, h) in enumerate(b.tolist()):
        box = (x, y, x + w, y + h)
        cells = [
            (cx, cy)
            for cx in range(int(x) // cell, int(x + w - 1) // cell + 1)
            for cy in range(int(y) // cell, int(y + h - 1) // cell + 1)
        ]
        nearby = {kept for spot in cells for kept in grid.get(spot, ())}
        if any(_overlap(box, kept) > 0.5 * w * h for kept in nearby):
            continue
        keep.append(i)
        for spot in cells:
            grid.setdefault(spot, []).append(box)
    return b[keep]


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """The area two (x1, y1, x2, y2) boxes share."""
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


_SAME_LINE = 0.25  # the middles of two letters on a line differ by at most this share of the taller one
_SAME_SIZE = 1.35  # and the taller one is at most this much taller


def _lines(letters: np.ndarray) -> list[np.ndarray]:
    """Groups of at least three letters side by side: similar height, on one line, close together.

    Letters are filed in bands by the height of their middles, so each one is only compared with
    the letters that could share its line, not with every letter above and below it on the page.
    """
    letters = letters[np.argsort(letters[:, 0], kind="stable")]
    n = len(letters)
    if n == 0:
        return []
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    xs, widths, heights = letters[:, 0].tolist(), letters[:, 2].tolist(), letters[:, 3].tolist()
    middles = (letters[:, 1] + letters[:, 3] / 2).tolist()
    band = float(np.median(letters[:, 3]))
    bands: dict[int, list[int]] = {}
    for i, middle in enumerate(middles):
        bands.setdefault(int(middle // band), []).append(i)  # in order of i, so sorted by x too
    for i in range(n):
        xi, wi, hi, mi = xs[i], widths[i], heights[i], middles[i]
        reach = _SAME_LINE * _SAME_SIZE * hi  # the furthest a partner's middle can be: it may be taller
        for key in range(int((mi - reach) // band), int((mi + reach) // band) + 1):
            row = bands.get(key, [])
            for k in range(bisect.bisect_right(row, i), len(row)):
                j = row[k]
                xj, wj, hj = xs[j], widths[j], heights[j]
                if xj - xi > 6 * hi:
                    break  # sorted by x: everything further right is too far away
                tall, short = max(hi, hj), min(hi, hj)
                gap = xj - (xi + wi)
                if gap > tall or gap < -0.3 * min(wi, wj):
                    continue
                if abs(mi - middles[j]) <= _SAME_LINE * tall and tall / short <= _SAME_SIZE:
                    parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [letters[members] for members in groups.values() if len(members) >= _LINE_MIN_LETTERS]


def _line_area(grey: np.ndarray, line: np.ndarray) -> float:
    """The line's area if it reads as text (dark-on-light or light-on-dark, with calm space around), else 0."""
    x1, y1 = int(line[:, 0].min()), int(line[:, 1].min())
    x2, y2 = int((line[:, 0] + line[:, 2]).max()), int((line[:, 1] + line[:, 3]).max())
    height = float(np.median(line[:, 3]))
    if x2 - x1 < 3 * height:
        return 0.0
    patch = grey[y1:y2, x1:x2]
    _, ink = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    values = patch.astype(np.float32)
    fg, bg = values[ink > 0], values[ink == 0]
    if len(fg) < 5 or len(bg) < 5:
        return 0.0
    between = len(fg) * len(bg) / values.size**2 * (fg.mean() - bg.mean()) ** 2
    if between / (values.var() + 1e-6) < _LINE_SEPARATION or abs(fg.mean() - bg.mean()) < _LINE_CONTRAST:
        return 0.0
    band = max(2, int(0.35 * height))
    around = [grey[max(0, y1 - band - 1) : max(0, y1 - 1), x1:x2], grey[y2 + 1 : y2 + band + 1, x1:x2]]
    calm = [part.std() for part in around if part.size > 10]
    if not calm or min(calm) >= _LINE_QUIET * values.std():
        return 0.0
    return (x2 - x1) * height


# --- Illustration -----------------------------------------------------------------------
# Two kinds of picture count, and the score is the stronger of the two:
# - flat digital art (cartoons, vector-style graphics): small blocks are almost all exactly two
#   tones, as at the border of two flat colour fills;
# - painted or printed art (such as a woodblock print): a few inks explain the colours almost
#   completely, colour changes are abrupt rather than gradual, and there are edges everywhere.
# Photos and AI renders have shading and soft transitions, so neither usually fits them. A photo
# of one object cut out on a plain background comes closest: about 40% of its busy blocks are two
# tones, against 51-66% for a cartoon of 1200 px (docs/RESEARCH.md §2b).

_BLOCK_TONES = 8  # block size (px, at _WORKING_EDGE) for the two-tone test
_BUSY_BLOCK = 8.0  # grey-level spread of a block worth testing
_TWO_TONE = 0.08  # leftover variance after splitting a block into its two best tones


def _illustration_score(small: np.ndarray) -> float:
    if min(small.shape[:2]) < 4 * _BLOCK_TONES:
        return 0.0
    smooth = cv2.medianBlur(small, 3)  # paper grain and sensor noise aren't detail here
    return max(_flat_art_score(smooth), _painted_art_score(small, smooth))


def _flat_art_score(smooth: np.ndarray) -> float:
    b = _BLOCK_TONES
    grey = _grey(smooth)
    h, w = grey.shape[0] // b * b, grey.shape[1] // b * b
    blocks = grey[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(-1, b * b)
    busy = blocks[blocks.std(axis=1) >= _BUSY_BLOCK]
    if len(busy) < 10:
        return 0.0
    leftover = _two_tone_leftover(busy)
    typical, two_tone_share = float(np.median(leftover)), float((leftover < _TWO_TONE).mean())
    return min(_ramp(typical, 0.14, 0.10), _ramp(two_tone_share, 0.41, 0.52))


def _two_tone_leftover(blocks: np.ndarray) -> np.ndarray:
    """Per block: the variance left after splitting its values into the best two groups, as a share."""
    values = np.sort(blocks, axis=1)
    n = values.shape[1]
    sums, squares = np.cumsum(values, axis=1), np.cumsum(values * values, axis=1)
    left = np.arange(1, n)
    l1, l2 = sums[:, :-1], squares[:, :-1]
    r1, r2 = sums[:, -1:] - l1, squares[:, -1:] - l2
    within = (l2 - l1**2 / left) + (r2 - r1**2 / (n - left))
    return within.min(axis=1) / n / (blocks.var(axis=1) + 1e-6)


def _painted_art_score(small: np.ndarray, smooth: np.ndarray) -> float:
    inks, colourful = _palette_fit(small)
    steps = smooth.astype(np.int16)
    jump = np.maximum(
        np.abs(np.diff(steps, axis=1)).max(axis=-1)[:-1, :], np.abs(np.diff(steps, axis=0)).max(axis=-1)[:, :-1]
    )
    gradual = float(((jump > 4) & (jump < 25)).mean())
    abrupt = float((jump >= 40).mean())
    grey = _grey(smooth)
    edges = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1)) / 8 > 15
    busy = float((cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).mean())
    return min(
        _ramp(inks, 0.89, 0.915) * _ramp(colourful, 3.0, 8.0),
        _ramp(abrupt / (gradual + 1e-3), 0.20, 0.32),
        _ramp(busy, 0.30, 0.40),
    )


_PALETTE_EDGE = 256
_INKS = 4


def _palette_fit(small: np.ndarray) -> tuple[float, float]:
    """How much of the colour variation four colours explain (0..1), and the mean chroma (Lab).

    A few rounds of k-means, started from brightness quartiles, so the answer never varies.
    """
    tiny = cv2.medianBlur(_shrink(small, _PALETTE_EDGE), 3)
    lab = cv2.cvtColor(tiny, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    total = float(((lab - lab.mean(axis=0)) ** 2).sum(axis=1).mean())
    chroma = float(np.hypot(lab[:, 1], lab[:, 2]).mean())
    if total < 1e-6:
        return 1.0, chroma
    order = np.argsort(lab[:, 0], kind="stable")
    centres = lab[order[((np.arange(_INKS) + 0.5) / _INKS * len(lab)).astype(int)]]
    for _ in range(12):
        nearest = ((lab[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
        for k in range(_INKS):
            members = lab[nearest == k]
            if len(members):
                centres[k] = members.mean(axis=0)
    leftover = float(((lab - centres[nearest]) ** 2).sum(axis=1).mean())
    return 1.0 - leftover / total, chroma
