"""Opening and saving pictures, plus the small previews the app shows.

Whatever comes in (phone HEIC, camera RAW, CMYK JPEG, 16-bit TIFF, animated GIF, SVG, EXR, sideways
EXIF photo), the rest of the app only ever sees upright 8-bit RGB, an alpha channel when some pixel
is actually see-through, and an RGB colour profile to put back on the result.

Decoders, tried in this order until one works:
1. Pillow: JPEG, PNG, WebP, AVIF, TIFF, BMP, GIF, JPEG 2000, ICO, TGA, PPM, DDS, QOI and the PSD
   composite; HEIC/HEIF through pillow-heif.
2. rawpy (LibRaw) for camera RAW, developed with the camera's white balance straight into sRGB.
3. resvg for SVG, drawn directly at the size asked for (vectors need no AI).
4. The bundled FFmpeg for EXR and Radiance HDR (brought down to standard range), and for DDS, QOI,
   PSD and TIFF files Pillow can't read.
5. On macOS, `sips` as the last resort: JPEG XL and anything else the Mac itself can open.
PDF, EPS and Illustrator files get a friendly "export it as PNG" message instead.

Colour: CMYK, grey and Lab pictures become sRGB through their own profile. RGB pictures keep their
profile (Display P3 from iPhones, Adobe RGB from cameras), which goes back onto the result, so the
most vivid colours aren't squeezed into sRGB.
"""

from __future__ import annotations

import contextlib
import errno
import functools
import gzip
import io
import json
import math
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TypeVar
from xml.parsers import expat

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener

from upscaler.config import SPACE_RESERVE_BYTES, SPACE_RESERVE_TEXT, Cancelled, UserError
from upscaler.sizing import describe_size, free_disk_bytes

register_heif_opener()  # iPhone photos (HEIC); pillow-heif also applies their rotation

T = TypeVar("T")

# The biggest picture the app opens: a square 16K one, so its own 16K results open too (Pillow's
# own guard refuses anything over 179 MP). Bigger files are refused with a friendly message, which
# also keeps a small but malicious upload from unpacking into many gigabytes.
MAX_OPEN_PIXELS = 16384 * 16384
Image.MAX_IMAGE_PIXELS = MAX_OPEN_PIXELS  # Pillow warns above this and refuses above twice it
# EXR and Radiance HDR are tone-mapped from 32-bit floats (12-16 bytes a pixel), so they stop at 8K:
# about 1 GB while converting, where a 16K one would need 4 GB.
_MAX_HDR_PIXELS = 8192 * 8192

# --- What can come in ------------------------------------------------------------

RAW_EXTS = frozenset({".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"})
_VECTOR_EXTS = frozenset({".svg", ".svgz"})
SUPPORTED_IMAGE_EXTS = frozenset(
    {
        ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".webp", ".heic", ".heif", ".avif",
        ".tif", ".tiff", ".bmp", ".dib", ".gif", ".jp2", ".j2k", ".jpf", ".ico", ".tga",
        ".ppm", ".pgm", ".pbm", ".pnm", ".psd", ".dds", ".qoi", ".exr", ".hdr",
    }  # fmt: skip
    | RAW_EXTS
    | _VECTOR_EXTS
    | ({".jxl"} if sys.platform == "darwin" else set())  # JPEG XL: only macOS can open it (via sips)
)
SUPPORTED_VIDEO_EXTS = frozenset(
    {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".mts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".ts"}
)

# Page-description documents: their renderers are AGPL or need Ghostscript, so people export a page instead.
_DOCUMENTS = {".pdf": "a PDF", ".ai": "an Illustrator", ".eps": "an EPS", ".epsf": "an EPS", ".ps": "a PostScript"}

# What a file is, from its name and first bytes; picks the decoders to try.
_RASTER, _RAW, _VECTOR, _HDR, _JXL = "raster", "raw", "vector", "hdr", "jxl"
_SNIFF_BYTES = 4096
_EXR_MAGIC = b"\x76\x2f\x31\x01"
_RADIANCE_MAGIC = (b"#?RADIANCE", b"#?RGBE")
_JXL_MAGIC = (b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
# Formats FFmpeg reads when Pillow can't (16-bit PSD, float TIFF, unusual DDS), by name or by first bytes.
_FFMPEG_EXTS = frozenset({".dds", ".qoi", ".psd", ".tif", ".tiff"})
_FFMPEG_MAGIC = (b"DDS ", b"qoif", b"8BPS", *_TIFF_MAGIC)
_MAGIC_LABELS = {_EXR_MAGIC: "EXR", b"DDS ": "DDS", b"qoif": "QOI", b"8BPS": "PSD"}
_PSD_HEADER_BYTES = 26
_PSD_ICC_RESOURCE = 0x040F  # Photoshop's image resource for the document's colour profile
_PSD_RESOURCE_MIN_BYTES = 12  # signature, id, an empty name (2 bytes with padding) and the length
_SUFFIX_LABELS = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF", ".heic": "HEIC", ".heif": "HEIC",
    ".webp": "WebP", ".jxl": "JPEG XL", ".jp2": "JPEG 2000", ".svgz": "SVG",
}  # fmt: skip
# Pillow's format names, as people know them.
_PILLOW_LABELS = {"MPO": "JPEG", "WEBP": "WebP", "HEIF": "HEIC", "JPEG2000": "JPEG 2000", "DIB": "BMP"}
# Formats whose extra "frames" aren't pages: Photoshop layers, phones' hidden gain maps, icon sizes.
_ONE_PICTURE_FORMATS = frozenset({"PSD", "MPO", "ICO", "ICNS"})
_ANIMATED_FORMATS = frozenset({"GIF", "WebP", "PNG", "AVIF"})
_DRAFTABLE_FORMATS = frozenset({"JPEG", "MPO"})  # can decode straight at 1/2, 1/4 or 1/8 size
_TRUNCATED = "image file is truncated"  # how Pillow says a file ran out before the picture did

_ORIENTATION = 0x0112
_TURNED_ORIENTATIONS = (5, 6, 7, 8)  # EXIF orientations that swap width and height
# Above this many pixels, inspect_image reports any alpha channel as see-through instead of decoding
# the whole picture to check (the full open still drops an alpha channel that is fully opaque).
_ALPHA_CHECK_PIXELS = 64_000_000
_ALPHA_PREVIEW_SIDE = 256  # an SVG is drawn this big to see whether its background is see-through

_CSS_DPI = 96.0  # SVG lengths in mm, pt or in are drawn at 96 per inch, as browsers do
_SVG_UNITS = {"px": 1.0, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4, "pt": 96 / 72, "pc": 16.0, "em": 16.0, "ex": 8.0}
_SVG_LENGTH = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(px|in|cm|mm|pt|pc|em|ex|%)?\s*$")
_MAX_SVG_BYTES = 64 * 10**6
# An XML start tag (name, attributes, end), and one name="value" attribute inside it.
_XML_START_TAG = re.compile(rb"""<([^\s/>]+)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*(/?>)""")
_XML_ATTRIBUTE = re.compile(rb"""([^\s=/>]+)\s*=\s*("[^"]*"|'[^']*')""")
_WEB_LINKS = ("http://", "https://", "mailto:")
_URL_SKIPPED = "".join(map(chr, range(33)))  # control characters and space, skipped before a URL's scheme

_SRGB_GAMMA = (2.4, 12.92)  # LibRaw's (power, toe slope) for the sRGB curve
_DECODE_TIMEOUT_SECONDS = 120
_REC709_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_HDR_CEILING = 65504.0  # the largest half-float, so a stray infinity doesn't swamp the picture
_HDR_WHITE_PERCENTILE = 99.9
_HDR_SAMPLE_PIXELS = 1_000_000
_HDR_BAND_PIXELS = 1 << 20  # tone-mapped about a million pixels at a time, so the working copies stay small
# Shrinks an HDR thumbnail inside FFmpeg. zscale keeps values brighter than white, which FFmpeg's
# ordinary scaler would clip, flattening the highlights before they are tone-mapped.
_HDR_SHRINK_FILTER = "zscale=w={width}:h={height}:filter=bilinear"
_EXR_DEEP_DATA = 0x800  # EXR version flag: "deep" pixels, which FFmpeg can't read
_EXR_MAX_ATTRIBUTES = 10_000
_EXR_MAX_NAME_BYTES = 256  # attribute and type names: 255 characters at most, plus the closing zero
_EXR_MAX_CHANNEL_LIST_BYTES = 1 << 20
_RADIANCE_MAX_HEADER_BYTES = 1 << 16
_RADIANCE_SIZE = re.compile(rb"[-+]([XY]) (\d+) [-+]([XY]) (\d+)")  # e.g. "-Y 1080 +X 1920"

CMYK_NOTE = "Converted from CMYK print colours to RGB screen colours."
HDR_NOTE = "HDR image converted to standard range."
RAW_NOTE = "Camera RAW developed with the camera's own colour settings. Edits made in other apps aren't included."
OUTSIDE_LINKS_NOTE = (
    "This SVG links to pictures outside the file, which were left out. Embed them in the SVG to include them."
)

# --- What can go out -------------------------------------------------------------

SUFFIXES = {"jpeg": ".jpg", "png": ".png", "tiff": ".tif"}  # the output formats offered today
OUTPUT_SUFFIXES = {**SUFFIXES, "webp": ".webp", "heic": ".heic", "avif": ".avif"}  # every format save_image writes
SAVE_FORMATS = (*SUFFIXES, "webp", "original")  # the choices save_image takes ("original": the source's own)
_ACCEPTED_SUFFIXES = {
    "jpeg": (".jpg", ".jpeg"), "png": (".png",), "tiff": (".tif", ".tiff"),
    "webp": (".webp",), "heic": (".heic", ".heif"), "avif": (".avif",),
}  # fmt: skip
_PIL_FORMATS = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF", "webp": "WEBP", "heic": "HEIF", "avif": "AVIF"}
_FORMAT_NAMES = {"jpeg": "JPEG", "png": "PNG", "tiff": "TIFF", "webp": "WebP", "heic": "HEIC", "avif": "AVIF"}
FORMATS_WITH_ALPHA = ("png", "tiff", "webp", "heic", "avif")
# "Same as original": source formats that can be written back, by any name they go by (case ignored).
_ORIGINAL_FORMATS = {
    "jpeg": "jpeg", "jpg": "jpeg", "mpo": "jpeg", "png": "png", "tiff": "tiff", "tif": "tiff",
    "webp": "webp", "heic": "heic", "heif": "heic", "avif": "avif",
}  # fmt: skip
# Format limits: JPEG and AVIF store sizes in 16 bits, WebP in 14; x265 (HEIC) stops at HEVC level 7.2.
_MAX_SIDE = {"jpeg": 65_535, "webp": 16_383, "avif": 65_536}
_MAX_PIXELS = {"heic": 142_606_336}

JPEG_QUALITY = 95
_LOSSY_QUALITY = {"webp": 95, "heic": 90, "avif": 90}  # each format's own scale; all visually lossless
MAX_DPI = 65_535  # JPEG stores DPI in 16 bits
# Above this, PNGs are compressed a little less (level 3 instead of Pillow's 6): a 16K PNG then
# saves in about half the time for a file only slightly bigger.
FAST_PNG_PIXELS = 100_000_000
_FAST_PNG_LEVEL = 3
_TIFF_PREDICTOR = 317  # TIFF tag
_HORIZONTAL_DIFFERENCING = 2
_X_RESOLUTION, _Y_RESOLUTION, _RESOLUTION_UNIT, _INCHES = 0x011A, 0x011B, 0x0128, 2  # EXIF tags
_PREVIEW_QUALITY = 85
_THUMBNAIL_QUALITY = 85
_CROP_QUALITY = 92
_CHECKER_SQUARE = 8  # px; transparent areas of thumbnails show as a light checkerboard
_CHECKER_SHADES = (255, 224)

# File systems allow 255 bytes per name; keep room for "-NN" and the temporary ".part" name.
_MAX_NAME_BYTES = 240
# A save that fails without saying why, with less than this left on the disk, ran out of space.
_FULL_DISK_BYTES = 64 * 10**6

_HIGH_BIT_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")
_WITHOUT_ALPHA = {"RGBA": "RGB", "LA": "L"}
_ICC_SPACE_FOR_MODE = {"L": "GRAY", "CMYK": "CMYK"}
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


class ImageFileError(Exception):
    """A picture couldn't be opened or saved. The message is written for the person using the app."""


@dataclass
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8, upright
    alpha: np.ndarray | None  # (H, W) uint8; None unless some pixel is see-through
    icc: bytes | None  # RGB colour profile of `rgb`, to embed in the result
    name: str  # the original file name
    format: str | None = None  # what the file was, e.g. "JPEG", "HEIC", "RAW (CR3)"; feeds save_image's "original"
    notes: list[str] = field(default_factory=list)  # e.g. "only the first page is upscaled", for the result


@dataclass
class ImageInfo:
    width: int  # upright, as the picture will be upscaled
    height: int
    format: str
    has_alpha: bool  # some pixel is see-through
    frames: int  # pages or animation frames; only the first is used
    is_vector: bool  # an SVG: drawn at full size directly, no AI needed
    warnings: list[str]


@dataclass
class _Picture:
    """A decoded picture before the shared clean-up (rotation, alpha, colour)."""

    image: Image.Image  # the first or main frame, loaded, as stored: EXIF rotation still to apply
    format: str
    frames: int = 1
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _File:
    path: Path
    suffix: str  # lower case
    head: bytes  # the first few kilobytes
    kind: str  # _RASTER, _RAW, _VECTOR, _HDR or _JXL

    @property
    def name(self) -> str:
        return self.path.name


class _NotDecoded(Exception):
    """This decoder can't read the file; try the next one."""


class _Incomplete(Exception):
    """The file stops part-way (a copy or download that didn't finish), so no decoder should try:
    macOS's sips would fill the missing part with flat grey."""


# --- Opening ----------------------------------------------------------------------


def load_image(path: Path, *, vector_size: tuple[int, int] | None = None) -> LoadedImage:
    """Open any supported picture as upright 8-bit RGB (plus alpha and profile).

    Multi-page and animated files give their first frame, with a note. SVGs are drawn at
    `vector_size` (width, height) when given, otherwise at their own size.
    """
    path = Path(path)

    def load() -> LoadedImage:
        with _open_picture(_sniff(path), vector_size=vector_size) as picture:
            icc = picture.image.info.get("icc_profile") or None
            image, alpha = _split_alpha(_upright(picture.image))
            rgb, icc = _to_rgb(image, icc)
            pixels = np.asarray(rgb, dtype=np.uint8)
        return LoadedImage(pixels, alpha, icc, path.name, picture.format, picture.notes)

    return _friendly(path, load)


def inspect_image(path: Path) -> ImageInfo:
    """Size, format and warnings for the upload screen, without decoding the picture where possible.

    Pillow formats read just the header (a picture with an alpha channel is decoded to check whether
    any pixel is really see-through); RAW reads LibRaw's sizes; SVG reads its width, height and
    viewBox; EXR and HDR read their headers (an EXR alpha channel counts as see-through). The formats
    only `sips` reads are decoded, as they have no cheap header path.
    """
    path = Path(path)

    def inspect() -> ImageInfo:
        file = _sniff(path)
        if file.kind == _VECTOR:
            return _inspect_svg(file)
        quick = None
        if file.kind == _RAW:
            quick = _inspect_raw(file)
        elif file.kind == _HDR:
            quick = _inspect_hdr(file)
        elif file.kind in (_RASTER, _JXL):
            quick = _inspect_with_pillow(file)
        if quick is not None:
            return quick
        with _open_picture(file) as picture:
            width, height = _upright_size(picture.image)
            has_alpha = _split_alpha(picture.image)[1] is not None
            return ImageInfo(width, height, picture.format, has_alpha, picture.frames, False, list(picture.notes))

    return _friendly(path, inspect)


def make_thumbnail(path: Path, max_side: int = 480) -> bytes:
    """A small upright JPEG of any supported picture, for browsers that can't show HEIC, RAW or TIFF.

    Big JPEGs decode at reduced size, RAW files at half size, EXR and HDR shrink inside FFmpeg, and
    SVGs are drawn at the thumbnail size directly. See-through areas show as a light checkerboard.
    """
    path = Path(path)

    def build() -> bytes:
        with _open_picture(_sniff(path), preview_side=max_side) as picture:
            icc = picture.image.info.get("icc_profile") or None
            image, alpha = _split_alpha(_upright(picture.image))
            thumbnail, icc = _to_rgb(image, icc)
            if alpha is not None:  # shrink colour and alpha together, so edges don't pick up hidden colours
                thumbnail = Image.fromarray(np.dstack((np.asarray(thumbnail, dtype=np.uint8), alpha)))
            # In place: `thumbnail` may be the decoded picture itself, which is closed after this block.
            thumbnail.thumbnail((max_side, max_side), Image.LANCZOS)
            if alpha is not None:
                thumbnail = _on_checkerboard(thumbnail)
            return _jpeg_bytes(thumbnail, _THUMBNAIL_QUALITY, icc)

    return _friendly(path, build)


def render_vector(path: Path, out_size: tuple[int, int]) -> np.ndarray:
    """Draw an SVG at exactly `out_size` (width, height) as (H, W, 4) uint8 RGBA.

    `load_image(path, vector_size=...)` gives the same drawing split into RGB and alpha.
    """
    width, height = out_size
    if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (width, height)):
        raise ValueError(f"The size to draw at must be two whole numbers above 0, not {out_size!r}.")
    if width * height > MAX_OPEN_PIXELS:
        raise UserError(f"{width:,} × {height:,} pixels is too big to draw.")
    path = Path(path)

    def render() -> np.ndarray:
        file = _sniff(path)
        if file.kind != _VECTOR:
            raise ImageFileError(f"{path.name} isn't a vector (SVG) picture.")
        svg, _notes = _read_svg(file)
        return _draw_svg(svg, (width, height), file.name)

    return _friendly(path, render)


def _friendly(path: Path, action: Callable[[], T]) -> T:
    """Run `action`, turning any trouble with the file into an ImageFileError worded for people."""
    try:
        return action()
    except _Incomplete as exc:
        raise ImageFileError(
            f"{path.name} seems to be incomplete, as if it didn't finish copying or downloading. "
            "Copy or download it again, then drop that in."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageFileError(f"{path.name} is too big for Re:Solution to open.") from exc
    except (OSError, ValueError, SyntaxError) as exc:  # SyntaxError: Pillow's "broken file" signal
        raise ImageFileError(_damaged_message(path.name)) from exc


@contextlib.contextmanager
def _open_picture(
    file: _File, *, vector_size: tuple[int, int] | None = None, preview_side: int | None = None
) -> Iterator[_Picture]:
    """Decode with the first decoder that can read `file`. `preview_side` asks only for a picture
    about that big (a faster decode where the format allows it)."""
    if file.kind == _VECTOR:
        picture = _decode_svg(file, vector_size, preview_side)
    else:
        picture = _first_decoder_that_works(file, preview_side)
    if picture.image.mode == "CMYK":
        picture.notes.append(CMYK_NOTE)
    try:
        yield picture
    finally:
        picture.image.close()


def _first_decoder_that_works(file: _File, preview_side: int | None) -> _Picture:
    for decoder in _decoders_for(file.kind):
        try:
            return decoder(file, preview_side)
        except _NotDecoded:
            continue
    raise ImageFileError(_failure_message(file))


def _decoders_for(kind: str) -> tuple[Callable[[_File, int | None], _Picture], ...]:
    if kind == _RAW:  # never Pillow: it would open the small preview many RAW files carry as a TIFF
        return _decode_raw, _decode_with_sips
    if kind == _HDR:
        return _decode_hdr_with_ffmpeg, _decode_with_sips
    return _decode_with_pillow, _decode_with_ffmpeg, _decode_with_sips


def _sniff(path: Path) -> _File:
    """Work out what kind of file this is, refusing documents and videos with a friendly message."""
    suffix = path.suffix.lower()
    with path.open("rb") as f:
        head = f.read(_SNIFF_BYTES)
    document = _DOCUMENTS.get(suffix) or _document_by_content(head)
    if document:
        raise ImageFileError(
            f"{path.name} is {document} document. Export the page as PNG or TIFF at the size you need, "
            "then drop that in."
        )
    if suffix in SUPPORTED_VIDEO_EXTS:
        raise ImageFileError(f"{path.name} is a video. Open the Videos tab to make it 4K.")
    if suffix in RAW_EXTS:
        kind = _RAW
    elif suffix in _VECTOR_EXTS or _looks_like_svg(head):
        kind = _VECTOR
    elif suffix == ".jxl" or head.startswith(_JXL_MAGIC):
        kind = _JXL
    elif suffix in (".exr", ".hdr") or head.startswith((_EXR_MAGIC, *_RADIANCE_MAGIC)):
        kind = _HDR
    else:
        kind = _RASTER
    return _File(path, suffix, head, kind)


def _document_by_content(head: bytes) -> str | None:
    if head.startswith(b"%PDF"):
        return "a PDF"
    if head.startswith(b"\xc5\xd0\xd3\xc6") or (head.startswith(b"%!PS") and b"EPSF" in head[:64]):
        return "an EPS"
    if head.startswith(b"%!PS"):
        return "a PostScript"
    return None


def _looks_like_svg(head: bytes) -> bool:
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return text.startswith(b"<") and b"<svg" in head and b"<html" not in head.lower()


def _on_mac() -> bool:
    return sys.platform == "darwin"


def _check_size(size: tuple[int, int], limit: int | None = None) -> None:
    width, height = size
    if width * height > (MAX_OPEN_PIXELS if limit is None else limit):
        raise Image.DecompressionBombError(f"{width}x{height}")


def _label(file: _File) -> str:
    """A short name for the file's format, from its first bytes or else its name."""
    for magic, label in _MAGIC_LABELS.items():
        if file.head.startswith(magic):
            return label
    if file.head.startswith(_RADIANCE_MAGIC):
        return "HDR"
    if file.head.startswith(_TIFF_MAGIC):
        return "TIFF"
    return _SUFFIX_LABELS.get(file.suffix) or file.suffix.lstrip(".").upper() or "picture"


def _failure_message(file: _File) -> str:
    """What to tell people when no decoder could read the file."""
    name = file.name
    if file.kind == _RAW:
        return (
            f"{name} couldn't be opened. Re:Solution can't read RAW files from this camera yet. Export it "
            "as JPEG or TIFF from your photo app, then drop that in."
        )
    if file.kind == _JXL and not _on_mac():
        return (
            f"{name} is a JPEG XL picture, which Re:Solution can only open on a Mac. Save it as PNG or "
            "JPEG first, then drop that in."
        )
    if file.suffix == ".psd" or file.head.startswith(b"8BPS"):
        return f"{name} couldn't be opened. In Photoshop, save a copy as PNG or TIFF, then drop that in."
    if file.suffix in SUPPORTED_IMAGE_EXTS or file.kind in (_JXL, _HDR):
        return _damaged_message(name)
    return (
        f"This file type isn't an image Re:Solution can read ({name}). JPEG, PNG, HEIC, WebP, TIFF, "
        "camera RAW and SVG all work."
    )


def _damaged_message(name: str) -> str:
    return f"{name} couldn't be opened as a picture. It may be damaged, or saved in a way Re:Solution can't read."


# --- Decoder: Pillow (and pillow-heif) -----------------------------------------------


def _decode_with_pillow(file: _File, preview_side: int | None) -> _Picture:
    image = _pillow_open(file.path)
    try:
        _check_size(image.size)
        label = _pillow_label(image.format)
        frames = _frame_count(image)
        if preview_side and image.format in _DRAFTABLE_FORMATS:
            image.draft(image.mode, (preview_side, preview_side))
        _pillow_load(image, from_file=True)
    except BaseException:
        image.close()
        raise
    return _Picture(image, label, frames, _frame_notes(label, frames))


def _pillow_open(source: Path | str | BinaryIO) -> Image.Image:
    try:
        return Image.open(source)
    except (Image.DecompressionBombError, MemoryError):
        raise
    except Exception as exc:  # broken files make Pillow's plugins raise all sorts
        raise _NotDecoded() from exc


def _pillow_load(image: Image.Image, *, from_file: bool = False) -> None:
    """Decode the pixels. `from_file`: `image` is the person's own file, so running out of data means
    the file is cut off (raises _Incomplete), not just that Pillow can't read it."""
    try:
        image.load()
    except (Image.DecompressionBombError, MemoryError):
        raise
    except Exception as exc:
        if from_file and isinstance(exc, OSError) and str(exc).startswith(_TRUNCATED):
            raise _Incomplete() from exc
        raise _NotDecoded() from exc


def _pillow_label(pil_format: str | None) -> str:
    return _PILLOW_LABELS.get(pil_format or "", pil_format or "picture")


def _frame_count(image: Image.Image) -> int:
    if image.format in _ONE_PICTURE_FORMATS:
        return 1
    try:
        return max(1, int(getattr(image, "n_frames", 1)))
    except Exception:  # a broken frame list: the first frame may still open
        return 1


def _frame_notes(label: str, frames: int) -> list[str]:
    if frames <= 1:
        return []
    if label in _ANIMATED_FORMATS:
        return [
            f"This {label} is animated. Only its first frame is upscaled, because animations aren't "
            "upscaled frame by frame yet."
        ]
    if label == "HEIC":
        return [f"This HEIC file holds {frames} pictures. Only the main one is upscaled."]
    return [f"This {label} has {frames} pages. Only the first page is upscaled."]


def _inspect_with_pillow(file: _File) -> ImageInfo | None:
    try:
        image = _pillow_open(file.path)
    except _NotDecoded:
        return None
    with image:
        _check_size(image.size)
        label = _pillow_label(image.format)
        frames = _frame_count(image)
        try:
            has_alpha = _has_see_through_pixels(image)
        except _NotDecoded:  # the header reads but the pixels don't: let the other decoders try
            return None
        notes = _frame_notes(label, frames) + ([CMYK_NOTE] if image.mode == "CMYK" else [])
        return ImageInfo(*_upright_size(image), label, has_alpha, frames, False, notes)


def _has_see_through_pixels(image: Image.Image) -> bool:
    if not image.has_transparency_data:
        return False
    if image.width * image.height > _ALPHA_CHECK_PIXELS:
        return True
    _pillow_load(image, from_file=True)
    return _split_alpha(image)[1] is not None


def _upright_size(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    # Pillow's TIFF reader turns the picture itself as it decodes, and already reports the turned size.
    if image.format == "TIFF" or _orientation(image) not in _TURNED_ORIENTATIONS:
        return width, height
    return height, width


def _orientation(image: Image.Image) -> int:
    # Pillow decodes a whole PNG to look for EXIF after the pixels (a decoded PNG has it in `info`
    # already). PNGs turned that way are rare, so a header-only look skips them; opening still turns them.
    if image.format == "PNG" and "exif" not in image.info:
        return 1
    try:
        return image.getexif().get(_ORIENTATION, 1)
    except Exception:  # a damaged EXIF block: the picture opens as stored
        return 1


# --- Decoder: camera RAW (rawpy / LibRaw) -----------------------------------------------


def _decode_raw(file: _File, preview_side: int | None) -> _Picture:
    import rawpy  # imported here, so a LibRaw problem can never stop ordinary pictures from opening

    try:
        with rawpy.imread(str(file.path)) as raw:
            _check_size(_raw_size(raw.sizes))
            # Camera white balance, sRGB colours with the sRGB tone curve (LibRaw's default curve is
            # BT.709's), 8 bits. LibRaw also turns the picture upright.
            rgb = raw.postprocess(
                use_camera_wb=True, gamma=_SRGB_GAMMA, output_bps=8, half_size=preview_side is not None
            )
    except (rawpy.LibRawError, OSError) as exc:
        raise _NotDecoded() from exc
    return _Picture(Image.fromarray(rgb), _raw_label(file), notes=[RAW_NOTE])


def _inspect_raw(file: _File) -> ImageInfo | None:
    import rawpy

    try:
        with rawpy.imread(str(file.path)) as raw:
            size = _raw_size(raw.sizes)
    except (rawpy.LibRawError, OSError):
        return None
    _check_size(size)
    return ImageInfo(size[0], size[1], _raw_label(file), False, 1, False, [RAW_NOTE])


def _raw_size(sizes) -> tuple[int, int]:
    """The developed picture's upright size: LibRaw stretches non-square pixels, then applies the turn."""
    width, height = sizes.width, sizes.height
    if sizes.pixel_aspect > 1.005:
        width = int(width * sizes.pixel_aspect + 0.5)
    elif sizes.pixel_aspect < 0.995:
        height = int(height / sizes.pixel_aspect + 0.5)
    return (height, width) if sizes.flip in (5, 6) else (width, height)


def _raw_label(file: _File) -> str:
    return f"RAW ({file.suffix.lstrip('.').upper()})"


# --- Decoder: SVG (resvg) -------------------------------------------------------------


def _decode_svg(file: _File, vector_size: tuple[int, int] | None, preview_side: int | None) -> _Picture:
    svg, notes = _read_svg(file)
    size = vector_size or _svg_declared_size(svg)
    if size and preview_side and not vector_size:
        size = _fit_within(size, preview_side)
    elif size and not vector_size:
        size = _fit_pixels(size, MAX_OPEN_PIXELS)
    if size:
        _check_size(size)
    rgba = _draw_svg(svg, size, file.name)
    _check_size((rgba.shape[1], rgba.shape[0]))
    return _Picture(Image.fromarray(rgba, "RGBA"), "SVG", notes=notes)


def _inspect_svg(file: _File) -> ImageInfo:
    svg, notes = _read_svg(file)
    size = _svg_declared_size(svg)
    if size:
        size = _fit_pixels(size, MAX_OPEN_PIXELS)  # as load_image will draw it
        drawn = _draw_svg(svg, _fit_within(size, _ALPHA_PREVIEW_SIDE), file.name)
    else:  # no usable size in the file: resvg sizes it from the drawing itself
        drawn = _draw_svg(svg, None, file.name)
        size = (drawn.shape[1], drawn.shape[0])
        _check_size(size)
    return ImageInfo(size[0], size[1], "SVG", bool(drawn[..., 3].min() < 255), 1, True, notes)


def _read_svg(file: _File) -> tuple[str, list[str]]:
    """The SVG's text with every link to another file removed, and a note when a picture was left out.

    resvg would otherwise read any picture an SVG names, so an SVG sent over the Wi-Fi could pull the
    host's private files into its result. Embedded pictures (data: links) and links inside the file stay.
    An SVG whose links can't all be taken out safely isn't drawn at all.
    """
    if file.path.stat().st_size > _MAX_SVG_BYTES:
        raise Image.DecompressionBombError(file.name)
    data = file.path.read_bytes()
    if data.startswith(b"\x1f\x8b"):  # .svgz
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as unpacked:
                data = unpacked.read(_MAX_SVG_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise ImageFileError(_svg_failure_message(file.name)) from exc
        if len(data) > _MAX_SVG_BYTES:
            raise Image.DecompressionBombError(file.name)
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    try:
        text, pictures_left_out = _without_outside_links(text)
    except _UnsafeSvg as exc:
        raise ImageFileError(_svg_failure_message(file.name)) from exc
    return text, [OUTSIDE_LINKS_NOTE] if pictures_left_out else []


class _UnsafeSvg(Exception):
    """An SVG with a link to another file that can't be taken out safely."""


def _without_outside_links(svg: str) -> tuple[str, bool]:
    """`svg` with each link to another file blanked in its tag, and whether any of them was a picture.

    expat finds the links as resvg will see them, and each is blanked in the tag expat found it in.
    Raises _UnsafeSvg for a link no tag spells out (one an entity or a DTD default adds), and for
    anything `_outside_links` refuses.
    """
    data = svg.encode("utf-8")
    edits: list[tuple[int, int, bytes]] = []  # (start, end, replacement), in document order
    pictures_left_out = False
    for offset, element, links in _outside_links(svg):
        tag = _XML_START_TAG.match(data, offset)
        if tag is None or tag.group(1).decode("utf-8") != element:
            raise _UnsafeSvg()  # the element came out of an entity, not from a tag at this spot
        blanked = set()
        for attribute in _XML_ATTRIBUTE.finditer(tag.group(2)):
            name = attribute.group(1).decode("utf-8")
            if name in links:
                start, end = (tag.start(2) + i for i in attribute.span(2))
                edits.append((start, end, attribute.group(2)[:1] * 2))  # the same quotes, nothing between
                blanked.add(name)
        if blanked != links.keys():
            raise _UnsafeSvg()  # a link the DTD adds as a default value
        # Web links aren't pictures resvg would draw anyway.
        pictures_left_out |= any(not link.strip().lower().startswith(_WEB_LINKS) for link in links.values())
    pieces, position = [], 0
    for start, end, replacement in edits:
        pieces += [data[position:start], replacement]
        position = end
    pieces.append(data[position:])
    return b"".join(pieces).decode("utf-8"), pictures_left_out


def _outside_links(svg: str) -> list[tuple[int, str, dict[str, str]]]:
    """Every start tag with links to other files, as (byte offset, element name, {attribute: link}).

    expat reads `svg` as resvg will, with entities expanded and DTD defaults filled in, so no spelling
    trick hides a link. Raises _UnsafeSvg for external entities (text pulled in from another file)
    and for anything expat can't read.
    """
    parser = expat.ParserCreate()
    found: list[tuple[int, str, dict[str, str]]] = []

    def on_start(element: str, attributes: dict[str, str]) -> None:
        links = {name: link for name, link in attributes.items() if _is_link(name) and not _is_inside_link(link)}
        if links:
            found.append((parser.CurrentByteIndex, element, links))

    def on_entity(_name: str, _is_parameter: bool, value: str | None, *_where: str | None) -> None:
        if value is None:  # its text lives in another file
            raise _UnsafeSvg()

    def refuse(*_args) -> None:
        raise _UnsafeSvg()

    parser.StartElementHandler = on_start
    parser.EntityDeclHandler = on_entity
    parser.ExternalEntityRefHandler = refuse
    try:
        parser.Parse(svg, True)  # a str, read as the UTF-8 resvg is given whatever the file declares
    except expat.ExpatError as exc:
        raise _UnsafeSvg() from exc
    return found


def _is_link(attribute: str) -> bool:
    """href, xlink:href, or href with any other prefix (any prefix can stand for XLink)."""
    return attribute.rpartition(":")[2].lower() == "href"


def _is_inside_link(link: str) -> bool:
    """Whether resvg can follow `link` without opening a file: no link at all, a part of the same SVG
    (#name), or a picture embedded as a data: URL.

    resvg reads anything else as a file path, even "#/../photo.png" or a data: link without its comma.
    """
    stripped = link.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return "/" not in link and "\\" not in link  # so at most a name inside the empty folder resvg gets
    # resvg's test for a data: URL: control characters and spaces skipped, any case, a comma before any "#".
    url = link.lstrip(_URL_SKIPPED)
    return url[:5].lower() == "data:" and "," in url[5:].partition("#")[0]


def _svg_declared_size(svg: str) -> tuple[int, int] | None:
    """The size resvg gives the SVG, in whole pixels; None when the file doesn't say (resvg then
    measures the drawing itself)."""
    root = _svg_root(svg)
    size = _svg_float_size(root[1]) if root else None
    return (max(1, int(size[0] + 0.5)), max(1, int(size[1] + 0.5))) if size else None


def _svg_float_size(attributes: dict[str, str]) -> tuple[float, float] | None:
    """The root's width and height in CSS pixels, following resvg's rules for missing values and %."""
    width_attr, height_attr = _svg_length(attributes.get("width")), _svg_length(attributes.get("height"))
    box = _svg_view_box(attributes.get("viewBox"))
    if box:
        width, height = _svg_resolve(width_attr, box[0]), _svg_resolve(height_attr, box[1])
        if width is not None and height is None:  # only one given: the other follows the viewBox's shape
            height = width * box[1] / box[0]
        elif height is not None and width is None:
            width = height * box[0] / box[1]
        elif width is None and height is None:
            width, height = box
    elif width_attr and height_attr and "%" not in (width_attr[1], height_attr[1]):
        width, height = _svg_resolve(width_attr, 0), _svg_resolve(height_attr, 0)
    else:
        return None
    if not width or not height or width <= 0 or height <= 0:
        return None
    return width, height


class _FoundRoot(Exception):
    pass


def _svg_root(svg: str) -> tuple[int, dict[str, str]] | None:
    """Where the outermost element starts (a byte offset into `svg` as UTF-8) and its attributes,
    entities expanded.

    Parsing stops there, and expat expands no external entities and caps runaway internal ones.
    """
    parser = expat.ParserCreate()

    def on_start(name: str, attributes: dict[str, str]) -> None:
        raise _FoundRoot(parser.CurrentByteIndex, attributes)

    parser.StartElementHandler = on_start
    try:
        parser.Parse(svg, True)  # a str, read as the UTF-8 resvg is given whatever the file declares
    except _FoundRoot as found:
        return found.args[0], found.args[1]
    except expat.ExpatError:
        return None
    return None


def _svg_at_size(svg: str, size: tuple[int, int]) -> str:
    """The SVG with its root element set to exactly `size`, so the drawing fills every pixel.

    resvg scales from the SVG's size rounded to whole pixels, so a fractional size (210 mm is 793.7 px)
    would leave a see-through strip several pixels wide along two edges of an 8K drawing. The viewBox
    (added from the old size if missing) keeps the content scaling; when the shapes agree, it is
    stretched to the last fraction of a pixel. Returns `svg` unchanged when the root can't be found.
    """
    data = svg.encode("utf-8")
    root = _svg_root(svg)
    tag = _XML_START_TAG.match(data, root[0]) if root else None
    old_size = _svg_float_size(root[1]) if root else None
    if tag is None or old_size is None:
        return svg
    box = _svg_view_box(root[1].get("viewBox"))
    attributes = _XML_ATTRIBUTE.findall(tag.group(2))
    replaced = {"width", "height"} | ({"viewBox"} if box is None else set())
    same_shape = box is None or abs(old_size[0] * box[1] - old_size[1] * box[0]) <= 1e-3 * old_size[0] * box[1]
    if same_shape:
        replaced.add("preserveAspectRatio")
    kept = [(name, value) for name, value in attributes if name.decode() not in replaced]
    added = [(b"width", f'"{size[0]}"'.encode()), (b"height", f'"{size[1]}"'.encode())]
    if box is None:
        added.append((b"viewBox", f'"0 0 {old_size[0]!r} {old_size[1]!r}"'.encode()))
    if same_shape:
        added.append((b"preserveAspectRatio", b'"none"'))
    new_tag = b"<" + tag.group(1) + b"".join(b" " + name + b"=" + value for name, value in kept + added) + tag.group(3)
    return (data[: tag.start()] + new_tag + data[tag.end() :]).decode("utf-8")


def _svg_length(value: str | None) -> tuple[float, str] | None:
    match = _SVG_LENGTH.match(value or "")
    return (float(match.group(1)), match.group(2) or "px") if match else None


def _svg_resolve(length: tuple[float, str] | None, reference: float) -> float | None:
    if length is None:
        return None
    number, unit = length
    return reference * number / 100 if unit == "%" else number * _SVG_UNITS[unit]


def _svg_view_box(value: str | None) -> tuple[float, float] | None:
    try:
        numbers = [float(n) for n in re.split(r"[\s,]+", (value or "").strip())]
    except ValueError:
        return None
    if len(numbers) != 4 or numbers[2] <= 0 or numbers[3] <= 0:
        return None
    return numbers[2], numbers[3]


def _fit_within(size: tuple[int, int], side: int) -> tuple[int, int]:
    scale = side / max(size)
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def _fit_pixels(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """`size`, or the same shape shrunk to at most `max_pixels`: a vector declared bigger than the app
    opens (a banner set out in millimetres) is drawn at the largest size that fits, not refused."""
    width, height = size
    if width * height <= max_pixels:
        return size
    scale = (max_pixels / (width * height)) ** 0.5
    fitted_width = max(1, min(int(width * scale), max_pixels))
    return fitted_width, max(1, min(int(height * scale), max_pixels // fitted_width))


def _draw_svg(svg: str, size: tuple[int, int] | None, name: str) -> np.ndarray:
    """(H, W, 4) RGBA; exactly `size` when given (resvg keeps the shape, so a pixel of rounding is stretched)."""
    import resvg_py  # imported here, like rawpy: only SVGs need it

    if size:
        svg = _svg_at_size(svg, size)
    try:  # checked again on exactly what resvg gets
        if _outside_links(svg):
            raise _UnsafeSvg()
    except _UnsafeSvg as exc:
        raise ImageFileError(_svg_failure_message(name)) from exc
    options = {"width": size[0], "height": size[1]} if size else {}
    # resvg looks for relative paths in an empty folder of its own, never the app's working folder.
    with tempfile.TemporaryDirectory(prefix="re-solution-") as empty_folder:
        try:
            png = resvg_py.svg_to_bytes(svg_string=svg, dpi=_CSS_DPI, resources_dir=empty_folder, **options)
        except Exception as exc:  # ValueError for a broken SVG; anything else resvg reports means the same
            raise ImageFileError(_svg_failure_message(name)) from exc
    with Image.open(io.BytesIO(png)) as drawn:
        rgba = drawn.convert("RGBA")
    if size and rgba.size != tuple(size):
        rgba = rgba.resize(size, Image.LANCZOS)
    return np.asarray(rgba)


def _svg_failure_message(name: str) -> str:
    return (
        f"{name} couldn't be drawn. It may be damaged or use features Re:Solution can't draw; "
        "export it as PNG at the size you need instead."
    )


# --- Decoder: the bundled FFmpeg --------------------------------------------------------


def _decode_with_ffmpeg(file: _File, preview_side: int | None) -> _Picture:
    if file.suffix not in _FFMPEG_EXTS and not file.head.startswith(_FFMPEG_MAGIC):
        raise _NotDecoded()  # only still-picture formats: FFmpeg would happily read a video or text file
    png = _run_ffmpeg(file, ["-f", "image2pipe", "-c:v", "png", "-pix_fmt", "rgba"])
    image = _pillow_open(io.BytesIO(png))
    _pillow_load(image)
    # FFmpeg carries a TIFF's profile over, but not a PSD's: read it from the file itself, or a
    # 16-bit Display P3 or Adobe RGB document would be read as sRGB and come out washed out.
    if not image.info.get("icc_profile") and file.head.startswith(b"8BPS"):
        icc = _psd_icc_profile(file.path)
        if icc:
            image.info["icc_profile"] = icc
    return _Picture(image, _label(file))


def _psd_icc_profile(path: Path) -> bytes | None:
    """The colour profile a Photoshop file (PSD or PSB) stores as image resource 0x040F, or None.

    The resources follow the 26-byte header and the colour-mode data. Each is "8BIM", a 2-byte id,
    a name padded to an even length, a 4-byte length, and the data, also padded to an even length.
    """
    try:
        with path.open("rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            f.seek(_PSD_HEADER_BYTES)
            colour_mode_length = _read_uint32(f)
            if colour_mode_length is None:
                return None
            f.seek(colour_mode_length, os.SEEK_CUR)
            resources_length = _read_uint32(f)
            if resources_length is None:
                return None
            end = min(f.tell() + resources_length, file_size)
            while f.tell() + _PSD_RESOURCE_MIN_BYTES <= end:
                signature, resource_id, name_length = struct.unpack(">4sHB", f.read(7))
                f.seek(name_length + (name_length + 1) % 2, os.SEEK_CUR)  # the name, padded to even
                length = _read_uint32(f)
                if length is None or f.tell() + length > end:
                    return None
                if signature == b"8BIM" and resource_id == _PSD_ICC_RESOURCE:
                    icc = f.read(length)
                    return icc if len(icc) == length and _icc_colour_space(icc) else None
                f.seek(length + length % 2, os.SEEK_CUR)
    except (OSError, struct.error):
        return None
    return None


def _read_uint32(f: BinaryIO) -> int | None:
    data = f.read(4)
    return struct.unpack(">I", data)[0] if len(data) == 4 else None


def _decode_hdr_with_ffmpeg(file: _File, preview_side: int | None) -> _Picture:
    """EXR and Radiance HDR hold linear light, often brighter than white: tone-map them to 8-bit sRGB.

    The header gives the size, so FFmpeg's floats (colour and alpha together) go straight into one
    buffer. A thumbnail is shrunk inside FFmpeg when its build has zscale, else after tone-mapping.
    """
    header = _hdr_header(file)
    if header is None:
        raise _NotDecoded()
    size, has_alpha = header
    _check_size(size, _hdr_pixel_limit())
    planes = None
    if preview_side and max(size) > preview_side:
        small = _fit_within(size, preview_side)
        shrink = _HDR_SHRINK_FILTER.format(width=small[0], height=small[1])
        with contextlib.suppress(_NotDecoded):  # an FFmpeg without zscale: decode at full size instead
            planes = _ffmpeg_planes(file, small, has_alpha, ["-vf", shrink])
    if planes is None:
        planes = _ffmpeg_planes(file, size, has_alpha, [])
    pixels = _hdr_to_8bit(planes)
    del planes  # the floats go back to the system before Pillow makes its own copy
    return _Picture(Image.fromarray(pixels), _label(file), notes=[HDR_NOTE])


def _inspect_hdr(file: _File) -> ImageInfo | None:
    header = _hdr_header(file)
    if header is None:
        return None
    (width, height), has_alpha = header
    _check_size((width, height), _hdr_pixel_limit())
    return ImageInfo(width, height, _label(file), has_alpha, 1, False, [HDR_NOTE])


def _hdr_pixel_limit() -> int:
    return min(_MAX_HDR_PIXELS, MAX_OPEN_PIXELS)


def _hdr_header(file: _File) -> tuple[tuple[int, int], bool] | None:
    """The (width, height) FFmpeg will give and whether there is an alpha channel, from the header;
    None when the header can't be read that way."""
    if file.head.startswith(_EXR_MAGIC):
        return _exr_header(file.path)
    if file.head.startswith(_RADIANCE_MAGIC):
        with file.path.open("rb") as f:
            return _radiance_header(f.read(_RADIANCE_MAX_HEADER_BYTES))
    return None


def _exr_header(path: Path) -> tuple[tuple[int, int], bool] | None:
    """An EXR's display window (FFmpeg gives the whole frame, black where nothing is stored) and
    whether it has an alpha channel. Multi-part files: the first part, which is what FFmpeg opens."""
    size, channels = None, None
    with path.open("rb") as f:
        start = f.read(8)
        version = int.from_bytes(start[4:8], "little")
        if len(start) < 8 or version & 0xFF != 2 or version & _EXR_DEEP_DATA:
            return None
        for _ in range(_EXR_MAX_ATTRIBUTES):  # each: name\0 type\0 int32 length, then the value
            name = _c_string(f)
            if not name:  # b"": the end of the header; None: a damaged one
                break
            kind, length_bytes = _c_string(f), f.read(4)
            length = int.from_bytes(length_bytes, "little", signed=True)
            if kind is None or len(length_bytes) < 4 or length < 0:
                return None
            if name == b"displayWindow" and kind == b"box2i" and length == 16:
                value = f.read(16)
                if len(value) < 16:
                    return None
                x_min, y_min, x_max, y_max = struct.unpack("<4i", value)
                size = (x_max - x_min + 1, y_max - y_min + 1)
            elif name == b"channels" and kind == b"chlist" and length <= _EXR_MAX_CHANNEL_LIST_BYTES:
                channels = _exr_channel_names(f.read(length))
            else:
                f.seek(length, os.SEEK_CUR)
            if size is not None and channels is not None:
                break
    if size is None or channels is None or min(size) < 1:
        return None
    return size, any(channel.lower() == b"a" for channel in channels)  # FFmpeg's own test for alpha


def _c_string(f: BinaryIO) -> bytes | None:
    """The text up to the next zero byte; None when there is none within the longest EXR name."""
    text = bytearray()
    while len(text) < _EXR_MAX_NAME_BYTES:
        byte = f.read(1)
        if not byte:
            return None
        if byte == b"\0":
            return bytes(text)
        text += byte
    return None


def _exr_channel_names(chlist: bytes) -> list[bytes]:
    """Names from an EXR channel list: each is name\\0 and 16 bytes of settings; a lone \\0 ends it."""
    names, position = [], 0
    while position < len(chlist) and chlist[position] != 0:
        end = chlist.find(b"\0", position)
        if end < 0:
            break
        names.append(chlist[position:end])
        position = end + 1 + 16
    return names


def _radiance_header(head: bytes) -> tuple[tuple[int, int], bool] | None:
    """A Radiance file's size from the line after its header, e.g. "-Y 1080 +X 1920". Whichever order
    and directions it gives, FFmpeg's picture is the X count wide and the Y count high."""
    _, blank, rest = head.partition(b"\n\n")
    match = _RADIANCE_SIZE.fullmatch(rest.split(b"\n", 1)[0].strip()) if blank else None
    if match is None or match[1] == match[3]:
        return None
    counts = {match[1]: int(match[2]), match[3]: int(match[4])}
    if min(counts.values()) < 1:
        return None
    return (counts[b"X"], counts[b"Y"]), False


def _ffmpeg_planes(file: _File, size: tuple[int, int], has_alpha: bool, filters: list[str]) -> np.ndarray:
    """FFmpeg's first frame as linear-light float planes (G, B, R and, with `has_alpha`, A) of `size`."""
    width, height = size
    channels = 4 if has_alpha else 3
    data = _run_ffmpeg(
        file,
        [*filters, "-f", "rawvideo", "-pix_fmt", "gbrapf32le" if has_alpha else "gbrpf32le"],
        max_pixels=_hdr_pixel_limit(),
        length=width * height * channels * 4,
    )
    return np.frombuffer(data, "<f4").reshape(channels, height, width)


def _run_ffmpeg(
    file: _File, output_args: list[str], *, max_pixels: int | None = None, length: int | None = None
) -> bytes | mmap.mmap:
    """The first frame of `file`, converted by `output_args` and read from FFmpeg's output.

    With `length`, the output must be exactly that many bytes, and is read straight into one buffer
    (a gigabyte for an 8K EXR, so never copied); a different amount means FFmpeg read the file
    differently from its header, and nothing is decoded.
    """
    with tempfile.TemporaryFile() as errors:  # a file, so FFmpeg never waits on a full pipe
        try:
            process = subprocess.Popen(
                _ffmpeg_command(file, output_args, MAX_OPEN_PIXELS if max_pixels is None else max_pixels),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=errors,
                creationflags=_no_console(),
            )
        except (OSError, RuntimeError) as exc:  # RuntimeError: no ffmpeg binary
            raise _NotDecoded() from exc
        timer = threading.Timer(_DECODE_TIMEOUT_SECONDS, process.kill)
        timer.start()
        try:
            with process.stdout:
                output = process.stdout.read() if length is None else _read_exactly(process.stdout, length)
                if output is None:  # not the amount expected: stop it, in case it's still writing
                    process.kill()
        except BaseException:
            process.kill()
            raise
        finally:
            process.wait()
            timer.cancel()
        errors.seek(0)
        if b"exceeds specified max pixel count" in errors.read():
            raise Image.DecompressionBombError(file.name)
    if process.returncode != 0 or not output:
        raise _NotDecoded()
    return output


def _ffmpeg_command(file: _File, output_args: list[str], max_pixels: int) -> list[str]:
    return [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-max_pixels", str(max_pixels), "-i", os.path.abspath(file.path),
        "-frames:v", "1", *output_args, "pipe:1",
    ]  # fmt: skip


def _read_exactly(stream: BinaryIO, length: int) -> mmap.mmap | None:
    """Exactly `length` bytes from `stream` until it ends, or None if it holds any other amount.

    The buffer is memory of its own (an anonymous map), which goes straight back to the system once
    it's no longer used; the usual allocator may keep a freed gigabyte for later.
    """
    buffer = mmap.mmap(-1, length)
    with memoryview(buffer) as view:
        filled = 0
        while filled < length:
            count = stream.readinto(view[filled:])
            if not count:
                break
            filled += count
    if filled < length or stream.read(1):
        buffer.close()
        return None
    return buffer


def _no_console() -> int:
    """Keeps Windows from flashing a console window for each helper program."""
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _hdr_to_8bit(planes: np.ndarray) -> np.ndarray:
    """(H, W, 3 or 4) uint8 sRGB from FFmpeg's linear-light planes. Pictures that stay within white
    convert exactly; brighter ones roll off gently (extended Reinhard, with the brightest 0.1% as
    white) instead of clipping flat. Converted in bands of rows, so only one float copy is ever whole."""
    channels, height, width = planes.shape
    step = max(1, int((height * width / _HDR_SAMPLE_PIXELS) ** 0.5))
    sample, _ = _linear_rgb(planes[:, ::step, ::step])
    white = max(1.0, float(np.percentile(sample @ _REC709_LUMA, _HDR_WHITE_PERCENTILE)))
    pixels = np.empty((height, width, channels), np.uint8)
    rows = max(1, _HDR_BAND_PIXELS // width)
    for top in range(0, height, rows):
        rgb, alpha = _linear_rgb(planes[:, top : top + rows])
        if white > 1:
            luminance = rgb @ _REC709_LUMA
            rgb *= ((1 + luminance / white**2) / (1 + luminance))[..., None]
        np.clip(rgb, 0, 1, out=rgb)
        encoded = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
        pixels[top : top + rows, :, :3] = np.rint(encoded * 255)
        if alpha is not None:
            pixels[top : top + rows, :, 3] = np.rint(alpha * 255)
    return pixels


def _linear_rgb(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(H, W, 3) float32 RGB from G, B, R(, A) planes: a copy, unpremultiplied (EXR colours are
    multiplied by alpha), with NaN as black and infinities as the brightest half-float. Plus the alpha."""
    rgb = np.stack((planes[2], planes[0], planes[1]), axis=-1)
    alpha = None
    if len(planes) == 4:
        alpha = np.clip(np.nan_to_num(planes[3]), 0, 1)
        with np.errstate(invalid="ignore", over="ignore"):
            rgb /= np.maximum(alpha, 1e-6)[..., None]
        rgb[alpha <= 0] = 0
    np.nan_to_num(rgb, copy=False, nan=0.0, posinf=_HDR_CEILING, neginf=0.0)
    np.clip(rgb, 0, _HDR_CEILING, out=rgb)
    return rgb, alpha


# --- Decoder: macOS sips (last resort) ---------------------------------------------------


def _decode_with_sips(file: _File, preview_side: int | None) -> _Picture:
    """Let macOS convert the file to PNG: JPEG XL, and anything else the Mac can open."""
    sips = shutil.which("sips") if _on_mac() else None
    if sips is None:
        raise _NotDecoded()
    source = os.path.abspath(file.path)
    _check_size(_sips_size(sips, source))
    with tempfile.TemporaryDirectory(prefix="re-solution-") as folder:
        converted = os.path.join(folder, "picture.png")
        result = _run_sips([sips, "-s", "format", "png", source, "--out", converted])
        if result.returncode != 0 or not os.path.isfile(converted):
            raise _NotDecoded()
        data = Path(converted).read_bytes()
    image = _pillow_open(io.BytesIO(data))  # keeps the profile and EXIF orientation sips copied over
    _pillow_load(image)
    return _Picture(image, _label(file))


def _sips_size(sips: str, source: str) -> tuple[int, int]:
    result = _run_sips([sips, "-g", "pixelWidth", "-g", "pixelHeight", source])
    output = result.stdout.decode("utf-8", "replace")
    width, height = re.search(r"pixelWidth:\s*(\d+)", output), re.search(r"pixelHeight:\s*(\d+)", output)
    if result.returncode != 0 or not width or not height:
        raise _NotDecoded()
    return int(width.group(1)), int(height.group(1))


def _run_sips(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, capture_output=True, timeout=_DECODE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        raise _NotDecoded() from exc


# --- Shared clean-up: rotation, alpha, colour -------------------------------------------


def _upright(image: Image.Image) -> Image.Image:
    """`image` turned as its EXIF says, in place: no full-size copy of a picture that needs no turning.
    Only for a picture the caller owns (the decoders' own, which `_open_picture` closes)."""
    try:
        ImageOps.exif_transpose(image, in_place=True)
    except Exception:  # a damaged EXIF block shouldn't stop the picture itself from opening
        pass
    return image


def _split_alpha(image: Image.Image) -> tuple[Image.Image, np.ndarray | None]:
    """The colour channels, and the alpha channel when some pixel is see-through.

    AI generators often save fully opaque RGBA; that alpha is dropped so the result can be a JPEG.
    """
    if not image.has_transparency_data:
        return image, None
    shade = image.info.get("transparency")
    if image.mode in _HIGH_BIT_DEPTH_MODES:  # 16-bit grey (PNG tRNS): converting would clip it to 8 bits first
        return image, (_key_alpha(np.asarray(image) == shade) if isinstance(shade, int) else None)
    if image.mode == "RGB" and isinstance(shade, tuple) and max(shade) > 255:
        # A 16-bit colour PNG opens as 8 bits (each value's high byte) but keeps its 16-bit transparent colour.
        return image, _key_alpha((np.asarray(image) == [value >> 8 for value in shade]).all(axis=-1))
    if image.mode not in _WITHOUT_ALPHA:  # palette, premultiplied, or one transparent colour (tRNS)
        try:
            image = image.convert("LA" if image.mode in ("L", "La") else "RGBA")
        except ValueError:  # a transparency Pillow can't apply: keep the picture, opaque
            return image, None
    alpha = np.asarray(image.getchannel("A"))
    colour = image.convert(_WITHOUT_ALPHA[image.mode])
    return colour, (alpha if alpha.min() < 255 else None)


def _key_alpha(transparent: np.ndarray) -> np.ndarray | None:
    """Alpha for a picture with one transparent colour: see-through where `transparent`, if anywhere."""
    return np.where(transparent, 0, 255).astype(np.uint8) if transparent.any() else None


def _to_rgb(image: Image.Image, icc: bytes | None) -> tuple[Image.Image, bytes | None]:
    """8-bit RGB, and the profile those RGB values are in (None means plain sRGB)."""
    image = _to_8bit(image)
    space = _icc_colour_space(icc)
    if image.mode == "RGB":
        return image, icc if space == "RGB" else None
    if image.mode in _ICC_SPACE_FOR_MODE and space == _ICC_SPACE_FOR_MODE[image.mode]:
        converted = _convert_with_profile(image, icc)
        if converted is not None:
            return converted, None
    if image.mode == "LAB":  # Pillow's own conversion would just copy the L, a and b numbers across
        return ImageCms.profileToProfile(image, _lab(), _srgb(), outputMode="RGB"), None
    rgb = image.convert("RGB")
    keeps_profile = space == "RGB" and image.mode not in ("L", "1", "CMYK")  # e.g. a palette image
    return rgb, icc if keeps_profile else None


def _to_8bit(image: Image.Image) -> Image.Image:
    """16-bit and 32-bit grey become 8-bit grey (16-bit colour PNGs already open as 8-bit RGB).

    White is 1.0 for floating-point pictures that stay within 0..1 (as scientific and HDR tools save
    them) and 65,535 for 16-bit ones. Values beyond 16 bits have no set white, so the brightest
    pixel becomes white.
    """
    if image.mode not in _HIGH_BIT_DEPTH_MODES:
        return image
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    brightest = float(values.max(initial=0.0, where=finite))
    if image.mode == "F" and brightest <= 1.0:
        white = 1.0
    elif image.mode.startswith("I;16") or 255 < brightest <= 65535:
        white = 65535.0
    elif brightest > 65535:
        white = brightest
    else:  # already 0..255
        white = 255.0
    if not finite.all():  # a stray infinity becomes white, a NaN black
        values = np.where(finite, values, np.where(values > 0, white, 0.0))
    return Image.fromarray(np.clip(np.rint(values * (255 / white)), 0, 255).astype(np.uint8))


def _icc_colour_space(icc: bytes | None) -> str | None:
    """'RGB', 'GRAY', 'CMYK', ... from the profile header (bytes 16-19), or None."""
    if not icc or len(icc) < 128:
        return None
    return icc[16:20].decode("ascii", "replace").strip()


@functools.cache
def _srgb() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@functools.cache
def _lab() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))


def _convert_with_profile(image: Image.Image, icc: bytes) -> Image.Image | None:
    """Grey or CMYK pixels to sRGB through their own profile, or None if the profile is unusable."""
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return ImageCms.profileToProfile(image, source, _srgb(), outputMode="RGB")
    except (ImageCms.PyCMSError, OSError, ValueError):
        return None


# --- Saving -----------------------------------------------------------------------


def save_image(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    path: Path,
    fmt: str,
    icc: bytes | None,
    dpi: int = 300,
    *,
    source_format: str | None = None,
) -> Path:
    """Save the picture and return the path written.

    `fmt`: "jpeg" (quality 95, no colour subsampling), "png", "tiff" (deflate), "webp" (up to 16,383 px;
    bigger raises ValueError with a friendly message), or "original": the source's own format when it
    can be written (JPEG, PNG, TIFF, WebP, HEIC, AVIF; `source_format` names it, e.g. LoadedImage.format),
    otherwise PNG. `resolve_format` gives the format this will write and the note to show for it.

    Every file records `dpi`, so it opens at the right print size. Pictures with transparency are
    saved as PNG unless the format holds transparency, so the returned path may have a different suffix.
    The file appears only once it is complete; a failed or cancelled save leaves nothing behind.
    """
    height, width = rgb.shape[:2]
    fmt, _note = resolve_format(fmt, source_format, (width, height), alpha=alpha is not None)
    check_dpi(dpi)
    if alpha is not None and fmt not in FORMATS_WITH_ALPHA:
        fmt = "png"  # JPEG can't hold transparency
    path = _with_suffix_for(Path(path), fmt)
    pixels = rgb if alpha is None else np.dstack((rgb, alpha))
    image = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8))
    options = _save_options(fmt, width * height, dpi)
    if icc:
        options["icc_profile"] = icc
    _write_atomically(path, lambda f: image.save(f, _PIL_FORMATS[fmt], **options))
    return path


def resolve_format(
    fmt: str, source_format: str | None = None, size: tuple[int, int] | None = None, *, alpha: bool = False
) -> tuple[str, str | None]:
    """The format `save_image` writes for `fmt`, and a note when "original" couldn't be kept.

    "original" becomes the source's own format when Re:Solution can write it and it holds a picture of
    `size` (width, height); otherwise PNG for formats it can't write, or for a picture too big for its
    own format JPEG when JPEG holds it (and there's no `alpha`), else PNG. Other formats come back
    unchanged, after checking that they can hold `size`; UserError (a ValueError), worded for people,
    when they can't or `fmt` is unknown.
    """
    if fmt == "original":
        if not source_format:
            return "png", "The original format isn't known, so this was saved as PNG."
        writable = _ORIGINAL_FORMATS.get(source_format.strip().lower())
        if writable is None:
            return "png", f"Re:Solution can't save {source_format} files, so this one was saved as PNG."
        if size and not _format_holds(writable, size):
            fallback = "jpeg" if not alpha and writable != "jpeg" and _format_holds("jpeg", size) else "png"
            name, fallback_name = _FORMAT_NAMES[writable], _FORMAT_NAMES[fallback]
            return fallback, f"{name} can't hold a picture this big, so it was saved as {fallback_name}."
        return writable, None
    if fmt not in OUTPUT_SUFFIXES:
        raise UserError(f"Unknown output format {fmt!r}; use one of {', '.join(SAVE_FORMATS)}.")
    if size and not _format_holds(fmt, size):
        raise UserError(_too_big_for_format(fmt, size))
    return fmt, None


def _format_holds(fmt: str, size: tuple[int, int]) -> bool:
    width, height = size
    side_limit, pixel_limit = _MAX_SIDE.get(fmt), _MAX_PIXELS.get(fmt)
    fits_side = side_limit is None or max(width, height) <= side_limit
    return fits_side and (pixel_limit is None or width * height <= pixel_limit)


# The formats people can choose, in the order they're offered.
_CHOOSABLE_FORMATS = ("jpeg", "png", "tiff", "webp")


def _too_big_for_format(fmt: str, size: tuple[int, int]) -> str:
    """Why `fmt` can't hold a picture of `size`, naming only the formats that can."""
    name, (width, height) = _FORMAT_NAMES[fmt], size
    if fmt in _MAX_SIDE:
        limit = f"bigger than {_MAX_SIDE[fmt]:,} pixels on a side"
    else:
        limit = "this big"
    others = [_FORMAT_NAMES[other] for other in _CHOOSABLE_FORMATS if other != fmt and _format_holds(other, size)]
    message = f"{name} can't hold pictures {limit}, and this one is {width:,} × {height:,}."
    if not others:
        return message
    choices = others[0] if len(others) == 1 else f"{', '.join(others[:-1])} or {others[-1]}"
    return f"{message} Choose {choices} instead."


def check_dpi(dpi: int) -> None:
    """Raise UserError (a ValueError) unless `dpi` is a whole number every format can store."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 1 <= dpi <= MAX_DPI:
        raise UserError(f"DPI must be a whole number from 1 to {MAX_DPI:,}, not {dpi!r}.")


def _save_options(fmt: str, pixels: int, dpi: int) -> dict[str, object]:
    if fmt == "jpeg":
        return {"quality": JPEG_QUALITY, "subsampling": 0, "dpi": (dpi, dpi)}
    if fmt == "tiff":
        # Deflate (ZIP) with horizontal prediction, as Photoshop's "ZIP with prediction": lossless,
        # and about a third smaller than deflate alone on photos.
        return {
            "compression": "tiff_deflate",
            "tiffinfo": {_TIFF_PREDICTOR: _HORIZONTAL_DIFFERENCING},
            "dpi": (dpi, dpi),
        }
    if fmt == "png":
        return {"dpi": (dpi, dpi), **({"compress_level": _FAST_PNG_LEVEL} if pixels > FAST_PNG_PIXELS else {})}
    # WebP, HEIC and AVIF have no DPI field of their own; they keep it in EXIF, as cameras and Photoshop do.
    return {"quality": _LOSSY_QUALITY[fmt], "exif": _dpi_exif(dpi)}


def _dpi_exif(dpi: int) -> bytes:
    exif = Image.Exif()
    exif[_X_RESOLUTION] = dpi
    exif[_Y_RESOLUTION] = dpi
    exif[_RESOLUTION_UNIT] = _INCHES
    return exif.tobytes()


def _with_suffix_for(path: Path, fmt: str) -> Path:
    if path.suffix.lower() in _ACCEPTED_SUFFIXES[fmt]:
        return path
    renamed = path.with_suffix(OUTPUT_SUFFIXES[fmt])
    # The caller only checked that `path` was free, not the renamed one.
    return unique_path(renamed.parent, renamed.stem, renamed.suffix) if renamed.exists() else renamed


def _write_atomically(path: Path, write: Callable[[BinaryIO], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        with temporary.open("wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ImageFileError(_save_failure_message(exc, path.parent)) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _save_failure_message(exc: OSError, folder: Path) -> str:
    if exc.errno == errno.ENOSPC or (exc.errno is None and _disk_is_full(folder)):
        return "The disk is full, so the upscaled picture couldn't be saved. Free up some space and try again."
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return f"The app isn't allowed to save pictures in {folder}."
    return f"Couldn't save the upscaled picture ({exc.strerror or exc})."


def _disk_is_full(folder: Path) -> bool:
    """Whether the disk is (all but) full, for errors that don't say why, such as libtiff's
    "encoder error -2". Called while the partial file still takes up the space it filled."""
    try:
        return free_disk_bytes(folder) < _FULL_DISK_BYTES
    except OSError:
        return False


def ensure_free_space(directory: Path, needed: int, *, keep_reserve: bool = False) -> None:
    """Refuse up front, rather than after a long upscale, when the result can't fit on the disk (with
    config.SPACE_RESERVE_BYTES to spare when `keep_reserve`: what the app always leaves free)."""
    free = free_disk_bytes(directory)
    if free >= needed + (SPACE_RESERVE_BYTES if keep_reserve else 0):
        return
    if keep_reserve:
        raise ImageFileError(
            f"There isn't enough free space to save this picture: it needs about {describe_size(needed)}, "
            f"and Re:Solution always leaves {SPACE_RESERVE_TEXT} free so the computer's other apps can still "
            f"save their work. The disk has {describe_size(free)} free. Free up some space and try again."
        )
    raise ImageFileError(
        f"There isn't enough free space to save this picture: it needs about {describe_size(needed)} "
        f"and the disk has {describe_size(free)} free. Free up some space and try again."
    )


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """`directory/stem+suffix`, or `stem-2`, `stem-3`, ... if that name is taken. Creates `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = _truncate_utf8(stem, _MAX_NAME_BYTES - len(suffix.encode()) - len("-999"))
    candidate, n = directory / f"{stem}{suffix}", 2
    while candidate.exists():
        candidate, n = directory / f"{stem}-{n}{suffix}", n + 1
    return candidate


def _truncate_utf8(text: str, max_bytes: int) -> str:
    return text.encode()[:max_bytes].decode("utf-8", "ignore")


# --- Previews ---------------------------------------------------------------------


def preview_jpeg(rgb: np.ndarray, max_side: int = 1600, *, icc: bytes | None = None) -> bytes:
    """A screen-sized JPEG of the whole picture. `icc` keeps wide-gamut colours right in the browser."""
    image = Image.fromarray(rgb)
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    return _jpeg_bytes(image, _PREVIEW_QUALITY, icc)


def compare_crops(src: np.ndarray, out: np.ndarray, size: int = 640, *, icc: bytes | None = None) -> tuple[bytes, bytes]:
    """Before/after crops at 100% of the output, taken at the same spot.

    "Before" is a plain Lanczos enlargement of `src`; "after" is `out`. The spot is the centre
    of the most detailed ninth of the picture, where the difference shows most.
    """
    src_h, src_w = src.shape[:2]
    out_h, out_w = out.shape[:2]
    scale_x, scale_y = out_w / src_w, out_h / src_h
    centre_x, centre_y = _most_detailed_centre(src)
    crop_w, crop_h = min(size, out_w), min(size, out_h)
    x0 = min(max(round(centre_x * scale_x - crop_w / 2), 0), out_w - crop_w)
    y0 = min(max(round(centre_y * scale_y - crop_h / 2), 0), out_h - crop_h)

    after = Image.fromarray(np.ascontiguousarray(out[y0 : y0 + crop_h, x0 : x0 + crop_w]))
    # Resizing just this box gives the same pixels as cropping a full-size Lanczos enlargement.
    source_box = (x0 / scale_x, y0 / scale_y, (x0 + crop_w) / scale_x, (y0 + crop_h) / scale_y)
    before = Image.fromarray(src).resize((crop_w, crop_h), Image.LANCZOS, box=source_box)
    return _jpeg_bytes(before, _CROP_QUALITY, icc), _jpeg_bytes(after, _CROP_QUALITY, icc)


def _most_detailed_centre(rgb: np.ndarray) -> tuple[float, float]:
    """Centre (x, y) of the ninth with the highest Laplacian variance. Ties go to the middle."""
    h, w = rgb.shape[:2]
    xs, ys = np.linspace(0, w, 4).astype(int), np.linspace(0, h, 4).astype(int)
    best, best_score = (1, 1), -1.0
    for row, col in [(1, 1)] + [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]:
        region = rgb[ys[row] : ys[row + 1], xs[col] : xs[col + 1]]
        score = _laplacian_variance(region)
        if score > best_score:
            best, best_score = (row, col), score
    row, col = best
    return (xs[col] + xs[col + 1]) / 2, (ys[row] + ys[row + 1]) / 2


def _laplacian_variance(rgb: np.ndarray) -> float:
    if rgb.shape[0] < 3 or rgb.shape[1] < 3:
        return 0.0
    grey = rgb.astype(np.float32) @ _LUMA
    laplacian = 4 * grey[1:-1, 1:-1] - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
    return float(laplacian.var())


def _on_checkerboard(rgba: Image.Image) -> Image.Image:
    """Flatten see-through areas onto the light checkerboard design apps use for transparency."""
    width, height = rgba.size
    rows, columns = np.indices((height, width)) // _CHECKER_SQUARE
    board = np.where((rows + columns) % 2 == 0, *_CHECKER_SHADES).astype(np.uint8)
    background = Image.fromarray(board).convert("RGBA")
    return Image.alpha_composite(background, rgba.convert("RGBA")).convert("RGB")


def _jpeg_bytes(image: Image.Image, quality: int, icc: bytes | None) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, icc_profile=icc)
    return buffer.getvalue()


# --- Zoomable previews (the compare viewer) ----------------------------------------
#
# Each finished picture gets a tile pyramid, so the viewer can show the whole picture (Fit) and
# every pixel (100%, 200%) while loading only what is on screen: 512 px JPEG tiles, level 0 at full
# size and each next level half as big, until the whole picture fits in 1024 px. Tile (col, row) of
# a level covers its pixels [col * 512, (col + 1) * 512) across and the same down; edge tiles are
# smaller. viewer.json describes the levels. The "before" side isn't stored: the viewer asks for
# each of its tiles, and before_tile makes it from the original, simply enlarged.

TILE_SIZE = 512
TILE_QUALITY = 88
VIEWER_FIT = 1024  # the smallest level fits in this many pixels
VIEWER_MANIFEST = "viewer.json"
_VIEWER_VERSION = 1
_TILE_WORKERS = 4  # JPEG encoding lets go of the GIL, so a few threads make a 16K pyramid quick


def pyramid_levels(width: int, height: int, tile_size: int = TILE_SIZE, fit: int = VIEWER_FIT) -> list[dict]:
    """The levels of a picture's tile pyramid: level 0 at full size, then half as big each time
    (rounded up) until the whole picture fits in `fit` pixels."""
    if width < 1 or height < 1:
        raise ValueError(f"A picture needs a size, not {width} x {height}")
    levels: list[dict] = []
    w, h = width, height
    while True:
        cols, rows = math.ceil(w / tile_size), math.ceil(h / tile_size)
        levels.append({"level": len(levels), "width": w, "height": h, "cols": cols, "rows": rows})
        if max(w, h) <= fit:
            return levels
        w, h = (w + 1) // 2, (h + 1) // 2


def write_tile_pyramid(
    rgb: np.ndarray,
    folder: Path,
    *,
    icc: bytes | None = None,
    source_size: tuple[int, int] | None = None,
    tile_size: int = TILE_SIZE,
    quality: int = TILE_QUALITY,
    cancel: threading.Event | None = None,
) -> dict:
    """Write `rgb` (H, W, 3 uint8) as a tile pyramid into `folder`: <level>/<col>_<row>.jpg plus
    viewer.json, which is returned. `source_size` is the picture before upscaling (for the viewer's
    "Before" label), `icc` goes into every tile so wide-gamut colours look right in the browser.

    Memory stays small even for 16K: level 0 is cut straight from `rgb`, and each smaller level is
    made from the one before it (a quarter of its size). The folder appears complete or not at all
    (it is written next to it, then renamed). Raises Cancelled when `cancel` is set part-way."""
    folder = Path(folder)
    height, width = rgb.shape[:2]
    levels = pyramid_levels(width, height, tile_size)
    part = folder.with_name(folder.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    try:
        with ThreadPoolExecutor(max_workers=_TILE_WORKERS, thread_name_prefix="viewer-tiles") as pool:
            current = rgb
            pending = _submit_tiles(pool, current, part / "0", tile_size, quality, icc, cancel)
            for spec in levels[1:]:
                # The next level is made while the tiles of this one are encoded.
                smaller = _half_size(current, (spec["width"], spec["height"]))
                _wait_for(pending, cancel)
                current = smaller
                pending = _submit_tiles(pool, current, part / str(spec["level"]), tile_size, quality, icc, cancel)
            _wait_for(pending, cancel)
        source_w, source_h = source_size or (width, height)
        manifest = {
            "version": _VIEWER_VERSION,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "levels": levels,
            "source_width": int(source_w),
            "source_height": int(source_h),
        }
        (part / VIEWER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        shutil.rmtree(folder, ignore_errors=True)
        os.replace(part, folder)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    return manifest


def read_viewer_manifest(folder: Path) -> dict | None:
    """A tile pyramid's viewer.json, or None if there's no complete pyramid in `folder`."""
    try:
        manifest = json.loads((Path(folder) / VIEWER_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("levels"), list) or not manifest["levels"]:
        return None
    return manifest


def _half_size(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`pixels` shrunk to `size` (about half), each new pixel the average of the ones it covers."""
    import cv2  # only needed here; it takes a moment to load

    return cv2.resize(np.ascontiguousarray(pixels), size, interpolation=cv2.INTER_AREA)


def _submit_tiles(
    pool: ThreadPoolExecutor,
    pixels: np.ndarray,
    folder: Path,
    tile_size: int,
    quality: int,
    icc: bytes | None,
    cancel: threading.Event | None,
) -> list[Future]:
    folder.mkdir()
    height, width = pixels.shape[:2]
    pending = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            path = folder / f"{x // tile_size}_{y // tile_size}.jpg"
            pending.append(pool.submit(_write_tile, pixels, path, x, y, tile_size, quality, icc, cancel))
    return pending


def _write_tile(
    pixels: np.ndarray,
    path: Path,
    x: int,
    y: int,
    tile_size: int,
    quality: int,
    icc: bytes | None,
    cancel: threading.Event | None,
) -> None:
    if cancel is not None and cancel.is_set():
        return
    tile = Image.fromarray(np.ascontiguousarray(pixels[y : y + tile_size, x : x + tile_size]))
    tile.save(path, "JPEG", quality=quality, icc_profile=icc)


def _wait_for(pending: list[Future], cancel: threading.Event | None) -> None:
    for future in pending:
        future.result()  # re-raises a tile that couldn't be written (a full disk, say)
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def tile_bounds(level: dict, col: int, row: int, tile_size: int = TILE_SIZE) -> tuple[int, int, int, int] | None:
    """The pixels (x0, y0, x1, y1) of `level` that tile (col, row) covers, or None if it has no such tile."""
    if not (0 <= col < level["cols"] and 0 <= row < level["rows"]):
        return None
    x0, y0 = col * tile_size, row * tile_size
    return x0, y0, min(x0 + tile_size, level["width"]), min(y0 + tile_size, level["height"])


def before_tile(source: Image.Image, level: dict, col: int, row: int, tile_size: int = TILE_SIZE) -> Image.Image:
    """Tile (col, row) of `level` for the viewer's "Before" side: the same part of the picture as the
    upscaled tile, taken from the original `source` and simply enlarged (Lanczos) to the tile's size.
    Resizing just this part gives the same pixels as cutting it from a whole enlarged picture."""
    bounds = tile_bounds(level, col, row, tile_size)
    if bounds is None:
        raise ValueError(f"Level {level['level']} has no tile {col}, {row}")
    x0, y0, x1, y1 = bounds
    scale_x, scale_y = source.width / level["width"], source.height / level["height"]
    box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
    # reducing_gap: a big shrink (the smallest levels of a large original) first halves cheaply.
    return source.resize((x1 - x0, y1 - y0), Image.LANCZOS, box=box, reducing_gap=3.0)


def tile_jpeg(image: Image.Image, icc: bytes | None = None, quality: int = TILE_QUALITY) -> bytes:
    """A viewer tile as JPEG bytes, made the same way as the stored ones."""
    return _jpeg_bytes(image, quality, icc)


def on_white(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """See-through areas shown on white, as the previews show them (the colour under them is often junk)."""
    if alpha is None:
        return rgb
    out = np.empty_like(rgb)
    for y in range(0, rgb.shape[0], 1024):  # in bands, so a big picture needs little extra memory
        a = alpha[y : y + 1024, :, None].astype(np.uint16)
        out[y : y + 1024] = (rgb[y : y + 1024] * a + 255 * (255 - a) + 127) // 255
    return out
