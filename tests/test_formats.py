"""Every kind of picture people drop in: phone, camera, designer, web and HDR formats.

All fixtures are generated here (Pillow, pillow-heif, the bundled ffmpeg, a hand-written DNG, and
macOS's own `sips` and colour profiles where a test needs them), so nothing private is checked in.
"""

import base64
import gzip
import io
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageFile

from upscaler import imageio_utils
from upscaler.imageio_utils import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_VIDEO_EXTS,
    ImageFileError,
    ImageInfo,
    inspect_image,
    load_image,
    make_thumbnail,
    render_vector,
    resolve_format,
    save_image,
)

ORIENTATION = 0x0112
X_RESOLUTION = 0x011A
PROFILES = Path("/System/Library/ColorSync/Profiles")
MAC_CMYK_PROFILE = PROFILES / "Generic CMYK Profile.icc"
MAC_P3_PROFILE = PROFILES / "Display P3.icc"
ON_MAC = sys.platform == "darwin" and shutil.which("sips") is not None

needs_mac = pytest.mark.skipif(not ON_MAC, reason="needs macOS's sips")


def gradient(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    return np.stack(np.broadcast_arrays(x + 0 * y, y + 0 * x, (x + y) / 2), axis=-1).astype(np.uint8)


def halves(width: int = 80, height: int = 40) -> np.ndarray:
    """Left half white, right half black: shows both size and orientation at a glance."""
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[:, : width // 2] = 255
    return pixels


def ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def sips(src: Path, fmt: str, dst: Path) -> None:
    subprocess.run(["sips", "-s", "format", fmt, str(src), "--out", str(dst)], check=True, capture_output=True)


def write_png16(path: Path, pixels: np.ndarray, transparent: int | tuple[int, int, int] | None = None) -> None:
    """A 16-bit PNG, grey for (H, W) pixels and RGB for (H, W, 3), optionally with one transparent
    value (tRNS). Pillow can read these but not write them."""
    h, w = pixels.shape[:2]
    rows = b"".join(b"\x00" + pixels[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    colour_type = 2 if pixels.ndim == 3 else 0
    header = struct.pack(">IIBBBBB", w, h, 16, colour_type, 0, 0, 0)
    trns = b""
    if transparent is not None:
        trns = chunk(b"tRNS", struct.pack(">3H" if colour_type else ">H", *np.atleast_1d(transparent)))
    chunks = chunk(b"IHDR", header) + trns + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunks)


def write_exr(path: Path, display: tuple[int, int, int, int], data: tuple[int, int, int, int], value: float) -> None:
    """A tiny uncompressed RGB EXR whose stored pixels (its data window, filled with `value`) can cover
    less than its frame (its display window), which FFmpeg can't write. Windows are (x0, y0, x1, y1)."""

    def attribute(name: str, kind: str, payload: bytes) -> bytes:
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(payload)) + payload

    x0, y0, x1, y1 = data
    width, height = x1 - x0 + 1, y1 - y0 + 1
    channels = b"".join(name + b"\0" + struct.pack("<iB3xii", 2, 0, 1, 1) for name in (b"B", b"G", b"R")) + b"\0"
    header = b"v/1\x01" + struct.pack("<I", 2) + b"".join([
        attribute("channels", "chlist", channels),
        attribute("compression", "compression", b"\0"),
        attribute("dataWindow", "box2i", struct.pack("<4i", *data)),
        attribute("displayWindow", "box2i", struct.pack("<4i", *display)),
        attribute("lineOrder", "lineOrder", b"\0"),
        attribute("pixelAspectRatio", "float", struct.pack("<f", 1)),
        attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0)),
        attribute("screenWindowWidth", "float", struct.pack("<f", 1)),
    ]) + b"\0"  # fmt: skip
    row = struct.pack(f"<{3 * width}f", *[value] * (3 * width))  # B, G and R values, one channel after another
    blocks = [struct.pack("<ii", y, len(row)) + row for y in range(y0, y1 + 1)]
    first_block = len(header) + 8 * height  # after the table of where each row starts
    offsets = struct.pack(f"<{height}Q", *(first_block + i * len(blocks[0]) for i in range(height)))
    path.write_bytes(header + offsets + b"".join(blocks))


def write_psd16(path: Path, pixels16: np.ndarray, icc: bytes | None = None) -> None:
    """A flat, uncompressed 16-bit RGB Photoshop file, which Pillow can't read, with `icc` as its colour
    profile (image resource 0x040F) after an unrelated resource with a name and an odd length."""

    def resource(resource_id: int, name: bytes, data: bytes) -> bytes:
        pascal = bytes([len(name)]) + name
        pascal += b"\0" * (len(pascal) % 2)  # the name is padded to an even length
        padded = data + b"\0" * (len(data) % 2)  # and so is the data
        return b"8BIM" + struct.pack(">H", resource_id) + pascal + struct.pack(">I", len(data)) + padded

    h, w, _ = pixels16.shape
    header = b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, h, w, 16, 3)  # version 1, 3 channels, 16 bits, RGB
    resources = resource(0x0400, b"x", b"abc") + (resource(0x040F, b"", icc) if icc else b"")
    planes = pixels16.transpose(2, 0, 1).astype(">u2").tobytes()  # R, G and B planes, one after another
    sections = struct.pack(">I", 0) + struct.pack(">I", len(resources)) + resources + struct.pack(">I", 0)
    path.write_bytes(header + sections + struct.pack(">H", 0) + planes)


def rgb_profile() -> bytes:
    """A wide-gamut RGB profile where the system has one (macOS's Display P3), otherwise Pillow's sRGB."""
    if MAC_P3_PROFILE.is_file():
        return MAC_P3_PROFILE.read_bytes()
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def write_pfm(path: Path, rgb: np.ndarray) -> None:
    """Linear float RGB as a Portable FloatMap (bottom row first, little-endian), for ffmpeg to turn into EXR."""
    h, w, _ = rgb.shape
    path.write_bytes(f"PF\n{w} {h}\n-1.0\n".encode() + rgb[::-1].astype("<f4").tobytes())


# A minimal DNG: one IFD holding a 16-bit RGGB sensor image, the tags LibRaw needs, and nothing else.
_SHORT, _LONG, _RATIONAL, _SRATIONAL, _BYTE, _ASCII = 3, 4, 5, 10, 1, 2


def write_dng(path: Path, cfa: np.ndarray, orientation: int = 1) -> None:
    h, w = cfa.shape
    data = cfa.astype("<u2").tobytes()
    entries = []

    def add(tag: int, kind: int, values) -> None:
        if kind == _ASCII:
            payload = values.encode() + b"\x00"
            count = len(payload)
        elif kind == _BYTE:
            payload, count = bytes(values), len(values)
        elif kind == _SHORT:
            payload, count = struct.pack(f"<{len(values)}H", *values), len(values)
        elif kind == _LONG:
            payload, count = struct.pack(f"<{len(values)}I", *values), len(values)
        else:  # (S)RATIONAL pairs
            pair = "ii" if kind == _SRATIONAL else "II"
            payload, count = struct.pack("<" + pair * len(values), *[x for v in values for x in v]), len(values)
        entries.append((tag, kind, count, payload))

    add(254, _LONG, [0])  # NewSubfileType: the main image
    add(256, _LONG, [w])
    add(257, _LONG, [h])
    add(258, _SHORT, [16])
    add(259, _SHORT, [1])  # uncompressed
    add(262, _SHORT, [32803])  # colour filter array
    add(271, _ASCII, "Re-Solution")
    add(272, _ASCII, "Test DNG")
    add(273, _LONG, [0])  # StripOffsets, patched below
    add(274, _SHORT, [orientation])
    add(277, _SHORT, [1])
    add(278, _LONG, [h])
    add(279, _LONG, [len(data)])
    add(284, _SHORT, [1])
    add(33421, _SHORT, [2, 2])  # CFARepeatPatternDim
    add(33422, _BYTE, [0, 1, 1, 2])  # CFAPattern: RGGB
    add(50706, _BYTE, [1, 4, 0, 0])  # DNGVersion
    add(50708, _ASCII, "Re-Solution Test DNG")
    add(50717, _LONG, [65535])  # WhiteLevel
    add(50721, _SRATIONAL, [(1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1)])  # ColorMatrix1
    add(50728, _RATIONAL, [(1, 1), (1, 1), (1, 1)])  # AsShotNeutral
    add(50778, _SHORT, [21])  # CalibrationIlluminant1: D65
    entries.sort()

    ifd_offset = 8
    extra_offset = ifd_offset + 2 + 12 * len(entries) + 4
    extra, fields = b"", []
    for tag, kind, count, payload in entries:
        if len(payload) <= 4:
            fields.append([tag, kind, count, payload.ljust(4, b"\x00")])
        else:
            fields.append([tag, kind, count, struct.pack("<I", extra_offset + len(extra))])
            extra += payload + b"\x00" * (len(payload) % 2)
    data_offset = extra_offset + len(extra)
    for field in fields:
        if field[0] == 273:
            field[3] = struct.pack("<I", data_offset)
    ifd = struct.pack("<H", len(fields)) + b"".join(struct.pack("<HHI", t, k, c) + v for t, k, c, v in fields)
    path.write_bytes(b"II*\x00" + struct.pack("<I", ifd_offset) + ifd + struct.pack("<I", 0) + extra + data)


def red_and_blue_dng(path: Path, orientation: int = 1) -> None:
    """64x48 sensor image: red on the left half, blue on the right."""
    cfa = np.full((48, 64), 6000, np.uint16)
    cfa[0::2, 0:32:2] = 50000  # red photosites, left half
    cfa[1::2, 33::2] = 50000  # blue photosites, right half
    write_dng(path, cfa, orientation)


def svg_text(width: str = "200", height: str = "100", body: str = "", extra: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 20 10" {extra}>'
        '<rect x="0" y="0" width="10" height="10" fill="#ff0000"/>'
        '<rect x="10" y="0" width="10" height="10" fill="#0000ff"/>'
        f"{body}</svg>"
    )


def private_picture(folder: Path, name: str = "secret.png") -> Path:
    """A lime picture standing in for a private file elsewhere on the host."""
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10), "lime").save(folder / name)
    return folder / name


def shows_private_picture(rgba: np.ndarray) -> bool:
    return bool(((rgba[..., 1] > 200) & (rgba[..., 0] < 50) & (rgba[..., 2] < 50)).any())


def assert_halves(rgb: np.ndarray, tolerance: int = 40) -> None:
    """Left half white and right half black, allowing for lossy formats and soft edges."""
    h, w = rgb.shape[:2]
    assert rgb[:, : w // 2 - 4].min() >= 255 - tolerance
    assert rgb[:, w // 2 + 4 :].max() <= tolerance


# --- Supported extensions --------------------------------------------------------------


def test_supported_extensions():
    expected_images = {
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif", ".tif", ".tiff", ".bmp", ".gif",
        ".jp2", ".ico", ".tga", ".ppm", ".psd", ".dds", ".qoi", ".exr", ".hdr", ".svg",
        ".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw",
    }  # fmt: skip
    assert expected_images <= SUPPORTED_IMAGE_EXTS
    assert SUPPORTED_VIDEO_EXTS == {
        ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".mts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".ts",
    }  # fmt: skip
    assert isinstance(SUPPORTED_IMAGE_EXTS, frozenset) and isinstance(SUPPORTED_VIDEO_EXTS, frozenset)
    for ext in SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS:
        assert ext.startswith(".") and ext == ext.lower()
    assert not SUPPORTED_IMAGE_EXTS & SUPPORTED_VIDEO_EXTS
    assert not {".pdf", ".eps", ".ai"} & SUPPORTED_IMAGE_EXTS
    assert (".jxl" in SUPPORTED_IMAGE_EXTS) == (sys.platform == "darwin")  # only macOS can open JPEG XL


# --- Phone and web formats -------------------------------------------------------------


def test_heic_roundtrip_load(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    if not pillow_heif.libheif_info().get("HEIF"):
        pytest.skip("this libheif has no HEIC encoder")
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(halves()).save(tmp_path / "iphone.heic", exif=exif.tobytes(), quality=90)

    loaded = load_image(tmp_path / "iphone.heic")
    info = inspect_image(tmp_path / "iphone.heic")

    assert loaded.rgb.shape == (80, 40, 3)  # rotated exactly once
    assert loaded.rgb[:35].mean() > 200 and loaded.rgb[45:].mean() < 50
    assert loaded.format == "HEIC" and loaded.notes == []
    assert (info.width, info.height, info.format, info.frames) == (40, 80, "HEIC", 1)
    assert not info.has_alpha and not info.is_vector and info.warnings == []


def test_avif_load(tmp_path):
    rgba = np.dstack([halves(), np.full((40, 80), 255, np.uint8)])
    rgba[:, :10, 3] = 0
    Image.fromarray(rgba).save(tmp_path / "web.avif", quality=95)

    loaded = load_image(tmp_path / "web.avif")
    info = inspect_image(tmp_path / "web.avif")

    assert loaded.rgb.shape == (40, 80, 3) and loaded.format == "AVIF"
    assert loaded.rgb[:, 14:36].mean() > 200 and loaded.rgb[:, 44:].mean() < 50
    assert loaded.alpha is not None and loaded.alpha[:, :8].max() < 30 and loaded.alpha[:, 14:].min() > 225
    assert (info.width, info.height, info.format, info.has_alpha) == (80, 40, "AVIF", True)


def test_webp_and_animated_webp(tmp_path):
    frames = [Image.new("RGB", (30, 20), colour) for colour in ("red", "lime", "blue")]
    frames[0].save(tmp_path / "moving.webp", save_all=True, append_images=frames[1:], duration=100, lossless=True)

    loaded = load_image(tmp_path / "moving.webp")
    info = inspect_image(tmp_path / "moving.webp")

    assert tuple(loaded.rgb[10, 15]) == (255, 0, 0)  # the first frame
    assert any("animated" in note for note in loaded.notes)
    assert (info.format, info.frames) == ("WebP", 3) and info.warnings == loaded.notes


def test_gif_animated_first_frame_note(tmp_path):
    frames = [Image.new("RGB", (30, 20), colour) for colour in ("red", "lime", "blue")]
    frames[0].save(tmp_path / "party.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

    loaded = load_image(tmp_path / "party.gif")
    info = inspect_image(tmp_path / "party.gif")

    assert loaded.rgb.shape == (20, 30, 3)
    assert tuple(loaded.rgb[10, 15]) == (255, 0, 0)  # the first frame
    assert len(loaded.notes) == 1
    assert "animated" in loaded.notes[0] and "first frame" in loaded.notes[0]
    assert (info.format, info.frames, info.warnings) == ("GIF", 3, loaded.notes)


def test_single_frame_gif_has_no_note(tmp_path):
    Image.new("P", (10, 10), 3).save(tmp_path / "still.gif")

    assert load_image(tmp_path / "still.gif").notes == []
    assert inspect_image(tmp_path / "still.gif").frames == 1


def test_iphone_style_mpo_counts_as_one_picture(tmp_path):
    # Phones save a hidden second picture (gain map, depth) inside some JPEGs; that isn't a page.
    main, hidden = Image.fromarray(gradient(40, 30)), Image.new("RGB", (40, 30), "blue")
    main.save(tmp_path / "photo.jpg", "MPO", save_all=True, append_images=[hidden])

    info = inspect_image(tmp_path / "photo.jpg")

    assert (info.format, info.frames, info.warnings) == ("JPEG", 1, [])
    assert load_image(tmp_path / "photo.jpg").notes == []


# --- TIFF, colour and bit depth ------------------------------------------------------------


def test_tiff_16bit_multipage_first_page_note(tmp_path):
    pages = [Image.fromarray(np.full((12, 16), value, np.uint16)) for value in (65535, 32896, 0)]
    pages[0].save(tmp_path / "scan.tif", save_all=True, append_images=pages[1:])

    loaded = load_image(tmp_path / "scan.tif")
    info = inspect_image(tmp_path / "scan.tif")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (12, 16, 3)
    assert (loaded.rgb == 255).all()  # the first page, 16-bit white -> 8-bit white
    assert loaded.notes == ["This TIFF has 3 pages. Only the first page is upscaled."]
    assert (info.width, info.height, info.format, info.frames) == (16, 12, "TIFF", 3)
    assert info.warnings == loaded.notes


def test_tiff_48bit_rgb_loads_as_8bit(tmp_path):
    pixels16 = np.random.default_rng(1).integers(0, 65536, (12, 16, 3), dtype=np.uint16)
    write_png16(tmp_path / "deep.png", pixels16)
    ffmpeg("-i", str(tmp_path / "deep.png"), "-pix_fmt", "rgb48le", str(tmp_path / "deep.tif"))

    loaded = load_image(tmp_path / "deep.tif")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (12, 16, 3)
    assert np.abs(loaded.rgb.astype(int) - pixels16 / 257).max() <= 1


def test_16bit_grey_png_with_a_transparent_shade(tmp_path):
    ramp = np.tile(np.array([0, 17000, 34000, 51000], np.uint16), (3, 1))
    write_png16(tmp_path / "mask.png", ramp, transparent=34000)

    loaded = load_image(tmp_path / "mask.png")

    assert list(loaded.rgb[0, :, 0]) == [0, 66, 132, 198]  # scaled from 16 bits, not clipped to white
    assert loaded.alpha is not None and list(loaded.alpha[0]) == [255, 255, 0, 255]
    assert inspect_image(tmp_path / "mask.png").has_alpha


def test_16bit_colour_png_with_a_transparent_colour(tmp_path):
    ramp = np.array([0, 17000, 34000, 51000], np.uint16)
    pixels = np.tile(np.stack([ramp, ramp // 2, 65535 - ramp], axis=-1), (3, 1, 1))
    write_png16(tmp_path / "key.png", pixels, transparent=tuple(int(v) for v in pixels[0, 2]))

    loaded = load_image(tmp_path / "key.png")

    np.testing.assert_array_equal(loaded.rgb, pixels >> 8)  # Pillow keeps each value's high byte
    assert loaded.alpha is not None and list(loaded.alpha[0]) == [255, 255, 0, 255]
    assert inspect_image(tmp_path / "key.png").has_alpha


def test_float_and_32bit_grey_tiffs_keep_their_brightness(tmp_path):
    ramp = np.linspace(0, 1, 64, dtype=np.float32)
    Image.fromarray(np.tile(ramp, (4, 1))).save(tmp_path / "float.tif")  # scientific and HDR tools: 0..1
    counts = np.tile(np.array([0, 3 << 19, 1 << 21], np.int32), (4, 1))
    Image.fromarray(counts).save(tmp_path / "counts.tif")  # 32-bit whole numbers with no set white

    floats = load_image(tmp_path / "float.tif").rgb
    deep = load_image(tmp_path / "counts.tif").rgb

    assert np.abs(floats[0, :, 0].astype(int) - np.rint(ramp * 255)).max() <= 1
    assert list(deep[0, :, 0]) == [0, 191, 255]  # scaled to the brightest pixel, not clipped to white


@pytest.mark.skipif(not MAC_CMYK_PROFILE.is_file(), reason="needs macOS's Generic CMYK profile")
def test_cmyk_with_icc_to_srgb(tmp_path):
    cmyk_icc = MAC_CMYK_PROFILE.read_bytes()
    colour = (153, 51, 0, 25)  # 60% cyan, 20% magenta, 10% black: a mid blue that sRGB can show
    Image.new("CMYK", (16, 16), colour).save(tmp_path / "brochure.jpg", quality=100, icc_profile=cmyk_icc)

    loaded = load_image(tmp_path / "brochure.jpg")

    lab = ImageCms.createProfile("LAB")
    cmyk = ImageCms.ImageCmsProfile(io.BytesIO(cmyk_icc))
    srgb = ImageCms.createProfile("sRGB")
    expected = _lab(ImageCms.profileToProfile(Image.new("CMYK", (1, 1), colour), cmyk, lab, outputMode="LAB"))
    got = _lab(ImageCms.profileToProfile(Image.fromarray(loaded.rgb[8:9, 8:9]), srgb, lab, outputMode="LAB"))
    naive_rgb = Image.new("CMYK", (1, 1), colour).convert("RGB")
    naive = _lab(ImageCms.profileToProfile(naive_rgb, srgb, lab, outputMode="LAB"))
    assert np.linalg.norm(got - expected) < 3  # ΔE*ab: the colour survived the trip through the profile
    assert np.linalg.norm(naive - expected) > 6  # and the test can tell a profile-less conversion apart
    assert loaded.icc is None  # sRGB now; the CMYK profile must never be attached to RGB pixels
    assert loaded.notes == ["Converted from CMYK print colours to RGB screen colours."]
    assert inspect_image(tmp_path / "brochure.jpg").warnings == loaded.notes


def _lab(image: Image.Image) -> np.ndarray:
    l, a, b = (float(v) for v in image.getpixel((0, 0)))
    return np.array([l * 100 / 255, a - 128, b - 128])


@pytest.mark.skipif(not MAC_P3_PROFILE.is_file(), reason="needs macOS's Display P3 profile")
def test_wide_gamut_rgb_keeps_its_profile(tmp_path):
    # iPhone (Display P3) and Adobe RGB pictures keep their own colours: the profile goes onto the
    # result instead of squeezing the colours into sRGB, which would dull the most vivid ones.
    p3 = MAC_P3_PROFILE.read_bytes()
    Image.fromarray(gradient(20, 10)).save(tmp_path / "vivid.png", icc_profile=p3)

    loaded = load_image(tmp_path / "vivid.png")

    assert loaded.icc == p3
    np.testing.assert_array_equal(loaded.rgb, gradient(20, 10))


def test_lab_tiff_converts_to_rgb(tmp_path):
    Image.new("LAB", (8, 4), (255, 128, 128)).save(tmp_path / "lab.tif")  # L=100: white

    loaded = load_image(tmp_path / "lab.tif")

    assert loaded.rgb.shape == (4, 8, 3) and loaded.rgb.min() > 245


@pytest.mark.parametrize(
    ("name", "save_format", "options"),
    [
        ("old.bmp", "BMP", {}),
        ("old.tga", "TGA", {}),
        ("scan.jp2", "JPEG2000", {"irreversible": False}),
        ("raw.ppm", "PPM", {}),
        ("texture.dds", "DDS", {}),
        ("quick.qoi", "QOI", {}),
    ],
)
def test_bmp_ico_tga_jp2_load(tmp_path, name, save_format, options):
    Image.fromarray(halves()).save(tmp_path / name, save_format, **options)

    loaded = load_image(tmp_path / name)

    assert loaded.rgb.shape == (40, 80, 3)
    assert_halves(loaded.rgb)


def test_ico_uses_largest_size(tmp_path):
    Image.fromarray(halves(64, 64)).save(tmp_path / "favicon.ico", sizes=[(16, 16), (32, 32), (64, 64)])

    loaded = load_image(tmp_path / "favicon.ico")
    info = inspect_image(tmp_path / "favicon.ico")

    assert loaded.rgb.shape == (64, 64, 3) and (info.width, info.height, info.frames) == (64, 64, 1)


@needs_mac
def test_psd_composite_loads(tmp_path):
    Image.fromarray(halves()).save(tmp_path / "flat.png")
    sips(tmp_path / "flat.png", "psd", tmp_path / "design.psd")

    loaded = load_image(tmp_path / "design.psd")
    info = inspect_image(tmp_path / "design.psd")

    assert loaded.rgb.shape == (40, 80, 3) and loaded.format == "PSD"
    assert_halves(loaded.rgb)
    assert (info.format, info.frames, info.warnings) == ("PSD", 1, [])


@needs_mac
def test_16bit_psd_falls_back_to_ffmpeg(tmp_path):
    pixels16 = halves().astype(np.uint16) * 257
    write_png16(tmp_path / "deep.png", pixels16)
    sips(tmp_path / "deep.png", "psd", tmp_path / "deep.psd")
    with pytest.raises(OSError):
        Image.open(tmp_path / "deep.psd")  # Pillow reads only 8-bit PSDs

    loaded = load_image(tmp_path / "deep.psd")

    assert loaded.rgb.shape == (40, 80, 3) and loaded.format == "PSD"
    assert_halves(loaded.rgb)


def test_16bit_psd_keeps_its_colour_profile(tmp_path):
    # FFmpeg decodes the 16-bit PSDs Pillow can't, but ignores the profile Photoshop stores, so Display
    # P3 or Adobe RGB colours would be read as sRGB and come out washed out.
    icc = rgb_profile()
    write_psd16(tmp_path / "poster.psd", halves().astype(np.uint16) * 257, icc)
    with pytest.raises(OSError):
        Image.open(tmp_path / "poster.psd")

    loaded = load_image(tmp_path / "poster.psd")

    assert loaded.format == "PSD" and loaded.icc == icc
    assert_halves(loaded.rgb)
    with Image.open(io.BytesIO(make_thumbnail(tmp_path / "poster.psd"))) as thumbnail:
        assert thumbnail.info.get("icc_profile") == icc


def test_16bit_psd_without_a_profile_is_srgb(tmp_path):
    write_psd16(tmp_path / "plain.psd", halves().astype(np.uint16) * 257)

    loaded = load_image(tmp_path / "plain.psd")

    assert loaded.icc is None
    assert_halves(loaded.rgb)


@pytest.mark.parametrize("cut", [27, 30, 40, 60, 200])
def test_psd_profile_reader_survives_damaged_files(tmp_path, cut):
    write_psd16(tmp_path / "poster.psd", halves().astype(np.uint16) * 257, rgb_profile())
    data = (tmp_path / "poster.psd").read_bytes()
    (tmp_path / "cut.psd").write_bytes(data[:cut])

    assert imageio_utils._psd_icc_profile(tmp_path / "cut.psd") is None


# --- Camera RAW --------------------------------------------------------------------------


def test_raw_dng_develops_upright_with_camera_white_balance(tmp_path):
    red_and_blue_dng(tmp_path / "camera.dng", orientation=6)

    loaded = load_image(tmp_path / "camera.dng")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (64, 48, 3)  # turned upright
    top, bottom = loaded.rgb[4:28].reshape(-1, 3).mean(0), loaded.rgb[36:60].reshape(-1, 3).mean(0)
    assert top[0] > 2 * top[2] and bottom[2] > 2 * bottom[0]  # red half on top after turning clockwise
    assert loaded.alpha is None and loaded.icc is None  # developed straight into sRGB
    assert loaded.format == "RAW (DNG)"
    assert len(loaded.notes) == 1 and "RAW" in loaded.notes[0]


def test_raw_inspect_reads_size_without_developing(tmp_path, monkeypatch):
    red_and_blue_dng(tmp_path / "camera.dng", orientation=6)

    def no_developing(*args, **kwargs):
        raise AssertionError("inspect must not develop the whole RAW file")

    monkeypatch.setattr(imageio_utils, "_decode_raw", no_developing)
    info = inspect_image(tmp_path / "camera.dng")

    assert (info.width, info.height, info.format, info.has_alpha, info.frames) == (48, 64, "RAW (DNG)", False, 1)


def test_raw_that_libraw_cannot_read_is_friendly(tmp_path, monkeypatch):
    (tmp_path / "broken.cr3").write_bytes(b"\x00" * 512)
    monkeypatch.setattr(imageio_utils, "_on_mac", lambda: False)  # no sips to fall back on

    with pytest.raises(ImageFileError, match="RAW"):
        load_image(tmp_path / "broken.cr3")


# --- Vectors ---------------------------------------------------------------------------------


def test_render_vector_draws_at_exact_size(tmp_path):
    (tmp_path / "logo.svg").write_text(svg_text())

    rgba = render_vector(tmp_path / "logo.svg", (1600, 800))
    stretched = render_vector(tmp_path / "logo.svg", (1601, 800))  # aspect off by a pixel still fills exactly

    assert rgba.dtype == np.uint8 and rgba.shape == (800, 1600, 4)
    assert tuple(rgba[400, 400]) == (255, 0, 0, 255) and tuple(rgba[400, 1200]) == (0, 0, 255, 255)
    assert stretched.shape == (800, 1601, 4)


def test_svg_inspect_and_load_use_its_own_size(tmp_path):
    (tmp_path / "logo.svg").write_text(svg_text(width="210mm", height="105mm"))

    info = inspect_image(tmp_path / "logo.svg")
    loaded = load_image(tmp_path / "logo.svg")
    big = load_image(tmp_path / "logo.svg", vector_size=(2000, 1000))

    assert (info.width, info.height) == (794, 397)  # 210 mm at 96 per inch, as browsers draw it
    assert (info.format, info.is_vector, info.has_alpha, info.frames) == ("SVG", True, False, 1)
    assert loaded.rgb.shape == (397, 794, 3) and loaded.alpha is None  # fully covered, so no see-through
    assert big.rgb.shape == (1000, 2000, 3) and loaded.format == "SVG"


def test_svg_transparent_background_keeps_alpha(tmp_path):
    (tmp_path / "icon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
        '<circle cx="10" cy="10" r="8" fill="red"/></svg>'
    )

    loaded = load_image(tmp_path / "icon.svg")

    assert inspect_image(tmp_path / "icon.svg").has_alpha
    assert loaded.alpha is not None and loaded.alpha[10, 30] == 0 and loaded.alpha[10, 10] == 255
    assert tuple(loaded.rgb[10, 10]) == (255, 0, 0)


@pytest.mark.parametrize(
    ("width", "height", "view_box", "expected"),
    [
        ("300", None, "0 0 30 10", (300, 100)),  # only a width: the height follows the viewBox shape
        (None, "300", "0 0 30 10", (900, 300)),
        ("100%", "100%", "0 0 640 480", (640, 480)),
        (None, None, "0,0,640,480", (640, 480)),
        ("2in", "1in", None, (192, 96)),
        ("72pt", "36pt", None, (96, 48)),
        ("100.6", "50.5", None, (101, 51)),
    ],
)
def test_svg_size_rules_match_resvg(tmp_path, width, height, view_box, expected):
    attributes = " ".join(
        f'{name}="{value}"' for name, value in (("width", width), ("height", height), ("viewBox", view_box)) if value
    )
    (tmp_path / "shape.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" {attributes}><rect width="5" height="5"/></svg>'
    )

    assert (inspect_image(tmp_path / "shape.svg").width, inspect_image(tmp_path / "shape.svg").height) == expected
    assert load_image(tmp_path / "shape.svg").rgb.shape[:2] == expected[::-1]


def test_svg_never_reads_other_files_on_this_computer(tmp_path):
    # A guest on the Wi-Fi could otherwise send an SVG that pulls the host's private pictures into its result.
    secret = private_picture(tmp_path / "private")
    body = (
        f'<image x="0" y="0" width="20" height="10" xlink:href="{secret}"/>'
        f"<image x='0' y='0' width='20' height='10' href='file://{secret}'/>"
    )
    (tmp_path / "sneaky.svg").write_text(svg_text(body=body))

    rgba = render_vector(tmp_path / "sneaky.svg", (200, 100))
    info = inspect_image(tmp_path / "sneaky.svg")
    loaded = load_image(tmp_path / "sneaky.svg")

    assert not shows_private_picture(rgba)
    assert tuple(rgba[50, 50]) == (255, 0, 0, 255)  # the rest of the drawing is still there
    assert len(info.warnings) == 1 and "left out" in info.warnings[0]
    assert loaded.notes == info.warnings


@pytest.mark.parametrize(
    "body",
    [
        # A quoted "#" inside another attribute's value hides the real link from a plain text search...
        """<image width="20" height="10" data-a="x href='#" href="{secret}" data-b='"'/>""",
        """<image width="20" height="10" data-a="x xlink:href='#" xlink:href="{secret}" data-b='"'/>""",
        # ...and so does one in the drawing's text.
        """<text>a href='#</text><image width="20" height="10" href="{secret}"/><text>'</text>""",
        # Any prefix can stand for XLink.
        """<image xmlns:q="http://www.w3.org/1999/xlink" width="20" height="10" q:href="{secret}"/>""",
    ],
)
def test_svg_links_hidden_from_a_text_search_are_left_out_too(tmp_path, body):
    secret = private_picture(tmp_path / "private")
    (tmp_path / "sneaky.svg").write_text(svg_text(body=body.format(secret=secret)))

    rgba = render_vector(tmp_path / "sneaky.svg", (200, 100))
    info = inspect_image(tmp_path / "sneaky.svg")

    assert not shows_private_picture(rgba)
    assert tuple(rgba[50, 50]) == (255, 0, 0, 255)
    assert len(info.warnings) == 1 and "left out" in info.warnings[0]
    assert load_image(tmp_path / "sneaky.svg").notes == info.warnings


@pytest.mark.parametrize(
    ("folder", "link"),
    [
        ("#", "#/../secret.png"),  # starts like a link inside the file, but resvg reads it as a path
        pytest.param(
            "data:",
            "data:/../secret.png",  # a data: link without its comma is a path to resvg too
            marks=pytest.mark.skipif(sys.platform == "win32", reason="Windows file names can't hold ':'"),
        ),
    ],
)
def test_svg_links_that_only_look_internal_are_left_out(tmp_path, monkeypatch, folder, link):
    private_picture(tmp_path)
    (tmp_path / folder).mkdir()
    monkeypatch.chdir(tmp_path)  # resvg looks for relative paths here unless told otherwise
    (tmp_path / "sneaky.svg").write_text(svg_text(body=f'<image width="20" height="10" href="{link}"/>'))

    rgba = render_vector(tmp_path / "sneaky.svg", (200, 100))

    assert not shows_private_picture(rgba)
    assert "left out" in inspect_image(tmp_path / "sneaky.svg").warnings[0]


def test_svg_drawing_never_looks_in_the_working_folder(tmp_path, monkeypatch):
    # resvg reads even a "#name" link on a picture as a file name, so it gets an empty folder to look in.
    private_picture(tmp_path, "#secret.png")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "odd.svg").write_text(svg_text(body='<image width="20" height="10" href="#secret.png"/>'))

    assert not shows_private_picture(render_vector(tmp_path / "odd.svg", (200, 100)))


@pytest.mark.parametrize(
    ("doctype", "body"),
    [
        # An entity holding a linked picture, spelled plainly or with a character reference.
        ("""<!DOCTYPE svg [<!ENTITY pic "<image width='20' height='10' href='{secret}'/>">]>""", "&pic;"),
        ("""<!DOCTYPE svg [<!ENTITY pic "<image width='20' height='10' &#104;ref='{secret}'/>">]>""", "&pic;"),
        # A link given as the attribute's default in the DTD rather than in the tag.
        ("""<!DOCTYPE svg [<!ATTLIST image href CDATA "{secret}">]>""", '<image width="20" height="10"/>'),
        # An external entity: another file pulled into the SVG.
        ("""<!DOCTYPE svg [<!ENTITY pic SYSTEM "{secret}">]>""", "&pic;"),
    ],
)
def test_svg_links_made_by_its_dtd_are_refused(tmp_path, doctype, body):
    secret = private_picture(tmp_path / "private")
    (tmp_path / "sneaky.svg").write_text(doctype.format(secret=secret) + svg_text(body=body))

    for action in (lambda path: render_vector(path, (200, 100)), inspect_image, load_image, make_thumbnail):
        with pytest.raises(ImageFileError, match="couldn't be drawn"):
            action(tmp_path / "sneaky.svg")


def test_svg_with_illustrator_entities_still_draws(tmp_path):
    # Illustrator writes its namespaces as entities; only entities that bring in other files are refused.
    doctype = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [\n'
        '\t<!ENTITY ns_svg "http://www.w3.org/2000/svg">\n\t<!ENTITY ns_xlink "http://www.w3.org/1999/xlink">\n]>\n'
    )
    svg = svg_text().replace("http://www.w3.org/2000/svg", "&ns_svg;")
    (tmp_path / "illustrator.svg").write_text(doctype + svg.replace("http://www.w3.org/1999/xlink", "&ns_xlink;"))

    rgba = render_vector(tmp_path / "illustrator.svg", (200, 100))

    assert tuple(rgba[50, 50]) == (255, 0, 0, 255) and tuple(rgba[50, 150]) == (0, 0, 255, 255)
    assert inspect_image(tmp_path / "illustrator.svg").warnings == []


def test_svg_embedded_pictures_still_draw(tmp_path):
    png = io.BytesIO()
    Image.new("RGB", (4, 4), "lime").save(png, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
    body = f'<image width="20" height="10" href="{uri}" preserveAspectRatio="none"/>'
    (tmp_path / "embedded.svg").write_text(svg_text(body=body))

    rgba = render_vector(tmp_path / "embedded.svg", (200, 100))

    assert tuple(rgba[50, 50][:3]) == (0, 255, 0)
    assert inspect_image(tmp_path / "embedded.svg").warnings == []


def test_svg_saved_as_utf16_fills_the_size_asked_for(tmp_path):
    text = '<?xml version="1.0" encoding="UTF-16"?>' + svg_text(width="210mm", height="105mm")
    (tmp_path / "windows.svg").write_bytes(text.encode("utf-16"))

    loaded = load_image(tmp_path / "windows.svg", vector_size=(2000, 1000))

    assert loaded.rgb.shape == (1000, 2000, 3)
    assert loaded.alpha is None  # no see-through strip along two edges
    assert not inspect_image(tmp_path / "windows.svg").has_alpha


def test_svgz_loads(tmp_path):
    (tmp_path / "logo.svgz").write_bytes(gzip.compress(svg_text().encode()))

    assert load_image(tmp_path / "logo.svgz").rgb.shape == (100, 200, 3)
    assert inspect_image(tmp_path / "logo.svgz").is_vector


def test_broken_svg_is_friendly(tmp_path):
    (tmp_path / "broken.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg' width='10'")

    with pytest.raises(ImageFileError, match="export it as PNG"):
        load_image(tmp_path / "broken.svg")


# --- HDR and other formats through ffmpeg ------------------------------------------------------


def test_exr_with_alpha_loads(tmp_path):
    rgba = np.dstack([halves(), np.full((40, 80), 255, np.uint8)])
    rgba[:, :10, 3] = 0
    Image.fromarray(rgba).save(tmp_path / "render.png")
    ffmpeg("-i", str(tmp_path / "render.png"), "-pix_fmt", "gbrapf32le", str(tmp_path / "render.exr"))

    loaded = load_image(tmp_path / "render.exr")
    info = inspect_image(tmp_path / "render.exr")

    assert loaded.rgb.shape == (40, 80, 3) and loaded.format == "EXR"
    assert loaded.rgb[:, 14:36].min() > 245 and loaded.rgb[:, 44:].max() < 10
    assert loaded.alpha is not None and loaded.alpha[:, :10].max() == 0 and loaded.alpha[:, 10:].min() == 255
    assert loaded.notes == ["HDR image converted to standard range."]
    assert (info.width, info.height, info.format, info.has_alpha, info.warnings) == (80, 40, "EXR", True, loaded.notes)


def hdr_scene(tmp_path: Path, height: int = 8, width: int = 32) -> list[Path]:
    """Four bands, left to right: black, mid grey, white, and a highlight 2 stops brighter than white,
    as scene.exr and scene.hdr."""
    linear = np.zeros((height, width, 3), np.float32)
    for band, value in enumerate([0.0, 0.18, 1.0, 4.0]):
        linear[:, band * width // 4 : (band + 1) * width // 4] = value
    write_pfm(tmp_path / "scene.pfm", linear)
    ffmpeg("-i", str(tmp_path / "scene.pfm"), str(tmp_path / "scene.exr"))
    ffmpeg("-i", str(tmp_path / "scene.pfm"), str(tmp_path / "scene.hdr"))
    return [tmp_path / "scene.exr", tmp_path / "scene.hdr"]


def assert_highlights_roll_off(rgb: np.ndarray, highest: int = 255) -> None:
    """For hdr_scene: white is no longer clipped flat against the brighter highlight."""
    height, width = rgb.shape[:2]
    black, grey, white, highlight = (int(rgb[height // 2, band * width // 4 + width // 8, 1]) for band in range(4))
    assert black < 3 and highlight >= highest
    assert 90 < grey < white < 235


def ffmpeg_runs(monkeypatch) -> list[list[str]]:
    """The command of every FFmpeg run imageio_utils makes from now on."""
    runs = []
    command = imageio_utils._ffmpeg_command

    def recorded(*args, **kwargs):
        runs.append(command(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(imageio_utils, "_ffmpeg_command", recorded)
    return runs


def test_hdr_highlights_roll_off_instead_of_clipping(tmp_path):
    for path in hdr_scene(tmp_path):
        loaded = load_image(path)
        assert_highlights_roll_off(loaded.rgb)
        assert loaded.notes == ["HDR image converted to standard range."]


def test_ffmpeg_formats_respect_the_size_limit(tmp_path, monkeypatch):
    Image.fromarray(halves()).save(tmp_path / "render.png")
    ffmpeg("-i", str(tmp_path / "render.png"), "-pix_fmt", "gbrpf32le", str(tmp_path / "render.exr"))
    monkeypatch.setattr(imageio_utils, "MAX_OPEN_PIXELS", 80 * 40 - 1)

    with pytest.raises(ImageFileError, match="too big"):
        load_image(tmp_path / "render.exr")


def test_hdr_inspect_reads_the_header_without_decoding(tmp_path, monkeypatch):
    rgba = np.dstack([halves(), np.full((40, 80), 255, np.uint8)])
    Image.fromarray(rgba).save(tmp_path / "render.png")
    ffmpeg("-i", str(tmp_path / "render.png"), "-pix_fmt", "gbrapf32le", str(tmp_path / "render.exr"))
    ffmpeg("-i", str(tmp_path / "render.png"), str(tmp_path / "render.hdr"))
    runs = ffmpeg_runs(monkeypatch)

    exr, hdr = inspect_image(tmp_path / "render.exr"), inspect_image(tmp_path / "render.hdr")

    assert runs == []
    # An EXR's alpha channel counts as see-through without decoding, as for very big PNGs.
    assert exr == ImageInfo(80, 40, "EXR", True, 1, False, ["HDR image converted to standard range."])
    assert hdr == ImageInfo(80, 40, "HDR", False, 1, False, ["HDR image converted to standard range."])


def test_exr_size_is_its_display_window(tmp_path):
    # Renders often store only the part of the frame with something in it (the data window). FFmpeg
    # gives the whole frame (the display window), black around that part, so that is the size.
    write_exr(tmp_path / "crop.exr", display=(0, 0, 63, 31), data=(16, 8, 47, 23), value=0.5)

    info = inspect_image(tmp_path / "crop.exr")
    loaded = load_image(tmp_path / "crop.exr")

    assert (info.width, info.height, info.has_alpha) == (64, 32, False)
    assert loaded.rgb.shape == (32, 64, 3)
    assert loaded.rgb[16, 32].min() > 150 and loaded.rgb[2, 2].max() == 0


def test_hdr_is_decoded_once_with_its_own_size_limit(tmp_path, monkeypatch):
    rgba = np.dstack([halves(), np.full((40, 80), 255, np.uint8)])
    rgba[:, :10, 3] = 0
    Image.fromarray(rgba).save(tmp_path / "render.png")
    ffmpeg("-i", str(tmp_path / "render.png"), "-pix_fmt", "gbrapf32le", str(tmp_path / "render.exr"))
    runs = ffmpeg_runs(monkeypatch)

    loaded = load_image(tmp_path / "render.exr")

    assert loaded.alpha is not None and loaded.alpha[:, :10].max() == 0
    assert len(runs) == 1  # colour and alpha come out together
    assert runs[0][runs[0].index("-max_pixels") + 1] == str(8192 * 8192)


def test_hdr_has_a_lower_size_limit(tmp_path, monkeypatch):
    # Floating-point pictures take 12-16 bytes a pixel while they are converted, so they stop at 8K.
    assert imageio_utils._MAX_HDR_PIXELS == 8192 * 8192
    paths = hdr_scene(tmp_path, 40, 80)
    monkeypatch.setattr(imageio_utils, "_MAX_HDR_PIXELS", 80 * 40 - 1)

    for path in paths:
        for action in (load_image, inspect_image, make_thumbnail):
            with pytest.raises(ImageFileError, match="too big for Re:Solution to open"):
                action(path)


def test_ffmpeg_refuses_an_hdr_bigger_than_its_header_says(tmp_path, monkeypatch):
    paths = hdr_scene(tmp_path, 40, 80)
    monkeypatch.setattr(imageio_utils, "_MAX_HDR_PIXELS", 80 * 40 - 1)
    monkeypatch.setattr(imageio_utils, "_hdr_header", lambda file: ((10, 10), False))

    for path in paths:
        with pytest.raises(ImageFileError, match="too big for Re:Solution to open"):
            load_image(path)


def test_hdr_thumbnail_is_shrunk_inside_ffmpeg_keeping_highlights(tmp_path, monkeypatch):
    paths = hdr_scene(tmp_path, 64, 256)
    runs = ffmpeg_runs(monkeypatch)

    for path in paths:
        runs.clear()
        thumb = decode(make_thumbnail(path, max_side=64))
        assert thumb.shape == (16, 64, 3), path.name
        assert len(runs) == 1 and "zscale=w=64:h=16:filter=bilinear" in runs[0], path.name
        assert_highlights_roll_off(thumb, highest=250)  # FFmpeg's ordinary scaler would clip them to white


def test_hdr_thumbnail_without_zscale_shrinks_after_decoding(tmp_path, monkeypatch):
    paths = hdr_scene(tmp_path, 64, 256)
    monkeypatch.setattr(imageio_utils, "_HDR_SHRINK_FILTER", "nosuchfilter=w={width}:h={height}")

    for path in paths:
        thumb = decode(make_thumbnail(path, max_side=64))
        assert thumb.shape == (16, 64, 3), path.name
        assert_highlights_roll_off(thumb, highest=250)


@needs_mac
def test_sips_is_the_last_resort(tmp_path):
    # ASTC textures: neither Pillow nor ffmpeg can read them, but macOS can.
    Image.fromarray(halves()).save(tmp_path / "flat.png")
    sips(tmp_path / "flat.png", "org.khronos.astc", tmp_path / "texture.astc")

    loaded = load_image(tmp_path / "texture.astc")
    info = inspect_image(tmp_path / "texture.astc")

    assert loaded.rgb.shape == (40, 80, 3)
    assert_halves(loaded.rgb, tolerance=60)
    assert (info.width, info.height) == (80, 40)


@needs_mac
def test_sips_keeps_orientation_and_profile(tmp_path, monkeypatch):
    p3 = MAC_P3_PROFILE.read_bytes() if MAC_P3_PROFILE.is_file() else None
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(halves()).save(tmp_path / "phone.jpg", exif=exif.tobytes(), icc_profile=p3, quality=95)
    monkeypatch.setattr(imageio_utils, "_decode_with_pillow", _never_decodes)

    loaded = load_image(tmp_path / "phone.jpg")

    assert loaded.rgb.shape == (80, 40, 3) and loaded.rgb[:35].mean() > 200 and loaded.rgb[45:].mean() < 50
    assert loaded.icc == p3


def _never_decodes(path, preview_side=None):
    raise imageio_utils._NotDecoded()


def test_jpeg_xl_off_mac_is_friendly(tmp_path, monkeypatch):
    (tmp_path / "modern.jxl").write_bytes(b"\xff\x0a" + b"\x00" * 64)
    monkeypatch.setattr(imageio_utils, "_on_mac", lambda: False)

    for action in (load_image, inspect_image, make_thumbnail):
        with pytest.raises(ImageFileError, match="JPEG XL"):
            action(tmp_path / "modern.jxl")


# --- Friendly refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content", "kind"),
    [
        ("flyer.pdf", b"%PDF-1.7\n", "PDF"),
        ("logo.eps", b"%!PS-Adobe-3.0 EPSF-3.0\n", "EPS"),
        ("misnamed.png", b"%PDF-1.4\n", "PDF"),  # recognised by what's inside, not just the name
    ],
)
def test_pdf_and_eps_explain_how_to_export(tmp_path, name, content, kind):
    (tmp_path / name).write_bytes(content + b"\x00" * 100)

    for action in (load_image, inspect_image, make_thumbnail):
        with pytest.raises(ImageFileError, match=f"{kind} document") as caught:
            action(tmp_path / name)
        assert "Export the page as PNG or TIFF at the size you need, then drop that in." in str(caught.value)


def test_unsupported_file_friendly_message(tmp_path):
    with zipfile.ZipFile(tmp_path / "letter.docx", "w") as docx:
        docx.writestr("word/document.xml", "<w:document/>")

    for action in (load_image, inspect_image, make_thumbnail):
        with pytest.raises(ImageFileError, match="This file type isn't an image Re:Solution can read"):
            action(tmp_path / "letter.docx")


def test_video_points_to_the_videos_tab(tmp_path):
    (tmp_path / "holiday.mov").write_bytes(b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 100)

    with pytest.raises(ImageFileError, match="Videos tab"):
        load_image(tmp_path / "holiday.mov")


def test_damaged_picture_keeps_the_damaged_message(tmp_path):
    (tmp_path / "holiday.jpg").write_text("not a picture")

    with pytest.raises(ImageFileError, match="couldn't be opened"):
        load_image(tmp_path / "holiday.jpg")


@pytest.mark.parametrize("name", ["holiday.jpg", "logo.png"])
def test_half_copied_picture_is_refused_on_every_computer(tmp_path, name):
    # macOS's sips "opens" a cut-off file by filling the missing part with flat grey, which would
    # then be upscaled for minutes into an 8K picture with a grey band.
    whole = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (600, 400, 3), np.uint8)).save(
        whole, "PNG" if name.endswith(".png") else "JPEG"
    )
    (tmp_path / name).write_bytes(whole.getvalue()[: len(whole.getvalue()) // 2])

    for action in (load_image, make_thumbnail):
        with pytest.raises(ImageFileError, match="incomplete") as raised:
            action(tmp_path / name)
        assert str(raised.value).startswith(name) and "again" in str(raised.value)


def test_svg_with_a_huge_declared_size_opens_at_the_biggest_size(tmp_path, monkeypatch):
    # A banner set out in millimetres can declare more than the app opens (16K squared); a vector
    # draws at any size, so it's shown at the largest size that fits instead of being refused.
    monkeypatch.setattr(imageio_utils, "MAX_OPEN_PIXELS", 1_000_000)  # 1 MP stands in for 16K squared
    (tmp_path / "banner.svg").write_text(svg_text(width="6000mm", height="3000mm"))  # 22,677 x 11,339 px

    info = inspect_image(tmp_path / "banner.svg")
    loaded = load_image(tmp_path / "banner.svg")

    assert info.is_vector and (info.width, info.height) == (1414, 707)
    assert loaded.rgb.shape[:2] == (707, 1414)
    assert tuple(loaded.rgb[0, 0]) == (255, 0, 0) and tuple(loaded.rgb[-1, -1]) == (0, 0, 255)
    with pytest.raises(ValueError):  # asking for too big a drawing is still refused
        render_vector(tmp_path / "banner.svg", (2000, 1000))


def test_raw_and_svg_respect_the_size_limit(tmp_path, monkeypatch):
    red_and_blue_dng(tmp_path / "camera.dng")
    (tmp_path / "poster.svg").write_text(svg_text())
    monkeypatch.setattr(imageio_utils, "MAX_OPEN_PIXELS", 1000)

    for action in (load_image, inspect_image):
        with pytest.raises(ImageFileError, match="too big"):
            action(tmp_path / "camera.dng")
    # A vector is drawn smaller instead (200 x 100 declared, fitted within 1000 pixels).
    assert (inspect_image(tmp_path / "poster.svg").width, inspect_image(tmp_path / "poster.svg").height) == (44, 22)
    assert load_image(tmp_path / "poster.svg").rgb.shape == (22, 44, 3)
    with pytest.raises(ValueError):
        render_vector(tmp_path / "poster.svg", (100, 100))


# --- inspect_image -------------------------------------------------------------------------------


def test_inspect_reads_only_the_header(tmp_path, monkeypatch):
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(gradient(300, 200)).save(tmp_path / "phone.jpg", exif=exif.tobytes())

    def no_decoding(self, *args, **kwargs):
        raise AssertionError("inspect must not decode a JPEG")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decoding)
    info = inspect_image(tmp_path / "phone.jpg")

    assert info == ImageInfo(
        width=200, height=300, format="JPEG", has_alpha=False, frames=1, is_vector=False, warnings=[]
    )


def test_inspect_alpha_means_actually_see_through(tmp_path):
    opaque = np.dstack([gradient(40, 30), np.full((30, 40), 255, np.uint8)])
    Image.fromarray(opaque).save(tmp_path / "chatgpt.png")
    see_through = opaque.copy()
    see_through[:5, :5, 3] = 0
    Image.fromarray(see_through).save(tmp_path / "logo.png")

    assert not inspect_image(tmp_path / "chatgpt.png").has_alpha  # saved as RGBA, but nothing is see-through
    assert inspect_image(tmp_path / "logo.png").has_alpha


def make_every_format(tmp_path: Path) -> list[Path]:
    """One small picture per format family, each 80x40 when upright."""
    pixels = halves()
    paths = []
    for name, fmt in [("a.jpg", "JPEG"), ("a.png", "PNG"), ("a.webp", "WEBP"), ("a.avif", "AVIF"), ("a.tif", "TIFF"),
                      ("a.bmp", "BMP"), ("a.gif", "GIF"), ("a.jp2", "JPEG2000"), ("a.tga", "TGA"), ("a.ppm", "PPM"),
                      ("a.dds", "DDS"), ("a.qoi", "QOI"), ("a.heic", "HEIF")]:  # fmt: skip
        Image.fromarray(pixels).save(tmp_path / name, fmt)
        paths.append(tmp_path / name)
    ffmpeg("-i", str(tmp_path / "a.png"), "-pix_fmt", "gbrpf32le", str(tmp_path / "a.exr"))
    ffmpeg("-i", str(tmp_path / "a.png"), str(tmp_path / "a.hdr"))
    red_and_blue_dng(tmp_path / "a.dng")  # 64x48
    (tmp_path / "a.svg").write_text(svg_text(width="80", height="40"))
    return [*paths, tmp_path / "a.exr", tmp_path / "a.hdr", tmp_path / "a.dng", tmp_path / "a.svg"]


def test_inspect_matches_load_for_every_format(tmp_path):
    for path in make_every_format(tmp_path):
        info = inspect_image(path)
        loaded = load_image(path)
        assert (info.height, info.width) == loaded.rgb.shape[:2], path.name
        assert info.format == loaded.format, path.name
        assert info.warnings == loaded.notes, path.name


@pytest.mark.parametrize("orientation", [1, 3, 5, 6, 7, 8])
def test_inspect_matches_load_for_turned_tiffs(tmp_path, orientation):
    # Pillow's TIFF reader turns these itself and already reports the turned size before decoding.
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    Image.fromarray(gradient(40, 20)).save(tmp_path / "scan.tif", exif=exif.tobytes())

    info = inspect_image(tmp_path / "scan.tif")
    loaded = load_image(tmp_path / "scan.tif")

    upright = (20, 40) if orientation >= 5 else (40, 20)
    assert (info.width, info.height) == upright
    assert loaded.rgb.shape[:2] == upright[::-1]


# --- make_thumbnail --------------------------------------------------------------------------------


def decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.format == "JPEG"
        return np.asarray(image.convert("RGB"))


def test_make_thumbnail_every_format(tmp_path):
    for path in make_every_format(tmp_path):
        thumb = decode(make_thumbnail(path, max_side=32))
        expected = (24, 32) if path.suffix == ".dng" else (16, 32)
        assert thumb.shape[:2] == expected, path.name


def test_make_thumbnail_shrinks_big_pictures_and_keeps_small_ones(tmp_path):
    Image.fromarray(gradient(2000, 1000)).save(tmp_path / "big.jpg")
    Image.fromarray(gradient(200, 100)).save(tmp_path / "small.png")

    assert decode(make_thumbnail(tmp_path / "big.jpg")).shape == (240, 480, 3)
    assert decode(make_thumbnail(tmp_path / "small.png")).shape == (100, 200, 3)


def test_make_thumbnail_vector_is_drawn_at_thumbnail_size(tmp_path):
    (tmp_path / "icon.svg").write_text(svg_text(width="20", height="10"))

    thumb = decode(make_thumbnail(tmp_path / "icon.svg", max_side=480))

    assert thumb.shape == (240, 480, 3)  # drawn crisply, not a 20 px picture blown up
    assert tuple(thumb[120, 100]) == pytest.approx((255, 0, 0), abs=6)


def test_make_thumbnail_turns_phone_photos_upright(tmp_path):
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(halves(800, 400)).save(tmp_path / "phone.jpg", exif=exif.tobytes(), quality=95)

    thumb = decode(make_thumbnail(tmp_path / "phone.jpg", max_side=100))

    assert thumb.shape == (100, 50, 3) and thumb[:45].mean() > 200 and thumb[55:].mean() < 50


def test_make_thumbnail_shows_transparency_as_a_checkerboard(tmp_path):
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(tmp_path / "empty.png")

    thumb = decode(make_thumbnail(tmp_path / "empty.png")).astype(int)

    assert thumb.min() > 180  # light, not black
    assert abs(thumb[2, 2, 0] - thumb[2, 10, 0]) > 15  # neighbouring squares differ


@pytest.mark.skipif(not MAC_P3_PROFILE.is_file(), reason="needs macOS's Display P3 profile")
def test_make_thumbnail_keeps_wide_gamut_profile(tmp_path):
    p3 = MAC_P3_PROFILE.read_bytes()
    Image.fromarray(gradient(40, 20)).save(tmp_path / "vivid.png", icc_profile=p3)

    with Image.open(io.BytesIO(make_thumbnail(tmp_path / "vivid.png"))) as thumb:
        assert thumb.info.get("icc_profile") == p3


# --- save_image: WebP and "same as original" ---------------------------------------------------------------


def test_save_webp_with_dpi_profile_and_alpha(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    alpha = np.full((30, 40), 255, np.uint8)
    alpha[:, :10] = 0

    saved = save_image(gradient(40, 30), alpha, tmp_path / "web.webp", "webp", icc, dpi=150)

    assert saved == tmp_path / "web.webp"
    with Image.open(saved) as image:
        assert image.format == "WEBP" and image.mode == "RGBA"
        assert image.info.get("icc_profile") == icc
        assert image.getexif()[X_RESOLUTION] == 150
        assert np.asarray(image)[:, :10, 3].max() == 0


def test_save_webp_renames_a_jpg_target(tmp_path):
    saved = save_image(gradient(16, 16), None, tmp_path / "x-4K-fast.jpg", "webp", None)

    assert saved == tmp_path / "x-4K-fast.webp" and saved.is_file()


def test_webp_too_large_friendly_error(tmp_path):
    wide = np.zeros((1, 16384, 3), np.uint8)

    with pytest.raises(ValueError, match="WebP") as caught:
        save_image(wide, None, tmp_path / "huge.webp", "webp", None)
    assert "16,383" in str(caught.value) and "JPEG, PNG or TIFF" in str(caught.value)
    assert list(tmp_path.iterdir()) == []
    save_image(np.zeros((1, 16383, 3), np.uint8), None, tmp_path / "ok.webp", "webp", None)  # the limit itself fits


@pytest.mark.parametrize(
    ("source_format", "suffix", "pil_format"),
    [
        ("JPEG", ".jpg", "JPEG"),
        ("PNG", ".png", "PNG"),
        ("TIFF", ".tif", "TIFF"),
        ("WebP", ".webp", "WEBP"),
        ("AVIF", ".avif", "AVIF"),
        ("HEIC", ".heic", "HEIF"),
    ],
)
def test_save_original_writes_the_source_format(tmp_path, source_format, suffix, pil_format):
    target = tmp_path / "photo-4K-fast.jpg"
    saved = save_image(gradient(32, 24), None, target, "original", None, source_format=source_format)

    assert saved.suffix == suffix and saved.is_file()
    with Image.open(saved) as image:
        assert image.format == pil_format and image.size == (32, 24)
        dpi = image.info.get("dpi") or (image.getexif().get(X_RESOLUTION),)
        assert round(dpi[0]) == 300  # DPI goes into every format


def test_save_original_heic_falls_back_or_writes(tmp_path, monkeypatch):
    saved = save_image(gradient(32, 24), None, tmp_path / "iphone.heic", "original", None, source_format="HEIC")
    assert saved.suffix == ".heic"
    with Image.open(saved) as image:
        assert image.format == "HEIF"

    # HEIC (HEVC) can't hold a 16K picture; the result falls back to JPEG, with a note saying so.
    assert resolve_format("original", "HEIC", (16384, 12312)) == (
        "jpeg",
        "HEIC can't hold a picture this big, so it was saved as JPEG.",
    )
    assert resolve_format("original", "HEIC", (16384, 12312), alpha=True)[0] == "png"
    monkeypatch.setitem(imageio_utils._MAX_PIXELS, "heic", 100)
    fallback = save_image(gradient(32, 24), None, tmp_path / "big.heic", "original", None, source_format="HEIC")
    assert fallback.suffix == ".jpg"


def test_save_original_too_big_for_jpeg_falls_back_to_png():
    assert resolve_format("original", "JPEG", (70000, 100)) == (
        "png",
        "JPEG can't hold a picture this big, so it was saved as PNG.",
    )
    assert resolve_format("original", "AVIF", (70000, 100))[0] == "png"  # JPEG can't hold it either
    assert resolve_format("original", "WebP", (20000, 100))[0] == "jpeg"


@pytest.mark.parametrize("source_format", ["RAW (CR3)", "GIF", "PSD", "SVG", "BMP", None])
def test_save_original_unwritable_is_png_with_note(tmp_path, source_format):
    fmt, note = resolve_format("original", source_format)

    assert fmt == "png" and "PNG" in note
    saved = save_image(gradient(8, 8), None, tmp_path / "x.jpg", "original", None, source_format=source_format)
    assert saved.suffix == ".png"


def test_resolve_format_passes_plain_choices_through():
    assert resolve_format("jpeg") == ("jpeg", None)
    assert resolve_format("webp", size=(16383, 9000)) == ("webp", None)
    assert resolve_format("original", "jpeg") == ("jpeg", None)  # case doesn't matter
    assert resolve_format("original", "HEIF") == ("heic", None)  # Pillow's own name works too
    with pytest.raises(ValueError, match="Unknown output format"):
        resolve_format("gif")


def test_loaded_format_feeds_save_original(tmp_path):
    Image.fromarray(gradient(32, 24)).save(tmp_path / "photo.webp", quality=90)
    loaded = load_image(tmp_path / "photo.webp")

    target = tmp_path / "out.jpg"
    saved = save_image(loaded.rgb, loaded.alpha, target, "original", loaded.icc, source_format=loaded.format)

    assert saved.suffix == ".webp"


def test_save_original_transparent_jpeg_target_stays_transparent(tmp_path):
    alpha = np.zeros((8, 8), np.uint8)

    saved = save_image(gradient(8, 8), alpha, tmp_path / "x.jpg", "original", None, source_format="WebP")

    with Image.open(saved) as image:
        assert image.format == "WEBP" and image.mode == "RGBA"


def test_too_big_for_a_format_names_only_formats_that_hold_it(tmp_path):
    from upscaler.config import UserError

    with pytest.raises(UserError) as webp:
        resolve_format("webp", None, (20_000, 100))
    assert str(webp.value).endswith("Choose JPEG, PNG or TIFF instead.")  # all three hold 20,000 px
    with pytest.raises(UserError) as jpeg:
        resolve_format("jpeg", None, (70_000, 100))  # beyond JPEG's 65,535, and so WebP's too
    assert "JPEG can't hold" in str(jpeg.value) and str(jpeg.value).endswith("Choose PNG or TIFF instead.")
    with pytest.raises(UserError) as heic:
        resolve_format("heic", None, (16_384, 12_288))
    assert str(heic.value).endswith("Choose JPEG, PNG or TIFF instead.")


def test_format_and_dpi_problems_are_user_errors():
    from upscaler.config import UserError

    with pytest.raises(UserError):
        resolve_format("gif")
    with pytest.raises(UserError):
        imageio_utils.check_dpi(0)
