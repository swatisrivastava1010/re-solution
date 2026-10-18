import io
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms, JpegImagePlugin

from upscaler import imageio_utils
from upscaler.imageio_utils import (
    ImageFileError,
    compare_crops,
    ensure_free_space,
    load_image,
    preview_jpeg,
    save_image,
    unique_path,
)

ORIENTATION = 0x0112
MAC_CMYK_PROFILE = Path("/System/Library/ColorSync/Profiles/Generic CMYK Profile.icc")


def gradient(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    return np.stack(np.broadcast_arrays(x + 0 * y, y + 0 * x, (x + y) / 2), axis=-1).astype(np.uint8)


def srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def write_png16_rgb(path: Path, pixels: np.ndarray) -> None:
    """A 16-bit-per-channel RGB PNG, which Pillow can read but not write."""
    h, w, _ = pixels.shape
    rows = b"".join(b"\x00" + pixels[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


# --- load_image ---------------------------------------------------------------


def test_load_rgb_png(tmp_path):
    pixels = gradient(40, 30)
    Image.fromarray(pixels).save(tmp_path / "plain.png")

    loaded = load_image(tmp_path / "plain.png")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (30, 40, 3)
    np.testing.assert_array_equal(loaded.rgb, pixels)
    assert loaded.alpha is None and loaded.icc is None
    assert loaded.name == "plain.png"


def test_load_rgba_opaque_drops_alpha(tmp_path):
    pixels = gradient(40, 30)
    rgba = np.dstack([pixels, np.full((30, 40), 255, np.uint8)])
    Image.fromarray(rgba).save(tmp_path / "chatgpt.png")

    loaded = load_image(tmp_path / "chatgpt.png")

    assert loaded.alpha is None
    np.testing.assert_array_equal(loaded.rgb, pixels)


def test_load_rgba_transparent_keeps_alpha(tmp_path):
    pixels = gradient(40, 30)
    alpha = np.full((30, 40), 255, np.uint8)
    alpha[:, :10] = 0
    alpha[:, 10:20] = 128
    Image.fromarray(np.dstack([pixels, alpha])).save(tmp_path / "logo.png")

    loaded = load_image(tmp_path / "logo.png")

    np.testing.assert_array_equal(loaded.alpha, alpha)
    np.testing.assert_array_equal(loaded.rgb, pixels)

    saved = save_image(loaded.rgb, loaded.alpha, tmp_path / "out" / "logo-4K-fast.jpg", "jpeg", None)

    assert saved.suffix == ".png" and saved.is_file()
    assert not (tmp_path / "out" / "logo-4K-fast.jpg").exists()
    with Image.open(saved) as reopened:
        assert reopened.format == "PNG" and reopened.mode == "RGBA"
        np.testing.assert_array_equal(np.asarray(reopened)[..., 3], alpha)


def test_load_palette_with_transparency_keeps_alpha(tmp_path):
    image = Image.new("P", (8, 4), 1)
    image.putpalette([0, 0, 0, 200, 30, 30] + [0] * 762)
    image.paste(0, (0, 0, 4, 4))
    image.save(tmp_path / "sticker.png", transparency=0)

    loaded = load_image(tmp_path / "sticker.png")

    assert loaded.alpha is not None
    assert (loaded.alpha[:, :4] == 0).all() and (loaded.alpha[:, 4:] == 255).all()
    assert tuple(loaded.rgb[0, 6]) == (200, 30, 30)


def test_load_exif_rotated(tmp_path):
    pixels = np.zeros((40, 80, 3), np.uint8)
    pixels[:, :40] = 255  # left half white
    exif = Image.Exif()
    exif[ORIENTATION] = 6  # "rotate 90° clockwise to display"
    Image.fromarray(pixels).save(tmp_path / "phone.jpg", exif=exif.tobytes(), quality=95)

    loaded = load_image(tmp_path / "phone.jpg")

    assert loaded.rgb.shape == (80, 40, 3)
    assert loaded.rgb[:35].mean() > 200 and loaded.rgb[45:].mean() < 50  # white is now on top


def test_load_heic(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    if not pillow_heif.libheif_info().get("HEIF"):
        pytest.skip("this libheif has no HEIC encoder")
    pixels = np.zeros((40, 80, 3), np.uint8)
    pixels[:, :40] = 255
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(pixels).save(tmp_path / "iphone.heic", exif=exif.tobytes(), quality=90)

    loaded = load_image(tmp_path / "iphone.heic")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (80, 40, 3)  # rotated exactly once
    assert loaded.rgb[:35].mean() > 200 and loaded.rgb[45:].mean() < 50


def test_load_16bit_png(tmp_path):
    rng = np.random.default_rng(0)
    pixels16 = rng.integers(0, 65536, (12, 16, 3), dtype=np.uint16)
    write_png16_rgb(tmp_path / "deep.png", pixels16)

    loaded = load_image(tmp_path / "deep.png")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (12, 16, 3)
    assert np.abs(loaded.rgb.astype(int) - pixels16 / 257).max() <= 1


def test_load_16bit_grayscale_png(tmp_path):
    values = np.linspace(0, 65535, 16 * 12).reshape(12, 16).astype(np.uint16)
    Image.fromarray(values).save(tmp_path / "deep-grey.png")

    loaded = load_image(tmp_path / "deep-grey.png")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (12, 16, 3)
    assert np.abs(loaded.rgb[..., 0].astype(int) - values / 257).max() <= 1
    assert (loaded.rgb[..., 0] == loaded.rgb[..., 2]).all()


def test_load_cmyk_jpeg(tmp_path):
    Image.new("CMYK", (20, 10), (0, 255, 255, 0)).save(tmp_path / "print.jpg", quality=95)

    loaded = load_image(tmp_path / "print.jpg")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (10, 20, 3)
    r, g, b = loaded.rgb[5, 10].astype(int)
    assert r > 230 and g < 25 and b < 25
    assert loaded.icc is None


@pytest.mark.skipif(not MAC_CMYK_PROFILE.is_file(), reason="needs macOS's Generic CMYK profile")
def test_load_cmyk_jpeg_with_profile_converts_to_srgb(tmp_path):
    cmyk_icc = MAC_CMYK_PROFILE.read_bytes()
    Image.new("CMYK", (20, 10), (0, 255, 255, 0)).save(tmp_path / "print.jpg", quality=95, icc_profile=cmyk_icc)

    loaded = load_image(tmp_path / "print.jpg")

    r, g, b = loaded.rgb[5, 10].astype(int)
    assert r > 180 and g < 100 and b < 100  # still red, through the profile
    assert loaded.icc is None  # a CMYK profile must never be attached to RGB pixels


def test_load_grayscale(tmp_path):
    values = np.linspace(0, 255, 16 * 12).reshape(12, 16).astype(np.uint8)
    Image.fromarray(values).save(tmp_path / "grey.png")

    loaded = load_image(tmp_path / "grey.png")

    assert loaded.rgb.dtype == np.uint8 and loaded.rgb.shape == (12, 16, 3)
    for channel in range(3):
        np.testing.assert_array_equal(loaded.rgb[..., channel], values)


def test_load_keeps_rgb_icc(tmp_path):
    icc = srgb_profile()
    Image.fromarray(gradient(20, 10)).save(tmp_path / "tagged.png", icc_profile=icc)

    assert load_image(tmp_path / "tagged.png").icc == icc


def test_load_opens_16k_but_refuses_bigger(tmp_path, monkeypatch):
    # Pillow on its own refuses anything over 179 MP, which would include the app's own 16K results.
    assert imageio_utils.MAX_OPEN_PIXELS >= 16384 * 12312
    Image.fromarray(gradient(40, 30)).save(tmp_path / "poster.png")
    monkeypatch.setattr(imageio_utils, "MAX_OPEN_PIXELS", 40 * 30)
    assert load_image(tmp_path / "poster.png").rgb.shape == (30, 40, 3)

    monkeypatch.setattr(imageio_utils, "MAX_OPEN_PIXELS", 40 * 30 - 1)
    with pytest.raises(ImageFileError, match="too big"):
        load_image(tmp_path / "poster.png")


def test_load_not_an_image_is_friendly(tmp_path):
    (tmp_path / "notes.png").write_text("not a picture")

    with pytest.raises(ImageFileError, match="couldn't be opened"):
        load_image(tmp_path / "notes.png")


# --- save_image -----------------------------------------------------------------


def test_save_jpeg_keeps_icc(tmp_path):
    icc = srgb_profile()

    saved = save_image(gradient(64, 48), None, tmp_path / "out.jpg", "jpeg", icc)

    assert saved == tmp_path / "out.jpg"
    with Image.open(saved) as reopened:
        assert reopened.format == "JPEG"
        assert reopened.info.get("icc_profile") == icc
        assert JpegImagePlugin.get_sampling(reopened) == 0  # 4:4:4, no colour subsampling
    assert [p.name for p in tmp_path.iterdir()] == ["out.jpg"]  # no temporary file left behind


def test_save_png_keeps_icc(tmp_path):
    icc = srgb_profile()

    saved = save_image(gradient(64, 48), None, tmp_path / "out.png", "png", icc)

    with Image.open(saved) as reopened:
        assert reopened.format == "PNG" and reopened.info.get("icc_profile") == icc
        np.testing.assert_array_equal(np.asarray(reopened), gradient(64, 48))


def test_forced_png_never_overwrites_an_existing_file(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"someone else's picture")
    alpha = np.zeros((10, 10), np.uint8)

    saved = save_image(np.zeros((10, 10, 3), np.uint8), alpha, tmp_path / "logo.jpg", "jpeg", None)

    assert saved == tmp_path / "logo-2.png"
    assert (tmp_path / "logo.png").read_bytes() == b"someone else's picture"


def test_save_failure_leaves_nothing_behind(tmp_path, monkeypatch):
    def full_disk(self, fp, format=None, **params):
        fp.write(b"half a picture")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Image.Image, "save", full_disk)

    with pytest.raises(ImageFileError, match="disk is full"):
        save_image(gradient(16, 16), None, tmp_path / "out.jpg", "jpeg", None)
    assert list(tmp_path.iterdir()) == []


def libtiff_write_error(self, fp, format=None, **params):
    fp.write(b"most of a 16K TIFF")
    raise OSError("encoder error -2 when writing image file")  # what libtiff reports, with no errno


def test_tiff_save_on_a_full_disk_says_so(tmp_path, monkeypatch):
    usage = type("Usage", (), {"total": 10**12, "used": 10**12, "free": 4096})
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage)
    monkeypatch.setattr(Image.Image, "save", libtiff_write_error)

    with pytest.raises(ImageFileError, match="disk is full"):
        save_image(gradient(16, 16), None, tmp_path / "poster.tif", "tiff", None)
    assert list(tmp_path.iterdir()) == []


def test_tiff_save_failure_with_free_space_is_not_called_a_full_disk(tmp_path, monkeypatch):
    usage = type("Usage", (), {"total": 10**12, "used": 0, "free": 10**12})
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage)
    monkeypatch.setattr(Image.Image, "save", libtiff_write_error)

    with pytest.raises(ImageFileError, match="Couldn't save") as caught:
        save_image(gradient(16, 16), None, tmp_path / "poster.tif", "tiff", None)
    assert "disk is full" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_save_failure_message_survives_an_unreadable_disk(tmp_path, monkeypatch):
    def unplugged(path):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(shutil, "disk_usage", unplugged)
    monkeypatch.setattr(Image.Image, "save", libtiff_write_error)

    with pytest.raises(ImageFileError, match="Couldn't save"):
        save_image(gradient(16, 16), None, tmp_path / "poster.tif", "tiff", None)


def test_save_tiff_dpi(tmp_path):
    icc = srgb_profile()

    saved = save_image(gradient(64, 48), None, tmp_path / "poster.tif", "tiff", icc)

    assert saved == tmp_path / "poster.tif"
    with Image.open(saved) as reopened:
        assert reopened.format == "TIFF" and reopened.mode == "RGB"
        assert reopened.info["dpi"] == (300, 300)
        assert reopened.info["compression"] == "tiff_adobe_deflate"  # code 8: Deflate (ZIP), as Photoshop writes
        assert reopened.tag_v2[317] == 2  # with horizontal prediction, for smaller files
        assert reopened.info.get("icc_profile") == icc
        np.testing.assert_array_equal(np.asarray(reopened), gradient(64, 48))
    assert [p.name for p in tmp_path.iterdir()] == ["poster.tif"]


def test_save_tiff_keeps_transparency(tmp_path):
    alpha = np.full((48, 64), 255, np.uint8)
    alpha[:, :20] = 0

    saved = save_image(gradient(64, 48), alpha, tmp_path / "logo.tif", "tiff", None, dpi=600)

    assert saved.suffix == ".tif"  # TIFF holds transparency, so no switch to PNG
    with Image.open(saved) as reopened:
        assert reopened.mode == "RGBA" and reopened.info["dpi"] == (600, 600)
        np.testing.assert_array_equal(np.asarray(reopened)[..., 3], alpha)


def test_save_jpeg_png_dpi(tmp_path):
    jpeg = save_image(gradient(32, 24), None, tmp_path / "a.jpg", "jpeg", None)
    png = save_image(gradient(32, 24), None, tmp_path / "b.png", "png", None, dpi=150)

    with Image.open(jpeg) as image:
        assert image.info["dpi"] == (300, 300)
    with Image.open(png) as image:
        assert tuple(round(d) for d in image.info["dpi"]) == (150, 150)  # PNG stores dots per metre


def test_save_dpi_on_forced_png(tmp_path):
    alpha = np.zeros((10, 10), np.uint8)

    saved = save_image(np.zeros((10, 10, 3), np.uint8), alpha, tmp_path / "logo.jpg", "jpeg", None, dpi=72)

    with Image.open(saved) as image:
        assert image.format == "PNG" and tuple(round(d) for d in image.info["dpi"]) == (72, 72)


@pytest.mark.parametrize("dpi", [0, -1, 70000, 300.5, True])
def test_save_rejects_bad_dpi(tmp_path, dpi):
    with pytest.raises(ValueError):
        save_image(gradient(8, 8), None, tmp_path / "x.jpg", "jpeg", None, dpi=dpi)
    assert list(tmp_path.iterdir()) == []


def test_save_png_over_100_megapixels_compresses_faster(tmp_path, monkeypatch):
    seen = []
    real_save = Image.Image.save

    def spy(self, fp, format=None, **params):
        seen.append(params.get("compress_level"))
        return real_save(self, fp, format, **params)

    monkeypatch.setattr(Image.Image, "save", spy)
    save_image(gradient(16, 16), None, tmp_path / "small.png", "png", None)
    monkeypatch.setattr(imageio_utils, "FAST_PNG_PIXELS", 100)
    save_image(gradient(16, 16), None, tmp_path / "huge.png", "png", None)

    assert seen == [None, 3]  # Pillow's default level for normal sizes


@pytest.mark.parametrize("error", [MemoryError(), KeyboardInterrupt()])
def test_interrupted_tiff_save_leaves_nothing_behind(tmp_path, monkeypatch, error):
    def interrupted(self, fp, format=None, **params):
        fp.write(b"the first 300 MB of a 16K picture")
        raise error

    monkeypatch.setattr(Image.Image, "save", interrupted)

    with pytest.raises(type(error)):
        save_image(gradient(16, 16), None, tmp_path / "poster.tif", "tiff", None)
    assert list(tmp_path.iterdir()) == []


def test_ensure_free_space(tmp_path, monkeypatch):
    usage = type("Usage", (), {"total": 10**12, "used": 0, "free": 50 * 10**6})
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage)

    ensure_free_space(tmp_path / "Upscaled", 40 * 10**6)  # fits; the folder needn't exist yet
    with pytest.raises(ImageFileError, match="free space"):
        ensure_free_space(tmp_path / "Upscaled", 60 * 10**6)
    with pytest.raises(ImageFileError, match="always leaves 2 GB free"):  # fits, but not with the reserve
        ensure_free_space(tmp_path / "Upscaled", 40 * 10**6, keep_reserve=True)
    assert not (tmp_path / "Upscaled").exists()


# --- unique_path ----------------------------------------------------------------


def test_unique_path_no_overwrite(tmp_path):
    names = []
    for _ in range(3):
        path = unique_path(tmp_path, "x", ".jpg")
        path.touch()
        names.append(path.name)

    assert names == ["x.jpg", "x-2.jpg", "x-3.jpg"]


def test_unique_path_unicode_spaces(tmp_path):
    stem = "Mum's photo 🌅 été (1)"

    first = unique_path(tmp_path / "Upscaled", stem, ".jpg")
    first.write_bytes(b"1")
    second = unique_path(tmp_path / "Upscaled", stem, ".jpg")
    second.write_bytes(b"2")

    assert first.name == f"{stem}.jpg" and second.name == f"{stem}-2.jpg"
    assert first.read_bytes() == b"1"


def test_unique_path_shortens_very_long_names(tmp_path):
    path = unique_path(tmp_path, "é" * 200, ".jpg")  # 400 bytes of UTF-8

    assert len(path.name.encode()) <= 255 and path.name.endswith(".jpg")
    path.touch()  # the file system accepts it


# --- previews -------------------------------------------------------------------


def decode(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.format == "JPEG"
        return np.asarray(image.convert("RGB"))


def test_preview_jpeg_fits_max_side():
    assert decode(preview_jpeg(gradient(4000, 1000))).shape == (400, 1600, 3)
    assert decode(preview_jpeg(gradient(300, 200))).shape == (200, 300, 3)
    assert decode(preview_jpeg(gradient(1000, 3000), max_side=600)).shape == (600, 200, 3)


def test_compare_crops_same_spot_in_most_detailed_region():
    src = np.full((300, 300, 3), 128, np.uint8)
    stripes = np.zeros((100, 100), np.uint8)
    stripes[:, ::4] = 255
    stripes[:, 1::4] = 255
    src[200:, 200:] = stripes[..., None]  # detail only in the bottom-right ninth
    out = np.repeat(np.repeat(src, 4, axis=0), 4, axis=1)

    before, after = compare_crops(src, out, size=200)

    after_px, before_px = decode(after), decode(before)
    assert after_px.shape == before_px.shape == (200, 200, 3)
    expected = out[900:1100, 900:1100].astype(int)  # centred on the bottom-right ninth
    assert np.abs(after_px.astype(int) - expected).mean() < 12
    assert abs(before_px.mean() - after_px.mean()) < 6  # the same spot, before and after


def test_compare_crops_small_output_uses_whole_image():
    src = gradient(50, 40)
    out = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)  # 100x80, smaller than the crop

    before, after = compare_crops(src, out, size=640)

    assert decode(before).shape == decode(after).shape == (80, 100, 3)
