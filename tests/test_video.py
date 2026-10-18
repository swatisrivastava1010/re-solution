import errno
import json
import os
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from upscaler.config import SPACE_RESERVE_BYTES, Cancelled
from upscaler.jobs import friendly_error
from upscaler.models import LoadedModel
from upscaler.video import NotEnoughDiskSpace, ffmpeg, pipeline, probe
from upscaler.video.ffmpeg import concat_and_mux, decode_cmd, encode_cmd, pick_encoder
from upscaler.video.pipeline import JoinFailed, VideoOptions, upscale_video
from upscaler.video.probe import VideoError, VideoInfo, output_size_4k, probe_video

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
RED_STRIPE_ON_LEFT = "color=c=blue:s=320x180:r=25:d=1,drawbox=x=0:y=0:w=40:h=180:color=red:t=fill"
H264 = ("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p")
DVD_MPEG2 = ("-c:v", "mpeg2video", "-q:v", "4", "-bf", "2", "-g", "15", "-f", "mpeg")  # as on a DVD
# A later file of a long camcorder or TV recording: its timestamps start well after zero.
STARTS_30S_IN = ("-output_ts_offset", "30")

# Each frame shows its own number: the left half is grey level n % 16, the right half n // 16.
_NUMBER_STEP = 14  # grey levels (16-235 range) per digit
NUMBERED_FRAMES = (
    "color=c=gray:s=64x36:r=25:d={seconds},format=yuv420p,"
    r"geq=lum='if(lt(X\,32)\,mod(N\,16)\,trunc(N/16))*14+23':cb=128:cr=128"
)


# --- Helpers ----------------------------------------------------------------


def run_ffmpeg(*args: str) -> None:
    subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def make_clip(
    path: Path,
    *,
    size: tuple[int, int] = (320, 180),
    seconds: float = 2.0,
    fps: int | str = 25,
    audio: bool = True,
    source: str | None = None,
    vf: str | None = None,
    video_args: tuple[str, ...] = H264,
    rotation: float | None = None,
    audio_codec: str = "aac",
) -> Path:
    """A test clip from ffmpeg's testsrc2 pattern, with a 440 Hz tone when `audio` is set.

    `rotation` adds a display matrix (counter-clockwise degrees, like phones write it)."""
    w, h = size
    inputs = ["-f", "lavfi", "-i", source or f"testsrc2=s={w}x{h}:r={fps}:d={seconds}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    filters = ["-vf", vf] if vf else []
    audio_args = ["-c:a", audio_codec] if audio else []
    target = path if rotation is None else path.with_name("unrotated-" + path.name)
    run_ffmpeg(*inputs, *filters, *video_args, *audio_args, str(target))
    if rotation is not None:
        run_ffmpeg("-display_rotation:v:0", str(rotation), "-i", str(target), "-c", "copy", str(path))
    return path


def inspect_output(path: Path) -> dict:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        cc = video.codec_context
        return {
            "codec": cc.name,
            "tag": cc.codec_tag,
            "size": (cc.width, cc.height),
            "pix_fmt": cc.pix_fmt,
            "colour": (cc.color_primaries, cc.color_trc, cc.colorspace),
            "frames": video.frames,
            "duration": container.duration / 1_000_000,
            "audio_streams": len(container.streams.audio),
            "audio_codecs": [a.codec_context.name for a in container.streams.audio],
        }


def first_frame_rgb(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_ndarray(format="rgb24")


def moov_before_mdat(path: Path) -> bool:
    data = path.read_bytes()
    return data.find(b"moov") < data.find(b"mdat")


def numbered_clip(path: Path, seconds: float = 3, **kwargs) -> Path:
    """A 25 fps clip whose frames show their own numbers (see frame_number)."""
    return make_clip(path, seconds=seconds, source=NUMBERED_FRAMES.format(seconds=seconds), **kwargs)


def frame_number(rgb: np.ndarray) -> int:
    """The number a NUMBERED_FRAMES frame shows, at any size."""
    h, w = rgb.shape[:2]
    rows = slice(h // 4, 3 * h // 4)
    low, high = (rgb[rows, w * eighth // 8 : w * (eighth + 2) // 8].mean() for eighth in (1, 5))
    step = _NUMBER_STEP * 255 / 219  # one digit, in full-range RGB
    return int(low // step) + 16 * int(high // step)


def frame_numbers(path: Path) -> list[int]:
    with av.open(str(path)) as container:
        return [frame_number(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]


def watch_frames(monkeypatch) -> list[int]:
    """The numbers of the frames the model is given, in order."""
    seen: list[int] = []
    enlarge = pipeline._FrameUpscaler.enlarge

    def watching(self, frame, cancel):
        seen.append(frame_number(frame))
        return enlarge(self, frame, cancel)

    monkeypatch.setattr(pipeline._FrameUpscaler, "enlarge", watching)
    return seen


def cancel_after_segments(job: Path, cancel: threading.Event, count: int = 1):
    """A progress callback that cancels the job once `count` segments are finished."""
    manifest = job / "manifest.json"

    def progress(fraction, message):
        if manifest.exists() and len(json.loads(manifest.read_text())["done"]) >= count:
            cancel.set()

    return progress


def interrupt_after_segments(src: Path, tmp_path: Path, count: int = 1, opts: VideoOptions | None = None) -> None:
    cancel = threading.Event()
    with pytest.raises(Cancelled):
        run_upscale(src, tmp_path, opts, cancel=cancel, progress=cancel_after_segments(tmp_path / "job", cancel, count))


class NearestNet(torch.nn.Module):
    """Stands in for a real model: 4x nearest-neighbour enlargement, on the CPU, in microseconds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=4, mode="nearest")


@pytest.fixture
def fake_model(monkeypatch):
    model = LoadedModel(net=NearestNet(), scale=4, dtype=torch.float32, tile=512, device=torch.device("cpu"))
    loaded = []
    monkeypatch.setattr(pipeline, "ensure_model", lambda spec, progress=None, cancel=None: Path("unused"))
    monkeypatch.setattr(pipeline, "load_model", lambda spec, device=None: loaded.append(spec.id) or model)
    return loaded


@pytest.fixture
def small_output(monkeypatch):
    """Upscale to a 480 px long edge instead of 3840, so CPU-only tests stay quick."""
    monkeypatch.setattr(pipeline, "output_size_4k", lambda info: probe.output_size_4k(info, long_edge=480))


@pytest.fixture
def fresh_encoder_cache():
    ffmpeg._working_encoder.cache_clear()
    yield
    ffmpeg._working_encoder.cache_clear()


def run_upscale(src: Path, tmp_path: Path, opts: VideoOptions | None = None, cancel=None, progress=None):
    return upscale_video(
        src,
        tmp_path / "out",
        tmp_path / "job",
        opts or VideoOptions(),
        progress=progress or (lambda fraction, message: None),
        cancel=cancel or threading.Event(),
    )


def info(**overrides) -> VideoInfo:
    values = dict(
        width=320, height=180, display_width=320, display_height=180, fps=Fraction(25), frames=50,
        duration=2.0, rotation=0, interlaced=False, hdr=False, has_audio=True, codec="h264",
    )
    return VideoInfo(**(values | overrides))


# --- Probe ------------------------------------------------------------------


def test_probe_basic(tmp_path):
    src = make_clip(tmp_path / "basic.mp4")
    got = probe_video(src)
    assert (got.width, got.height) == (320, 180)
    assert (got.display_width, got.display_height) == (320, 180)
    assert got.fps == Fraction(25)
    assert got.frames == 50
    assert got.duration == pytest.approx(2.0, abs=0.05)
    assert got.rotation == 0
    assert not got.interlaced
    assert not got.hdr
    assert got.has_audio
    assert got.codec == "h264"


def test_probe_rotated_portrait(tmp_path):
    # Phones store portrait video sideways and add "turn it 90 degrees clockwise" (-90 counter-clockwise).
    src = make_clip(tmp_path / "phone.mp4", rotation=-90)
    got = probe_video(src)
    assert (got.width, got.height) == (320, 180)
    assert (got.display_width, got.display_height) == (180, 320)
    assert got.rotation == 90


def test_probe_rotated_the_other_way(tmp_path):
    got = probe_video(make_clip(tmp_path / "phone.mp4", rotation=90, audio=False))
    assert got.rotation == 270
    assert (got.display_width, got.display_height) == (180, 320)


def test_probe_no_audio(tmp_path):
    assert not probe_video(make_clip(tmp_path / "silent.mp4", audio=False)).has_audio


def test_probe_non_square_pixels(tmp_path):
    src = make_clip(tmp_path / "anamorphic.mp4", size=(160, 120), vf="setsar=2", audio=False)
    got = probe_video(src)
    assert (got.width, got.height) == (160, 120)
    assert (got.display_width, got.display_height) == (320, 120)


def test_probe_variable_frame_rate_uses_the_nominal_rate(tmp_path):
    # A 30 fps clip with five dropped frames, timestamps kept: variable frame rate, like phones record.
    src = make_clip(
        tmp_path / "vfr.mp4",
        fps=30,
        audio=False,
        vf=r"select='not(between(n\,10\,14))'",
        video_args=("-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
    )
    got = probe_video(src)
    assert got.fps == Fraction(30)
    assert got.frames == 60


def test_probe_interlaced(tmp_path):
    src = make_clip(
        tmp_path / "camcorder.mp4",
        size=(320, 240),
        audio=False,
        vf="tinterlace=interleave_top,setfield=tff",
        video_args=("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-flags", "+ildct+ilme", "-x264-params", "tff=1"),
    )
    assert probe_video(src).interlaced


def hdr_clip(path: Path) -> Path:
    to_pq = (
        "zscale=tin=bt709:pin=bt709:min=bt709:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=tv:npl=100,"
        "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv"
    )
    video_args = ("-c:v", "libx265", "-x265-params", "log-level=error", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1")
    return make_clip(path, seconds=0.5, audio=False, vf=to_pq, video_args=video_args)


def test_probe_hdr(tmp_path):
    assert probe_video(hdr_clip(tmp_path / "hdr.mp4")).hdr


def test_probe_rejects_non_video(tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("not a video")
    with pytest.raises(VideoError, match="video"):
        probe_video(bogus)


# --- Output size --------------------------------------------------------------


def test_output_size_landscape():
    assert output_size_4k(info()) == (3840, 2160)


def test_output_size_portrait_uses_display_dims():
    assert output_size_4k(info(width=1920, height=1080, display_width=1080, display_height=1920, rotation=90)) == (
        2160,
        3840,
    )


def test_output_size_odd_dims():
    w, h = output_size_4k(info(width=1080, height=1081, display_width=1080, display_height=1081))
    assert (w, h) == (3836, 3840)
    assert w % 2 == 0 and h % 2 == 0


def test_output_size_never_below_source():
    assert output_size_4k(info(width=4097, height=2161, display_width=4097, display_height=2161)) == (4098, 2162)


# --- ffmpeg commands ----------------------------------------------------------


def test_decode_cmd_is_constant_frame_rate_rgb_with_autorotate():
    cmd = decode_cmd(Path("in.mp4"), info(fps=Fraction(30000, 1001)), start_frame=300, frames=300)
    joined = " ".join(cmd)
    filters = cmd[cmd.index("-vf") + 1]
    assert "-noautorotate" not in cmd
    # Frames are numbered by the fps filter from the start of the file, never found with a seek: after
    # a seek, MPEG-TS and DVD files keep shifted timestamps and frames get repeated or skipped.
    assert "-ss" not in cmd
    assert filters.startswith("fps=fps=30000/1001:start_time=0,trim=start_frame=300,setpts=PTS-STARTPTS,scale=")
    assert "-fps_mode passthrough" in joined
    assert "-frames:v 300" in joined
    assert "-pix_fmt rgb24" in joined and cmd[-1] == "-"
    assert "bwdif" not in joined and "zscale" not in joined


def test_decode_cmd_reads_to_the_end_without_a_frame_count():
    cmd = decode_cmd(Path("in.mp4"), info(), start_frame=0, frames=None)
    assert "-frames:v" not in cmd and "-ss" not in cmd
    assert "trim" not in cmd[cmd.index("-vf") + 1]


def test_decode_cmd_deinterlaces_and_squares_pixels():
    dv = info(width=720, height=480, display_width=640, display_height=480, interlaced=True)
    cmd = decode_cmd(Path("in.mp4"), dv, 0, 10)
    filters = cmd[cmd.index("-vf") + 1]
    assert filters.startswith("bwdif")
    assert "scale=640:480" in filters
    assert "setsar=1" in filters


def test_decode_cmd_tone_maps_hdr():
    filters = " ".join(decode_cmd(Path("in.mp4"), info(hdr=True), 0, 10))
    assert "zscale=t=linear:npl=100" in filters and "tonemap=tonemap=hable" in filters
    assert "zscale=t=bt709:m=bt709:r=tv" in filters


def test_hdr_refused_without_zscale(monkeypatch):
    monkeypatch.setattr(ffmpeg, "has_zscale", lambda: False)
    with pytest.raises(VideoError, match="HDR"):
        decode_cmd(Path("in.mp4"), info(hdr=True), 0, 10)


def test_encode_cmd_tags_bt709_hevc_10bit(fresh_encoder_cache):
    cmd = encode_cmd(Path("seg.mp4"), 3840, 2160, Fraction(25))
    joined = " ".join(cmd)
    assert "-f rawvideo -pix_fmt rgb24 -s 3840x2160 -r 25/1 -i -" in joined
    assert "-tag:v hvc1" in joined
    assert "scale=out_color_matrix=bt709:out_range=tv" in joined
    for flag in ("-colorspace", "-color_primaries", "-color_trc"):
        assert cmd[cmd.index(flag) + 1] == "bt709"
    assert cmd[cmd.index("-color_range") + 1] == "tv"


@pytest.mark.skipif(sys.platform != "darwin", reason="VideoToolbox is macOS only")
def test_pick_encoder_uses_videotoolbox_on_mac(fresh_encoder_cache):
    args = pick_encoder()
    assert args[args.index("-c:v") + 1] == "hevc_videotoolbox"
    assert "main10" in args and "p010le" in args


def test_pick_encoder_falls_back_to_x265(monkeypatch, fresh_encoder_cache):
    def only_x265(args):
        return "libx265" in args

    monkeypatch.setattr(ffmpeg, "_encoder_works", only_x265)
    args = pick_encoder()
    assert args[args.index("-c:v") + 1] == "libx265"
    assert "yuv420p10le" in args and "hvc1" in args


def test_no_working_encoder_is_explained_kindly(monkeypatch, fresh_encoder_cache):
    monkeypatch.setattr(ffmpeg, "_encoder_works", lambda args: False)
    with pytest.raises(VideoError, match="couldn't create video files") as caught:
        pick_encoder()
    assert friendly_error(caught.value) == str(caught.value)  # shown as written, not "Something went wrong"
    assert "HEVC" not in str(caught.value) and "ffmpeg" not in str(caught.value)


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are a POSIX idea")
def test_ffmpeg_runs_in_its_own_process_group():
    # Ctrl+C in the Terminal goes to the whole foreground process group. ffmpeg must not get it, or it quits
    # and a job that should pause (and carry on later) fails instead; the app stops ffmpeg itself.
    child = pipeline._spawn(
        [sys.executable, "-c", "import os; print(os.getpgrp())"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
    )
    group = int(child.communicate()[0])
    assert group != os.getpgrp()


def test_ffmpeg_on_windows_gets_no_window_and_its_own_process_group(monkeypatch):
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)
    monkeypatch.setattr(ffmpeg.sys, "platform", "win32")
    assert ffmpeg.process_options() == {"creationflags": 0x08000000 | 0x00000200}


# The FFmpeg that imageio-ffmpeg bundles for Linux crashes (signal 11) when it starts decoding an MPEG-TS
# file part way through. Macs and Windows, the platforms Re:Solution supports, decode these files fine.
LINUX_TS_CRASH = pytest.mark.skipif(
    sys.platform.startswith("linux"), reason="the bundled Linux FFmpeg crashes seeking inside MPEG-TS files"
)


def decoded_frames(cmd: list[str], w: int, h: int) -> np.ndarray:
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


@pytest.mark.parametrize(
    "name, fps, video_args, audio_codec",
    [
        # 29.97 fps in a QuickTime file with a 600 timescale (as iPhones write): frame times aren't whole ticks.
        ("ntsc.mov", "30000/1001", (*H264, "-video_track_timescale", "600"), "aac"),
        pytest.param("camcorder.mts", 25, H264, "aac", marks=LINUX_TS_CRASH),  # MPEG-TS, timestamps start at 1.4 s
        pytest.param("spanned.mts", 25, (*H264, *STARTS_30S_IN), "aac", marks=LINUX_TS_CRASH),
        ("dvd.mpg", 25, DVD_MPEG2, "mp2"),  # MPEG program stream
        ("phone.mp4", 30, (*H264, "-fps_mode", "vfr"), "aac"),  # variable frame rate, see below
    ],
)
def test_decoding_from_a_later_frame_matches_decoding_from_the_start(tmp_path, name, fps, video_args, audio_codec):
    # A resumed job starts decoding at its first unfinished frame. That frame, and every one after it, must be
    # exactly the frame a run from the start would have had there, or the video stutters at the join.
    vf = r"select='not(between(n\,10\,14))'" if name == "phone.mp4" else None  # five dropped frames
    src = make_clip(tmp_path / name, size=(96, 64), seconds=3, fps=fps, vf=vf, video_args=video_args,
                    audio_codec=audio_codec)
    got = probe_video(src)
    w, h = got.display_width, got.display_height
    whole = decoded_frames(decode_cmd(src, got, 0, None), w, h)
    assert len(whole) == pytest.approx(got.frames, abs=1)  # the probe's count is an estimate
    for start in (12, got.frames // 3, got.frames - 5):
        assert np.array_equal(decoded_frames(decode_cmd(src, got, start, None), w, h), whole[start:])
        assert np.array_equal(decoded_frames(decode_cmd(src, got, start, 4), w, h), whole[start : start + 4])


def test_encode_then_decode_keeps_colours(tmp_path, fresh_encoder_cache):
    # Flat patches of saturated colour: a BT.601/BT.709 or range mix-up shifts these by 10+ levels.
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (40, 40, 40), (230, 230, 230), (200, 120, 60)]
    frame = np.concatenate([np.full((64, 64, 3), c, np.uint8) for c in colours], axis=1)
    h, w = frame.shape[:2]
    segment = tmp_path / "seg.mp4"
    encoder = subprocess.Popen(encode_cmd(segment, w, h, Fraction(25)), stdin=subprocess.PIPE)
    encoder.communicate(frame.tobytes() * 5)
    assert encoder.returncode == 0

    decoded = subprocess.run(decode_cmd(segment, probe_video(segment), 0, 1), capture_output=True, check=True).stdout
    back = np.frombuffer(decoded, np.uint8).reshape(h, w, 3)
    for i, colour in enumerate(colours):
        centre = back[16:48, i * 64 + 16 : i * 64 + 48].reshape(-1, 3).mean(axis=0)
        assert np.abs(centre - colour).max() < 4, (colour, centre)


def test_concat_and_mux_joins_segments_and_adds_audio(tmp_path):
    src = make_clip(tmp_path / "src.mp4", seconds=2)
    segments = []
    for i, start in enumerate((0, 1)):
        segment = tmp_path / f"seg{i}.mp4"
        run_ffmpeg("-ss", str(start), "-t", "1", "-i", str(src), "-an", "-c:v", "libx265", "-x265-params",
                   "log-level=error", "-tag:v", "hvc1", str(segment))
        segments.append(segment)
    dst = tmp_path / "joined.mp4"
    concat_and_mux(segments, src, dst, has_audio=True)
    out = inspect_output(dst)
    assert out["tag"] == "hvc1" and out["audio_streams"] == 1
    assert out["duration"] == pytest.approx(2.0, abs=0.1)
    assert moov_before_mdat(dst)


def test_concat_and_mux_converts_audio_quicktime_cannot_play(tmp_path):
    src = tmp_path / "pcm.mov"
    run_ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x64:r=25:d=1", "-f", "lavfi", "-i", "sine=duration=1",
               "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "pcm_s16le", str(src))
    segment = tmp_path / "seg.mp4"
    run_ffmpeg("-i", str(src), "-an", "-c:v", "libx265", "-x265-params", "log-level=error", "-tag:v", "hvc1",
               str(segment))
    dst = tmp_path / "out.mp4"
    concat_and_mux([segment], src, dst, has_audio=True)
    assert inspect_output(dst)["audio_codecs"] == ["aac"]


def test_join_failure_is_explained_kindly(tmp_path):
    src = make_clip(tmp_path / "src.mp4", seconds=0.5)
    damaged = tmp_path / "segment.mp4"
    damaged.write_bytes(bytes(1000))
    with pytest.raises(VideoError, match="put the finished video together") as caught:
        concat_and_mux([damaged], src, tmp_path / "out.mp4", has_audio=True)
    assert friendly_error(caught.value) == str(caught.value)


# --- Whole pipeline (CPU stand-in model, small output) --------------------------


def test_upscale_video_no_audio(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "silent.mp4", audio=False)
    result = run_upscale(src, tmp_path)
    out = inspect_output(result.output_path)
    assert result.output_path.name == "silent-4K-fast.mp4"
    assert out["audio_streams"] == 0
    assert out["size"] == (480, 270) == (result.width, result.height)
    assert out["duration"] == pytest.approx(2.0, abs=0.1)
    assert fake_model == ["fast"]


def test_upscale_video_short(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "short.mp4", seconds=0.5)
    result = run_upscale(src, tmp_path)
    out = inspect_output(result.output_path)
    assert out["frames"] == probe_video(src).frames
    assert out["audio_streams"] == 1
    assert out["duration"] == pytest.approx(probe_video(src).duration, abs=0.1)


def test_upscale_video_output_plays_in_quicktime(tmp_path, fake_model, small_output):
    result = run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    out = inspect_output(result.output_path)
    assert out["codec"] == "hevc" and out["tag"] == "hvc1"
    assert out["pix_fmt"] == "yuv420p10le"
    assert out["colour"] == (1, 1, 1)  # BT.709 primaries, transfer and matrix
    assert moov_before_mdat(result.output_path)


def test_rotated_phone_video_comes_out_upright(tmp_path, fake_model, small_output):
    # Stored sideways with a red stripe on the left; turned 90 degrees clockwise the stripe is on top.
    src = make_clip(tmp_path / "phone.mp4", source=RED_STRIPE_ON_LEFT, audio=False, rotation=-90)
    result = run_upscale(src, tmp_path)
    assert (result.width, result.height) == (270, 480)
    assert probe_video(result.output_path).rotation == 0
    frame = first_frame_rgb(result.output_path).astype(int)
    top, bottom = frame[:20].reshape(-1, 3).mean(axis=0), frame[-100:].reshape(-1, 3).mean(axis=0)
    assert top[0] > 180 and top[2] < 80
    assert bottom[2] > 180 and bottom[0] < 80


def test_odd_dims_give_even_4k_output(tmp_path, fake_model):
    src = make_clip(
        tmp_path / "odd.mp4", size=(99, 101), seconds=0.2, audio=False,
        video_args=("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p"),
    )
    result = run_upscale(src, tmp_path)
    assert (result.width, result.height) == (3764, 3840)
    assert inspect_output(result.output_path)["size"] == (3764, 3840)


def test_variable_frame_rate_video_keeps_its_length(tmp_path, fake_model, small_output):
    src = make_clip(
        tmp_path / "vfr.mp4",
        fps=30,
        vf=r"select='not(between(n\,10\,14))'",
        video_args=("-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"),
    )
    out = inspect_output(run_upscale(src, tmp_path).output_path)
    assert out["frames"] == 60
    assert out["duration"] == pytest.approx(2.0, abs=0.1)


def test_hdr_video_is_converted_with_a_note(tmp_path, fake_model, small_output):
    result = run_upscale(hdr_clip(tmp_path / "hdr.mp4"), tmp_path)
    assert any("HDR" in note for note in result.notes)
    assert inspect_output(result.output_path)["colour"] == (1, 1, 1)
    assert first_frame_rgb(result.output_path).mean() > 30  # tone-mapped, not black


def test_interlaced_video_is_deinterlaced_with_a_note(tmp_path, fake_model, small_output):
    src = make_clip(
        tmp_path / "camcorder.mp4",
        size=(320, 240),
        vf="tinterlace=interleave_top,setfield=tff",
        video_args=("-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-flags", "+ildct+ilme", "-x264-params", "tff=1"),
    )
    result = run_upscale(src, tmp_path)
    assert any("interlaced" in note for note in result.notes)
    assert inspect_output(result.output_path)["frames"] == probe_video(src).frames


def test_preview_seconds(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "long.mp4", seconds=3)
    result = run_upscale(src, tmp_path, VideoOptions(preview_seconds=1))
    out = inspect_output(result.output_path)
    assert result.output_path.name == "long-4K-fast-preview1s.mp4"
    assert out["frames"] == 25
    assert out["duration"] == pytest.approx(1.0, abs=0.1)
    assert any("preview" in note.lower() for note in result.notes)


def test_resume_skips_done_segments(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "twelve.mp4", size=(64, 36), seconds=12)
    manifest = tmp_path / "job" / "manifest.json"
    cancel = threading.Event()

    def cancel_after_first_segment(fraction, message):
        if manifest.exists() and json.loads(manifest.read_text())["done"]:
            cancel.set()

    with pytest.raises(Cancelled):
        run_upscale(src, tmp_path, cancel=cancel, progress=cancel_after_first_segment)
    first = tmp_path / "job" / "segments" / "segment-00000.mp4"
    mtime = first.stat().st_mtime_ns
    assert not list((tmp_path / "out").glob("*.mp4"))

    messages, mtimes = [], set()

    def watch(fraction, message):
        messages.append(message)
        if first.exists():
            mtimes.add(first.stat().st_mtime_ns)

    result = run_upscale(src, tmp_path, progress=watch)
    assert mtimes == {mtime}  # segment 0 was kept as it was until the final join...
    assert not any("frame 1 of" in m for m in messages)  # ...and never redone
    out = inspect_output(result.output_path)
    assert out["frames"] == 300
    assert out["duration"] == pytest.approx(12.0, abs=0.1)


@pytest.mark.parametrize(
    "name, video_args, audio_codec",
    [
        pytest.param("camcorder.mts", H264, "aac", marks=LINUX_TS_CRASH),
        pytest.param("spanned.mts", (*H264, *STARTS_30S_IN), "aac", marks=LINUX_TS_CRASH),
        ("dvd.mpg", DVD_MPEG2, "mp2"),
    ],
)
def test_camcorder_and_dvd_video_keeps_every_frame_in_order(
    tmp_path, fake_model, small_output, monkeypatch, name, video_args, audio_codec
):
    # MPEG-TS (.mts, .m2ts) and DVD (.mpg) files cut into 1-second segments. Seeking to each segment once
    # froze the picture at every boundary, then skipped frames, so the sound drifted out of step.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    src = numbered_clip(tmp_path / name, video_args=video_args, audio_codec=audio_codec)
    result = run_upscale(src, tmp_path)
    assert frame_numbers(result.output_path) == list(range(75))
    out = inspect_output(result.output_path)
    assert out["audio_streams"] == 1
    assert out["duration"] == pytest.approx(3.0, abs=0.1)


@LINUX_TS_CRASH
def test_resumed_camcorder_video_carries_on_at_the_right_frame(tmp_path, fake_model, small_output, monkeypatch):
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    src = numbered_clip(tmp_path / "camcorder.mts", video_args=(*H264, *STARTS_30S_IN))
    interrupt_after_segments(src, tmp_path)
    seen, messages = watch_frames(monkeypatch), []
    result = run_upscale(src, tmp_path, progress=lambda fraction, message: messages.append(message))
    assert seen == list(range(25, 75))  # segment 0 kept; the rest made from the right frames
    assert frame_numbers(result.output_path) == list(range(75))
    assert "Picking up where it left off" in messages


def test_resumed_preview_stops_at_the_preview_length(tmp_path, fake_model, small_output, monkeypatch):
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    src = numbered_clip(tmp_path / "clip.mp4", seconds=4)
    interrupt_after_segments(src, tmp_path, opts=VideoOptions(preview_seconds=2.4))
    seen = watch_frames(monkeypatch)
    result = run_upscale(src, tmp_path, VideoOptions(preview_seconds=2.4))
    assert seen == list(range(25, 60))
    assert frame_numbers(result.output_path) == list(range(60))


def test_damaged_segment_is_redone_on_resume(tmp_path, fake_model, small_output, monkeypatch):
    # A power cut can leave a segment that's recorded as finished but cut short. Resuming must redo it
    # (and, since frames are read in order, everything after it) instead of failing at the very end.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    src = numbered_clip(tmp_path / "clip.mp4")
    interrupt_after_segments(src, tmp_path, count=2)
    first = tmp_path / "job" / "segments" / "segment-00000.mp4"
    data = first.read_bytes()
    first.write_bytes(data[: len(data) // 2] + bytes(len(data) - len(data) // 2))
    seen = watch_frames(monkeypatch)
    result = run_upscale(src, tmp_path)
    assert seen == list(range(75))
    assert frame_numbers(result.output_path) == list(range(75))


def test_resume_with_a_different_encoder_starts_over(tmp_path, fake_model, small_output, monkeypatch):
    # Segments from two encoders can't be joined without re-encoding: the second one's frames come out
    # garbled. E.g. NVIDIA's encoder works on the first run but not after a driver update.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    x265 = ffmpeg._ENCODERS["libx265"]
    x265_fast = tuple("ultrafast" if arg == "medium" else arg for arg in x265)
    src = numbered_clip(tmp_path / "clip.mp4")
    monkeypatch.setattr(ffmpeg, "_working_encoder", lambda: x265)
    interrupt_after_segments(src, tmp_path)
    monkeypatch.setattr(ffmpeg, "_working_encoder", lambda: x265_fast)
    seen = watch_frames(monkeypatch)
    result = run_upscale(src, tmp_path)
    assert seen == list(range(75))
    assert frame_numbers(result.output_path) == list(range(75))


def test_too_little_disk_space_to_join_is_explained_before_joining(tmp_path, fake_model, small_output, monkeypatch):
    joins = []
    monkeypatch.setattr(pipeline, "concat_and_mux", lambda *args, **kwargs: joins.append(args))
    monkeypatch.setattr(pipeline, "_check_space_for_segment", lambda *args: None)  # the segments had room
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: SPACE_RESERVE_BYTES - 500_000_000)
    with pytest.raises(NotEnoughDiskSpace, match="Free up about 1 GB") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    assert isinstance(caught.value, VideoError)
    assert friendly_error(caught.value) == str(caught.value)
    assert not joins
    assert list((tmp_path / "job" / "segments").glob("segment-*.mp4"))  # the finished work is still there


def test_too_little_disk_space_to_join_says_how_much_to_free(tmp_path, monkeypatch):
    segment = tmp_path / "segment-0.mp4"
    segment.write_bytes(bytes(1_000_000))
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: SPACE_RESERVE_BYTES)  # only the space kept free
    with pytest.raises(NotEnoughDiskSpace) as caught:
        pipeline._check_space_to_join([segment], tmp_path, tmp_path, seconds=10)
    # How to carry on (Retry in the app, the same command again on the command line) is added by the caller.
    assert str(caught.value) == "There isn't enough free disk space to finish this video. Free up about 1 GB."
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: 0)
    with pytest.raises(NotEnoughDiskSpace, match="Free up about 3 GB"):  # the 2 GB always kept free, too
        pipeline._check_space_to_join([segment], tmp_path, tmp_path, seconds=10)


def test_disk_filling_up_part_way_keeps_the_finished_segments(tmp_path, fake_model, small_output, monkeypatch):
    # The encoder fails on the second segment because the disk is full: the first segment is kept, and
    # the error says it's about disk space, so the job can carry on from it once there's room.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    real_encode_cmd = pipeline.encode_cmd
    started = []

    def encode_cmd(*args):
        started.append(args)
        return real_encode_cmd(*args) if len(started) == 1 else FAILING_COMMAND

    monkeypatch.setattr(pipeline, "encode_cmd", encode_cmd)
    monkeypatch.setattr(pipeline, "_check_space_for_segment", lambda *args: None)  # the space went mid-segment
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: 10_000_000)
    with pytest.raises(NotEnoughDiskSpace, match="Free up some space") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=2), tmp_path)
    assert friendly_error(caught.value) == str(caught.value)
    assert [p.name for p in (tmp_path / "job" / "segments").glob("segment-*.mp4")] == ["segment-00000.mp4"]
    assert not list((tmp_path / "out").glob("*"))


def test_a_segment_isnt_started_when_it_would_eat_into_the_space_kept_free(
    tmp_path, fake_model, small_output, monkeypatch
):
    # The space was checked when the video was added, but other jobs and apps may have used it since:
    # each segment checks again, and stops while 2 GB are still free, keeping the segments done so far.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    free = iter([10**12, SPACE_RESERVE_BYTES])  # room for the first segment, then not for the second
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: next(free))
    with pytest.raises(NotEnoughDiskSpace, match=r"Free up about \d+ GB") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=2), tmp_path)
    assert friendly_error(caught.value) == str(caught.value)
    assert [p.name for p in (tmp_path / "job" / "segments").glob("segment-*.mp4")] == ["segment-00000.mp4"]
    assert not list((tmp_path / "out").glob("*"))


def test_the_disk_filling_up_while_joining_keeps_the_segments(tmp_path, fake_model, small_output, monkeypatch):
    # The space check before joining passed, then something else (a guest's upload, another app) filled
    # the disk while ffmpeg was joining: the finished segments stay, to join again once there's room.
    free = {"bytes": 10**12}
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: free["bytes"])

    def join_fills_the_disk(segments, src, dst, has_audio, *, duration=None):
        dst.write_bytes(b"half a video")
        free["bytes"] = 1_000_000
        raise VideoError("Couldn't put the finished video together. Please try again.")

    monkeypatch.setattr(pipeline, "concat_and_mux", join_fills_the_disk)
    with pytest.raises(NotEnoughDiskSpace, match="Free up some space"):
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    assert list((tmp_path / "job" / "segments").glob("segment-*.mp4"))
    assert (tmp_path / "job" / "manifest.json").is_file()
    assert not (tmp_path / "job" / "output.part.mp4").exists()  # the half-made video's space is freed
    assert not list((tmp_path / "out").glob("*"))


def test_a_join_that_fails_for_another_reason_keeps_the_segments_too(tmp_path, fake_model, small_output, monkeypatch):
    def join_fails(segments, src, dst, has_audio, *, duration=None):
        raise VideoError("Couldn't put the finished video together. Please try again.")

    monkeypatch.setattr(pipeline, "concat_and_mux", join_fails)
    src = make_clip(tmp_path / "clip.mp4", seconds=1)
    with pytest.raises(JoinFailed, match="Couldn't put the finished video together") as caught:
        run_upscale(src, tmp_path)
    assert isinstance(caught.value, VideoError) and friendly_error(caught.value) == str(caught.value)
    assert list((tmp_path / "job" / "segments").glob("segment-*.mp4"))

    monkeypatch.setattr(pipeline, "concat_and_mux", concat_and_mux)  # carrying on only joins them again
    upscaled = []
    monkeypatch.setattr(pipeline._FrameUpscaler, "enlarge", lambda *args: upscaled.append(1))
    result = run_upscale(src, tmp_path)
    assert upscaled == [] and inspect_output(result.output_path)["frames"] == 25


def test_a_full_disk_anywhere_along_the_way_keeps_the_segments(tmp_path, fake_model, small_output, monkeypatch):
    # E.g. saving the list of finished segments after the first one: a plain "disk full" from Python.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    mark_done = pipeline._Manifest.mark_done

    def full_after_the_first(self, index, frames):
        if index:
            raise OSError(errno.ENOSPC, "No space left on device")
        mark_done(self, index, frames)

    monkeypatch.setattr(pipeline._Manifest, "mark_done", full_after_the_first)
    with pytest.raises(NotEnoughDiskSpace, match="Free up some space") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=2), tmp_path)
    assert isinstance(caught.value.__cause__, OSError)
    assert json.loads((tmp_path / "job" / "manifest.json").read_text())["done"] == {"0": 25}


def test_the_encoder_saying_the_disk_is_full_keeps_the_segments(tmp_path, fake_model, small_output, monkeypatch):
    # Space can come back the moment ffmpeg stops (another upload gives up and is removed), so what
    # ffmpeg said counts too, not just how much is free afterwards.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    real_encode_cmd = pipeline.encode_cmd
    started = []
    says_full = [sys.executable, "-c", "import sys; sys.stderr.write('No space left on device'); sys.exit(1)"]

    def encode_cmd(*args):
        started.append(args)
        return real_encode_cmd(*args) if len(started) == 1 else says_full

    monkeypatch.setattr(pipeline, "encode_cmd", encode_cmd)
    with pytest.raises(NotEnoughDiskSpace):
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=2), tmp_path)
    assert [p.name for p in (tmp_path / "job" / "segments").glob("segment-*.mp4")] == ["segment-00000.mp4"]


def test_saving_to_another_drive_that_fails_keeps_the_segments(tmp_path, fake_model, small_output, monkeypatch):
    def move_fails(source, target):
        Path(target).write_bytes(b"half a copy")
        raise OSError(errno.EIO, "Input/output error")  # e.g. the drive was unplugged

    monkeypatch.setattr(pipeline.shutil, "move", move_fails)
    with pytest.raises(JoinFailed, match="Couldn't save the finished video in the out folder"):
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    assert not list((tmp_path / "out").glob("*"))  # no half copy left behind
    assert not (tmp_path / "job" / "output.part.mp4").exists()
    assert list((tmp_path / "job" / "segments").glob("segment-*.mp4"))


def test_model_download_can_be_cancelled(tmp_path, fake_model, small_output, monkeypatch):
    seen = {}

    def ensure_model(spec, progress=None, cancel=None):
        seen["cancel"] = cancel
        return Path("unused")

    monkeypatch.setattr(pipeline, "ensure_model", ensure_model)
    cancel = threading.Event()
    run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path, cancel=cancel)
    assert seen["cancel"] is cancel


def test_enough_disk_space_counts_the_finished_segments(tmp_path, monkeypatch):
    segments = []
    for i in range(3):
        segments.append(tmp_path / f"segment-{i}.mp4")
        segments[-1].write_bytes(bytes(1_000_000))
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: 4_000_000_000)
    pipeline._check_space_to_join(segments, tmp_path, tmp_path, seconds=60)
    monkeypatch.setattr(pipeline, "_free_bytes", lambda path: 3_000_000)
    with pytest.raises(NotEnoughDiskSpace):
        pipeline._check_space_to_join(segments, tmp_path, tmp_path, seconds=60)


def test_changed_options_start_over(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "clip.mp4", size=(64, 36), seconds=1)
    job = tmp_path / "job"
    job.mkdir()
    (job / "segments").mkdir()
    stale = job / "segments" / "segment-00000.mp4"
    stale.write_bytes(b"left over from a Fast run")
    (job / "manifest.json").write_text(json.dumps({"key": {"mode": "fast"}, "done": {"0": 25}}))
    result = run_upscale(src, tmp_path, VideoOptions(mode="general"))
    assert inspect_output(result.output_path)["frames"] == 25
    assert fake_model == ["video_general"]


def test_cancel_leaves_no_output(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "clip.mp4", seconds=1)
    cancel = threading.Event()

    def cancel_soon(fraction, message):
        if "frame 3 " in message:
            cancel.set()

    with pytest.raises(Cancelled):
        run_upscale(src, tmp_path, cancel=cancel, progress=cancel_soon)
    assert not list((tmp_path / "out").glob("*"))
    assert not list((tmp_path / "job" / "segments").glob("*.part"))


FAILING_COMMAND = [sys.executable, "-c", "import sys; sys.exit(1)"]


def test_encoder_failure_is_explained_kindly(tmp_path, fake_model, small_output, monkeypatch):
    monkeypatch.setattr(pipeline, "encode_cmd", lambda *args: FAILING_COMMAND)  # e.g. the disk filled up
    with pytest.raises(VideoError, match="free disk space") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    assert friendly_error(caught.value) == str(caught.value)  # shown as written, not "Something went wrong"
    assert not list((tmp_path / "out").glob("*"))
    assert not list((tmp_path / "job" / "segments").glob("*.mp4*"))


def test_decoder_failure_is_explained_kindly(tmp_path, fake_model, small_output, monkeypatch):
    monkeypatch.setattr(pipeline, "decode_cmd", lambda *args: FAILING_COMMAND)
    with pytest.raises(VideoError, match="damaged") as caught:
        run_upscale(make_clip(tmp_path / "clip.mp4", seconds=1), tmp_path)
    assert friendly_error(caught.value) == str(caught.value)


def test_video_damaged_part_way_is_not_saved_as_finished(tmp_path, fake_model, small_output, monkeypatch):
    # The decoder gives some frames, then fails (damage further into the file). The frames it gave
    # must not be kept as a finished segment, or a retry would join a video with a piece missing.
    monkeypatch.setattr(pipeline, "SEGMENT_SECONDS", 1)
    src = numbered_clip(tmp_path / "clip.mp4")
    frame_bytes = 64 * 36 * 3
    gives_up = f"import sys; sys.stdout.buffer.write(bytes({frame_bytes} * 30)); sys.exit(1)"
    monkeypatch.setattr(pipeline, "decode_cmd", lambda *args: [sys.executable, "-c", gives_up])
    with pytest.raises(VideoError, match="damaged"):
        run_upscale(src, tmp_path)
    saved = json.loads((tmp_path / "job" / "manifest.json").read_text())["done"]
    assert saved == {"0": 25}  # the whole first segment, and nothing after it
    assert not list((tmp_path / "job" / "segments").glob("*.part"))


def test_progress_reports_frames_and_time_left(tmp_path, fake_model, small_output):
    updates = []
    run_upscale(make_clip(tmp_path / "clip.mp4", seconds=2), tmp_path, progress=lambda f, m: updates.append((f, m)))
    fractions = [f for f, _ in updates]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("frame 50 of 50" in m for _, m in updates)
    assert any("left" in m for _, m in updates)


def test_output_names_never_overwrite(tmp_path, fake_model, small_output):
    src = make_clip(tmp_path / "clip.mp4", seconds=0.2, audio=False)
    names = []
    for job in ("job1", "job2"):
        result = upscale_video(src, tmp_path / "out", tmp_path / job, VideoOptions(),
                               progress=lambda f, m: None, cancel=threading.Event())
        names.append(result.output_path.name)
    assert names == ["clip-4K-fast.mp4", "clip-4K-fast-2.mp4"]


def test_already_4k_video_is_refused_kindly(tmp_path, fake_model):
    src = make_clip(tmp_path / "big.mp4", size=(3840, 2160), seconds=0.08, audio=False)
    with pytest.raises(VideoError, match="already 4K"):
        run_upscale(src, tmp_path)


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        run_upscale(tmp_path / "x.mp4", tmp_path, VideoOptions(mode="photoreal"))


# --- Real model ------------------------------------------------------------------


@pytest.mark.gpu
def test_upscale_video_fast_e2e(tmp_path):
    src = make_clip(tmp_path / "clip.mp4", size=(320, 180), seconds=2, fps=25)
    result = run_upscale(src, tmp_path, VideoOptions(mode="fast"))
    out = inspect_output(result.output_path)
    assert result.output_path.name == "clip-4K-fast.mp4"
    assert out["size"] == (3840, 2160) == (result.width, result.height)
    assert out["codec"] == "hevc" and out["tag"] == "hvc1"
    assert out["audio_streams"] == 1
    assert out["duration"] == pytest.approx(2.0, abs=0.1)
    assert moov_before_mdat(result.output_path)
