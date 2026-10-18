import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from upscaler.config import SPACE_RESERVE_BYTES
from upscaler.hardware import Hardware
from upscaler.sizing import (
    DEFAULT_PHOTOREAL_LEVEL,
    PHOTOREAL_LEVELS,
    PRESETS,
    SEEDVR2_MAX_LONG_EDGE,
    SMALL_IMAGE_WARNING,
    TWO_PASS_NOTE,
    describe_size,
    estimate,
    estimated_file_bytes,
    free_disk_bytes,
    pass_seconds,
    photoreal_route,
    plan_passes,
    print_size,
    target_size,
)

MAC_16GB = Hardware("mps", "Apple M4", 16, "mac", True)
MAC_8GB = Hardware("mps", "Apple M1", 8, "mac", True)
PC_NVIDIA = Hardware("cuda", "RTX 4070", 12, "windows", False)
PC_NO_GPU = Hardware("cpu", "Intel", 32, "windows", False)
DiskUsage = namedtuple("DiskUsage", "total used free")


def test_target_size_8k_landscape():
    assert target_size(2048, 1539, "8k") == (8192, 6156)


def test_target_size_8k_portrait():
    assert target_size(941, 1672, "8k") == (4610, 8192)


def test_target_size_6k():
    assert target_size(2048, 1539, "6k") == (6144, 4617)


def test_target_size_4k():
    assert target_size(512, 385, "4k") == (3840, 2888)


def test_target_size_8k_tv_portrait():
    # Portrait fits within 4320x7680.
    assert target_size(941, 1672, "8k_tv") == (4320, 7676)


def test_target_size_8k_tv_landscape():
    # 2048x1539 is taller than 16:9, so the 4320 height is the limit.
    assert target_size(2048, 1539, "8k_tv") == (5749, 4320)


def test_target_size_scale():
    assert target_size(1000, 500, "3x") == (3000, 1500)


def test_target_size_never_shrinks():
    assert target_size(9000, 6000, "8k") == (9000, 6000)


def test_target_size_exact_for_vectors():
    # A vector draws at any size, so "8K" means 8K even when its own size is bigger.
    assert target_size(9000, 6000, "8k", exact=True) == (8192, 5461)
    assert target_size(20000, 10000, "16k_tv", exact=True) == (15360, 7680)
    assert target_size(9000, 6000, "2x", exact=True) == (18000, 12000)
    assert target_size(1000, 500, "4k", exact=True) == target_size(1000, 500, "4k")


def test_estimate_for_a_vector(roomy_disk):
    est = estimate(20000, 10000, "8k", "photoreal", "jpeg", MAC_16GB, vector=True)

    assert est["out"] == (8192, 4096)
    assert est["passes"] == 0  # drawn directly, no AI
    assert est["seconds"] == pytest.approx(8192 * 4096 / 1e6 * 0.035, abs=0.1)


def test_target_size_rounds_half_up():
    # 3 * 3840 / 2560 = 4.5 exactly; banker's rounding would give 4.
    assert target_size(2560, 3, "4k") == (3840, 5)


def test_target_size_16k():
    assert target_size(2048, 1539, "16k") == (16384, 12312)


def test_target_size_16k_tv_landscape():
    # Taller than 16:9, so the 8640 height is the limit.
    assert target_size(2048, 1539, "16k_tv") == (11498, 8640)


def test_target_size_16k_tv_portrait():
    # Portrait fits within 8640x15360.
    assert target_size(941, 1672, "16k_tv") == (8640, 15352)


def test_target_size_unknown_preset():
    with pytest.raises(ValueError):
        target_size(100, 100, "32k")


def test_presets_cover_every_size_in_ui_order():
    assert list(PRESETS) == ["4k", "6k", "8k", "8k_tv", "16k", "16k_tv", "2x", "3x", "4x"]
    assert PRESETS["16k"] == "16K (16384 px)"
    assert all(isinstance(label, str) and label for label in PRESETS.values())


def test_plan_one_pass():
    assert plan_passes((2048, 1539), (8192, 6156)).passes == 1


def test_plan_lanczos_up_to_5x():
    plan = plan_passes((941, 1672), (4610, 8192))  # 4.9x
    assert plan.passes == 1
    assert plan.second_pass_scale is None
    assert plan.warning is None


def test_plan_exactly_5x_is_one_pass():
    plan = plan_passes((1000, 500), (5000, 2500))
    assert (plan.passes, plan.second_pass_scale) == (1, None)


def test_plan_16k_two_pass_fast():
    plan = plan_passes((2048, 1539), (16384, 12312))
    assert plan.passes == 2
    assert plan.second_pass_scale == 2.0
    assert "16K" in plan.warning and "small" not in plan.warning  # 2048 px isn't small


def test_plan_small_source():
    plan = plan_passes((800, 600), (8192, 6144))
    assert plan.passes == 2
    assert plan.second_pass_scale == 2.56
    assert "small" in plan.warning


def test_plan_small_source_to_16k():
    plan = plan_passes((512, 385), (16384, 12320))
    assert (plan.passes, plan.second_pass_scale) == (2, 8.0)
    assert "small" in plan.warning


def test_plan_nothing_to_do_when_already_at_target():
    plan = plan_passes((9000, 6000), (9000, 6000))
    assert plan.passes == 0
    assert plan.second_pass_scale is None
    assert plan.warning is None


# --- print size -----------------------------------------------------------------


def test_print_size_300dpi():
    size = print_size(16384, 12312, 300)
    assert size == {"inches": (54.6, 41.0), "cm": (138.7, 104.2), "dpi": 300}


def test_print_size_72dpi():
    assert print_size(3840, 2160, 72) == {"inches": (53.3, 30.0), "cm": (135.5, 76.2), "dpi": 72}


@pytest.mark.parametrize("dpi", [0, -300])
def test_print_size_rejects_bad_dpi(dpi):
    with pytest.raises(ValueError):
        print_size(100, 100, dpi)


# --- Photo-real routes -----------------------------------------------------------


def test_photoreal_levels():
    assert PHOTOREAL_LEVELS == ("balanced", "rich", "gentle")
    assert DEFAULT_PHOTOREAL_LEVEL == "balanced"


def test_photoreal_route_2k_to_8k_per_level():
    src, dst = (2048, 1539), (8192, 6156)
    balanced = photoreal_route(src, dst, "balanced")
    assert (balanced.seedvr2_input, balanced.seedvr2_scale, balanced.fast_pass) == (src, 2, True)
    rich = photoreal_route(src, dst, "rich")
    assert (rich.seedvr2_input, rich.seedvr2_scale, rich.fast_pass) == (src, 4, False)
    gentle = photoreal_route(src, dst, "gentle")
    assert (gentle.seedvr2_input, gentle.seedvr2_scale, gentle.fast_pass) == (src, 2, False)
    assert balanced.warning is rich.warning is gentle.warning is None
    assert photoreal_route(src, dst) == balanced  # the default level


def test_photoreal_route_16k_per_level():
    src, dst = (2048, 1539), (16384, 12312)
    rich = photoreal_route(src, dst, "rich")
    assert (rich.seedvr2_scale, rich.fast_pass, rich.warning) == (4, True, TWO_PASS_NOTE)
    for level in ("balanced", "gentle"):  # two steps by design, so no two-pass note
        route = photoreal_route(src, dst, level)
        assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass, route.warning) == (src, 2, True, None)


def test_photoreal_route_small_source_warns():
    for level in PHOTOREAL_LEVELS:
        route = photoreal_route((800, 600), (8192, 6144), level)
        assert route.fast_pass and route.warning == SMALL_IMAGE_WARNING


@pytest.mark.parametrize("level", PHOTOREAL_LEVELS)
@pytest.mark.parametrize(
    ("src", "size"),
    [
        ((5000, 3750), "16k"),
        ((2048, 1539), "6k"),
        ((3000, 2000), "4k"),
        ((3200, 12), "8k"),
        ((941, 1672), "8k"),
        ((5000, 3750), "6k"),
        ((8064, 6048), "8k"),
        ((16000, 12000), "16k"),
    ],
)
def test_photoreal_route_never_makes_a_picture_longer_than_8k_or_the_size(src, size, level):
    dst = target_size(*src, size)
    route = photoreal_route(src, dst, level)
    made = tuple(side * route.seedvr2_scale for side in route.seedvr2_input)
    limit = max(max(dst), SEEDVR2_MAX_LONG_EDGE)
    assert max(made) <= limit + route.seedvr2_scale // 2  # rounding of the smaller copy only
    if max(src) * route.seedvr2_scale <= limit:
        assert route.seedvr2_input == src  # nothing thrown away when it fits


def test_photoreal_route_5000px_to_16k_never_runs_4x_to_20000px():
    src, dst = (5000, 3750), (16384, 12288)
    balanced = photoreal_route(src, dst, "balanced")
    assert (balanced.seedvr2_input, balanced.seedvr2_scale, balanced.fast_pass) == (src, 2, True)
    rich = photoreal_route(src, dst, "rich")  # still SeedVR2 4x, on a copy shrunk so 4x lands on 16K
    assert (rich.seedvr2_input, rich.seedvr2_scale, rich.fast_pass) == ((4096, 3072), 4, False)
    gentle = photoreal_route(src, dst, "gentle")
    assert (gentle.seedvr2_input, gentle.seedvr2_scale, gentle.fast_pass) == (src, 2, False)


@pytest.mark.parametrize("level", PHOTOREAL_LEVELS)
@pytest.mark.parametrize(("src", "size"), [((3000, 2000), "4k"), ((4032, 3024), "6k"), ((2048, 1539), "6k")])
def test_photoreal_route_keeps_every_real_pixel_up_to_8k(src, size, level):
    # SeedVR2 gets the whole picture, even when its result is bigger than needed (6000 px for 4K,
    # 8064 px for 6K): the final resize brings it down, rather than SeedVR2 re-inventing detail
    # thrown away beforehand.
    route = photoreal_route(src, target_size(*src, size), level)
    assert route.seedvr2_input == src


def test_photoreal_route_phone_photo_keeps_its_detail_at_6k_and_8k():
    src = (4032, 3024)
    for size in ("6k", "8k"):
        route = photoreal_route(src, target_size(*src, size))
        assert (route.seedvr2_input, route.seedvr2_scale) == (src, 2)


def test_photoreal_route_shrinks_a_big_picture_only_as_far_as_8k():
    # 2x of 5000 px would be 10000 px for a 6K picture: the copy is shrunk so SeedVR2 makes 8K.
    for level in PHOTOREAL_LEVELS:
        route = photoreal_route((5000, 3750), target_size(5000, 3750, "6k"), level)
        assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((4096, 3072), 2, False)


def test_photoreal_route_shrinks_to_the_size_when_it_is_bigger_than_8k():
    route = photoreal_route((16000, 12000), target_size(16000, 12000, "16k"))
    assert (route.seedvr2_input, route.seedvr2_scale) == ((8192, 6144), 2)  # SeedVR2 makes exactly 16K


@pytest.mark.parametrize("level", ["balanced", "gentle"])
@pytest.mark.parametrize(("src", "size"), [((1024, 768), "16k"), ((1500, 1125), "16k"), ((640, 480), "8k")])
def test_photoreal_route_small_picture_runs_4x_first(src, size, level):
    # 2x, then the Fast model's 4x, would leave a blurry stretch: 1024 px to 16K is 16x.
    route = photoreal_route(src, target_size(*src, size), level)
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == (src, 4, True)
    assert route.warning == SMALL_IMAGE_WARNING


def test_photoreal_route_2x_while_the_fast_model_reaches():
    assert photoreal_route((1000, 500), (10000, 5000), "balanced").seedvr2_scale == 2  # 2x, then 5x at most
    assert photoreal_route((1000, 500), (10100, 5050), "balanced").seedvr2_scale == 4
    assert photoreal_route((1640, 1230), target_size(1640, 1230, "16k"), "gentle").seedvr2_scale == 2  # 9.99x


def test_photoreal_route_rich_uses_2x_within_lanczos_reach():
    route = photoreal_route((3300, 12), target_size(3300, 12, "8k"), "rich")
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((3300, 12), 2, False)


def test_photoreal_route_rich_keeps_a_big_photo_whole_just_past_lanczos_reach():
    # 4x would mean shrinking 3200 px to 2048 px first (8K is the cap); 2x keeps every real pixel
    # and leaves a 1.28x resize.
    route = photoreal_route((3200, 2400), target_size(3200, 2400, "8k"), "rich")
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((3200, 2400), 2, False)
    # Further than that, Rich still means SeedVR2 4x, on a copy shrunk so it lands on 8K.
    route = photoreal_route((3000, 2000), target_size(3000, 2000, "8k"), "rich")
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((2048, 1365), 4, False)


def test_photoreal_route_rich_small_picture_keeps_4x_at_the_same_factor():
    route = photoreal_route((1000, 750), (2560, 1920), "rich")  # 2.56x too, but 4x needs no shrink
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((1000, 750), 4, False)


def test_photoreal_route_rich_4x_to_4_9x_resizes_the_rest():
    route = photoreal_route((941, 1672), (4610, 8192), "rich")
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((941, 1672), 4, False)


def test_photoreal_route_balanced_exact_2x_needs_no_fast_pass():
    route = photoreal_route((1000, 500), (2000, 1000), "balanced")
    assert (route.seedvr2_input, route.seedvr2_scale, route.fast_pass) == ((1000, 500), 2, False)


def test_photoreal_route_gentle_adds_fast_only_beyond_5x():
    assert not photoreal_route((1000, 500), (5000, 2500), "gentle").fast_pass  # 2x, then Lanczos 2.5x
    assert photoreal_route((1000, 500), (5100, 2550), "gentle").fast_pass


def test_photoreal_route_rejects_unknown_level():
    with pytest.raises(ValueError, match="detail level"):
        photoreal_route((100, 100), (400, 400), "extreme")


# --- estimate -------------------------------------------------------------------


@pytest.fixture
def roomy_disk(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: DiskUsage(10**12, 0, 10**12))


def test_estimate_16k_warns_size(roomy_disk):
    est = estimate(2048, 1539, "16k", "photoreal", "tiff", MAC_16GB, level="rich")

    assert est["out"] == (16384, 12312)
    assert est["megapixels"] == 201.7
    assert est["file_mb"] == pytest.approx(201.7 * 1.8, abs=0.1)
    assert est["print"] == print_size(16384, 12312, 300)
    assert est["passes"] == 2
    # Photo-real at 4x on 3.15 MP, then Fast on the 50.4 MP pass-1 result, then finishing and saving.
    assert est["seconds"] == pytest.approx(3.152 * 400 + 50.43 * 0.5 + 201.7 * 0.06, rel=0.01)
    assert any("16K files are very large" in w for w in est["warnings"])
    assert any(f"about {round(est['seconds'] / 60)} minutes" in w for w in est["warnings"])
    assert not any("free space" in w for w in est["warnings"])


def test_estimate_disk_warning(monkeypatch):
    checked = []

    def nearly_full(path):
        checked.append(path)
        return DiskUsage(500 * 10**9, 499 * 10**9, 900 * 10**6)  # 900 MB free

    monkeypatch.setattr(shutil, "disk_usage", nearly_full)

    est = estimate(2048, 1539, "16k", "fast", "png", MAC_16GB)  # about 323 MB, needs 3x that

    assert any(w.startswith("Not enough free space") for w in est["warnings"])
    assert checked


def test_estimate_disk_warning_counts_the_space_kept_free(monkeypatch):
    # Room for three times the file isn't enough when it would eat into the 2 GB the app always leaves
    # free: the picture would be refused when its turn came, so say so before it starts.
    file_bytes = estimated_file_bytes(16384, 12312, "png")
    free = {"bytes": 3 * file_bytes + SPACE_RESERVE_BYTES - 1}
    monkeypatch.setattr(shutil, "disk_usage", lambda path: DiskUsage(10**12, 0, free["bytes"]))

    est = estimate(2048, 1539, "16k", "fast", "png", MAC_16GB)
    warning = next(w for w in est["warnings"] if w.startswith("Not enough free space"))
    assert "always leaves 2 GB free" in warning and warning.endswith(", or choose JPEG.")
    assert est["room_mb"] == round((free["bytes"] - SPACE_RESERVE_BYTES) / 1e6, 1)  # for a whole batch

    free["bytes"] = 100_000_000  # less than the space kept free: no room for any result
    assert estimate(2048, 1539, "4k", "fast", "jpeg", MAC_16GB)["room_mb"] == 0


def test_estimate_without_a_readable_disk_skips_the_space_warning(monkeypatch, tmp_path):
    def missing_drive(path):
        raise FileNotFoundError(2, "No such file or directory")  # e.g. an unplugged drive on Windows

    monkeypatch.setattr(shutil, "disk_usage", missing_drive)

    est = estimate(2048, 1539, "16k", "fast", "png", MAC_16GB, out_dir=tmp_path)

    assert est["out"] == (16384, 12312)
    assert not any("free space" in w for w in est["warnings"])
    assert est["room_mb"] is None


def test_estimate_disk_checks_nearest_existing_folder(monkeypatch, tmp_path):
    checked = []
    monkeypatch.setattr(shutil, "disk_usage", lambda path: checked.append(path) or DiskUsage(1, 0, 10**12))

    estimate(2048, 1539, "8k", "fast", "jpeg", MAC_16GB, out_dir=tmp_path / "not" / "made" / "yet")

    assert checked == [tmp_path]
    assert not (tmp_path / "not").exists()


def test_estimate_small_fast_job_has_no_warnings(roomy_disk):
    est = estimate(2048, 1539, "4k", "fast", "jpeg", MAC_16GB)

    assert est["out"] == (3840, 2886)
    assert est["passes"] == 1
    assert est["seconds"] == pytest.approx(3.152 * 0.5 + 11.08 * 0.035, abs=0.1)
    assert est["file_mb"] == pytest.approx(3840 * 2886 * 0.2 / 1e6, abs=0.1)
    assert est["warnings"] == []


def test_estimate_print_uses_dpi(roomy_disk):
    est = estimate(2048, 1539, "8k", "fast", "jpeg", MAC_16GB, dpi=150)
    assert est["print"] == print_size(8192, 6156, 150)


def test_pass_seconds_photoreal_2x_is_a_quarter_of_the_work():
    four, _ = pass_seconds((2048, 1539), (8192, 6156), "photoreal", MAC_16GB, level="rich")
    two, _ = pass_seconds((4032, 3024), (8192, 6144), "photoreal", MAC_16GB, level="rich")  # 2.03x: SeedVR2 runs 2x
    assert four == pytest.approx(3.152 * 400, rel=0.01)
    assert two == pytest.approx(12.19 * 400 / 4, rel=0.01)


def test_pass_seconds_scale_with_hardware():
    def seconds(hw):
        return pass_seconds((2048, 1539), (16384, 12312), "faithful", hw)

    first, second = seconds(MAC_16GB)
    assert second == pytest.approx(50.43 * 0.5, rel=0.01)
    assert seconds(PC_NVIDIA) == pytest.approx((first / 4, second / 4))
    assert seconds(PC_NO_GPU) == pytest.approx((first * 15, second * 15))


def test_pass_seconds_per_photoreal_level():
    src, dst = (2048, 1539), (8192, 6156)
    rich = pass_seconds(src, dst, "photoreal", MAC_16GB, level="rich")
    balanced = pass_seconds(src, dst, "photoreal", MAC_16GB, level="balanced")
    gentle = pass_seconds(src, dst, "photoreal", MAC_16GB, level="gentle")
    assert rich == pytest.approx((3.152 * 400, 0.0), rel=0.01)
    # Balanced: a quarter of the 4x work, then the Fast model on SeedVR2's 2x result (12.6 MP).
    assert balanced == pytest.approx((3.152 * 400 / 4, 12.61 * 0.5), rel=0.01)
    assert gentle == pytest.approx((3.152 * 400 / 4, 0.0), rel=0.01)
    assert pass_seconds(src, dst, "photoreal", MAC_16GB) == balanced  # the default level


def test_pass_seconds_rich_counts_the_shrunk_copy():
    first, second = pass_seconds((5000, 3750), (16384, 12288), "photoreal", MAC_16GB, level="rich")
    assert first == pytest.approx(4096 * 3072 / 1e6 * 400, rel=0.01) and second == 0


def test_pass_seconds_small_picture_to_16k_counts_seedvr2_4x():
    src, dst = (1024, 768), target_size(1024, 768, "16k")
    first, second = pass_seconds(src, dst, "photoreal", MAC_16GB)  # Balanced
    assert first == pytest.approx(0.786 * 400, rel=0.01)  # the full 4x work, not a quarter
    assert second == pytest.approx(0.786 * 16 * 0.5, rel=0.01)  # Fast on SeedVR2's 4096 x 3072


def test_pass_seconds_count_the_whole_picture_when_it_is_kept():
    first, _ = pass_seconds((3000, 2000), (3840, 2560), "photoreal", MAC_16GB)
    assert first == pytest.approx(6.0 * 400 / 4, rel=0.01)  # 2x on all 6 MP


def test_estimate_per_photoreal_level(roomy_disk):
    finishing = 50.43 * 0.035
    balanced = estimate(2048, 1539, "8k", "photoreal", "jpeg", MAC_16GB)
    rich = estimate(2048, 1539, "8k", "photoreal", "jpeg", MAC_16GB, level="rich")
    gentle = estimate(2048, 1539, "8k", "photoreal", "jpeg", MAC_16GB, level="gentle")
    assert balanced["seconds"] == pytest.approx(315.2 + 6.3 + finishing, rel=0.01)  # about 5 minutes
    assert rich["seconds"] == pytest.approx(1260.8 + finishing, rel=0.01)  # about 21 minutes
    assert gentle["seconds"] == pytest.approx(315.2 + finishing, rel=0.01)
    assert (balanced["passes"], rich["passes"], gentle["passes"]) == (2, 1, 1)
    assert balanced["out"] == rich["out"] == gentle["out"] == (8192, 6156)
    assert any("21 minutes" in w for w in rich["warnings"])
    assert not any("minutes" in w for w in balanced["warnings"])


def test_estimate_rejects_unknown_level(roomy_disk):
    with pytest.raises(ValueError):
        estimate(2048, 1539, "8k", "photoreal", "jpeg", MAC_16GB, level="extreme")


def test_estimate_level_is_ignored_for_other_modes(roomy_disk):
    assert estimate(2048, 1539, "8k", "fast", "jpeg", MAC_16GB, level="rich") == estimate(
        2048, 1539, "8k", "fast", "jpeg", MAC_16GB
    )


def test_estimate_scales_with_hardware(roomy_disk):
    seconds = {
        hw.device: estimate(2048, 1539, "8k", "faithful", "jpeg", hw)["seconds"]
        for hw in (MAC_16GB, PC_NVIDIA, PC_NO_GPU)
    }
    assert seconds["cuda"] < seconds["mps"] < seconds["cpu"]
    assert seconds["mps"] == pytest.approx(3.152 * 45 + 50.43 * 0.035, rel=0.01)


def test_estimate_photoreal_without_enough_memory_uses_faithful(roomy_disk):
    est = estimate(2048, 1539, "8k", "photoreal", "jpeg", MAC_8GB)

    assert est["seconds"] == pytest.approx(estimate(2048, 1539, "8k", "faithful", "jpeg", MAC_8GB)["seconds"])
    assert any("Faithful" in w for w in est["warnings"])


def test_estimate_cloud_time_is_mostly_network(roomy_disk):
    mac = estimate(2048, 1539, "8k", "cloud:fal:topaz_precision", "jpeg", MAC_16GB)["seconds"]
    cpu = estimate(2048, 1539, "8k", "cloud:fal:topaz_precision", "jpeg", PC_NO_GPU)["seconds"]
    assert mac == cpu == pytest.approx(3.152 * 20 + 50.43 * 0.035, rel=0.01)


def test_estimate_already_big_enough(roomy_disk):
    est = estimate(9000, 6000, "8k", "photoreal", "png", MAC_16GB)

    assert est["out"] == (9000, 6000) and est["passes"] == 0
    assert est["seconds"] < 5  # just saving a copy


@pytest.mark.parametrize(
    ("size", "mode", "fmt"), [("10k", "fast", "jpeg"), ("8k", "magic", "jpeg"), ("8k", "fast", "gif")]
)
def test_estimate_rejects_unknown_choices(roomy_disk, size, mode, fmt):
    with pytest.raises(ValueError):
        estimate(2048, 1539, size, mode, fmt, MAC_16GB)


# --- file sizes and disk space ------------------------------------------------------


def test_estimated_file_bytes_counts_transparency():
    assert estimated_file_bytes(1000, 1000, "png") == 1_600_000
    assert estimated_file_bytes(1000, 1000, "tiff", alpha=True) == 2_400_000  # four channels, not three


@pytest.mark.parametrize(
    ("num_bytes", "text"), [(0, "1 MB"), (362_600_000, "363 MB"), (999_000_000, "999 MB"), (1_234_000_000, "1.2 GB")]
)
def test_describe_size(num_bytes, text):
    assert describe_size(num_bytes) == text


def test_free_disk_bytes_checks_nearest_existing_folder(monkeypatch, tmp_path):
    checked = []
    monkeypatch.setattr(shutil, "disk_usage", lambda path: checked.append(path) or DiskUsage(1, 0, 123))

    assert free_disk_bytes(tmp_path / "not" / "made") == 123
    assert checked == [tmp_path]


def test_sizing_is_light_to_import():
    # Pure maths: no Pillow (whose global size limit imageio_utils changes) and no torch.
    code = "import sys, upscaler.sizing; print(sorted(m for m in ('PIL', 'numpy', 'torch') if m in sys.modules))"
    root = Path(__file__).resolve().parents[1]
    loaded = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert loaded.stdout.strip() == "[]"


def test_sizing_problems_worded_for_people_are_user_errors():
    from upscaler.config import UserError
    from upscaler.sizing import mode_for_hardware, photoreal_route

    with pytest.raises(UserError):
        target_size(100, 100, "10k")
    with pytest.raises(UserError):
        photoreal_route((100, 100), (800, 800), "extreme")
    with pytest.raises(UserError):
        mode_for_hardware("magic", Hardware("mps", "Apple M4", 16, "mac", True))
    with pytest.raises(UserError):
        estimated_file_bytes(10, 10, "gif")
