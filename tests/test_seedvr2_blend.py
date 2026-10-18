import numpy as np
import pytest

from upscaler.seedvr2.blend import blend_tiles, feather_weights, tile_grid


def _cut_tiles(image: np.ndarray, boxes, scale: int):
    return [(box, image[box[1] * scale : box[3] * scale, box[0] * scale : box[2] * scale]) for box in boxes]


def test_tile_grid_covers_without_gaps():
    w, h = 2048, 1539
    boxes = tile_grid(w, h, 512, 64)

    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 <= 512 and y1 - y0 <= 512
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_tile_grid_neighbours_overlap_by_at_least_the_overlap():
    boxes = tile_grid(2048, 1539, 512, 64)
    xs = sorted({(x0, x1) for x0, _, x1, _ in boxes})
    ys = sorted({(y0, y1) for _, y0, _, y1 in boxes})
    for (_, prev_end), (next_start, _) in zip(xs, xs[1:]):
        assert prev_end - next_start >= 64
    for (_, prev_end), (next_start, _) in zip(ys, ys[1:]):
        assert prev_end - next_start >= 64


def test_tile_grid_uses_one_tile_size_so_every_tile_scales_alike():
    boxes = tile_grid(2048, 1539, 512, 64)
    assert len({(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes}) == 1


def test_tile_grid_uses_the_fewest_tiles_with_little_wasted_overlap():
    w, h = 2048, 1539
    boxes = tile_grid(w, h, 512, 64)
    processed = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)

    assert len(boxes) == 5 * 4
    assert processed / (w * h) < 1.3  # full 512 px tiles would process 1.66x the image


def test_tile_grid_small_image_is_one_tile():
    assert tile_grid(300, 200, 512, 64) == [(0, 0, 300, 200)]
    assert tile_grid(2048, 300, 512, 64)[0][3] == 300


def test_feather_weights_ramp_only_on_interior_edges():
    weights = feather_weights(8, 10, 4, (True, False, False, True))  # left and bottom are interior

    assert weights.shape == (8, 10)
    assert (weights > 0).all() and (weights <= 1).all()
    assert weights[0, 0] < weights[0, 3] < weights[0, 4] == 1.0  # left edge ramps up
    assert weights[0, -1] == 1.0  # right edge is the image border: no ramp
    assert weights[-1, 5] < weights[0, 5] == 1.0  # bottom ramps, top does not


def test_feather_weights_sum_to_one():
    w, h, scale = 300, 200, 2
    boxes = tile_grid(w, h, 128, 32)
    tiles = [(box, np.full(((box[3] - box[1]) * scale, (box[2] - box[0]) * scale, 3), 137, np.uint8)) for box in boxes]

    out = blend_tiles(tiles, w * scale, h * scale, scale, ramp=32 * scale)

    assert out.shape == (h * scale, w * scale, 3) and out.dtype == np.uint8
    assert (out == 137).all()


def test_blend_tiles_no_seam():
    w, h, scale = 333, 250, 4
    yy, xx = np.mgrid[0 : h * scale, 0 : w * scale].astype(np.float32)
    gradient = np.stack([xx / (w * scale) * 255, yy / (h * scale) * 255, (xx + yy) / ((w + h) * scale) * 255], axis=-1)
    gradient = gradient.round().astype(np.uint8)
    boxes = tile_grid(w, h, 96, 24)

    out = blend_tiles(_cut_tiles(gradient, boxes, scale), w * scale, h * scale, scale, ramp=24 * scale)

    assert np.abs(out.astype(int) - gradient.astype(int)).max() <= 1


def test_blend_tiles_hides_the_tile_border():
    """Where two tiles disagree, the result fades from one to the other instead of jumping."""
    w, h, scale = 200, 64, 1
    boxes = tile_grid(w, h, 128, 48)
    assert len(boxes) == 2
    tiles = [(box, np.full((box[3] - box[1], box[2] - box[0], 3), v, np.uint8)) for box, v in zip(boxes, (100, 200))]

    row = blend_tiles(tiles, w, h, scale, ramp=48)[0, :, 0].astype(int)

    assert row[0] == 100 and row[-1] == 200
    assert np.all(np.diff(row) >= 0)
    assert np.abs(np.diff(row)).max() <= 5


def test_blend_tiles_rejects_wrongly_sized_tile():
    with pytest.raises(ValueError):
        blend_tiles([((0, 0, 10, 10), np.zeros((39, 40, 3), np.uint8))], 40, 40, 4, ramp=8)
