"""
test_infer.py

Tests for infer.py that need neither a trained model nor the CAMUS dataset.

The segmentation model is replaced by a trivial threshold function, and the
"echo sequence" is a bright ellipse that shrinks from frame to frame (a heart
contracting, drastically simplified). What is tested is the plumbing around the
model: frame ordering, registration-based propagation, volume and EF
arithmetic, and the output format -- not the model's accuracy.

Run from the repository root:
    pytest
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import infer


def make_sequence(n_frames=4, size=96, start=(30, 20), end=(20, 14)):
    """A bright ellipse on a dark background whose radii shrink linearly."""
    yy, xx = np.mgrid[:size, :size]
    cy = cx = size // 2
    frames = []
    for k in range(n_frames):
        t = k / (n_frames - 1)
        ry = start[0] + (end[0] - start[0]) * t
        rx = start[1] + (end[1] - start[1]) * t
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        frames.append(np.where(inside, 200.0, 20.0))
    return np.stack(frames).astype(np.float32)


def threshold_segmenter(frame):
    """Stand-in for the network: label 1 (LV cavity) wherever the frame is bright."""
    return (frame > 100).astype(np.float32)


# --- ordering ---------------------------------------------------------------

def test_read_ed_es_converts_to_zero_based(tmp_path):
    cfg = tmp_path / "Info_4CH.cfg"
    cfg.write_text("ED: 1\nES: 12\nNbFrame: 20\nImageQuality: Good\n")
    assert infer.read_ed_es(str(cfg)) == (0, 11)


def test_ed_first_reverses_when_es_comes_first(tmp_path):
    cfg = tmp_path / "Info_4CH.cfg"
    cfg.write_text("ED: 18\nES: 1\n")
    seq = np.arange(18)[:, None, None] * np.ones((18, 2, 2))
    ordered = infer.ed_first(seq, str(cfg))
    assert ordered[0, 0, 0] == 17 and ordered[-1, 0, 0] == 0


def test_ed_first_keeps_order_when_ed_comes_first(tmp_path):
    cfg = tmp_path / "Info_4CH.cfg"
    cfg.write_text("ED: 1\nES: 18\n")
    seq = np.arange(18)[:, None, None] * np.ones((18, 2, 2))
    ordered = infer.ed_first(seq, str(cfg))
    assert ordered[0, 0, 0] == 0 and ordered[-1, 0, 0] == 17


def test_ed_first_without_cfg_changes_nothing():
    seq = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
    assert np.array_equal(infer.ed_first(seq, None), seq)


# --- the pipeline -----------------------------------------------------------

def test_estimate_ef_on_contracting_sequence():
    seq = make_sequence()
    result = infer.estimate_ef(seq, (1.0, 1.0), seq, (1.0, 1.0),
                               threshold_segmenter, ef_offset=0.0)

    assert set(result) == {"edv_ml", "esv_ml", "ef_raw_percent", "ef_percent",
                           "ef_offset", "ef_class", "warnings"}
    # The ellipse shrinks, so the end-systolic volume must be smaller.
    assert result["edv_ml"] > result["esv_ml"] > 0
    assert 0.0 < result["ef_raw_percent"] < 100.0
    assert result["ef_class"] in ("Normal", "Reduced")


def test_ef_offset_is_subtracted():
    seq = make_sequence()
    result = infer.estimate_ef(seq, (1.0, 1.0), seq, (1.0, 1.0),
                               threshold_segmenter, ef_offset=2.0)
    assert abs(result["ef_raw_percent"] - result["ef_percent"] - 2.0) < 0.11  # rounding to 0.1


def test_reversed_sequence_gives_negative_ef():
    """Feeding ES -> ED order is exactly the bug the cfg handling exists to prevent."""
    seq = make_sequence()[::-1].copy()
    result = infer.estimate_ef(seq, (1.0, 1.0), seq, (1.0, 1.0),
                               threshold_segmenter, ef_offset=0.0)
    assert result["ef_raw_percent"] < 0.0
    assert result["warnings"], "a negative EF must come with a warning"


def test_empty_segmentation_is_an_error_not_a_class():
    """No LV found must raise, not return NaN labelled 'Normal'."""
    seq = make_sequence()

    def empty(frame):
        return np.zeros_like(frame)

    with pytest.raises(infer.SegmentationError):
        infer.estimate_ef(seq, (1.0, 1.0), seq, (1.0, 1.0), empty)
