"""
infer.py

Run the full pipeline on ONE new patient and print the ejection fraction.

    2CH + 4CH half_sequence (NIfTI)
        -> segmentation model on the ED frame     -> ED mask
        -> frame-to-frame registration            -> ES mask
        -> Simpson biplane                        -> EDV, ESV
        -> calibration offset                     -> EF -> Normal / Reduced

Unlike run_pipeline.py, this script needs no CAMUS folder layout, no
ground-truth masks and no hard-coded paths: you point it at two files.

The model can be an ONNX file (needs only onnxruntime -- no PyTorch) or a
PyTorch .pt checkpoint (needs torch and timm). Which one is chosen by the file
extension.

Usage:
    python infer.py --model models/pvt_rta.onnx \\
        --seq-4ch patient0001_4CH_half_sequence.nii.gz \\
        --seq-2ch patient0001_2CH_half_sequence.nii.gz \\
        --cfg-4ch Info_4CH.cfg --cfg-2ch Info_2CH.cfg

--cfg-* are optional. They give the true ED/ES frame numbers; without them the
first frame is assumed to be ED. That assumption is WRONG for about 2 % of
CAMUS sequences (they run ES -> ED), which flips EDV and ESV, so pass the cfg
files whenever you have them.

Research prototype: not a medical device, not for clinical use.
"""

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _folder in ("data", "models", "evaluation"):
    sys.path.insert(0, os.path.join(_ROOT, _folder))

import numpy as np
import SimpleITK as sitk

from loader import resize_image
from processing import propagate_mask
from volume import lv_volume_biplane_ml, ejection_fraction, classify_ef

INPUT_SIZE = 256
# Calibration offset measured on the training set for the PVTv2 + reverse
# attention model. It belongs to that model: a different model (or a quantized
# copy of it) needs its own offset.
DEFAULT_EF_OFFSET = 0.95


# --- model backends ---------------------------------------------------------

def _make_logits_fn(model_path):
    """
    Return a function: float32 array (1, 1, 256, 256) -> logits (1, 4, 256, 256).
    ONNX Runtime and PyTorch are imported only when needed, so an ONNX-only
    install (e.g. the Docker image) never needs PyTorch.
    """
    if model_path.endswith(".onnx"):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        return lambda x: session.run(None, {input_name: x})[0]

    import torch
    from pvt_rta import PVTReverseAttentionUNet
    model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()

    def run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    return run


def make_segmenter(model_path):
    """
    Return segment(frame) -> mask, where `frame` is one 2D echo frame at its
    original size and `mask` has the same size with labels 0-3.

    The network works on 256x256 inputs scaled to [0, 1] (as in training), but
    registration and volume measurement need the original resolution, so the
    predicted mask is resized back with nearest-neighbour interpolation to keep
    the labels whole numbers.
    """
    logits_fn = _make_logits_fn(model_path)

    def segment(frame):
        original_shape = frame.shape
        resized = resize_image(frame.astype(np.float32), (INPUT_SIZE, INPUT_SIZE),
                               is_mask=False)
        x = (resized / 255.0).astype(np.float32)[None, None]
        prediction = logits_fn(x).argmax(axis=1)[0]
        return resize_image(prediction.astype(np.float32), original_shape, is_mask=True)

    return segment


# --- input reading ----------------------------------------------------------

def read_sequence(path):
    """Read a half_sequence NIfTI. Returns (frames, H, W) array and (x, y) spacing in mm."""
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image), image.GetSpacing()[:2]


def read_ed_es(cfg_path):
    """
    Read the ED and ES frame numbers from an Info_*.cfg file.
    The cfg numbers frames from 1; numpy indexes from 0.
    """
    info = {}
    with open(cfg_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return int(info["ED"]) - 1, int(info["ES"]) - 1


def ed_first(seq, cfg_path=None):
    """
    Return the sequence ordered so that index 0 is ED and the last index is ES.
    Reverses it when the cfg says ES comes first. Without a cfg, the sequence
    is returned unchanged.
    """
    if cfg_path is None:
        return seq
    ed_index, es_index = read_ed_es(cfg_path)
    return seq[::-1] if ed_index > es_index else seq


# --- the pipeline -----------------------------------------------------------

class SegmentationError(RuntimeError):
    """Raised when no LV cavity is found, so no volume or EF can be computed."""


def estimate_ef(seq_4ch, spacing_4ch, seq_2ch, spacing_2ch, segment,
                ef_offset=DEFAULT_EF_OFFSET):
    """
    The full pipeline on two ED-first sequences.

    Args:
        seq_4ch, seq_2ch: (frames, H, W) arrays, index 0 = ED, last = ES
        spacing_*: (x, y) pixel spacing in mm
        segment: function frame -> mask (see make_segmenter)
        ef_offset: calibration offset subtracted from the raw EF

    Returns a dict with EDV, ESV, raw and calibrated EF and the class.
    """
    ed_4ch = segment(seq_4ch[0])
    ed_2ch = segment(seq_2ch[0])

    es_4ch = propagate_mask(seq_4ch, ed_4ch, spacing_4ch)[-1]
    es_2ch = propagate_mask(seq_2ch, ed_2ch, spacing_2ch)[-1]

    edv = lv_volume_biplane_ml(ed_4ch, spacing_4ch, ed_2ch, spacing_2ch)
    esv = lv_volume_biplane_ml(es_4ch, spacing_4ch, es_2ch, spacing_2ch)

    # With no LV in the ED mask the EF is NaN, and classify_ef(NaN) would
    # silently answer "Normal" (NaN < 50 is False). A failed segmentation must
    # be reported as a failure, never as a clinical class.
    if not edv > 0:
        raise SegmentationError(
            "No LV cavity found in the ED mask, so no EF can be computed. "
            "Check the input files and that the model matches this pipeline.")

    ef_raw = ejection_fraction(edv, esv)
    ef = ef_raw - ef_offset

    warnings = []
    if ef_raw < 0:
        warnings.append("Negative EF: ESV is larger than EDV. The sequence is probably "
                        "in ES -> ED order; pass the Info_*.cfg files.")

    return {
        "edv_ml": round(float(edv), 1),
        "esv_ml": round(float(esv), 1),
        "ef_raw_percent": round(float(ef_raw), 1),
        "ef_percent": round(float(ef), 1),
        "ef_offset": ef_offset,
        "ef_class": classify_ef(ef),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate LV ejection fraction from a 2CH + 4CH echo sequence.")
    parser.add_argument("--model", required=True, help="model file: .onnx or .pt")
    parser.add_argument("--seq-4ch", required=True, help="4CH half_sequence .nii.gz")
    parser.add_argument("--seq-2ch", required=True, help="2CH half_sequence .nii.gz")
    parser.add_argument("--cfg-4ch", help="Info_4CH.cfg (gives true ED/ES frame numbers)")
    parser.add_argument("--cfg-2ch", help="Info_2CH.cfg (gives true ED/ES frame numbers)")
    parser.add_argument("--ef-offset", type=float, default=DEFAULT_EF_OFFSET,
                        help="calibration offset subtracted from the raw EF "
                             f"(default {DEFAULT_EF_OFFSET}, measured for the FP32 model)")
    args = parser.parse_args()

    if args.cfg_4ch is None or args.cfg_2ch is None:
        print("WARNING: no Info_*.cfg given -- assuming the first frame is ED. "
              "This is wrong for ~2 % of CAMUS sequences.", file=sys.stderr)

    seq_4ch, sp_4ch = read_sequence(args.seq_4ch)
    seq_2ch, sp_2ch = read_sequence(args.seq_2ch)
    seq_4ch = ed_first(seq_4ch, args.cfg_4ch)
    seq_2ch = ed_first(seq_2ch, args.cfg_2ch)

    try:
        result = estimate_ef(seq_4ch, sp_4ch, seq_2ch, sp_2ch,
                             make_segmenter(args.model), args.ef_offset)
    except SegmentationError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
