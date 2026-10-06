"""
quantize_onnx.py

Post-training static INT8 quantization of the exported ONNX model.

Quantization stores weights (and activations) as 8-bit integers instead of
32-bit floats: the file gets about 4x smaller and, on CPUs with fast integer
instructions, inference can get faster. The price is a small numerical error,
so the result must be MEASURED, not assumed -- benchmark_onnx.py compares Dice
against ground truth for both models.

"Static" means the activation ranges are measured in advance on a calibration
set. That set should look like the real inputs, so by default it is a random
sample of the preprocessed training images (data/processed/training/*.npz).
Calibrating on random noise would give the wrong ranges, so --synthetic exists
only to smoke-test the script (e.g. in CI, where the dataset is not available).

Usage:
    python quantize_onnx.py
    python quantize_onnx.py --calib-dir data/processed/training --n-calib 100
    python quantize_onnx.py --synthetic        # no dataset; for testing only
"""

import argparse
import os

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

INPUT_SIZE = 256


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds one (1, 1, 256, 256) float32 image at a time to the calibrator."""

    def __init__(self, input_name, images):
        self._batches = iter([{input_name: img[None, None].astype(np.float32)}
                              for img in images])

    def get_next(self):
        return next(self._batches, None)


def load_calibration_images(calib_dir, n, seed=0):
    """A reproducible random sample of n preprocessed training images (values in [0, 1])."""
    files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {calib_dir}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [np.load(os.path.join(calib_dir, files[i]))["image"] for i in chosen]


def synthetic_images(n, seed=0):
    """Random smooth images. For smoke tests only -- NOT a valid calibration set."""
    rng = np.random.default_rng(seed)
    return [rng.random((INPUT_SIZE, INPUT_SIZE)).astype(np.float32) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="models/pvt_rta.onnx", help="FP32 ONNX model")
    parser.add_argument("--out", default="models/pvt_rta_int8.onnx", help="quantized output")
    parser.add_argument("--calib-dir", default="data/processed/training",
                        help="folder of preprocessed .npz files used for calibration")
    parser.add_argument("--n-calib", type=int, default=100, help="number of calibration images")
    parser.add_argument("--synthetic", action="store_true",
                        help="calibrate on random noise (testing only)")
    args = parser.parse_args()

    import onnxruntime as ort
    input_name = ort.InferenceSession(
        args.model, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    if args.synthetic:
        print("WARNING: calibrating on random noise -- the quantized model is for testing only.")
        images = synthetic_images(args.n_calib)
    else:
        images = load_calibration_images(args.calib_dir, args.n_calib)
    print(f"Calibrating on {len(images)} images")

    # Shape inference first: it makes quantization more reliable. Skipped
    # (with a message) when ONNX Runtime cannot preprocess this particular graph.
    model_in = args.model
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        prepared = args.out.replace(".onnx", "_prep.onnx")
        quant_pre_process(args.model, prepared)
        model_in = prepared
    except Exception as err:  # noqa: BLE001 - any failure just means "skip this optional step"
        print(f"Pre-processing skipped ({type(err).__name__}); quantizing the original graph.")

    quantize_static(
        model_in,
        args.out,
        ImageCalibrationReader(input_name, images),
        quant_format=QuantFormat.QDQ,
        per_channel=True,                 # one scale per output channel: more accurate for conv weights
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )

    if model_in != args.model and os.path.exists(model_in):
        os.remove(model_in)

    fp32_mb = os.path.getsize(args.model) / 1e6
    int8_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out}: {int8_mb:.2f} MB (FP32: {fp32_mb:.2f} MB, "
          f"FP32 / INT8 size ratio {fp32_mb / int8_mb:.1f}x)")


if __name__ == "__main__":
    main()
