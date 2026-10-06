"""
export_onnx.py

Exports the trained PVTv2 + reverse-attention segmentation model to ONNX and
checks that the exported model gives the same answer as the PyTorch original.

An ONNX file can be run without PyTorch (ONNX Runtime, C++, C#, embedded
devices), which is how a model is usually deployed inside device software.

Usage (from the repository root):
    python export_onnx.py
    python export_onnx.py --weights models/pvt_rta_both_views.pt --out models/pvt_rta.onnx

The input is fixed at 1 x 1 x 256 x 256 in H and W (the size the network was
trained on); the batch dimension is dynamic.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import numpy as np
import torch

from pvt_rta import PVTReverseAttentionUNet

INPUT_SIZE = 256
OPSET = 17


def load_model(weights_path=None):
    """
    Build the model and (optionally) load trained weights.

    pretrained=False: the weights come from the checkpoint, so downloading
    ImageNet weights would be wasted work (and would need internet access).
    With weights_path=None the weights stay random -- only useful for testing
    the export machinery itself, e.g. in CI where no checkpoint exists.
    """
    model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model


def export(model, out_path, opset=OPSET):
    """Export `model` to `out_path` as ONNX with a dynamic batch dimension."""
    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)

    kwargs = dict(
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )

    # Newer PyTorch versions default to the "dynamo" exporter, which needs an
    # extra package. The classic exporter is used where it can be selected,
    # because it is the long-established path; older versions only have it.
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, out_path, **kwargs)

    return out_path


def check_parity(model, onnx_path, n_samples=4, seed=0, atol=1e-3):
    """
    Run the same random inputs through PyTorch and ONNX Runtime and compare.

    Two things are reported:
      - the largest absolute difference between the raw logits;
      - the fraction of pixels whose predicted class (argmax) is identical.
    The second is what matters for segmentation; the first shows how close the
    two implementations are numerically.

    Returns (max_abs_diff, pixel_agreement). Raises AssertionError when the
    logits differ by more than `atol`.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.random((n_samples, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"image": x})[0]

    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    agreement = float((torch_logits.argmax(1) == onnx_logits.argmax(1)).mean())

    assert max_diff <= atol, f"ONNX and PyTorch logits differ by {max_diff:.2e} (> {atol})"
    return max_diff, agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default="models/pvt_rta_both_views.pt",
                        help="trained PyTorch weights")
    parser.add_argument("--out", default="models/pvt_rta.onnx", help="output ONNX file")
    parser.add_argument("--random-weights", action="store_true",
                        help="export an untrained model (testing only, e.g. in CI)")
    args = parser.parse_args()

    if args.random_weights:
        print("WARNING: random weights -- this exports the architecture only.")
        torch.manual_seed(0)
    model = load_model(None if args.random_weights else args.weights)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    export(model, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Exported {args.out} ({size_mb:.1f} MB)")

    max_diff, agreement = check_parity(model, args.out)
    print(f"Parity with PyTorch: max |logit difference| = {max_diff:.2e}, "
          f"identical predicted class on {agreement * 100:.3f} % of pixels")


if __name__ == "__main__":
    main()
