"""
benchmark_onnx.py

Compares the deployment variants of the segmentation model:

    PyTorch (optional)  |  ONNX Runtime FP32  |  ONNX Runtime INT8

on three things: file size, CPU latency (one 256x256 image at a time, which is
what the pipeline does), and -- when a preprocessed dataset folder is given --
segmentation accuracy against ground truth.

Speed alone is not a result. A quantized model that is 3x faster but loses 5
Dice points is a regression, so accuracy is measured with the same Dice
definition as the rest of the project (evaluation/metrics.py).

Usage:
    python benchmark_onnx.py --fp32 models/pvt_rta.onnx --int8 models/pvt_rta_int8.onnx \\
        --weights models/pvt_rta_both_views.pt --data data/processed/validation \\
        --save results/onnx_benchmark.md

Latency numbers depend on the CPU, so the CPU and thread count are printed
with the table. INT8 is only faster on CPUs with fast 8-bit integer
instructions; on others it can be slower than FP32.
"""

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import numpy as np

from metrics import dice_score

CLASS_NAMES = {1: "LV cavity", 2: "Myocardium", 3: "Left atrium"}


def make_ort_runner(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda x: session.run(None, {name: x})[0]


def make_torch_runner(weights, threads):
    import torch
    from pvt_rta import PVTReverseAttentionUNet
    torch.set_num_threads(threads)
    model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.eval()

    def run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    return run


def time_runner(run, runs, warmup, seed=0):
    """Median and 95th-percentile latency in ms for a single 1x1x256x256 input."""
    x = np.random.default_rng(seed).random((1, 1, 256, 256), dtype=np.float32)
    for _ in range(warmup):          # first calls are slower (allocation, caching)
        run(x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def load_samples(data_dir, limit):
    """(image, mask) pairs from preprocessed .npz files; image in [0, 1], mask labels 0-3."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))[:limit]
    samples = []
    for f in files:
        d = np.load(os.path.join(data_dir, f))
        samples.append((d["image"].astype(np.float32), d["mask"].astype(np.int64)))
    return samples


def evaluate(run, samples):
    """Per-class mean Dice vs ground truth, plus the predictions for later comparison."""
    scores = {c: [] for c in CLASS_NAMES}
    predictions = []
    for image, mask in samples:
        pred = run(image[None, None]).argmax(axis=1)[0]
        predictions.append(pred)
        for c in CLASS_NAMES:
            scores[c].append(dice_score(pred, mask, c))
    return {c: float(np.mean(v)) for c, v in scores.items()}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fp32", default="models/pvt_rta.onnx")
    parser.add_argument("--int8", default=None, help="quantized model (skipped if omitted)")
    parser.add_argument("--weights", default=None, help="PyTorch weights (skipped if omitted)")
    parser.add_argument("--data", default=None,
                        help="preprocessed split folder, e.g. data/processed/validation")
    parser.add_argument("--n-samples", type=int, default=100, help="images used for accuracy")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--save", default=None, help="also write the table to this file")
    args = parser.parse_args()

    variants = {}
    if args.weights:
        variants["PyTorch FP32"] = (make_torch_runner(args.weights, args.threads), None)
    variants["ONNX Runtime FP32"] = (make_ort_runner(args.fp32, args.threads), args.fp32)
    if args.int8:
        variants["ONNX Runtime INT8"] = (make_ort_runner(args.int8, args.threads), args.int8)

    samples = load_samples(args.data, args.n_samples) if args.data else None

    rows, predictions = [], {}
    for name, (run, path) in variants.items():
        median, p95 = time_runner(run, args.runs, args.warmup)
        size = f"{os.path.getsize(path) / 1e6:.2f}" if path else "-"
        row = {"name": name, "size": size, "median": f"{median:.1f}", "p95": f"{p95:.1f}"}
        if samples:
            dice, predictions[name] = evaluate(run, samples)
            row.update({CLASS_NAMES[c]: f"{dice[c]:.3f}" for c in CLASS_NAMES})
        rows.append(row)

    header = ["Variant", "Size (MB)", "Median ms", "p95 ms"]
    if samples:
        header += list(CLASS_NAMES.values())
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [r["name"], r["size"], r["median"], r["p95"]]
        if samples:
            cells += [r[n] for n in CLASS_NAMES.values()]
        lines.append("| " + " | ".join(cells) + " |")

    notes = [
        "",
        f"CPU: {platform.processor() or platform.machine()}, {args.threads} thread(s), "
        f"batch 1, 256x256, {args.runs} timed runs after {args.warmup} warm-up runs.",
    ]
    if samples:
        notes.append(f"Accuracy: mean Dice on {len(samples)} images from {args.data}.")
        reference = "ONNX Runtime FP32"
        for name in predictions:
            if name != reference:
                same = np.mean([(a == b).mean()
                                for a, b in zip(predictions[name], predictions[reference])])
                notes.append(f"{name} predicts the same class as {reference} on "
                             f"{same * 100:.2f} % of pixels.")
    else:
        notes.append("No --data given: latency and size only, accuracy not measured.")

    report = "\n".join(lines + notes)
    print(report)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
