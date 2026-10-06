"""
test_onnx.py

Tests for the deployment path: PyTorch -> ONNX export, ONNX Runtime inference
and INT8 quantization.

These use a model with RANDOM weights. That is deliberate: the questions here
are "does the exported graph compute the same function as the PyTorch model?"
and "does the pipeline run?", which do not depend on the weights being trained.
Accuracy of the trained model is measured by benchmark_onnx.py, not here.

The whole file is skipped when PyTorch / timm / ONNX Runtime are not installed,
so the lightweight test job (numpy + SimpleITK only) is unaffected.

Run from the repository root:
    pytest tests/test_onnx.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import export_onnx


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A random-weight model, exported once and shared by all tests in this file."""
    torch.manual_seed(0)
    model = export_onnx.load_model(None)
    path = str(tmp_path_factory.mktemp("onnx") / "model.onnx")
    export_onnx.export(model, path)
    return model, path


def test_onnx_matches_pytorch(exported):
    model, path = exported
    max_diff, agreement = export_onnx.check_parity(model, path, n_samples=2)
    assert max_diff < 1e-3
    assert agreement > 0.999


def test_onnx_accepts_a_different_batch_size(exported):
    _, path = exported
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).random((3, 1, 256, 256), dtype=np.float32)
    out = session.run(None, {"image": x})[0]
    assert out.shape == (3, 4, 256, 256)


def test_int8_model_runs_and_has_the_same_output_shape(exported, tmp_path):
    """
    Quantization must produce a model that loads, runs and returns finite
    logits of the same shape. Calibration here is on noise, so the numerical
    agreement with FP32 is NOT asserted -- on real data that is measured by
    benchmark_onnx.py.
    """
    _, fp32_path = exported
    int8_path = str(tmp_path / "model_int8.onnx")

    import quantize_onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        fp32_path, int8_path,
        quantize_onnx.ImageCalibrationReader("image", quantize_onnx.synthetic_images(4)),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
    )

    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(2).random((1, 1, 256, 256), dtype=np.float32)
    out = session.run(None, {"image": x})[0]
    assert out.shape == (1, 4, 256, 256)
    assert np.isfinite(out).all()
    assert os.path.getsize(int8_path) < os.path.getsize(fp32_path)
