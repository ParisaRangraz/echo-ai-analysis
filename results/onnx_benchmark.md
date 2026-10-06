| Variant | Size (MB) | Median ms | p95 ms | LV cavity | Myocardium | Left atrium |
|---|---|---|---|---|---|---|
| PyTorch FP32 | - | 330.0 | 407.4 | 0.949 | 0.874 | 0.906 |
| ONNX Runtime FP32 | 20.23 | 50.2 | 63.8 | 0.949 | 0.874 | 0.906 |
| ONNX Runtime INT8 | 5.83 | 64.6 | 77.2 | 0.948 | 0.872 | 0.903 |

CPU: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, 12 thread(s), batch 1, 256x256, 50 timed runs after 10 warm-up runs.
Accuracy: mean Dice on 100 images from data/processed/validation.
PyTorch FP32 predicts the same class as ONNX Runtime FP32 on 100.00 % of pixels.
ONNX Runtime INT8 predicts the same class as ONNX Runtime FP32 on 98.92 % of pixels.
