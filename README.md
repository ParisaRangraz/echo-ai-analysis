# Echo AI Analysis

An end-to-end pipeline for echocardiographic image analysis — denoising,
segmentation, motion-based mask propagation, and left ventricular ejection
fraction (LVEF) classification — built on the public CAMUS dataset.

Starting from a raw echo sequence, the pipeline segments the heart on the
end-diastole frame, tracks that segmentation through the cardiac cycle with
deformable registration, measures left ventricular volumes, and classifies
ejection fraction as normal or reduced.

Three segmentation architectures were built and compared end to end: a U-Net, a
PVTv2 transformer encoder with an attention-gated decoder, and the same encoder
with a reverse-attention decoder.

`NOTES.md` documents the experiments behind these results, including the
hypotheses that were tested and rejected, and the bugs found along the way.

## Motivation

This project extends prior work in quantitative ultrasound and RF signal
processing into a documented, reproducible deep-learning pipeline, applying
established methods from the literature to a standard public benchmark.

## Pipeline

```
ED frame (2CH and 4CH)
    -> segmentation model          -> predicted ED mask
    -> frame-to-frame registration -> predicted ES mask
    -> Simpson biplane             -> EDV, ESV
    -> calibration                 -> EF -> Normal / Reduced
```

Each module was first validated in isolation, using ground-truth masks as
input, so its own error could be measured without contamination from the stages
before it. Only then were they connected end to end.

## Results

### Held-out test set — the headline numbers

50 patients the pipeline had never seen, evaluated once, after every decision
was frozen. No ground-truth masks anywhere in the path; they are used only for
scoring. Model: PVTv2 + reverse attention. EF offset: measured on training.

| Metric | Value |
|---|---|
| LV Dice, ED mask (4CH) | 0.946 +/- 0.020 |
| LV Dice, propagated ES mask (4CH) | 0.882 +/- 0.051 |
| LV Dice, ED mask (2CH) | 0.935 +/- 0.042 |
| LV Dice, propagated ES mask (2CH) | 0.871 +/- 0.053 |
| EF bias | -4.7 % |
| EF SD of the difference | 6.8 % |
| EF mean absolute error | 6.6 % |
| Normal/Reduced agreement | **40 / 50 (80 %)** |

Against validation (0.951 / 0.900 / 0.946 / 0.868; bias -4.1 %, MAE 5.2 %,
44/50), segmentation held up almost unchanged while EF degraded more. That gap
is the optimism that validation numbers carry when every decision was made on
them, and it is reported rather than hidden.

One thing transferred well: the calibration offset was measured on training
patients, and the bias moved only 0.6 points between validation and a completely
new cohort.

### Architecture comparison — end to end, on validation

| Metric | U-Net | PVTv2 | PVTv2 + reverse attention |
|---|---|---|---|
| LV Dice, ED mask (4CH) | 0.946 | **0.951** | 0.951 |
| LV Dice, propagated ES mask (4CH) | 0.903 | **0.908** | 0.900 |
| LV Dice, ED mask (2CH) | 0.943 | **0.947** | 0.946 |
| LV Dice, propagated ES mask (2CH) | **0.878** | 0.870 | 0.868 |
| EF bias | -4.2 % | **-4.0 %** | -4.1 % |
| EF SD of the difference | 6.0 % | **5.6 %** | 5.7 % |
| EF mean absolute error | 5.3 % | **5.2 %** | **5.2 %** |
| Normal/Reduced agreement | 44 / 50 | 44 / 50 | 44 / 50 |

Three architectures spanning 2.0 M to 5.0 M parameters, with real differences in
segmentation quality, give **identical clinical classification** and mean
absolute errors within 0.1 points of each other.

That is the main engineering finding of the project: segmentation was never the
bottleneck. The EF error's random component had a standard deviation of exactly
6.4 % for all three models on the calibration set — it does not depend on the
segmentation network. The limits are the geometric assumptions in Simpson's
method and the accuracy lost while propagating the mask to ES.

Propagating from a *perfect* ED mask scores 0.916 at ES; propagating from a
model's own mask scores 0.900-0.908. Little is lost, which indicates
segmentation error is carried forward rather than amplified — registration
follows the motion in the image, not the shape of the mask it is moving.

## Dataset

This project uses the **CAMUS** dataset, which is **not** included in this
repository — see `data/README.md` for how to download it. The official
training / validation / testing split files are used as-is (400 / 50 / 50
patients).

## Preprocessing

- Images vary widely in size (height 292-973 px, width 323-1181 px in the
  training set); all are resized to 256x256 for the networks.
- Masks use nearest-neighbor interpolation, so class labels stay whole numbers;
  images use linear interpolation.
- Pixel intensities are normalized from [0, 255] to [0, 1].
- Results are cached to disk as one `.npz` file per patient per view.
- Registration and volume measurement work on the original, non-resized frames,
  because they need true physical pixel spacing.
- Sequence ordering is read from each patient's `Info_*.cfg` rather than
  assumed: 19 of 800 training sequences run ES to ED, and the ordering can
  differ between the two views of the same patient.

## Module 1: Denoising

A Lee filter (adaptive local-variance smoothing, ref. 2) and a small
convolutional autoencoder trained self-supervised.

| Method | PSNR | SSIM |
|---|---|---|
| Lee filter, window_size=5 | 30.93 dB | 0.876 |
| Lee filter, window_size=11 | 26.95 dB | 0.714 |

A larger window removes more speckle but changes the image more, and SSIM falls
faster than PSNR. Since sharp boundaries matter for segmentation, the smaller
window is the better default.

**Not currently used in the pipeline.** The segmentation models were trained on
raw frames, so feeding them denoised input would change their input
distribution; a fair test would require retraining on denoised data. Left as
future work rather than quietly enabled.

**Limitation.** Real ultrasound has no clean reference image — speckle is part
of the physics, not added noise. PSNR and SSIM here measure *how much each
method changed the image*, not improvement toward a ground truth.

## Module 2: Segmentation

All models were trained on the same data, with the same loss (CrossEntropy +
Dice combined), for the same 15 epochs, on ED frames from both apical views
(800 samples), on CPU.

- **U-Net** (ref. 4) — standard 4-level architecture: double 3x3 convolutions
  with BatchNorm, bilinear upsampling, skip connections. Adam, lr = 1e-3.
- **PVTv2 + attention gates** (refs. 5, 7) — a PVTv2-b0 encoder pretrained on
  ImageNet, with a U-Net-style decoder whose skip connections pass through
  attention gates. Discriminative learning rates: 1e-4 for the pretrained
  encoder, 1e-3 for the randomly initialised decoder.
- **PVTv2 + reverse attention** (ref. 6) — the same, with the attention map
  inverted before it modulates the skip connection. Where an attention gate
  emphasises what the model is already confident about, reverse attention
  emphasises what it is not: the periphery.

### Architecture comparison (4CH validation, 50 patients)

| Model | Params | LV cavity | Myocardium | Left atrium |
|---|---|---|---|---|
| U-Net | 2.0 M | 0.947 | 0.871 | 0.866 |
| PVTv2 + attention gate | 4.5 M | 0.951 | 0.860 | 0.895 |
| PVTv2 + boundary-weighted loss | 4.5 M | 0.948 | 0.862 | 0.901 |
| PVTv2 + reverse attention (conv) | 5.0 M | 0.951 | **0.872** | **0.902** |
| PVTv2 + reverse attention (transformer) | 5.8 M | **0.952** | 0.872 | 0.901 |

The transformer encoder's clearest gain is the left atrium — in 2CH it raises
Dice from 0.853 to 0.906 and cuts the standard deviation to a third. That
structure lies deepest in the sector, where local texture alone is ambiguous and
global context helps most.

Its one weakness was the myocardium, a thin ring whose Dice depends almost
entirely on boundary precision. A boundary-weighted loss did not fix it
(0.860 → 0.862, inside the noise). Reverse attention did (0.872), overtaking the
U-Net for the first time — and it adds no resolution, which showed the fine
detail had been present all along and simply discarded by the decoder.

Going from 5.0 M to 5.8 M parameters produced no further gain, so the
improvement came from the mechanism rather than the capacity. The convolutional
variant was chosen: equal accuracy, fewer parameters.

## Module 3: Registration and mask propagation

Direct ED-to-ES registration was considered and rejected: contraction between
those two frames is large and non-rigid, which no affine transform can
represent. Instead, **consecutive frames** are registered with a deformable
B-spline transform (ref. 10), and the ED mask is carried forward step by step.

CAMUS annotates every frame, so propagation is not filling a gap in this
dataset. Instead it *simulates* the clinical situation where only one frame is
annotated — which then allows the propagated masks to be scored against real
annotations at every frame.

### Validation results (50 patients, 974 frames)

| Method | LV Dice at ES |
|---|---|
| Copy the ED mask unchanged | 0.783 +/- 0.082 |
| Frame-to-frame registration | 0.916 +/- 0.032 |

Mean Dice across all frames: 0.945 +/- 0.019. Registration beat the baseline in
50/50 patients and halved the standard deviation.

Direct segmentation of the ES frame was also tested as an alternative. It gave a
slightly better mean Dice (0.912 vs 0.903) but roughly double the spread, with
failures severe enough to produce physiologically implausible EF values.
Propagation was kept: it starts from a good mask and only deforms it, so it
cannot fail as badly.

### Prior work
Mask propagation via registration or motion tracking is an established
technique, not a contribution of this project; recent examples are refs. 11 and
12. This project implements a simpler, classical (SimpleITK) version of the same
idea.

## Module 4: Volume and ejection fraction

Volumes are computed with Simpson's method of disks, the method recommended for
2D echocardiography by the ASE/EACVI guidelines (ref. 13). Two variants were
evaluated on ground-truth masks: mono-plane (4CH only, circular disks) and
biplane (one diameter from each view, elliptical disks — the clinical standard,
and the basis of the CAMUS reference values).

| | Mono-plane | Biplane |
|---|---|---|
| Bias | +4.1 % | +8.1 % |
| SD of the difference | 8.3 % | **3.5 %** |
| Correlation with reference | 0.837 | **0.971** |

Mono-plane has the smaller bias but is far less reliable: on one patient it gave
20.6 % against a reference of 57.0 %. **Biplane was chosen**, because systematic
error can be corrected while random error cannot.

Classification applies the standard clinical threshold from the same guidelines
(EF below 50 % is reduced) to the computed value. It is a fixed rule rather than
a trained classifier: the dataset provides EF values, not Normal/Reduced labels,
so a classifier trained on labels derived from EF would only reproduce the
threshold.

### Calibration

Simpson's method overestimates EF systematically, so a fixed offset was measured
on **training** patients and applied to validation and test. Fitting the
correction on a set and then reporting improvement on that same set would be
circular.

The offset had to be measured twice. The first was derived from ground-truth
masks (+6.33 %), and applying it to pipeline output produced a large negative
bias: predicted masks behave differently, particularly at ES. Re-measured under
real pipeline conditions it was +1.13 %, +1.04 % and +0.95 % for the three
models.

**Limitations.**
- The correction transfers imperfectly between cohorts. It measured about +1 %
  on training, yet the residual bias is about -4 % on both validation and test.
  The offset is consistent across the two unseen cohorts, but it does not
  reproduce the training measurement.
- The bias is nearly cancelled by coincidence rather than design: the volume
  method overestimates EF, while the propagated ES mask stays slightly too large
  and underestimates it. These are independent errors that happen to oppose each
  other.
- The source of the overestimate is not explained. Three hypotheses were tested
  and rejected — a tilted long axis, mono-plane versus biplane, and re-orienting
  the disks along the measured LV axis. The remaining candidates are the
  definition of the LV base in the masks, and the possibility that the reference
  EF was measured with clinical software rather than derived from these masks at
  all.

## Repository layout

```
data/          loading, preprocessing, image processing (denoising, registration)
models/        network architectures and loss functions
evaluation/    metrics and clinical measurements
tests/         unit tests for the core functions
results/       CSV outputs from evaluation runs
*.py           top-level scripts: training, evaluation, full pipeline
```

## Running it

```
pip install -r requirements.txt
python data/run_preprocessing.py     # one-time: builds the .npz cache
python train_segmentation.py         # U-Net
python train_pvt.py                  # PVTv2 + attention gates
python train_pvt_rta.py              # PVTv2 + reverse attention
python evaluate_segmentation.py
python calibrate_pipeline_ef.py      # measures the EF offset on training
python run_pipeline.py               # end-to-end
pytest                               # unit tests (no dataset required)
```

## Planned work

- Investigate the source of the EF bias rather than only calibrating it — the
  measured bottleneck, unlike segmentation. The disk-orientation hypothesis has
  been tested and rejected; the definition of the LV base remains open.
- Wire the denoising module into the pipeline, which requires retraining the
  segmentation model on denoised frames for a fair comparison.
- Figures and a demo notebook.

## Attribution and reuse

The code in this repository is original. Where a method comes from the
literature it is cited below, and each implementation was written from the
published description rather than copied from an authors' repository — this
applies in particular to the reverse-attention decoder, which follows the
description in RTA-Former (ref. 6) but is not derived from its source code.

The dataset is not redistributed here; see `data/README.md` for how to obtain it
and for its own terms of use.

If this work is useful in your own, please cite it. GitHub's "Cite this
repository" button uses the `CITATION.cff` file at the repository root.

## References

### Dataset
1. Leclerc, S., Smistad, E., Pedrosa, J., Østvik, A. et al. (2019). *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.* IEEE Transactions on Medical Imaging, 38(9), 2198-2210.

### Denoising
2. Lee, J.-S. (1980). *Digital Image Enhancement and Noise Filtering by Use of Local Statistics.* IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-2(2), 165-168.
3. Odena, A., Dumoulin, V. & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts.* Distill. — the resize-convolve fix used in the decoder.

### Segmentation architectures
4. Ronneberger, O., Fischer, P. & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
5. Oktay, O., Schlemper, J., Le Folgoc, L. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL.
6. Li, Z., Yi, S., Uneri, A., Niu, S. & Jones, C. (2024). *RTA-Former: Reverse Transformer Attention for Polyp Segmentation.* IEEE EMBC.
7. Wang, W., Xie, E., Li, X. et al. (2022). *PVT v2: Improved Baselines with Pyramid Vision Transformer.* Computational Visual Media, 8, 415-424.
8. Milletari, F., Navab, N. & Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV. — the Dice loss.
9. Dice, L. R. (1945). *Measures of the Amount of Ecologic Association Between Species.* Ecology, 26(3), 297-302.

### Registration and mask propagation
10. Rueckert, D., Sonoda, L. I., Hayes, C. et al. (1999). *Nonrigid Registration Using Free-Form Deformations: Application to Breast MR Images.* IEEE TMI, 18(8), 712-721.
11. EchoSAM2 (2026). *Enhancing anatomical consistency in few-shot cardiac ultrasound segmentation via sequence-wise memory and registration.* Frontiers in Medicine.
12. *Point Tracking as a Temporal Cue for Robust Myocardial Segmentation in Echocardiography Videos.* arXiv:2601.09207.

### Clinical measurement
13. Lang, R. M., Badano, L. P., Mor-Avi, V. et al. (2015). *Recommendations for Cardiac Chamber Quantification by Echocardiography in Adults: An Update from the American Society of Echocardiography and the European Association of Cardiovascular Imaging.* J Am Soc Echocardiogr, 28(1), 1-39.
14. Ferraz, S., Coimbra, M., Pedrosa, J. et al. (2022). *Beyond Simpson's Rule: Accounting for Orientation and Ellipticity Assumptions.* Ultrasound in Medicine & Biology, 48(12), 2476-2485.
15. Bland, J. M. & Altman, D. G. (1986). *Statistical Methods for Assessing Agreement Between Two Methods of Clinical Measurement.* The Lancet, 327(8476), 307-310.
16. Ouyang, D., He, B., Ghorbani, A. et al. (2020). *Video-based AI for beat-to-beat assessment of cardiac function.* Nature, 580, 252-256.

### Software
17. Lowekamp, B. C., Chen, D. T., Ibáñez, L. & Blezek, D. (2013). *The Design of SimpleITK.* Frontiers in Neuroinformatics, 7:45.
18. Wightman, R. (2019). *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models
19. Paszke, A., Gross, S., Massa, F. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
20. van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J. et al. (2014). *scikit-image: image processing in Python.* PeerJ, 2:e453.

## Project log

- [x] Data access and preprocessing pipeline
- [x] Denoising module (classical + learned)
- [x] Segmentation: U-Net, both views, validated
- [x] Segmentation: PVTv2 + attention decoder
- [x] Segmentation: boundary-weighted loss (tested, rejected)
- [x] Segmentation: reverse attention, convolutional and transformer variants
- [x] Registration + mask propagation (validated)
- [x] EF calculation, biplane, with calibration
- [x] End-to-end runs with all three architectures
- [x] Disk orientation along the true LV long axis (tested, rejected)
- [x] Unit tests + CI workflow
- [x] Final test-set evaluation
- [ ] Figures and demo
- [ ] Denoising wired into the pipeline

## Author

Parisa Rangraz, Ph.D. — Biomedical Engineer

## License

MIT — see `LICENSE`. You may use, modify and redistribute the code, including
commercially, provided the copyright notice and licence text are retained.

The MIT licence does not require citation, but citation is requested: see
"Attribution and reuse" above and `CITATION.cff`.

The CAMUS dataset is **not** covered by this licence and is not included in this
repository. It carries its own terms, which prohibit redistribution; see
`data/README.md`.
