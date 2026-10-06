# Engineering Notes

A running log of the experiments behind this project: what was tried, what the
result was, and what was concluded — including the attempts that did not work.

README.md reports the final results. This file records how they were reached.

---

## Data

### Verifying the dataset structure instead of assuming it
The loader was written only after inspecting the actual archive contents. This
caught two differences from the initial assumption:
- the distribution is NIfTI (`.nii.gz`), not the raw/mhd format assumed;
- there is no `training/` or `testing/` folder — all 500 patients sit flat under
  `database_nifti/`, and the official split is defined by text files in
  `database_split/` (400 / 50 / 50).

Reading those split files programmatically also corrected an early plan to take
"the first 450 patients" as training: the official split is randomised, not
sequential.

### Correcting a wrong assumption about annotations
The registration module was originally motivated by the claim that only the ED
frame is annotated. Checking `half_sequence_gt` directly disproved this: every
frame carries a full mask (labels 0-3). The module's framing was rewritten to be
accurate — see "Registration" below.

**Still open:** whether the intermediate-frame masks are drawn manually or
produced semi-automatically. This is not yet documented either way, and should be
verified in the official CAMUS documentation before any claim is made about it.

### Image size varies widely
Training set: height 292-973 px, width 323-1181 px. All frames are resized to
256x256 for the networks. Masks use nearest-neighbor interpolation so class
labels stay whole numbers; a check on `np.unique` after resizing confirmed the
labels remained exactly [0, 1, 2, 3].

Registration deliberately works on the original, non-resized frames, because it
needs true physical pixel spacing.

---

## Module 1: Denoising

### Checkerboard artifacts from ConvTranspose2d
The first autoencoder decoder used `ConvTranspose2d(kernel_size=3, stride=2)`.
The loss curve looked fine, but the output showed a faint checkerboard pattern,
clearest near the image edges.

Cause: when kernel size is not evenly divisible by stride, overlapping kernels
cover some output pixels more than others — a known failure mode.

Fix: `Upsample` (nearest) followed by a stride-1 `Conv2d`, which separates
resizing from learning. The artifact disappeared.

**Lesson:** a decreasing loss says nothing about artifacts. Look at the output.

### Trade-off in the classical filter
Lee filter on one frame:

| window_size | PSNR | SSIM |
|---|---|---|
| 5 | 30.93 dB | 0.876 |
| 11 | 26.95 dB | 0.714 |

A larger window removes more speckle but changes the image more, and SSIM (which
is structure-sensitive) falls faster than PSNR. Since sharp boundaries matter for
the segmentation step, the smaller window is the better default.

### An honest limitation
Real ultrasound has no clean reference image — speckle is part of the physics, not
added noise. PSNR/SSIM here are computed between the original and the filtered
image, so they measure *how much the filter changed the image*, not improvement
toward a ground truth. The same caveat applies to the CNN denoiser, which is
trained self-supervised with the input as its own target.

### Not wired into the pipeline
This module was built and validated, but `run_pipeline.py` feeds raw frames
straight to the segmentation network. Inserting denoising would not be a
one-line change: the models were trained on raw images, so denoised input would
no longer match their training distribution, and a fair test would require
retraining on denoised data too. Left as future work rather than quietly
switched on.

---

## Module 2: Segmentation

### The first architecture could not work, and no loss function could fix it

| Attempt | Result | Diagnosis |
|---|---|---|
| 2-level U-Net + CrossEntropy | Predicted almost only background | Class imbalance |
| + Dice loss | All classes appeared, but scattered and noisy | Imbalance fixed; predictions resembled a brightness map |
| + 30 epochs | Improved, then plateaued around Dice loss 0.49 | Not simply undertrained |
| + Combined loss, lr = 1e-4 | Plateaued by epoch 6 | Learning rate too low |
| 4-level U-Net + BatchNorm, combined loss, lr = 1e-3 | Loss 1.70 -> 0.17, clean boundaries | **Receptive field was the real limit** |

The root cause: with two downsampling steps, each output pixel saw roughly a
15-pixel neighbourhood of a 256x256 image. Local brightness is not enough to tell
these structures apart — the LV, RV and atria are all dark regions. The model had
no way to know *where in the heart* a pixel was.

**Lesson:** when a model plateaus, ask whether it has access to the information
the task requires, before tuning the loss or the optimizer.

### A visualization bug that hid the real behaviour
`plt.imshow` rescales colours per image. A prediction containing only classes 0
and 1 rendered class 1 in the colour ground truth used for class 3, making the
output look more wrong than it was. Fixed with `vmin=0, vmax=3` on both panels.

**Lesson:** verify the measuring instrument, not just the model.

### Validation results (4CH-only model, 50 patients, 4CH ED frames)

| Structure | Dice (mean +/- std) |
|---|---|
| LV cavity | 0.937 +/- 0.032 |
| Myocardium | 0.852 +/- 0.063 |
| Left atrium | 0.838 +/- 0.087 |

The myocardium is a thin ring, so boundary errors cost proportionally more Dice
than they do for compact regions. The left atrium has the widest spread, likely
because it sits near the edge of the ultrasound sector where image quality drops
and part of it can fall outside the field of view.

---

## Module 3: Registration and mask propagation

### Why not register ED directly to ES
Rejected before implementation. Contraction between ED and ES is large and
non-rigid: different parts of the wall move by different amounts in different
directions. An affine transform cannot represent that, and even deformable
registration is unreliable over such a large displacement.

Instead, consecutive frames are registered — small, physically plausible motion —
and the mask is carried forward step by step.

### Reframing the module honestly
Since CAMUS annotates every frame, propagation is not filling a gap in this
dataset. What it does is *simulate* the clinical situation where only one frame is
annotated, which then allows the propagated masks to be scored against real
annotations at every frame. That makes the evaluation stronger, not weaker.

### The first test looked like a failure
Registering frame 0 to frame 1 gave LV Dice 0.984, slightly *below* the 0.986
obtained by copying the mask unchanged. Motion between adjacent frames is so small
that there is almost nothing to correct, and registration adds a little error.

The single-pair test was simply the wrong test. Running the full sequence showed
the two curves crossing around frame 4, after which the gap widens steadily.

**Lesson:** test a method where it is supposed to matter.

### Speed: full-resolution registration was too slow
Larger validation images (up to 973x1181) made each registration evaluate the
similarity metric over more than a million pixels per optimizer step. Windows
Efficiency Mode throttling VS Code made it worse.

Fix: random metric sampling at 10% of pixels, with a fixed seed for
reproducibility. Accuracy check on the same patient: ES Dice 0.881 with sampling
vs 0.886 without — negligible.

**Lesson:** after any speed optimization, re-run the original test and show the
accuracy held.

### Validation results (50 patients, 974 frames)

| Method | LV Dice at ES |
|---|---|
| Copy ED mask unchanged | 0.783 +/- 0.082 |
| Frame-to-frame registration | 0.916 +/- 0.032 |

Mean Dice over all frames: 0.945 +/- 0.019. Registration beat the baseline in
50/50 patients, and halved the standard deviation — it is more consistent across
patients, not only more accurate on average.

The baseline matters: without it, 0.916 would be an uninterpretable number.

Accuracy decays gradually toward ES because each step is applied to the previous
estimate, so small errors accumulate (drift), and each resampling of the mask adds
a little quantization error.

**Open question:** the no-registration baseline declines almost perfectly
linearly across frames. Real wall motion is not that uniform, which raises the
question of how the intermediate-frame masks were produced. Not investigated yet.

### Direct ES segmentation, tested as an alternative
If a model can segment the ES frame directly, propagation is unnecessary for EF
(which needs only two frames, not the whole cycle). Tested on the same 50
patients:

| | Propagated | Direct |
|---|---|---|
| ES Dice, 4CH | 0.903 +/- 0.036 | 0.912 +/- **0.067** |
| ES Dice, 2CH | 0.878 +/- 0.068 | 0.891 +/- **0.107** |

Direct segmentation has the slightly better mean and roughly double the spread.
Its failures are severe: one patient scored 0.548 and produced an EF of 76%
against a reference of 53%; another produced 91.6% against 57%, a value that is
barely physiological.

Propagation cannot fail that badly, because it starts from a good ED mask and
only deforms it. **Propagation was kept**, and the same reasoning applies as in
the mono-plane vs biplane comparison: stability beats a marginally better mean.

Propagation is also still required for anything that needs the intermediate
frames, such as a volume-time curve.

---

## Module 4: Volume and ejection fraction

### First result, using ground-truth masks

| Quantity | Value |
|---|---|
| EDV | 111.0 mL |
| ESV | 41.3 mL |
| EF (computed, 4CH only) | 62.8 % |
| EF (CAMUS reference) | 54 % |

Ground-truth masks were used deliberately, to test the volume calculation in
isolation before mixing in segmentation or registration error.

The volumes are physiologically plausible, so the method is not grossly wrong,
but an 8.8-point EF gap needed explaining.

### Hypothesis 1, tested and rejected: a tilted long axis
Simpson's method treats each image row as a disk perpendicular to the LV long
axis. If the LV were tilted, row widths would overestimate the true diameters —
and since diameter is squared, the volume error would grow quickly.

Tested by plotting the LV centre line per row and the row-width profile. The
centre line was nearly vertical and the width profile smooth. **Rejected.**

### Hypothesis 2, tested and rejected: mono-plane vs biplane
The reference EF is identical (54) in both view files for patient0001 even
though the frame counts differ (18 vs 20), confirming it is a biplane
measurement combining 2CH and 4CH.

The mono-plane calculation uses the 4CH width as *both* diameters of each disk
(a circular cross-section). The clinical biplane method uses one diameter from
each view (an elliptical cross-section). The expectation was that this would
close the gap.

Result for the same patient: EDV 90.3 mL, ESV 31.3 mL, EF **65.3 %** — the
volumes dropped as predicted, but EF moved *further* from the reference.

**Why:** EF is a ratio. Any change that scales ED and ES proportionally leaves
it unchanged. Our ESV/EDV ratio is 0.35 against the reference's implied 0.46,
so whatever is wrong affects ES differently from ED. **Rejected as the
explanation for the gap** — though the comparison below shows biplane is still
the better method, for a different reason.

**Lesson:** when the target metric is a ratio, ask whether a candidate fix
affects numerator and denominator differently. If it does not, it cannot move
the metric.

### Comparing the two methods across 50 validation patients

| | Mono-plane | Biplane |
|---|---|---|
| Bias | +4.1 % | +8.1 % |
| SD of difference | 8.3 % | **3.5 %** |
| Mean absolute error | 7.2 % | 8.1 % |
| Correlation with reference | 0.837 | **0.971** |
| Same Normal/Reduced class | 42/50 | 39/50 |

Mono-plane looks better on bias alone, but that is misleading. Its SD is more
than twice as large and its correlation is far weaker — on patient0291 it gave
20.6 % against a reference of 57.0 %, a 36-point error that would read as severe
heart failure rather than normal function. Biplane gave 65.6 % for the same
patient.

**Biplane was chosen.** Systematic error can be corrected; random error cannot.
Its worse class agreement is purely a consequence of the larger bias pushing
borderline patients across the 50 % threshold.

**Lesson:** a single headline metric can rank two methods the wrong way round.
Bias, spread, correlation and the clinically relevant decision each answer a
different question.

### Calibrating the systematic offset
The offset was measured on **training** patients (+6.33 %) and applied as a fixed
constant to validation. Fitting the correction on the validation set and then
reporting improvement on that same set would have been circular.

Validation results, both using ground-truth masks:

| | Before | After |
|---|---|---|
| Bias | +8.1 % | +1.7 % |
| SD of difference | 3.5 % | 3.5 % |
| Mean absolute error | 8.1 % | **3.2 %** |
| Same Normal/Reduced class | 39/50 | **47/50** |

Two details worth noting:

- **SD did not change.** A constant offset shifts the whole distribution; it
  cannot reduce scatter. If SD had improved, something would have been wrong.
- **Residual bias is +1.7, not 0.** The training offset (+6.33) differs from the
  validation bias (+8.1) by about 1.8 points, so the correction does not transfer
  perfectly between patient groups.

**This offset did not transfer to the full pipeline.** It was measured with
ground-truth masks; model-predicted masks behave differently, particularly at
ES, and applying it to pipeline output produced a large *negative* bias instead.
The calibration had to be re-measured under the conditions it would be used in.

**Lesson:** a correction is only valid under the conditions it was fitted in.
Ground-truth masks and predicted masks are different conditions.

### Hypothesis 3, tested and rejected: disks along the wrong axis
Simpson's method stacks disks perpendicular to the LV long axis, but this
implementation stacks them along image rows, which is only correct if the
ventricle happens to be vertical in the image. Ferraz et al. (2022) measured
this assumption directly and report that re-orienting the disks improves both
accuracy and precision, which made it the strongest remaining candidate.

Implemented by detecting the axis and rotating the mask upright, then reusing
the existing row-based calculation — equivalent to tilting the disks, and much
simpler than resampling along an arbitrary direction.

Finding the axis took three attempts, and the first two failed the same way:

1. **Lowest LV row as the base.** When the ventricle is tilted, that row touches
   only one corner of the mitral plane, not its middle. ES tilt came out at
   23.1°, far more than the visible tilt.
2. **Annulus from the LV/atrium boundary, apex as the furthest point from it.**
   The base was now correct, but the apex landed on a corner of the rounded
   dome, and the ED tilt jumped from 4.8° to 19.9°.
3. **A least-squares line through the per-row LV centres.** Correct: ED 14.1°,
   ES 11.1°, with both endpoints on the centre line.

**Lesson:** an extreme point of a wide, rounded shape lands on a corner, not on
the axis. Fitting through the whole shape lets every row vote, so no single
pixel can tilt the result.

With the axis found correctly, the result was still negative:

| | Image-aligned disks | Axis-aligned disks |
|---|---|---|
| EDV | 90.3 mL | 95.6 mL |
| ESV | 31.3 mL | 32.5 mL |
| EF | 65.3 % | 66.0 % |

Reference: 54 %. The rotation moved EF *further* away.

The reason is the lesson from Hypothesis 2, again: rotation scaled both volumes
by similar amounts (EDV +5.9 %, ESV +3.8 %), and EF is a ratio. Nearest-neighbour
rotation also enlarges masks slightly through edge stair-stepping, which is why
both volumes grew rather than shrank.

**Three hypotheses tested, three rejected.** What remains is the definition of
the LV base in the masks, and the possibility that the reference EF was measured
with clinical software rather than derived from these masks at all — in which
case no formula applied to them reproduces it exactly. Calibration was kept as
the practical answer, with the bias reported openly.

---

## A wrong assumption that survived four modules

### The symptom
Running the pipeline end to end produced an EF bias of -9.4 %, and one patient
came out with a *negative* ejection fraction — physically impossible, since it
means the computed ES volume exceeded the ED volume.

### The investigation
Plotting LV volume across the cardiac cycle for that patient showed the curve
rising from 35 mL to 84 mL — for the **ground-truth** masks, not only ours. The
heart appeared to be filling rather than contracting.

The first hypothesis was that the volume formula was wrong. Counting raw LV
pixels per frame ruled that out: the area also grew from frame 0 to the last
frame, with no formula involved.

Checking `Info_4CH.cfg` for that patient gave the answer:

```
ED frame = 19,  ES frame = 1,  NbFrame = 19
```

The sequence ran **ES to ED**, the reverse of what every script assumed.

### Why it went unnoticed for so long
Early on, the ED/ES ordering *was* verified — `half_sequence` frame 0 was
compared with the standalone ED file using `np.array_equal`, and the last frame
with the ES file. Both matched. But that was done for `patient0001` only, and
the result was then treated as a property of the dataset rather than of one
patient. Every module built afterwards inherited the assumption.

An intermediate check made it worse: a script counting the ordering across
splits reported zero reversed sequences, which seemed to settle the question.
It had only been run on validation. Re-running it on training gave:

| Split | ED first | ES first |
|---|---|---|
| training | 781 | **19** |
| validation | 100 | 0 |

19 of 800 training sequences are reversed (2.4 %), and the ordering can differ
between the two views of the *same* patient — patient0006 is reversed in 4CH but
normal in 2CH.

### What it did and did not affect
- **Validation results were not affected**: that split contains no reversed
  sequences, so the segmentation and registration numbers reported above stand.
- **The EF calibration was affected**: it was measured on training patients,
  which do contain reversed sequences, and patient0006 fell within the first 50.

### The fix
Two functions in `loader.py`: one reads the true ED and ES frame numbers from
the cfg file, the other loads a sequence and reverses it when needed, so every
downstream step can safely assume index 0 is ED.

Verified across all 900 sequences: exactly one exception remains
(`patient0185`, 2CH), where the cfg says ED is first but the LV area is larger
in the last frame. Either a labelling inconsistency or an unusual case; noted
and left alone.

### Lessons
- **A check on one sample is not a check on the dataset.** The original
  verification was sound; generalising from it was not.
- **A negative result from a verification script deserves the same scrutiny as
  a positive one.** The "zero reversed sequences" output was believed without
  asking which split it had run on.
- **Metadata that is read should be used.** The cfg file was parsed early on,
  and the ED/ES fields were printed and discussed — and then not used. The
  correct answer was sitting in the data the whole time.
- **Physically impossible outputs are worth chasing.** A negative ejection
  fraction could have been dismissed as a bad segmentation on a hard case. It
  was the visible end of a bug affecting an entire split.

---

## Extending to both views

Segmentation was originally trained on 4CH ED frames only. Biplane EF needs an LV
mask in both views, so the preprocessing cache was rebuilt with the view in the
filename (`patient0001_4CH.npz`, `patient0001_2CH.npz`) and the model retrained
on both views together — 800 training samples instead of 400.

The previous 4CH-only checkpoint was kept rather than overwritten, so the two
could be compared directly on the same 4CH validation data:

| Structure | 4CH-only model | Both-views model |
|---|---|---|
| LV cavity | 0.937 +/- 0.032 | **0.947 +/- 0.027** |
| Myocardium | 0.852 +/- 0.063 | **0.871 +/- 0.041** |
| Left atrium | 0.838 +/- 0.087 | **0.866 +/- 0.080** |

Training on both views improved accuracy on 4CH alone, in every structure, with
lower variance — more data and more view diversity helped, even though the task
itself became harder.

**Comparison note:** these two models must be compared on 4CH data only. The
both-views model is evaluated on 100 sequences rather than 50, so a headline
"combined" number would change both the model and the evaluation set at once.

---

## Trying a transformer: PVTv2 with an attention-gated decoder

### The motivation
The receptive-field failure earlier in the project is a structural property of
CNNs: context is built up layer by layer. Self-attention gives every patch
access to the whole image from the first stage. The left atrium — the weakest
and least stable structure for the U-Net — is exactly the kind of region where
local texture is ambiguous and global position should help.

The encoder is PVTv2 rather than a plain ViT because a U-Net decoder needs
feature maps at several resolutions, which PVTv2's pyramid structure provides
and a plain ViT does not.

### A learning rate that was right for one half of the model
The first training run used a single lr = 1e-4 for the whole network, on the
reasoning that the encoder is pretrained and large updates would wash out its
ImageNet features. Loss fell far too slowly:

| epoch | U-Net | PVT at 1e-4 |
|---|---|---|
| 1 | 1.31 | 1.60 |
| 2 | 0.62 | 1.40 |
| 3 | 0.40 | 1.31 |
| 4 | 0.32 | 1.23 |

The reasoning was only half right. The encoder is pretrained; the decoder and
attention gates are not — they start from random weights and need to learn
quickly. One learning rate cannot serve both.

Fix: discriminative learning rates, 1e-4 for the encoder and 1e-3 for
everything else. Loss then fell normally, reaching 0.089 at epoch 15 against the
U-Net's 0.167.

**Lesson:** when part of a model is pretrained and part is not, they are two
different optimisation problems sharing one loop.

### Validation results

| Structure | View | U-Net | PVTv2 |
|---|---|---|---|
| LV cavity | 4CH | 0.947 | **0.951** |
| Myocardium | 4CH | **0.871 +/- 0.041** | 0.860 +/- 0.054 |
| Left atrium | 4CH | 0.866 +/- 0.080 | **0.895 +/- 0.051** |
| LV cavity | 2CH | 0.943 | **0.948** |
| Myocardium | 2CH | 0.865 | **0.869** |
| Left atrium | 2CH | 0.853 +/- **0.168** | **0.906 +/- 0.062** |

The prediction held. The left atrium in 2CH — the worst and least stable result
for the U-Net — improves most, with its standard deviation cut to a third.

The one regression is the myocardium in 4CH: a thin ring whose Dice depends
almost entirely on boundary precision.

---

## Chasing the myocardium: two hypotheses, one right

The transformer's only regression was the myocardium. Two explanations were
plausible, and they call for opposite fixes.

**Hypothesis A — an attention problem.** Dice and cross-entropy treat every
pixel equally, so most of the training signal comes from easy interior pixels.
Telling the model that boundaries matter more should help.

**Hypothesis B — an information problem.** The PVTv2 encoder reduces resolution
fourfold at its first stage. If the fine boundary detail is already gone, no
amount of emphasis can recover it, and only an architectural change would help.

### Testing A: a boundary-weighted loss
A weight map was built from the ground-truth mask: a morphological gradient
(max-pooling as dilation, minus its negative counterpart as erosion) marks the
edges, then a second pooling spreads the weight into a band around them. Errors
inside that band count three times as much.

The map was inspected visually before any training — weights ranged from 1.0 to
3.0, covering 10.7 % of pixels, and the band traced the mask edges exactly.
Verifying the tool first was worth it: a weight map that highlighted the wrong
pixels would have trained the model to care about the wrong thing, and the loss
curve would not have revealed it.

Result on the target structure: myocardium 0.860 -> **0.862**, a change of 0.002
against a standard deviation of 0.054. In 2CH the left atrium's spread got
*worse* (0.062 -> 0.103).

**Hypothesis A rejected**, and Hypothesis B looked confirmed. That conclusion was
recorded — and it turned out to be wrong.

### Testing B: reverse attention
The architectural fix came from RTA-Former (Li, Yi, Uneri, Niu & Jones, EMBC
2024), which targets exactly this problem in polyp segmentation. Its core idea
is one line: subtract the attention map from 1.

An ordinary attention gate asks "which regions matter?", and the model answers
with the structure's interior, because that is what drives most of the loss.
Inverting the map leaves the regions the model is *least* confident about —
which is where the boundary is. Those features are refined separately and added
back, so the decoder receives both the confident interior and a processed
boundary signal.

Two differences from the paper, stated for honesty: the paper reuses a
transformer stage inside the reverse branch, and adds a learnable feature
fusion. Neither was implemented in the first version.

| Structure (4CH) | U-Net | PVT | PVT + boundary loss | PVT + reverse attention |
|---|---|---|---|---|
| LV cavity | 0.947 | 0.951 | 0.948 | **0.951** |
| Myocardium | 0.871 | 0.860 | 0.862 | **0.872** |
| Left atrium | 0.866 | 0.895 | 0.901 | **0.902** |

**Hypothesis B was also wrong.** Reverse attention adds no resolution
whatsoever, yet it closed the myocardium gap and, for the first time, beat the
U-Net on that structure.

So the information was never missing. The decoder was simply not using it: the
ordinary attention gates were suppressing the periphery, which is precisely the
part the myocardium consists of.

**Lesson:** "the loss did not fix it, so the information must be missing" is a
tempting inference and an invalid one. A third possibility — the information is
present but discarded downstream — was not considered until a paper suggested it.

### Reproducing the paper's ablation
The paper reports that the transformer version of reverse attention gains
roughly twice what a convolutional one does, so the convolutional result alone
would have been ambiguous: a null result could mean "reverse attention does not
help here" or "we tested the weak variant". Both were therefore run.

| Model | Params | LV (4CH) | Myocardium (4CH) | Left atrium (4CH) |
|---|---|---|---|---|
| U-Net | 2.0 M | 0.947 | 0.871 | 0.866 |
| PVT + attention gate | 4.5 M | 0.951 | 0.860 | 0.895 |
| PVT + boundary loss | 4.5 M | 0.948 | 0.862 | 0.901 |
| PVT + reverse attention (conv) | 5.0 M | 0.951 | **0.872** | **0.902** |
| PVT + reverse attention (transformer) | 5.8 M | **0.952** | 0.872 | 0.901 |

The transformer variant gained nothing, and was slightly worse in 2CH
(myocardium 0.868 vs 0.875). This does not reproduce the paper's ablation, and
the differences in setup are worth stating: the reverse branch here uses a
self-attention block written for this project rather than a reused PVT stage,
attention is computed on a reduced 16x16 grid to stay affordable on CPU, the
paper used PVTv2-B5 (~250 M parameters) against B0 here, and the task is
different.

The null result does answer one question cleanly. Parameter counts rise at every
step of the table, so a natural objection is that the myocardium fix came from
capacity rather than mechanism. If that were true, 5.8 M should have beaten
5.0 M. It did not. **The gain came from reverse attention, not from size.**

The convolutional variant was chosen: equal accuracy, fewer parameters, simpler.

---

## End-to-end results: three architectures, one outcome

Every stage run together — model-predicted masks, propagated by registration,
no ground truth anywhere in the path.

| Metric | U-Net | PVT | PVT + RA |
|---|---|---|---|
| LV Dice, ED (4CH) | 0.946 | 0.951 | 0.951 |
| LV Dice, ES (4CH) | 0.903 | **0.908** | 0.900 |
| LV Dice, ED (2CH) | 0.943 | 0.947 | 0.946 |
| LV Dice, ES (2CH) | **0.878** | 0.870 | 0.868 |
| EF bias | -4.2 % | **-4.0 %** | -4.1 % |
| EF SD | 6.0 % | **5.6 %** | 5.7 % |
| EF mean absolute error | 5.3 % | 5.2 % | 5.2 % |
| Normal/Reduced agreement | 44/50 | 44/50 | 44/50 |

Three architectures, spanning 2.0 M to 5.0 M parameters and real differences in
segmentation quality, produce **identical clinical classification** and mean
absolute errors within 0.1 points of each other.

Two details make the reason concrete:

- The EF calibration offset measured on training was +1.13, +1.04 and +0.95 for
  the three models, and the standard deviation of the EF error was **exactly
  6.4 % for all three**. The random part of the EF error does not depend on the
  segmentation model at all.
- Reverse attention was the best model on segmentation, yet slightly *worse* on
  the propagated ES mask (0.900 vs 0.908). A better starting mask does not
  guarantee a better mask after nineteen registration steps.

The bottleneck was never segmentation. It is the geometric assumptions in
Simpson's method, and the accuracy lost while propagating the mask to ES.

**Lesson:** the component that is most interesting to improve is not necessarily
the one limiting the result. Measuring where the error actually comes from
should precede choosing what to optimise.

---

## The test set, used once

Every number above was measured on validation, and every decision — architecture,
loss, learning rates, propagation versus direct segmentation, mono-plane versus
biplane, the calibration method — was made by looking at those numbers. They are
therefore optimistic by an unknown amount. The test set exists to measure that
amount.

It was run once, with the pipeline frozen: PVTv2 + reverse attention, the
training-derived offset, no changes afterwards.

| Metric | Validation | Test |
|---|---|---|
| LV Dice, ED (4CH) | 0.951 +/- 0.022 | 0.946 +/- 0.020 |
| LV Dice, ES (4CH) | 0.900 +/- 0.035 | 0.882 +/- 0.051 |
| LV Dice, ED (2CH) | 0.946 +/- 0.023 | 0.935 +/- 0.042 |
| LV Dice, ES (2CH) | 0.868 +/- 0.067 | 0.871 +/- 0.053 |
| EF bias | -4.1 % | -4.7 % |
| EF SD | 5.7 % | 6.8 % |
| EF mean absolute error | 5.2 % | 6.6 % |
| Normal/Reduced agreement | 44/50 | **40/50** |

Segmentation barely moved. EF degraded more: the mean absolute error rose by 1.4
points and four more patients crossed the classification threshold the wrong way.

That pattern is consistent with where the tuning happened. Architecture choices
were made on segmentation Dice, and those transferred. The EF pipeline has more
free choices downstream — which variant of Simpson's method, how the calibration
offset is measured, propagation versus direct segmentation — and each was picked
by looking at validation EF. The optimism accumulated there.

One thing transferred cleanly: the calibration offset was fitted on training and
moved the bias by only 0.6 points between two entirely different cohorts
(-4.1 % to -4.7 %). Its failure mode is not instability but a persistent offset
from the training measurement, which the README states as a limitation.

**Lesson:** the size of the validation-to-test drop is itself a measurement. It
says how many decisions were made on validation and how much they were worth.

---

## Deployment: ONNX, quantization, Docker, CI

### Why
Until this point the model existed only as a PyTorch `.pt` checkpoint, and every
script that used it needed PyTorch and timm installed. In medical imaging a
model usually runs inside device or clinical software, not inside a Python
research environment. ONNX is a framework-independent format that ONNX Runtime
(and C++/C# runtimes) can execute, so exporting to it removes the PyTorch
dependency from inference.

### What was added
- `export_onnx.py` exports the PVTv2 + reverse-attention model (fixed 256x256
  input, dynamic batch) and immediately compares its output with PyTorch's on
  the same random inputs: largest logit difference and the fraction of pixels
  with an identical predicted class.
- `quantize_onnx.py` does static INT8 quantization (QDQ format, per-channel
  weights) calibrated on a random sample of real preprocessed training images.
- `benchmark_onnx.py` reports file size, CPU latency and per-structure Dice for
  PyTorch, ONNX FP32 and ONNX INT8.
- `infer.py` runs the whole pipeline on one patient from two NIfTI files and
  prints EDV, ESV, EF and the class as JSON.
- `Dockerfile` builds an inference image that contains only the inference code
  and ONNX Runtime; the model and the data are mounted, not baked in.
- CI now has three jobs: the fast unit tests, an ONNX job, and a Docker build.

### Decisions
- **Calibration data.** Static quantization measures activation ranges on a
  calibration set, so the set must look like the real inputs. Random noise gives
  the wrong ranges; it is allowed only as a smoke test (`--synthetic`).
- **Quantization is judged on Dice, not on speed.** A quantized model that is
  faster but segments worse is a regression, so the benchmark reports Dice
  against ground truth for every variant. INT8 is also only faster on CPUs with
  fast 8-bit integer instructions.
- **The EF offset belongs to the model it was measured for.** The +0.95 offset
  was measured for the FP32 PyTorch model. A quantized copy is a different model
  and would need its own offset, which is why `infer.py` takes `--ef-offset`.
- **CI uses random weights.** Checkpoints are not in the repository, so the CI
  job tests that the export computes the same function as PyTorch and that the
  scripts run. It says nothing about the trained model's accuracy.

### Found while writing it
- **A failed segmentation could be reported as "Normal".** If the model finds no
  LV, the EDV is 0, `ejection_fraction` returns NaN, and `classify_ef(NaN)`
  returns "Normal" because `NaN < 50` is False. `infer.py` now raises an error
  instead of printing a class. The same behaviour still exists in
  `evaluation/volume.py` and has not been changed there.
- **A reversed sequence is visible in the output.** If a sequence is fed in
  ES -> ED order, EF comes out negative, as it did in the bug described earlier.
  `infer.py` reads the ED/ES frame numbers from `Info_*.cfg` when given, and adds
  a warning to its output when the EF is negative.
- **`timm` was missing from `requirements.txt`**, although the model files
  import it. A fresh install would have failed.
- **`data/loader.py` contains absolute Windows paths** to the local dataset
  folder. `infer.py` does not depend on them, but they are in a public
  repository and should become a configuration setting.

### Results
Run on the trained PVTv2 + reverse-attention model, on 100 validation images
(50 patients x 2 views, ED frames), 12-thread Intel CPU, batch 1, median of 50
timed runs after 10 warm-up runs.

| Variant | Size (MB) | Median ms | p95 ms | LV cavity | Myocardium | Left atrium |
|---|---|---|---|---|---|---|
| PyTorch FP32 | - | 330.0 | 407.4 | 0.949 | 0.874 | 0.906 |
| ONNX Runtime FP32 | 20.23 | 50.2 | 63.8 | 0.949 | 0.874 | 0.906 |
| ONNX Runtime INT8 | 5.83 | 64.6 | 77.2 | 0.948 | 0.872 | 0.903 |

- ONNX FP32 reproduces PyTorch exactly: the predicted class is identical on
  100.00 % of pixels and the Dice values are the same to three decimals. It is
  about 6.6x faster.
- INT8 shrank the file 3.5x, but it was **slower** than FP32 (64.6 vs 50.2 ms)
  and cost 0.001-0.003 Dice per structure; it agrees with FP32 on 98.92 % of
  pixels. The expectation that quantization speeds up inference did not hold on
  this CPU. The cause was not investigated, so none is claimed.
- Only segmentation Dice was measured. The effect of INT8 on end-to-end EF was
  not, and its EF offset has not been re-measured.

### Status
Done on the real model: ONNX export (confirmed by the PyTorch-vs-ONNX
agreement above), INT8 quantization, and the benchmark. Not yet confirmed:
`infer.py` on a real patient, the Docker build, and the new CI jobs on GitHub.

**Lesson:** a deployment path that only exists on paper is not evidence. The
parity check turns "I exported it" into "the exported model computes the same
function". And an optimisation has to be measured on the hardware it will run
on: the quantized model was smaller and slightly less accurate, and also slower.

---

## Conventions adopted along the way

- **Change one thing at a time.** Every fix above was isolated, so its effect
  could be attributed.
- **Always compare against the simplest possible method.** A metric with no
  baseline is not evidence.
- **Look at the output, not only the metric.** Several problems in this project
  were visible in images before they were visible in numbers.
- **Test the measurement code before trusting it.** The volume formula was tested
  on ground-truth masks; the boundary weight map was inspected before training on
  it; the visualization was found to be wrong before the model was blamed.
- **Fit corrections on training data, report on validation — and fit them under
  the conditions they will be used in.**
- **Prefer stability over a marginally better mean.** Twice — biplane vs
  mono-plane, and propagation vs direct ES segmentation — the option with the
  worse headline number was chosen because its failures were smaller.
- **Prefer the simpler model when results are equal.** The convolutional reverse
  attention matched the transformer variant with fewer parameters.
- **A failed fix narrows the cause; it does not identify it.** Twice the
  conclusion drawn from a null result was too strong and later proved wrong.
- **Check that the end metric is sensitive to the component being improved**
  before spending effort on it.
- **Touch the test set once.** Everything else is validation.
- **Cache expensive steps.** Preprocessing writes one `.npz` per patient per view;
  trained models are saved to disk, so evaluation does not require retraining.
- **Long runs write results to CSV after every patient** and save a model
  checkpoint after every epoch, so hours of computation survive an interruption.
