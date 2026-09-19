# How the methods work

Protocol schema: **1**. All models store their extraction configuration and feature order.
Changes to feature definitions require a schema increment and retraining.
For callable functions and their return values, see the [Python method reference](api.md).

## The main idea

A fractal dimension describes how measured detail changes when you look at smaller
scales. Imagine covering the edges in an image with square tiles. Halving each tile's
width needs about twice as many tiles for a line, giving dimension 1, or four times
as many for a filled area, giving dimension 2. The package repeats this measurement
at several scales and fits a line to the logarithms of scale and count.

For grayscale images, brightness becomes height: a flat-colored image is a flat
surface with dimension 2, while a rough surface can have a larger estimated dimension.
This surface dimension measures a different object from the dimension of image edges.
The reports keep both measurements, their counts at each scale, and the quality of
the fit so you can inspect what produced the estimate.

The detection pipeline combines several observations:

1. **Normalize the image** so samples use the same resolution and brightness range.
2. **Separate fine detail** by subtracting a blurred or low-rank reconstruction.
   This residual emphasizes small variations while reducing broad scene structure.
3. **Measure structure at several scales**: dimensions describe scaling, lacunarity
   describes uneven mass distribution, and entropy describes variation in residual values.
4. **Measure repetition** in the frequency representation and search for small image
   blocks that resemble transformed, downsampled larger blocks.
5. **Learn from labeled examples**. A standardized logistic regression combines these
   measurements into a synthetic-image probability. Each contribution is reported.

There is no universal fractal-dimension cutoff for AI images. Cameras, generators,
content and editing can all change these measurements. Training learns associations
in your dataset; independent evaluation checks whether those associations transfer.
Prediction reuses the exact measurement configuration saved with the trained model.

## Preprocessing and residuals

The default protocol orients file pixels using EXIF, composites alpha over white,
converts RGB to luminance with weights `(0.299, 0.587, 0.114)`, center-crops a square,
and resizes to 128×128 using bicubic interpolation. Values are clipped to `[0,1]`.
It does not linearize gamma or apply ICC color management. Use consistently decoded
arrays when your experiment requires a different color pipeline.

### Gaussian residual

With `residual="gaussian"`, blur the grayscale image using a Gaussian filter, then
subtract the blurred image from the original. The filter uses `gaussian_sigma=1.0`
by default and reflects pixels at the boundary. Increasing sigma makes the reference
smoother and leaves coarser variations in the residual.
The result emphasizes edges and fine texture; it also contains scene detail, not just noise.

### SVD residual

With `residual="svd"`, subtract each column's mean, factor the centered image as
`U × S × Vᵀ`, and reconstruct it using the largest `svd_rank` singular components.
Subtract that reconstruction from the centered image. Rank defaults to 8; increasing
it removes more of the image's dominant structure. This is a per-image low-rank
residual, not PCA fitted across a dataset.

### Residual normalization

Let `A = max(abs(residual))`. The residual surface is `residual/(2A) + 0.5`; its mass
measure is `abs(residual)/A`. Amplitudes at or below `1e-12` are treated as numerically
empty: the surface becomes constant 0.5, the mass becomes zero, and a diagnostic warning
is emitted. Residual standard deviation retains absolute amplitude independently of this
normalization. Empty mass dimensions/lacunarity use zero sentinels with explicit diagnostics.

## Fractal dimensions

All regressions use at least three scales and ordinary least squares. Default scales
are 2, 4, 8, 16, and 32 pixels. Scales must be increasing, distinct, divide the analysis
size, and be no larger than half the image width. Grids start at the top-left corner;
there is no padding or hidden selection of the most favorable scales.

`estimate_dimension` fits `log N(s) = D × log(1/s) + intercept`; the slope `D` is the
dimension. Zero counts are excluded from the fit. Fewer than three positive counts
produce an undefined estimate. Counts are always retained for inspection.

### Binary box counting

For each size `s`, count nonempty `s × s` tiles, obtaining `N(s)`. Regress
`log N(s)` on `log(1/s)`. This estimates the dimension of a binary set, between zero
and two in ideal cases. The detector constructs an edge mask from finite-difference
gradient magnitude strictly above its configured quantile (default 0.75) and a `1e-12`
floor. It does **not** box-count the entire nonzero grayscale image as a binary set.
The outputs are `edge_dimension`, its fit quality `edge_dimension_r2`, and the
validity flag below. A higher dimension means the selected edges occupy space more densely
across the measured scales.

An empty mask has an undefined dimension (`value=null`), not a measured dimension of
zero. The feature vector contains a zero sentinel and an `edge_dimension_valid` flag.
A single occupied point has dimension zero; a full line has dimension one; a filled
plane has dimension two. These and a Sierpinski carpet are tested analytically.

### Differential box counting

Grayscale and normalized residual surfaces use a vertical bin height `h = s/M`, where
`M` is image width. For each tile with intensity minimum `a` and maximum `b`, count
`floor(b/h) − floor(a/h) + 1` vertical boxes. Exact intensity 1 maps to the final valid
vertical bin. Sum over tiles and fit the same logarithmic regression.

This measures a surface in three-dimensional `(x,y,intensity)` space. A constant image
has dimension two. Rough surfaces generally approach three; finite estimates are not
clamped into a theoretical interval. Binary-set and grayscale-surface dimensions describe
different objects and should not be compared as if they were the same quantity.

Applied to grayscale, this produces `gray_dimension` and `gray_dimension_r2`.
Applied to the normalized residual surface, it produces `residual_dimension` and
`residual_dimension_r2`. The latter measures the shape of fine-detail variation after
its amplitude has been normalized.

Each `DimensionEstimate` stores its value, R², slope standard error, scales, counts,
and an undefined reason if applicable. The standard error is the regression residual
estimate, not a statistical confidence interval for image provenance. R² below 0.95
produces a fit warning. A good fit does not establish that an image is a mathematical fractal.

## Mass distribution and spectral features

### Lacunarity: unevenness of residual mass

Sum the normalized absolute residual inside each tile, including empty tiles. At each
scale compute `E[m²]/E[m]²`, where `m` is a tile's mass. For nonempty mass, a value of
1 means equal mass in every tile; larger values mean more uneven concentration.
`lacunarity_mean` and `lacunarity_std` summarize this curve across scales; the complete
curve is retained in diagnostics. Empty mass uses a zero sentinel. This is a fixed-grid
estimator, not a sliding-box estimator.

### Rényi dimensions: how residual mass spreads across scales

At each scale, divide each positive tile mass by the total mass to obtain probabilities
`p`. Compute the three entropy curves below using natural logarithms. Each reported
dimension is the slope of its entropy curve against `log(1/s)`.

| Feature | Entropy curve | What it measures |
| --- | --- | --- |
| `residual_d0` | `log(number of positive-mass boxes)` | Scaling of occupied area; all occupied tiles count equally. |
| `residual_d1` | `−sum(p log p)` | Scaling of the mass distribution, weighted by each tile's probability. |
| `residual_d2` | `−log(sum(p²))` | Scaling with greater emphasis on tiles holding concentrated mass. |

These are three Rényi dimensions, not a recovered continuous multifractal spectrum.
Empty residual mass gives zero curves and zero feature values, with `mass_empty=true`
in diagnostics. The stored `renyi_entropy_curves` contain `[H0,H1,H2]` for each scale.

### Residual standard deviation: strength of fine detail

`residual_std` is the population standard deviation of the signed residual **before**
amplitude normalization. It measures variation in original normalized-brightness units.
Two residuals can have the same fractal dimension while differing strongly in amplitude;
this feature preserves that difference.

### Residual entropy: variety of residual values

Bin the normalized residual surface into 64 equal-width bins on `[0,1]`. Convert bin
counts to probabilities and calculate `−sum(p × log2(p))`, ignoring empty bins.
`residual_entropy` is measured in bits: one occupied bin gives 0, and equal occupancy
of all 64 gives 6. It measures value diversity; it does not use the pixels' spatial order.

### Spectral self-similarity: repeated frequency patterns

For spectral self-similarity, compute `log(1 + abs(FFT2(residual)))`, zero its DC bin,
and subtract its mean. Correlate this array with horizontal and vertical circular shifts
of half and quarter of the image width; average the two axis correlations at each shift.
For the centered spectrum `A`, each correlation is `sum(A × shifted(A)) / sum(A²)`.
The outputs are `spectrum_similarity_half` and `spectrum_similarity_quarter`.
Values near 1 indicate similar patterns after the shift, values near 0 indicate little
linear similarity, and negative values indicate opposing deviations from the mean.
An energy denominator at or below `1e-20` produces a zero sentinel.
These measure translational spectral repetition. No conjugate reflection comparison is
used, since real images already have Hermitian Fourier symmetry. These handcrafted
features are inspired by spectral-fractal research; they do not reproduce Fractal-CNN.

## Range/domain self-similarity

The optional compression feature family follows the range/domain idea and normalized
array convention of [Fast-Fractal-Compression](https://github.com/Ja-Gk-00/Fast-Fractal-Compression).

1. Partition the image into nonoverlapping range blocks (default 8×8).
2. Enumerate double-sized domain blocks with configurable stride (default 8).
3. If there are too many domains, retain `max_domains` evenly spaced candidates in
   row-major order (default 64). Selection is deterministic.
4. Downsample each domain by averaging each 2×2 cell.
5. Apply the configured D4 transforms in FastFractal's canonical order.
6. For each range, minimize MSE of `s × domain + offset`, constraining `abs(s) <= s_clip`
   (default 0.95). A zero-variance domain uses `s=0`.

For range pixels `r` and candidate domain pixels `d`, the fitted contrast is
`s = clip(cov(d,r)/var(d), -s_clip, s_clip)` and the offset is
`mean(r) − s × mean(d)`. Domain variance at or below `1e-15` uses `s=0`.
Keep the candidate with the lowest mean squared pixel error. Low error means the range
is well explained by a transformed, smaller copy of a larger region.

| ID | Transform |
| --- | --- |
| 0 | Identity |
| 1 | 90° clockwise |
| 2 | 180° |
| 3 | 90° counterclockwise |
| 4 | Horizontal reflection |
| 5 | Transpose, then 180° |
| 6 | Vertical reflection |
| 7 | Transpose |

Features summarize MSE (mean, standard deviation, 90th percentile) and absolute contrast
(mean and standard deviation). Diagnostics retain every range's error, signed contrast,
offset, and chosen domain index. `domain_index // len(transform_ids)` selects the recorded
domain position; the remainder selects the transform. Range order is row-major.

The feature keys are `compression_mse_mean`, `compression_mse_std`,
`compression_mse_p90`, `compression_contrast_mean`, and `compression_contrast_std`.
The 90th percentile describes the more difficult block matches; contrast describes
how strongly the chosen domain's brightness variations are scaled. Setting
`compression=false` omits all five features, leaving 16 features instead of 21.

This is an independent fixed-partition self-similarity measurement, not an `.ffc` codec.
It does not implement FastFractal's quantization, quadtree splitting, entropy pruning,
top-k shortlist, or iterative decoding, and does not claim byte-format compatibility.
The affine error is a one-step fit error, not a decompression reconstruction error.

## Training and prediction

### Explicit dataset splits

`train_splits` uses your train/validation/test assignments directly. It rejects shared
decoded images or groups across splits, fits the scaler and classifier only on train,
and evaluates the optional validation/test splits independently. It performs no automatic
tuning, random holdout, or refit. [Data configurations](data.md) expose the same behavior.

### Holdout selection

`train` uses a seeded, class-stratified split when samples have no groups. With groups,
it tries up to 64 seeded group splits and takes the first containing both classes on
both sides. A group stays entirely on one side; `test_size` then refers to the fraction
of groups. Exact decoded duplicates are rejected. The held-out images are used only
for evaluation, and the returned model is fitted only on the training portion.

### Standardization and logistic regression

For each feature, learn its mean and standard deviation from the fitting samples,
then calculate `z = (value − mean) / scale`. Constant features use scale 1.
Logistic regression learns weights and an intercept by minimizing classification
log loss with regularization. Smaller `regularization_c` penalizes large weights more
strongly; `class_weight="balanced"` increases the influence of the less common class.
Fitting fails if it does not converge within `max_iter`.

Prediction repeats extraction with the saved settings and reuses the saved scaler:
`score = intercept + sum(weight × z)` and `probability = 1 / (1 + exp(−score))`.
The label is `synthetic` when that probability is at least `threshold` (default 0.5).

### Prediction explanation

Each `weight × z` term is the feature's contribution to the log-odds score. Contributions
are sorted by absolute magnitude; positive values favor synthetic and negative values
favor real. The text explanation includes the grayscale fractal dimension and the three
largest contributions. They explain the model's calculation, not the image's actual origin.

### Evaluation

`evaluate` predicts labeled images and compares results with their labels. It requires
both classes and rejects images or groups used for fitting. The metrics are:

| Metric | Calculation / meaning | Better direction |
| --- | --- | --- |
| Accuracy | Fraction of correctly classified images at the saved threshold. | Higher |
| Balanced accuracy | Mean of recall for real images and recall for synthetic images. | Higher |
| ROC-AUC | How well synthetic probabilities rank synthetic images above real images across thresholds. | Higher |
| Average precision | Precision averaged over increases in recall, treating synthetic as positive. | Higher |
| Log loss | Negative log probability assigned to the true class; confidently wrong predictions cost more. | Lower |
| Brier score | Mean squared difference between synthetic probability and the 0/1 label. | Lower |

The confusion matrix has actual labels in rows and predicted labels in columns, both
ordered `[real, synthetic]`. Probabilities are not separately calibrated. See
[usage](usage.md#train-save-predict-evaluate) for independent-test and split guidance.

## Performance and configuration

| Feature setting | Default | Constraints |
| --- | --- | --- |
| `image_size` | 128 | 32–1024; square analysis resolution |
| `scales` | `[2,4,8,16,32]` | At least three divisors of the analysis size |
| `residual` | `gaussian` | `gaussian` or `svd` |
| `gaussian_sigma` | 1.0 | `(0,10]` |
| `svd_rank` | 8 | Positive, smaller than analysis size |
| `edge_quantile` | 0.75 | `(0,1)` |
| `compression` | true | Include range/domain features |
| `block_size` | 8 | 2–32; divides analysis size; doubled domain fits |
| `stride` | 8 | Positive |
| `max_domains` | 64 | 1–1024, before transforms |
| `s_clip` | 0.95 | `(0,1)` |
| `transform_ids` | `[0,1,2,3,4,5,6,7]` | Distinct, nonempty subset |
| `backend` | `auto` | `auto`, `native`, `python` |
| `max_input_pixels` | 40,000,000 | Maximum decoded input pixel count |

Box statistics and the exhaustive affine search have C kernels. Buffers are validated
and copied before releasing the GIL; the C implementation does not depend on NumPy's
C ABI. NumPy/scikit-learn/SciPy already supply compiled FFT, SVD, filtering, and fitting.
The remaining Python code mainly handles orchestration and reporting, so mypyc is not
an additional build dependency. Benchmark before adding another compilation layer.

Affine search work is proportional to `ranges × candidate_domains × block_pixels`.
Larger images, small strides, and many domains can become expensive. The domain cap
controls search size; use the supplied benchmark to choose a budget. Thread workers
parallelize images. For SVD-heavy batches, limit BLAS threads (`OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) to avoid oversubscription. Floating-point
roundoff can vary across platforms; nearly tied domain matches may differ.
