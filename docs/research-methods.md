# Research methods and presets

The existing `extract`, `train`, `Detector`, and schema-1 JSON models remain supported.
New handcrafted features are opt-in. Learned models use the optional `neural` extra
and the additive `NeuralDetector` / `train_neural` API. They also run through the
existing CLI and MLflow Projects launcher.

All presets use seed **1337**, automatic CPU selection and automatic training-image
sizing unless explicitly overridden. For measured results in best-first order and
required configuration fields, see [choose and run a method](method-catalog.md).

## Which method does what?

Each row has a `-standard.toml` and a `-deep.toml` file in `configs/methods/`.
Names identify feature families or complete pipelines; constituent statistics are
listed together when they share the same estimator.

| Config prefix | Computation | Standard → deep |
| --- | --- | --- |
| `gray_surface` | Differential box counts on grayscale intensity; fit log counts against negative log box size. Returns dimension and fit quality. | Five scales → six scales; automatic image size in both |
| `residual_surface` | The same surface estimator on a signed residual rescaled to `[0,1]`. | More scales at automatically selected resolution |
| `edge_box` | Threshold gradient magnitude, count occupied boxes, fit the support dimension. Includes an explicit validity flag. | More scales; explore `edge_quantile` separately |
| `renyi` | Fit mass entropies for orders 0, 1, 2 against scale to obtain generalized dimensions. | More scales |
| `lacunarity` | Variation of box mass: `1 + variance / mean²`. | Nonoverlapping boxes → every valid sliding window, using a summed-area table |
| `statistics` | Residual standard deviation and histogram Shannon entropy. These are supporting statistics, not fractal dimensions. | Same automatic size; `entropy_bins` remains 64 to avoid sparse histograms |
| `spectral` | Correlate the log Fourier magnitude with half- and quarter-image circular shifts. | Same automatic size; still a handcrafted statistic |
| `compression` | Match each range block against downsampled domain blocks under D4 transformations and a bounded affine intensity fit. | 64 candidate domains/stride 8 → 256/stride 4; all eight transformations |
| `gaussian_residual` | Subtract Gaussian-smoothed intensity, then extract the existing feature families except compression. | More scales; sigma remains configurable |
| `median_residual` | Subtract the local median, then measure the residual. | Median window 3 → 7 |
| `pca_residual` | Column-center each image matrix and remove its leading singular components. | Removed rank 8 → 28 |
| `multifractal` | Fit a partition function over positive and negative moment orders; derive the full spectrum. | Seven → nine orders, more spatial scales |
| `low_correlation` | Repeat dimension, entropy, mass-variation and multifractal measurements after removing several PCA ranks. | Rank 28 → ranks 24, 26, 28, 30, 32; sliding lacunarity |
| `combined` | Combine the preceding handcrafted families with PCA residuals and compression. | More measurements, denser domain search and stronger classifier regularization |
| `srm_fractal` | Fixed SRM filters → learned residual CNN → Fourier magnitude → recursive quadrant products/averages → linear classifier. | Two residual blocks → eight blocks; automatic image size in both |
| `median_fractal` | Median residual → normalized Fourier magnitude → learned Fractal Units → classifier. | Longer validation-controlled training; automatic image size in both |
| `gaussian_fractal` | The preceding neural pipeline with a fixed Gaussian residual. | Same training changes |
| `dncnn_fractal` | Jointly train a DnCNN residual predictor and the spectral Fractal-CNN. | Same changes; retain 17 layers and 64 denoiser channels |

These are starting points, **not established best settings**. “Deep” increases
measurement detail or training budget, not guaranteed accuracy. Both neural presets
keep three fractal levels; blindly increasing recursion can discard useful spatial
information. Handcrafted deep presets reduce logistic `C` from 1 to 0.1, increasing
regularization. Select using validation data and compare against the standard preset.

## Full multifractal spectrum

For each box side `s`, sum nonnegative residual magnitudes into masses and normalize
them to probabilities `p_i`. With `epsilon = s / image_size`, fit

```text
log(sum(p_i ** q)) = tau(q) * log(epsilon) + intercept
alpha(q) = derivative of tau with respect to q
f(alpha) = q * alpha - tau
D(q) = tau(q) / (q - 1)
```

At `q=1`, fit `sum(p_i * log(p_i))` directly instead of dividing by zero. The
derivative uses second-order finite differences on the configured order grid.
`logsumexp` prevents underflow/overflow in the partition sum. Empty boxes are excluded;
negative orders therefore describe the occupied support, not its empty surroundings.
An optional probability floor clips and renormalizes occupied masses. It changes the
estimator and should be recorded and checked for sensitivity; the default is zero.

The prediction vector contains each generalized dimension, spectrum width, peak
location/height, dimension span, minimum fit R² and a validity flag. Diagnostics retain
all orders, scales, partition curves, slopes, standard errors, occupancy, alpha and
spectrum values. An empty residual has zero-valued sentinel features and `valid=0`.
There is no interpretation of these sentinels as measured zero dimensions. A narrow
scale range or poor fit can make a seemingly precise spectrum unreliable.

## Mapping the supplied papers to code

### Xiao et al., arXiv 2503.08484v2

`networks.FractalNetwork(architecture="srm_fractal")` implements the structural
pipeline of equations 16–22: learn spatial representations after SRM filtering,
take their Fourier magnitude, batch-normalize it, multiply the four corresponding
quadrants and globally average the product at each level. Average the quadrants to
form the next level, concatenate level features, and learn a binary head.
[Paper](https://arxiv.org/abs/2503.08484v2).

Implementation choices are explicit: three conventional SRM high-pass filters on
grayscale input, a configurable residual backbone, and one sigmoid logit. The paper's
crop/pad protocol and much larger ProGAN training corpus are not reproduced here;
the package retains its existing center-crop/resize input contract. The standard
backbone is smaller; the deep preset uses eight residual blocks. Neither the existing
circular-shift correlations nor a fractal dimension is a substitute for this learned
network. The paper's benchmark score is not a score for this implementation.

### Das and Samuel, supplied SSRN 7291274

`residual_fractal` supports median, Gaussian and DnCNN residuals. DnCNN predicts the
residual directly and trains from scratch with the classifier. Its learning rate is
one tenth of the main rate by default. A Fractal Unit applies shared convolutions to
four quadrants, multiplies them, convolves the product, and pools it for classification.
Averaged quadrant responses are downsampled between levels. The classifier uses a
128-unit hidden layer and dropout. This also follows the authors' published
[implementation](https://github.com/dadoktor8/FractalCNN-ImageDetection/blob/main/Fractal_CNN.py).

Adam, weight decay, label smoothing, gradient clipping, cosine warm restarts and
validation-loss early stopping are configurable. Training-only augmentations apply
JPEG compression, blur and half-resolution resampling independently with probability
0.1 each. The JPEG minimum quality is configurable. The implementation uses a single
logit, equivalent to a two-class softmax score difference. The supplied paper has
inconsistent image-size descriptions; presets choose size from training images unless
explicitly overridden and do not claim exact numerical reproduction. No author checkpoint is downloaded.

### Xie et al., arXiv 2604.17268v1

`low_correlation` implements per-image matrix PCA residual analysis, including the
paper's rank range 24–32, fractal surface dimension, lacunarity, entropy and multifractal
descriptors. One SVD is reused across ranks. Retained energy and scale-fit diagnostics
make it possible to inspect what each rank removal did.
[Paper](https://arxiv.org/abs/2604.17268v1).

Our exact centering, residual rescaling, finite box estimator and empty-box policy are
specified above and in [the original methods guide](methods.md). These choices should
not be confused with a dataset-wide PCA basis or a guarantee of agreement with the
authors' numerical tables. Automatic face alignment/cropping is not included; supply
consistently prepared faces for a face-domain comparison. COCO versus AI-GenBench is
a broader, different task. The original grayscale dimension remains available for
comparison with residual dimensions.

## Parameters worth exploring

| Family | Small, useful exploration | What to inspect |
| --- | --- | --- |
| Residuals | Gaussian sigma 0.6, 1.0, 1.5; odd median windows 3, 5, 7; PCA ranks 8, 24, 28, 32 | Retained energy and stability after JPEG/resizing |
| Box dimensions | 128/256 px; scales 2–32/2–64; edge quantiles 0.65, 0.75, 0.85 | R² and slope errors, not only classifier accuracy |
| Multifractal | Orders −2…2 versus −4…4; zero versus a small mass floor | Sensitivity to tiny occupied masses and derivative endpoints |
| Lacunarity | Tiled versus sliding; sliding stride 1, 2, 4 | Extra cost and shift sensitivity |
| Compression | 64/128/256 domains; strides 8/4; contrasts 0.8/0.95 | Validation benefit per second of extraction |
| Linear classifier | `regularization_c`: 0.1, 1, 10 | Validation AUC, then log loss; feature count breaks remaining ties |
| Neural networks | Levels 2/3; channels 16/32; learning rates 2e-5/5e-5/1e-4 | Several seeds, validation loss, per-generator results |

Change one small, prespecified set at a time. Neural depth must fit the image: residual
Fractal Units halve each axis and also pool between levels. Invalid geometries are
rejected. `SearchConfig` accepts explicit feature candidates and regularization values,
limits a search to 100 trials, reuses extracted features across classifier settings,
and never reads `data.test`. The selected artifact records validation IDs/groups so
later evaluation cannot silently call that data independent test data. Search does
not yet automate neural hyperparameter sweeps.

## Run a preset

```console
uv sync --locked --extra neural --extra mlflow
uv run python examples/method_configs.py --train D:/datasets/train.csv --validation D:/datasets/validation.parquet
uv run fractal-detector run configs/methods/low_correlation-standard.toml
uv run fractal-detector run configs/methods/dncnn_fractal-deep.toml
```

Without uv, use `python -m pip install -e ".[neural,mlflow]"`, then run `python`
and `fractal-detector` directly. Generating presets writes all 36 files; keep manually
edited configurations in a separate directory. Relative paths resolve from each config
file's directory. Do not provide `--test` while comparing presets. Use a separate
`evaluate` task once the model and parameters are fixed.

Every preset uses `workers="auto"`, resolved from `os.process_cpu_count()` (24 logical
CPUs on the development machine). Image extraction limits BLAS to one thread per
image worker to avoid nested SVD thread pools. Neural training uses the resolved count
for PyTorch CPU operations and image loading; `device="auto"` selects CUDA only when
the installed PyTorch build supports it. The verified local installation is CPU-only.
For GPU installation, choose a wheel compatible with your driver using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). CPU and CUDA
training need not produce bit-identical weights. Lower batch size if GPU memory is
insufficient. Run training jobs serially within a process: thread limits and PyTorch
random-state controls are process-level resources.

Neural prediction accepts the same directory/Parquet/source iterators in bounded
batches. Training still materializes sample metadata, embedded bytes and handcrafted
measurement records, but decodes neural image tensors one batch at a time. For very
large training corpora, prefer path manifests; prediction is the streaming workflow.

## Python examples

```python
from fractal_detector import FeatureConfig, extract

measurement = extract(
    "image.jpg",
    FeatureConfig(
        residual="svd",
        svd_rank=28,
        compression=False,
        multifractal=True,
        low_correlation=True,
        pca_ranks=(24, 28, 32),
        lacunarity_mode="sliding",
    ),
)
print(measurement.diagnostics["multifractal"])
print(measurement.diagnostics["low_correlation"])
```

```python
from fractal_detector import NeuralConfig, read_manifest
from fractal_detector.neural import NeuralDetector, train_neural

result = train_neural(
    read_manifest("train.csv", require_labels=True),
    read_manifest("validation.csv", require_labels=True),
    neural_config=NeuralConfig(architecture="residual_fractal", residual="dncnn"),
    workers="auto",
)
result.detector.save("neural-model.json")
prediction = NeuralDetector.load("neural-model.json").predict("image.jpg")
```

Neural model files store validated metadata and checksummed safetensors bytes inside
JSON, not Python pickle. Prediction reports retain independently measured fractal
dimensions. Neural contributions sum exactly to the final head's log-odds, but describe
hidden activations; they are not an attribution of the score to measured dimensions.

For CLI neural prediction, add `[neural]` with `device="auto"` to an ordinary prediction
config and point `model` to a neural model. Architecture and preprocessing come from
the artifact; prediction-time neural settings other than `device` are ignored.
