# Local research validation

Measured on 24 September 2026 using seed **1337** and balanced COCO/AI-GenBench
data: **600 training, 200 validation and 200 test images**, each split balanced 1:1.
Decoded-image and source-group overlap checks passed. Extraction used all 24 logical
CPUs through `workers="auto"`, with one BLAS thread per worker.

## Validation ranking

Ten candidate configurations, each with C = 0.1, 1, 10, were fitted on training data
and selected by validation ROC-AUC. The search never reads test images. This table
shows the best C per candidate, strongest first. All 30 trials and provenance
digests are in [the JSON report](research-results.json).

| Rank | Configuration | C | Features | Validation AUC | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | `configs/ranked/01-gaussian_256.toml` | 10 | 29 | 0.9084 | 0.820 |
| 2 | `configs/ranked/02-combined_256.toml` | 1 | 59 | 0.8928 | 0.825 |
| 3 | `configs/ranked/03-gaussian_multifractal.toml` | 1 | 29 | 0.8219 | 0.760 |
| 4 | `configs/ranked/04-legacy.toml` | 1 | 21 | 0.8135 | 0.720 |
| 5 | `configs/ranked/05-gaussian_sigma_0_6.toml` | 10 | 29 | 0.8087 | 0.705 |
| 6 | `configs/ranked/06-median_multifractal.toml` | 0.1 | 29 | 0.8085 | 0.745 |
| 7 | `configs/ranked/07-gaussian_sigma_1_5.toml` | 1 | 29 | 0.8078 | 0.730 |
| 8 | `configs/ranked/08-low_correlation.toml` | 0.1 | 30 | 0.7791 | 0.735 |
| 9 | `configs/ranked/09-pca_rank_8.toml` | 0.1 | 29 | 0.7751 | 0.690 |
| 10 | `configs/ranked/10-pca_rank_28.toml` | 0.1 | 29 | 0.7568 | 0.705 |

The selected configuration uses 256 pixels, scales 2–64, Gaussian sigma 1,
multifractal features, no compression, and C=10. Its validation AUC is **0.9084**
and accuracy **0.820**. The 128-pixel baseline with C=1 achieves AUC **0.8135**
and accuracy **0.720**. More features were not automatically better: the 59-feature
PCA/sliding-lacunarity combination did not win.

These comparisons explicitly fix image size. They do not measure the new automatic
size presets or establish a ranking of isolated families and learned models.
See [all methods, arguments and commands](method-catalog.md).

## Resolution control and existing test split

A post-selection control uses the same 256-pixel processing and C=10 without
multifractal features: validation AUC **0.8863**, accuracy **0.815**, log loss
**0.4168**. Much of the gain is associated with resolution; adding multifractal
features improves AUC by 0.0221 in this comparison. This control did not change the
selection and is not a multi-seed significance test.

After selection, the existing 200-image test split yields **AUC 0.8913, accuracy
0.830, average precision 0.9029, log loss 0.4180 and Brier score 0.1281**. The
128-pixel baseline on that split yields AUC **0.8151**, accuracy **0.760**.
This test split was inspected in earlier notebook work, so this is an exploratory
comparison. Thirty trials on 200 validation images create selection optimism.
Higher resolution can expose source/compression differences; unseen-source testing,
confidence intervals and replication remain necessary.

## Reproduce

From the project directory, use the seed-1337 notebook sample receipt:

```console
uv run python examples/research_search.py outputs/notebook-demo/sample-1337-500-9be55ad838ce6753/preparation.json --output outputs/research-search-1337
```

This writes `model.json`, `search.json`, configuration, environment, split provenance
and structured extraction events. Validation IDs/groups used for selection are stored
in the model. To fit the chosen settings on your own data, edit `[data]` in
`configs/research-selected.toml`, then run:

```console
fractal-detector run configs/research-selected.toml
```

The files under `configs/ranked/` reproduce each candidate's best settings. The
standard/deep TOML presets under `configs/methods/` support broader exploration.

## Learned models

Tests train all four neural variants, check DnCNN gradients, reproducibility,
checkpoint roundtrips, explanations, train-only size selection and split separation.
Tiny neural training checks establish execution correctness, not an accuracy ranking.
Full standard/deep neural presets still need a larger training study. The supplied
papers' scores are not presented as local results. CUDA has not been validated with
the local CPU-only PyTorch wheel.

## Software checks

The complete suite passed **202 tests** with **97.54% line/branch coverage** after
the sizing, seed and TOML changes. Checks ran on Windows/Python 3.14.7. Cross-platform
CI is configured; hosted runners were not executed in this session.
