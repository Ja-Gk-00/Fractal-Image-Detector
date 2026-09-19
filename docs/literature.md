# Literature and implementation choices

The supplied `Literature` directory was reviewed before implementation. Source PDFs remain
outside the distributable package. These papers motivate experiments; their published
accuracy numbers are not measurements of this package.

| Supplied paper | Relevance and scope |
| --- | --- |
| [Yao et al., 1803.09403v2](https://arxiv.org/abs/1803.09403v2) | Uses high-pass residuals and a CNN to distinguish rendered graphics from photographs. Supports suppressing content before analysis. Its task and CNN differ from this detector. |
| [Xiao et al., 2503.08484v1](https://arxiv.org/abs/2503.08484v1) and [v2](https://arxiv.org/abs/2503.08484v2) | Studies repeated spectral structure induced by image generation and learns it with Fractal-CNN. Motivates residual spectra and repetition statistics here. Both provided versions were inspected; the implemented statistics are not its learned branch fusion network. |
| [Wang et al., 2504.09451v2](https://arxiv.org/abs/2504.09451v2) | FractalForensics embeds and recovers fractal watermarks to detect/localize changes. It requires proactive watermarking and cannot be directly applied to arbitrary unwatermarked user images. Watermark embedding/localization is outside this package. |
| [Mittal et al., 2511.07329v4](https://arxiv.org/abs/2511.07329v4) | Explores recursive neural architecture generation and evaluation. Useful context for experiment organization; it does not establish a passive image-origin criterion. |
| [Xie et al., 2604.17268v1](https://arxiv.org/abs/2604.17268v1) | Examines low-correlation residuals, fractal dimension, lacunarity, entropy, and multifractal quantities. Motivates the low-rank residual option and interpretable feature families. This package specifies its own finite-grid estimators and does not claim exact reproduction. |
| [Anderson and Farrell, Improving Fractal Pre-training, WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Anderson_Improving_Fractal_Pre-Training_WACV_2022_paper.html) | Uses generated fractal imagery for visual pretraining. Fractal structure is not exclusive to photographic or synthetic origins; generated textures are suitable mathematical/software fixtures, not a proxy for real detection validation. |
| [Das and Samuel, SSRN 7231443](https://ssrn.com/abstract=7231443) | Compares residual extraction and adds a learned denoiser to FractalCNN. The supplied preprint reports sensitivity to preprocessing and corruption. Motivates configurable residuals and independent robustness evaluation; no DnCNN or reported accuracy is reproduced. |

## Relationship to the reference package

[Fast-Fractal-Compression](https://github.com/Ja-Gk-00/Fast-Fractal-Compression) informs
the normalized array API, explicit backend selection, typed result objects, configuration
examples, C acceleration, uv dependency management, tests, and benchmark organization.
The new package uses a `src/` layout. Range/domain sizes, averaging, affine contrast fitting,
and D4 transform IDs remain recognizable. It has no runtime dependency on the reference
repository and does not read or write its compressed bitstream.

## Evidence still needed

The unit tests establish estimator behavior and software consistency. They do not establish
generalization to AI-generated images. For an empirical study, record generator versions,
camera/image sources, content groups, class counts, and all preprocessing. Keep exact and
near-duplicates, related prompts, subjects, and source images within one split. Reserve
unseen generators and independently sourced photographs for a final test. Report results
by generator and image source as well as aggregate metrics; repeat under resizing, JPEG
compression, denoising, and cropping. Compare individual feature families and a non-fractal
baseline using the same split. A small number of representative, independent tests is
more informative than reusing published scores from a different architecture.
