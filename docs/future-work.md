# Future work

The highest-value improvements concern evaluation and preservation of forensic traces,
not simply adding more feature columns.

1. **Independent generator and source holdouts.** Hold out whole generators, capture
   sources, subjects and related prompts. Add perceptual duplicate detection alongside
   decoded-image hashes and supplied groups. Match real/fake resolution and compression
   distributions so that acquisition differences do not dominate the score.
2. **A larger learned-model study.** Train both Fractal-CNN variants on substantially
   larger balanced corpora, use several seeds and group-aware validation, and compare
   against the regularized handcrafted baseline. Report uncertainty and per-generator
   errors. A passing training test does not demonstrate the papers' reported accuracy.
3. **Preserve native-resolution traces.** Add configurable crop/pad and multi-patch
   aggregation, with split-safe patch grouping. Compare these against the current
   center-crop/resize contract before changing defaults. Add opt-in face alignment for
   comparisons to face-domain literature, including failures and crop provenance.
4. **Robustness and calibration.** Evaluate fixed JPEG, blur, resampling and screenshot
   perturbations on independent data. Fit probability calibration and operational
   thresholds using a separate calibration split. Report false-positive rates at
   selected operating points and allow abstention for out-of-distribution images.
5. **More reliable fractal estimation.** Compare grid offsets, scale-window stability,
   detrended multifractal estimators and direct spectrum estimators. Bootstrap spatial
   blocks to estimate uncertainty. Study negative-order sensitivity, empty boxes and
   intensity normalization explicitly; reject unreliable fits instead of treating
   all finite dimensions as equally informative.
6. **Neural parameter search.** Add bounded, multi-seed searches with early trial
   termination, GPU scheduling and checkpoint resume. Keep validation used for early
   stopping separate from final selection where data permits. The current automated
   search covers handcrafted features and logistic regularization only.
7. **Training at dataset scale.** Replace materialized embedded-byte samples with a
   lazy indexed Parquet dataset, persist feature caches keyed by image and config
   hashes, and checkpoint extraction. Prediction already streams; training metadata
   and measurement storage are still proportional to dataset size.
8. **Profile before compiling more code.** Existing box statistics and affine matching
   use C; new partition functions, integral images, SVD and neural layers use compiled
   NumPy/SciPy/PyTorch operations. Benchmark selective feature computation, multiscale
   kernel reuse and GPU transfer overlap. Add another C kernel or mypyc only when a
   measured Python bottleneck justifies its maintenance and build cost.
9. **Better neural explanations.** Add carefully validated perturbation and attribution
   maps, compare them with measured residual structure, and test their stability.
   Current final-layer contributions are exact algebraic decompositions but do not
   explain which image structures caused a decision.

Each improvement should include an ablation with the same frozen splits, preprocessing
and compute budget. Retain the smaller model when extra complexity has no consistent
validation benefit.
