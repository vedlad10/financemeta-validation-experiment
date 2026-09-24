# FinanceMeta validation experiment — frozen one-page specification

**Question.** For one next-day equity-return task, does shuffled random cross-validation report a materially more favorable validation result than a leakage-safe purged walk-forward protocol when the feature set, preprocessing, model family, hyperparameter grid, and final lockbox are held constant?

**Asset, sample, and target.** Use vendor-adjusted daily OHLCV for Apple common stock (`AAPL.US`) from 2010-01-01 through 2025-12-31. A row is formed immediately after close on trading day *t*. Its target is the next-day log close return, `log(Close[t+1] / Close[t])`. The development sample ends 2023-12-31. Dates from 2024-01-01 through 2025-12-31 form a lockbox that is not read during feature, model, split, or threshold decisions.

**Causal features.** Use only information available by close *t*: cumulative log returns over 1, 2, 3, 5, 10, and 20 trading days; rolling 5- and 20-day standard deviation of daily log returns; `log(High/Low)`; `log(Open/previous Close)`; and the 20-day rolling z-score of log volume. Missing warm-up rows are dropped. Every transformation is fit inside each training fold through a `StandardScaler`; no full-sample normalization is permitted.

**Model family.** Ridge regression only: `StandardScaler -> Ridge`. The sole tuned hyperparameter is `alpha ∈ {0.1, 1, 10, 100}`. Both validation schemes receive the same rows, features, model code, grid, and deterministic seed.

**Validation schemes.** (1) **Random:** shuffled 5-fold `KFold`, seed 20260924. (2) **Safe:** five contiguous, expanding walk-forward folds. The first 50% of the development sample is the minimum training history; each validation block is strictly later than its training data. Because the label spans one future trading day, remove the final one training row before every validation block (**purge = 1**). Exclude the first row after each completed validation block from all later training/test membership (**embargo = 1**). This embargo is conservative in a forward-only design but is fixed here to make the boundary rule explicit. Assert in code that `max(train date) < min(validation date)` and that purged/embargoed rows are absent.

**Selection and final comparison.** Within each scheme, choose the alpha with the highest mean validation primary metric; ties choose the larger alpha. Refit that scheme's selected pipeline on all eligible development rows (the safe fit removes the final purge row) and evaluate exactly once on the common lockbox. Do not tune after viewing lockbox results.

**Primary metric.** Zero-benchmark out-of-sample \(R^2 = 1 - \sum(y-\hat y)^2 / \sum y^2\), pooled across out-of-fold predictions. Secondary diagnostics are Pearson information coefficient, sign accuracy, RMSE, and fold dispersion.

**Economic diagnostic.** The statistical forecast is primary, so transaction costs cannot rescue or invalidate the primary result. For context only, form a daily long/short position `sign(prediction)` and subtract **5 bps per one-way unit of turnover**: `0.0005 * abs(position[t] - position[t-1])`. Report annualized net Sharpe; do not optimize the cost or trading rule.

**Predeclared decision/failure rule.** The hypothesis “random splitting is optimistically biased” is supported only if both conditions hold: (a) random mean validation OOS \(R^2\) exceeds safe mean validation OOS \(R^2\) by at least **0.002**, and (b) random validation optimism—`validation R2 - lockbox R2`—exceeds safe validation optimism by at least **0.002**. Otherwise the hypothesis fails. A disappearance or reversal under the safe protocol is reported unchanged; no alternative asset, dates, features, costs, thresholds, seeds, or models may be substituted after the comparison.

**Reproducibility.** The run records the SHA-256 of `config.json`, raw-data hash/source, package versions, fold boundaries, all alpha-level validation scores, lockbox scores, and the decision-rule outcome. If live data download fails, an explicit synthetic regime-shift smoke test may verify the code path, but it cannot answer the empirical question and must be labeled as such.
