# FinanceMeta experiment 2 — preregistration only

**Status:** protocol only; no experiment-2 model has been fit and no experiment-2 validation or lockbox score has been computed. This file must exist in its own Git commit before any new empirical outcome is viewed. All code, data capture, scores, predictions, plots, and conclusions belong in later commits.

**Relationship to experiment 1.** Experiment 1 remains frozen at `ec6c0df3772a6c510650bc7f7b0b306a7356cc2d`. Nothing in this extension changes that commit or its result. Experiment 1 should be described as “protocol declared as frozen before outcome, but exact preregistration not independently timestamp-verifiable,” not as cryptographically preregistered.

## Primary question and hypothesis

For the fixed AAPL next-day-return task below, does a size-matched random training protocol report a materially higher validation score than a leakage-safe ordered protocol when both protocols use the same validation rows, exactly the same number of training rows, the same features, preprocessing, model family, hyperparameter grid, and metric?

The directional hypothesis is that temporal disorder is optimistic. Let

`delta = pooled_validation_oos_r2(random_selected) - pooled_validation_oos_r2(ordered_selected)`.

The hypothesis is supported only when `delta >= 0.002`. It fails otherwise, including when the difference is smaller, zero, or negative.

## Frozen data scope and label-availability boundary

- **Asset:** Apple Inc. common stock, NASDAQ ticker `AAPL` only. There is no primary multi-asset panel.
- **Provider:** Yahoo Finance chart endpoint, daily interval. Use adjusted close as `Close`; multiply raw `Open`, `High`, and `Low` by that date's `adjusted_close / raw_close` factor; retain reported volume without price-adjustment scaling.
- **Requested raw range:** 2010-01-01 through 2026-08-31, inclusive.
- **Supervised row:** features are observed after the close on trading day `t`; the label is the log close return from `t` to the next trading day, and `label_date` is that next trading day.
- **Development labels:** 2010-02-03 through 2025-12-31, inclusive. On the expected calendar this is 4,003 supervised rows, with feature dates 2010-02-02 through 2025-12-30.
- **One-time lockbox labels:** 2026-01-02 through 2026-08-31, inclusive. Their expected feature dates are 2025-12-31 through 2026-08-28. No 2026 price or score may be inspected before this preregistration commit.
- A row belongs to development or lockbox by `label_date`, never merely by feature date. No training label may cross into the lockbox.
- After this commit, the raw response may be downloaded exactly once and cached unchanged. Its URL, retrieval timestamp, byte length, and SHA-256 must be recorded before model fitting. If the expected calendar boundaries or 4,003-row development sample are not reproduced, stop without fitting and preregister any remedy in a new commit.

## Frozen features and preprocessing

Every feature uses information available by the close of feature date `t` only:

1. cumulative log close returns over 1, 2, 3, 5, 10, and 20 trading days;
2. rolling standard deviation of one-day log returns over 5 and 20 trading days, with `ddof=1`;
3. `log(High[t] / Low[t])`;
4. `log(Open[t] / Close[t-1])`;
5. the 20-day rolling z-score of `log1p(Volume)`, with rolling standard deviation `ddof=1`.

Warm-up rows with missing or non-finite values are dropped. A `StandardScaler` is fit independently inside each training fold and is never fit on validation or lockbox rows.

## Frozen model family and grid

The only model family is `StandardScaler -> Ridge regression`. The only tuned hyperparameter is

`alpha in {0.1, 1.0, 10.0, 100.0}`.

The deterministic base seed is `20260926`. Software is frozen to Python 3.11.0, NumPy 2.4.0, pandas 2.3.3, and scikit-learn 1.8.0. No additional model family, feature, target, asset, seed search, or hyperparameter value is permitted in the primary test.

## Common validation rows

The final 2,002 development rows are divided into five disjoint, chronological validation blocks. Both protocols predict exactly these rows:

| Fold | Rows | Feature dates | Label dates |
|---:|---:|---|---|
| 1 | 401 | 2018-01-12 to 2019-08-16 | 2018-01-16 to 2019-08-19 |
| 2 | 401 | 2019-08-19 to 2021-03-22 | 2019-08-20 to 2021-03-23 |
| 3 | 400 | 2021-03-23 to 2022-10-20 | 2021-03-24 to 2022-10-21 |
| 4 | 400 | 2022-10-21 to 2024-05-24 | 2022-10-24 to 2024-05-28 |
| 5 | 400 | 2024-05-28 to 2025-12-30 | 2024-05-29 to 2025-12-31 |

The dates and row counts are fixed. A data discrepancy triggers the stop rule above rather than an automatic boundary change.

## Size-matched training protocols

Both protocols use exactly **2,000 effective training rows in every fold**.

### Ordered protocol

For each validation block, take the 2,001 most recent eligible development rows whose feature dates precede the first validation feature date. Purge the immediately preceding row because its one-day label becomes available on the first validation feature date. The remaining 2,000 rows form a strictly earlier rolling training window. Assert `max(train_label_date) < min(validation_feature_date)` and `max(train_feature_date) < min(validation_feature_date)`.

### Random protocol

Use the identical validation block. From all development rows outside that block, exclude:

- the same one-row pre-block purge boundary; and
- the first 20 feature rows after the block as an embargo, because 20 days is the longest feature lookback.

Uniformly sample exactly 2,000 rows without replacement from the remaining candidates. Sampling for fold `k` uses NumPy `SeedSequence([20260926, k])`; sampled indices are sorted only for storage, not treated as chronological. Rows later than the validation block are intentionally eligible after the embargo. This future eligibility is the protocol difference being tested.

No resampling is allowed after scores are viewed. Fold membership and boundary audits must be written before metric calculation.

## One primary validation statistic

The sole primary validation statistic is zero-benchmark pooled out-of-sample R-squared over the concatenated predictions from all five common validation blocks:

`pooled_oos_r2 = 1 - sum((y - prediction)^2) / sum(y^2)`.

This same statistic is used everywhere in the primary analysis:

1. For each protocol, select the alpha with the highest `pooled_oos_r2` across all of its validation predictions.
2. Ties select the larger alpha.
3. Compare the selected protocols using `delta` defined above.
4. Apply the fixed `+0.002` decision threshold to that same `delta`.

Mean-fold R-squared may be reported only as a secondary diagnostic. It cannot select alpha, determine the primary conclusion, or replace pooled R-squared.

## Lockbox and no-retuning rule

After fold membership, all alpha-level validation predictions, selected alphas, and the primary `delta` are stored, each selected pipeline is refit on the same complete development sample. The 2026 lockbox is then evaluated exactly once. Lockbox metrics are pooled zero-benchmark OOS R-squared, RMSE, Pearson information coefficient, sign accuracy, and an explicitly secondary long/short diagnostic using 5 basis points per one-way unit of turnover.

Lockbox results cannot change alpha selection, features, preprocessing, training windows, purge/embargo rules, threshold, cost, dates, or the primary conclusion. There is no post-lockbox retuning.

## Negative-result and reporting rules

- If `delta < 0.002`, report that the preregistered optimism hypothesis failed.
- A zero or negative `delta`, negative validation R-squared, or negative lockbox R-squared must be retained unchanged.
- A favorable secondary metric cannot rescue a failed primary test.
- A favorable `delta` does not make random splitting temporally valid and does not establish a profitable signal.
- Missing data, execution failure, or an invariant violation is reported as a failed run, not repaired after outcomes are inspected.
- No alternative asset, date window, feature, model, alpha grid, seed, threshold, fold partition, or benchmark may be substituted in response to the result.

## Separately planned robustness extension

The robustness extension is not part of the primary test and cannot change its conclusion. Only after the primary result is committed unchanged, repeat the same five validation blocks and every other rule with **1,500 effective training rows per fold** instead of 2,000 (ordered candidate window 1,501 followed by the same one-row purge; random sample exactly 1,500). Report its pooled R-squared `delta` as sensitivity to training-window length. Do not pool it with the primary test, use it for alpha selection in the primary test, or present it as a replacement result.

## Commit separation requirement

This preregistration commit must contain no experiment-2 source code that computes scores, no new empirical data, no predictions, no plots, and no result summary. The first commit containing any experiment-2 outcome must be a later descendant of this preregistration commit and must cite its SHA.
