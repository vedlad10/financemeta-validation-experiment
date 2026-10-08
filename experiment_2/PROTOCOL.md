# Experiment 2 — rolling-origin successor protocol

**Status: protocol only.** No experiment-2 model has been fitted, no validation or lockbox score exists, and no 2026 data has been downloaded or inspected. The result-bearing comparison stays on hold until this protocol is reviewed and authorized.

**Frozen reference:** `ec6c0df3772a6c510650bc7f7b0b306a7356cc2d`, unchanged. Nothing in experiment 2 edits that commit or its results.
**Preregistration this extends:** `9899f44`, "Preregister size-matched experiment 2 protocol".
**Artifacts in this commit:** this file, `config_experiment_2.json`, `plan_folds.py`, `fold_plan.csv`, `fold_plan_audit.txt`, `PROTOCOL_DIFF.md`.

---

## 1. Question and hypothesis

For the fixed AAPL next-day-return task, does a **size-matched random training protocol** report a materially higher validation score than a **leakage-safe rolling-origin protocol**, when both predict exactly the same validation rows using the same count of training rows, the same features, preprocessing, model family, grid and metric?

    delta = pooled_validation_oos_r2(random_selected) − pooled_validation_oos_r2(ordered_selected)

The hypothesis "temporal disorder is optimistic" is supported only if `delta >= 0.002`. It fails otherwise, including when the difference is zero or negative.

## 2. Availability boundary: feature date and label date

Every row has two dates, and the distinction governs every rule below.

| | meaning |
|---|---|
| `feature_date` *t* | all features are observed at or before the close of *t* |
| `label_date` | the next trading day; the label is `log(Close[label_date] / Close[t])` |

- A row belongs to development or lockbox **by `label_date`**, never by feature date.
- `label_date` is taken from the **trading calendar**, not from the next row of the feature table. Taking the successor row instead silently drops the final development row, which is how the first version of `plan_folds.py` lost exactly one row.
- Development: `label_date` from 2010-02-03 to 2025-12-31 inclusive — **4,003 rows**, verified against the cached raw file.
- Lockbox: `label_date` from 2026-01-02 to 2026-08-31 inclusive. Not downloaded. One download, after authorization, recorded by URL, retrieval timestamp, byte length and SHA-256 before any fitting.

## 3. Features, preprocessing, model family

Unchanged from the frozen reference and inherited by reference, not restated: eleven causal features, `StandardScaler -> Ridge`, `alpha ∈ {0.1, 1.0, 10.0, 100.0}`, scaler fitted inside each training fold only. `plan_folds.py` imports `make_dataset` from the frozen `experiment.py` so the feature definitions cannot drift from experiment 1 by restatement.

Longest feature lookback: **20 rows**. Label horizon: **1 row**. These two numbers determine the boundary exclusions in §5.

## 4. Validation blocks — the exact windows

The final 2,002 development rows form five disjoint, chronological blocks. **Both protocols predict exactly these rows.** Verified against the cached calendar:

| Fold | Rows | Validation feature dates | Validation label dates |
|---:|---:|---|---|
| 1 | 401 | 2018-01-12 → 2019-08-16 | 2018-01-16 → 2019-08-19 |
| 2 | 401 | 2019-08-19 → 2021-03-22 | 2019-08-20 → 2021-03-23 |
| 3 | 400 | 2021-03-23 → 2022-10-20 | 2021-03-24 → 2022-10-21 |
| 4 | 400 | 2022-10-21 → 2024-05-24 | 2022-10-24 → 2024-05-28 |
| 5 | 400 | 2024-05-28 → 2025-12-30 | 2024-05-29 → 2025-12-31 |

These reproduce the preregistered table exactly. A future discrepancy triggers the stop rule in §11, not a silent re-cut.

## 5. Training windows — the exact windows

Both arms use exactly **2,000 effective training rows per fold**.

### Ordered arm (rolling origin)

For each block: take the 2,001 most recent development rows whose feature dates precede the block's first feature date, then drop the single row adjacent to the block, because its label lands on that first validation feature date. The origin advances with each fold; the window rolls rather than expands.

| Fold | Training feature dates | Last training label | Rows from after the block begins |
|---:|---|---|---:|
| 1 | 2010-02-02 → 2018-01-10 | 2018-01-11 | 0 |
| 2 | 2011-09-02 → 2019-08-15 | 2019-08-16 | 0 |
| 3 | 2013-04-11 → 2021-03-19 | 2021-03-22 | 0 |
| 4 | 2014-11-10 → 2022-10-19 | 2022-10-20 | 0 |
| 5 | 2016-06-14 → 2024-05-23 | 2024-05-24 | 0 |

Asserted per fold, in code: `max(train label_date) < min(validation feature_date)` and `max(train feature_date) < min(validation feature_date)`.

### Random arm (the contrast under test)

Same validation block. From all development rows outside it, exclude:

- **purge = 1 row immediately before** the block: its label lands on the first validation feature date;
- **embargo = 20 rows immediately after** the block: their features reach back across the validation window.

The asymmetry is deliberate and follows from §3: before the block the only leak channel is the one-day label, after it the twenty-day feature lookback. Then draw exactly 2,000 rows uniformly without replacement from the remaining candidates. **Rows later than the block stay eligible: that future eligibility is the treatment being measured.**

### Boundary exclusions, both arms

| Side of block | Rows excluded | Reason |
|---|---|---|
| before | 1 | label horizon |
| after | 20 (random arm) | longest feature lookback |
| inside | all | validation rows are never trained on |

## 6. Refit cadence

- **Per fold:** each (arm, alpha) pipeline is refitted once at each of the five origins, on that fold's training rows only. Five fits per (arm, alpha) pair; **20 fits per arm** across the four-alpha grid; **40 total** before the two final refits. No pipeline is carried across folds.
- **Once before lockbox:** after `delta` is computed and stored, each arm's selected pipeline is refitted on the complete development sample. The ordered arm excludes the single purge row adjacent to the lockbox (training on 4,002 rows); the random arm trains on all 4,003 rows. This asymmetry arises because the ordered arm must maintain `max(train_label_date) < min(lockbox_feature_date)`, whereas the random arm has no temporal ordering to protect. The reviewer should confirm this is acceptable, or decide that both arms purge identically (see `PROTOCOL_DIFF.md` §5.4).
- Nothing is refitted after any lockbox number is seen.

## 7. Primary metric and selection

Pooled zero-benchmark out-of-sample R², over the concatenated predictions of all five blocks:

    pooled_oos_r2 = 1 − sum((y − yhat)^2) / sum(y^2)

Used for everything primary: alpha selection within each arm (ties take the larger alpha), the comparison, and the threshold. Mean-fold R², Pearson IC, sign accuracy, RMSE and fold dispersion are secondary diagnostics; none may select alpha or change the conclusion.

## 8. Transaction-cost treatment

Secondary, contextual, and incapable of changing the primary result. Daily position `sign(prediction)`, cost `0.0005 × abs(position[t] − position[t−1])` (5 bps per one-way unit of turnover), reported as annualized net Sharpe on the lockbox only. The cost level and the trading rule are fixed in advance and are not optimized. A favourable cost-adjusted number cannot rescue a failed primary test.

## 9. Seeds and determinism

| Purpose | Value |
|---|---|
| base seed | `20260926` |
| random-arm sampling, fold *k* | `numpy.random.default_rng(SeedSequence([20260926, k]))` |
| ordered arm | no stochastic component |
| Ridge | deterministic solver; no seed dependence |

Software is pinned in `config_experiment_2.json` (Python 3.11.0, NumPy 2.4.0, pandas 2.3.3, scikit-learn 1.8.0). Fold membership is recorded as a SHA-256 digest per fold in `fold_plan.csv`, so the draw can be checked without shipping index lists.

## 10. Failure criterion

The predeclared decision rule, applied to the primary statistic only:

- `delta >= 0.002` → the optimism hypothesis is **supported**.
- `delta < 0.002`, including zero and negative → the hypothesis **fails**, and that is reported unchanged.
- A negative validation or lockbox R² is reported unchanged.
- Missing data, an execution failure, or a violated invariant is reported as a **failed run**, not repaired after outcomes are seen.
- No substitution of asset, dates, features, model, grid, seed, threshold, cost or fold partition in response to a result.

## 11. Stop rule

If, after the authorized download, the development sample is not 4,003 rows, or any validation block boundary in §4 differs, or any assertion in §5 fails: **stop before fitting**, report the discrepancy, and register any remedy in a new commit. Do not re-cut windows to fit the data.

## 12. Known properties a reviewer should weigh

1. **The treatment intensity is not constant across folds.** The number of training rows drawn from after the validation block falls fold by fold: 887, 654, 434, 214, 0. By fold 5 the two arms draw zero future rows because the block ends at the development boundary. However, **fold 5 is not a pure ordering-only control**. Even with zero future rows, the random arm samples 2,000 rows from the entire pre-block span (feature dates 2010-02-03 through 2024-05-23), while the ordered arm takes the 2,000 most recent rows (2016-06-14 through 2024-05-23). In the index check the two arms share approximately 1,111 of 2,000 rows; the remaining ~889 rows differ. The random arm includes older data (2010–2016) that is absent from the ordered window, introducing a **training-composition confound**: any score difference in fold 5 comes from training on different market regimes (recency vs breadth), not from look-ahead. The pooled `delta` therefore averages five different treatment strengths, the last of which is zero look-ahead but non-zero composition difference. **Per-fold delta will be reported alongside the pooled statistic** as a diagnostic, without changing the primary rule.
2. **Fold 1 has zero slack.** Its ordered window begins at development row 0: 2,000 training rows plus 1 purge row exactly consume everything before the first block. One missing bar at the start makes fold 1 infeasible, which is a design fragility rather than a bug. Two mitigations are available if the reviewer prefers: reduce the per-fold training size to 1,900 (leaving ~100 rows of slack) or anchor the blocks by date instead of by row count. **Neither is adopted unilaterally**, because both change the preregistered design.
3. **Planning used the experiment-1 cache**, which ends 2025-12-31 and covers the whole development span. The lockbox calendar is unverified until the single authorized download.
4. **Embargo label-price overlap.** The 20-row embargo protects against features reaching back into the validation window, but the last validation label uses `Close[b+1]` (where `b` is the last validation feature index). The first admitted post-embargo row at `b+21` has `ret_20 = log(Close[b+21] / Close[b+1])`, which shares `Close[b+1]` with the last validation target. The current exclusion covers validation features but not validation labels. A 21-row embargo (`lookback + label_horizon = 20 + 1`) would remove this overlap. This is a review decision (see `PROTOCOL_DIFF.md` §5.3).

## 13. What is verified as of this commit

`python experiment_2/plan_folds.py` reproduces, from the cached raw file and the frozen feature code:

- 4,003 development rows, label dates 2010-02-03 → 2025-12-31;
- the five validation blocks in §4, matching the preregistration exactly;
- the five ordered training windows in §5;
- all seven boundary assertions per fold, passing;
- a per-fold count of random-arm rows drawn from after the block.

The script fits no model and computes no score. `fold_plan_audit.txt` is its output, verbatim.

## 14. Authorization gate

On written authorization from the review route, and not before: download the 2026 lockbox once, record its hashes, run the five folds for both arms across the grid, store fold membership and all alpha-level validation predictions, compute `delta`, apply §10, then evaluate the lockbox exactly once.

## 15. Provider cache preservation

The experiment-1 cache (`data/aapl_yahoo_chart.json`, SHA-256 `5d7b5348b28e6fab88d19f3e135910eeb6256ab3981242ece079e2dfc366067d`) covers 2010-01-01 through 2025-12-31 and is preserved unchanged. The 2026 lockbox download:

1. is stored as a separate file (e.g., `data/aapl_yahoo_chart_2026.json`), not overwriting the existing cache;
2. has its URL, retrieval timestamp, byte length, and SHA-256 recorded before any fitting;
3. is compared against the experiment-1 cache for overlapping dates — if any historical adjusted close differs by more than `1e-6` relative, this is flagged and recorded as a provider revision before fitting, not silently accepted.

This protects against retroactive historical-price revisions changing the development sample after the protocol and fold plan were verified.

## 16. Inference limits

The experiment uses:

- **one asset** (AAPL common stock) — the result cannot generalize to other equities, higher-frequency data, cross-sectional panels, or longer label horizons;
- **one model family** (Ridge regression) — more adaptive model searches may behave differently under temporal disorder;
- **one sampling seed per fold** (`SeedSequence([20260926, k])`) — a different seed produces a different random draw and potentially a different delta; the single-seed design provides no uncertainty interval around the observed delta;
- **one effect threshold** (0.002) — this is a pre-specified practical-significance boundary, not a statistical significance test; no p-value or confidence interval is computed.

The `.002` threshold tells the reader whether the observed gap exceeds a pre-committed minimum, but it does not quantify sampling variability. These limits must be stated in the final write-up and cannot be resolved by the experiment design itself.
