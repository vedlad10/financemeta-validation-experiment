# Protocol diff — experiment 2 against the frozen reference

**Frozen reference:** `ec6c0df3772a6c510650bc7f7b0b306a7356cc2d` (`SPEC.md`, `config.json`, `experiment.py`).
**Preregistration:** `9899f44`. **This commit:** the successor protocol, `PROTOCOL.md` plus `config_experiment_2.json`.

Nothing in `ec6c0df` is edited. The frozen experiment and its results stand as published.

---

## 1. What changed, and why

| # | Frozen reference (`ec6c0df`) | Experiment 2 | Why |
|---|---|---|---|
| 1 | **Validation rows differ between arms.** Random uses shuffled 5-fold `KFold`; safe uses five expanding walk-forward blocks. The two arms score different rows. | **One set of validation rows, used by both arms** (§4). | A difference in score could previously come from scoring different periods. Now the only difference is which rows are allowed into training. |
| 2 | **Training sizes differ between arms.** Random trains on ~2,801 rows; safe on 1,750 → 3,147 rows, growing by fold. | **2,000 training rows in every fold of both arms** (§5). | Removes sample size as a competing explanation for any gap. |
| 3 | **Expanding window** for the safe arm: every fold starts at 2010-02-02. | **Rolling origin:** the window start advances 2010-02-02 → 2011-09-02 → 2013-04-11 → 2014-11-10 → 2016-06-14. | Matches deployment, where a model is refitted on a bounded recent history, and keeps the training size fixed. |
| 4 | **Embargo = 1 row** after each validation block, applied only to the safe arm. | **Embargo = 20 rows** after the block in the random arm, with the reason stated: 20 is the longest feature lookback. Purge stays 1, the label horizon. | The frozen value was explicitly "conservative in a forward-only design… fixed to make the boundary rule explicit". Once rows after the block become eligible, 1 row is too few: a row 5 days later still has features reaching across the block. |
| 5 | **Membership implied by row position.** | **Membership by `label_date`**, with `label_date` read from the trading calendar (§2). | A row whose features sit in 2025 but whose label lands in 2026 belongs to the lockbox. The calendar rule also prevents the off-by-one that drops the last development row. |
| 6 | **Refit cadence not stated separately.** | **Stated:** once per fold at each origin, plus one final refit before the lockbox (§6). | Requested explicitly for review. |
| 7 | **Primary metric pooled, selection also pooled** (same in both). | **Unchanged**, and now stated once and used everywhere: alpha selection, comparison, threshold (§7). | Prevents selecting on mean-fold R² while reporting pooled, or vice versa. |
| 8 | **Decision rule has two conditions** (validation gap ≥ 0.002 **and** optimism gap ≥ 0.002). | **One condition:** `delta >= 0.002` on the pooled primary statistic (§10). | The second condition needed the lockbox to decide the primary claim, which puts a lockbox number inside the primary decision. Experiment 2 decides on validation alone and uses the lockbox only as a one-shot report. |
| 9 | **Transaction costs** described as a context diagnostic. | **Unchanged in substance**, restated as explicitly secondary and non-rescuing (§8). | — |
| 10 | **Seed** `20260924`, one global seed. | Base seed `20260926`, plus `SeedSequence([seed, fold])` per fold (§9). | Per-fold streams keep folds independent and reproducible one at a time. |
| 11 | Lockbox 2024-01-01 → 2025-12-31. | Lockbox 2026-01-02 → 2026-08-31; the old lockbox period is now inside development. | The frozen lockbox has been read; it cannot serve as a lockbox again. |

## 2. What deliberately did not change

- The question, asset and target.
- All eleven features and the preprocessing, inherited by importing `make_dataset` from the frozen `experiment.py` rather than restating it.
- Model family `StandardScaler -> Ridge` and the grid `{0.1, 1.0, 10.0, 100.0}`.
- The primary metric, pooled zero-benchmark OOS R².
- The effect threshold, 0.002.
- The transaction-cost level, 5 bps per one-way unit of turnover, secondary only.
- The no-retuning rule and the negative-result reporting rules.

## 3. Files

| File | Role |
|---|---|
| `PROTOCOL.md` | the successor protocol |
| `config_experiment_2.json` | machine-readable parameters |
| `plan_folds.py` | enumerates windows and audits boundaries; fits nothing |
| `fold_plan.csv` | every window, with a membership digest per fold |
| `fold_plan_audit.txt` | the invariant checks, verbatim output |
| `PROTOCOL_DIFF.md` | this file |

## 4. Open points for the methods reviewer

1. **Fold 1 has zero slack** (`PROTOCOL.md` §12.2). Keep 2,000 training rows, or drop to 1,900 for a margin? Changing it alters the preregistered design, so it needs a decision rather than a default.
2. **Treatment intensity decays across folds** (887 → 654 → 434 → 214 → 0 rows drawn from after the block). Fold 5 contributes a zero-treatment comparison to a pooled statistic meant to measure the treatment. Options: keep pooled as primary and report per-fold delta as a diagnostic (current choice), or drop fold 5 from the primary pool (a design change, not adopted unilaterally).
3. **Embargo length.** 20 rows follows from the longest feature lookback. If the reviewer prefers the stricter `lookback + label horizon = 21`, that is a one-line change and should be settled before authorization.
