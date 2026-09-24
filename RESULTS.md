# Frozen empirical result

Run status: **empirical result; predeclared hypothesis failed**. No asset, period, feature, model, cost, split, threshold, or seed was changed after the lockbox was viewed.

| Quantity | Random 5-fold | Purged walk-forward |
|---|---:|---:|
| Selected Ridge alpha | 100 | 100 |
| Mean validation OOS R² | -0.012781 | -0.004347 |
| Pooled validation OOS R² | -0.011781 | -0.008369 |
| 2024–2025 lockbox OOS R² | -0.007703 | -0.007527 |
| Lockbox information coefficient | -0.019319 | -0.017540 |
| Lockbox sign accuracy | 51.70% | 51.70% |

The random-minus-safe mean validation difference was **-0.008434**, opposite the hypothesized optimistic random-split advantage and below the predeclared `+0.002` requirement. Random validation optimism (`validation R² - lockbox R²`) was **-0.005077**; safe validation optimism was **+0.003180**. Their difference was **-0.008258**, also opposite the required direction. Both conditions failed.

## Interpretation

Random folds were structurally invalid for this forecasting question: each fold trained on thousands of dates later than its earliest validation observation. Yet, on this exact AAPL/Ridge task, that look-ahead structure did **not** make validation look better. It made the mean estimate more pessimistic. The safe scheme estimated the lockbox more closely in absolute R² terms, but both final R² values were negative, so neither model beat the zero-return forecast on squared error.

This is a useful negative result. It rejects the broad claim that random splitting must inflate performance in every next-day equity task, while still supporting the methodological point that time-ordered evaluation is the defensible protocol. The experiment provides no evidence of a tradable predictive edge. The 5-bps trading diagnostic is secondary and unstable because both Ridge forecasts cluster near zero; it must not override the primary metric.

## Audit trail and limitations

- `config.json` SHA-256: `da8dee3d99d9179164a67e031f3b4715cc275081f64a42a2464b9af4b1cdedde`.
- Development: 3,502 observations, 2010-02-02 through 2023-12-29.
- Lockbox: 501 observations, 2024-01-02 through 2025-12-30.
- Data: Yahoo Finance chart endpoint; raw response SHA-256 is recorded in `results/summary.json`.
- Scope is intentionally narrow: one large-cap equity, daily OHLCV-derived features, one linear model family, one lockbox period. It cannot establish that random splitting is safe in other assets, higher-frequency data, cross-sectional panels, longer label horizons, or more adaptive model searches.

Machine-readable scores, fold boundaries, predictions, environment versions, and the decision rule are in `results/`.
