# Preregistration seal

- Frozen before the empirical AAPL download and lockbox run on **2026-09-24 (Asia/Calcutta)**.
- `config.json` SHA-256: `da8dee3d99d9179164a67e031f3b4715cc275081f64a42a2464b9af4b1cdedde`
- The synthetic smoke test was run before sealing only to verify split invariants and output generation. It was not used to alter the asset, dates, features, model, grid, metrics, purge/embargo rule, cost assumption, or failure threshold.
- The empirical lockbox is 2024-01-01 through 2025-12-31 and is evaluated once by `python experiment.py`.

## Pre-data operational amendment

The first empirical command stopped before parsing or displaying any price or lockbox observation because Stooq returned a JavaScript verification page rather than CSV. Before any empirical result was visible, the transport was changed to Yahoo's chart endpoint. Yahoo's adjustment factor is applied to OHLC; volume remains raw. No asset, date, feature, target, model, hyperparameter, split, metric, cost, threshold, or decision rule changed. The frozen `config.json` hash above is unchanged.
