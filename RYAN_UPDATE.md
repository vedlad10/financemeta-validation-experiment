# Ready-to-send update to Ryan

Hi Ryan,

I completed the narrow first experiment and preserved the falsifying result without retuning.

The task predicts AAPL's next-day log close return from causal daily price/volume features using the same standardized Ridge family under both protocols. I froze the asset, 2010–2023 development period, untouched 2024–2025 lockbox, alpha grid, one-day purge, one-day embargo, zero-benchmark OOS R², 5-bps secondary cost assumption, and a two-part failure rule before the empirical run.

The predeclared hypothesis failed. Mean validation OOS R² was -0.01278 under shuffled random 5-fold validation and -0.00435 under purged walk-forward, so random splitting was worse by 0.00843 rather than better by the required 0.002. On the common lockbox, OOS R² was -0.00770 for the random-selected model and -0.00753 for the safe-selected model. Neither beat the zero-return forecast. Random validation was pessimistic relative to its lockbox, not optimistic, and both decision-rule conditions failed.

The result does not make random splitting valid—the fold audit confirms that it trained on future dates—but it does falsify the claim that random splitting necessarily inflates apparent performance for this specific task. The safe estimate was closer to the lockbox, and there is no evidence of predictive edge here.

Included are the one-page frozen spec, three related primary references, source code, raw-data/config hashes, fold boundaries, alpha-level scores, lockbox predictions, plot, and machine-readable summary.

Best,  
Ved
