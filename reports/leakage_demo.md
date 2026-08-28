# What the leakage audit is worth

Identical LightGBM config (seed=42), identical split, two feature sets.

| Feature set | n features | Test AUC |
| --- | ---: | ---: |
| Naive (leaks included) | 83 | **0.9998** |
| Audited (clean) | 61 | **0.7147** |

Top drivers of the naive model: `total_rec_prncp`, `loan_amnt`, `funded_amnt_inv`, `last_pymnt_amnt`, `last_fico_range_high`, `settlement_amount`

Every one of those is a loan-servicing field that does not exist at
decision time. An AUC near 1.0 on consumer credit default is not a strong
model -- it is the signature of a leak. The audited figure is the one that
describes performance on a decision that could actually be made.
