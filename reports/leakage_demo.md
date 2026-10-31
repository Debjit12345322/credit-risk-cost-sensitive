# What the leakage audit is worth

Identical LightGBM pipeline, identical split (seed 42), two feature sets. The
audited row is the project's actual baseline model.

| feature set | features | test AUC |
| --- | --- | --- |
| Naive (leaky columns added) | 101 | 0.99997 |
| Audited | 79 | 0.72058 |

Top drivers of the naive model (by gain): `recoveries`, `collection_recovery_fee`, `last_fico_range_high`, `total_rec_prncp`, `last_pymnt_amnt`, `loan_amnt`.

These are servicing fields that do not exist at decision time. An AUC near 1.0
on consumer credit default is not a strong model; it is the signature of a leak.

## The same quantity, measured before and after the outcome

| column | mean_if_repaid | mean_if_defaulted | gap |
| --- | --- | --- | --- |
| `fico_range_low` (at application) | 694.0 | 685.5 | 8.5 |
| `last_fico_range_low` (re-pulled after origination) | 691.6 | 519.8 | 171.8 |

## Columns that are the label in disguise

- `debt_settlement_flag == 'Y'`: 4,217 loans, default rate 100.0%. A settlement only exists for a loan that already went bad.
- `recoveries > 0`: 30,555 loans, default rate 100.0%. Recoveries are post-charge-off collections.
