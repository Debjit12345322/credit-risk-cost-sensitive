# Credit Default Risk as a Cost-Sensitive Decision

A LendingClub default model where the object of study is the **decision**, not the
classifier. Built on the 2014 origination vintage (223,103 resolved loans, 18.45%
default rate) from the perspective of a platform investor choosing which listed
loans to fund.

## Results

| | Logistic Regression | LightGBM |
| --- | ---: | ---: |
| Test AUC | 0.7173 | 0.7198 |
| Brier score | 0.1364 | 0.1361 |
| Mean predicted PD | 0.1838 | 0.1831 |
| Actual base rate | 0.1845 | 0.1845 |

**Decision performance** (LightGBM, per application, on 55,776 held-out loans):

| Strategy | Threshold | Cost / application | Decline rate |
| --- | ---: | ---: | ---: |
| Decline everything | — | $2,381.71 | 100% |
| Fund everything | — | $1,006.05 | 0% |
| Default cutoff | 0.500 | $981.06 | 2.5% |
| **Cost-optimal (fitted out-of-fold)** | **0.470** | **$979.15** | 3.6% |
| Cost-optimal, per-term | 0.355 / 0.470 | **$972.56** | 5.0% |
| *Test-fitted oracle (unattainable)* | *0.460* | *$978.83* | *3.9%* |
| Naive 14:1 cost assumption | 0.065 | **$2,058.39** | 84.2% |

Cost is measured as **regret against a perfect decision**: correctly funding a good
loan scores zero rather than a profit. $979/application is how much worse than an
oracle, not money lost.

## The headline finding

**The cost matrix matters more than the model.**

Logistic regression and gradient boosting differ by 0.0025 AUC — worth a couple of
dollars per application. But adopting the standard industry assumption for this
problem (loss given default of 55–75%, thin margin on repayment, implying roughly
14:1 asymmetry) drives the optimiser to a threshold of 0.065, declines 84% of
applications, and **more than doubles cost per application**.

That assumption is wrong, and the data says so:

| Outcome | n | Mean funded | Mean received | Net | % of funded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Defaulted | 41,162 | $15,568 | $10,167 | −$5,401 | **−34.8%** |
| Fully paid | 181,941 | $14,354 | $17,250 | +$2,896 | **+19.1%** |

Loss given default is ~35%, not 55–75%: the quoted figure describes loss on
principal *outstanding at charge-off*, but a defaulting borrower typically pays for
a year or more first, and recoveries add a little back. And profit on a repaid loan
is ~19%, not a thin margin — these are 13–15% APR loans held for years. Both
corrections compress the asymmetry to **1.8:1**.

The whole decision layer hinges on one number that most projects assume rather than
measure.

## What this model actually does

Within any single LendingClub grade, AUC drops from 0.72 to roughly **0.63**:

| Grade | n | Default rate | AUC | Cost / app |
| --- | ---: | ---: | ---: | ---: |
| A | 9,077 | 5.1% | 0.6364 | $246 |
| B | 14,958 | 12.0% | 0.6310 | $561 |
| C | 15,605 | 19.5% | 0.6307 | $991 |
| D | 9,850 | 27.4% | 0.6375 | $1,532 |
| E | 4,468 | 34.9% | 0.6309 | $2,059 |
| F | 1,451 | 37.8% | 0.6455 | $2,169 |
| G | 367 | 47.7% | 0.5568 | $2,948 |

Most of the ranking power is re-deriving LendingClub's own grade. Conditional on
grade, the other 70 features add modestly. The honest description is *a funding
decision layered on top of LendingClub's underwriting*, not a model that beats it.
`grade`, `sub_grade` and `int_rate` were kept deliberately — an investor genuinely
observes them at decision time — but this is the cost of that choice.

## Method

### Leakage audit

All 152 source columns are classified explicitly in `src/leakage_audit.py`, each
with a written rationale. The test: *would this value exist, with this value, in the
lender's system at the moment of the decision?*

| Bucket | Columns |
| --- | ---: |
| `KEEP` — legitimate at origination | 70 |
| `EMPTY_OR_CONST` — 0% fill or constant in 2014 | 35 |
| `LEAK_OUTCOME` — hardship / settlement | 21 |
| `LEAK_PAYMENT` — servicing fields | 13 |
| `IDENTIFIER` — IDs, free text, `zip_code` | 6 |
| `LEAK_MONITORING` — credit data re-pulled post-origination | 3 |
| `ANALYSIS_ONLY` — segments only, never modelled | 2 |
| `TARGET` | 2 |

`audit_columns()` **raises** on any unclassified column, so schema drift fails loudly
rather than silently admitting an unvetted field.

What the audit is worth — same model, same seed, same split:

| Feature set | n | Test AUC |
| --- | ---: | ---: |
| Naive (leaks included) | 83 | **0.9998** |
| Audited | 61 | **0.7147** |

Top drivers of the naive model: `total_rec_prncp`, `last_pymnt_amnt`,
`last_fico_range_high`, `settlement_amount` — all servicing fields. An AUC near 1.0
on consumer credit is not a strong model, it is the signature of a leak.

The clearest single example:

| | Repaid | Defaulted | Gap |
| --- | ---: | ---: | ---: |
| `fico_range_low` (at application) | 694.0 | 685.5 | 8.5 |
| `last_fico_range_low` (post-origination) | 691.6 | 519.8 | **171.8** |

Same quantity, but the second is re-pulled during servicing. Its 172-point gap
measures an outcome that has already happened.

### Vintage and target

**Matured ≠ resolved.** A loan matures when its scheduled term elapses; it resolves
when it reaches a terminal state, which usually happens earlier through prepayment
or charge-off. 97.2% of resolved 60-month loans resolved *before* month 60, at a
median of ~25 months.

2014 was chosen as the most recent vintage where nearly all loans have resolved:
36-month 100% (162,570/162,570), 60-month 16.3% censored. 2017 was rejected — 59% of
those loans are still "Current", and dropping them is not outcome-neutral, since
defaults resolve fast while healthy loans stay Current to the end.

Target uses terminal states only: `Charged Off`/`Default` → 1, `Fully Paid` → 0.
`Current`, `Late`, and `In Grace Period` are dropped rather than folded into the
positive class — a late loan may still cure, and labelling it as default asserts an
outcome that has not been observed.

### Missingness

Nine columns carry missing values. Measured default rate by missingness:

| Column | n missing | Lift (pp) | p |
| --- | ---: | ---: | ---: |
| `emp_length` | 11,554 | **+5.56** | 7.9e-51 |
| `mths_since_recent_inq` | 20,185 | **−5.18** | 5.7e-73 |
| `bc_util` | 2,482 | +1.76 | 2.7e-02 |
| `revol_util` | 115 | +1.55 | 7.6e-01 *(n.s.)* |
| `mths_since_last_major_derog` | 159,591 | −1.33 | 2.4e-13 |
| `mths_since_last_record` | 183,144 | −1.31 | 1.1e-09 |
| `mths_since_recent_bc_dlq` | 163,991 | −0.99 | 1.1e-07 |
| `mths_since_last_delinq` | 109,646 | −0.88 | 8.8e-08 |
| `mths_since_recent_revol_delinq` | 142,779 | −0.56 | 1.2e-03 |

Note the **sign flip**. Missing `mths_since_last_delinq` means *never delinquent* →
lower risk. Missing `emp_length` means the borrower supplied no employment
information → higher risk. Same mechanism, opposite meaning; no single imputation
strategy expresses both.

All nine get an indicator, including the weak ones. The table above was computed on
the full dataset, so using it to select features would let outcome information
influence model design — the same error the leakage audit exists to prevent, one
level up. Nine extra columns on 70 is cheap insurance against that story.

Interesting side result: `emp_length`'s *value* barely predicts anything (17.6%–19.4%
default across all eleven buckets, non-monotonic), while its *missingness* separates
18.2% from 23.7%.

### Models

Both baselines receive the missingness indicators, so the comparison is between
pipelines each given its idiomatic preprocessing — not between algorithms on
identical matrices, which is not available since handling missingness *is* part of
what a GBM does.

- **Logistic regression**: median imputation, standardisation, one-hot encoding, all
  inside a `Pipeline` so parameters are fitted on training folds only.
- **LightGBM**: native NaN handling and native categoricals. Modest, untuned
  hyperparameters — tuning would require a separate validation fold, and the
  project's contribution is the decision layer.

**No class weighting or resampling.** This is deliberate. `class_weight='balanced'`
shifts predicted probabilities away from the true base rate — a balanced model
outputs ~0.5 for an average loan instead of ~0.18 — which would break both the
expected-cost threshold and the calibration check. Asymmetric cost is a property of
the *decision*, not the training data; handling it by reweighting *and* by moving the
threshold double-counts the same asymmetry.

### Calibration

Both models are well calibrated with no post-hoc correction. Across ten quantile
bins the largest deviation is 1.7pp, in the top bin; bins 1–9 agree within ~0.5pp.
Calibration also holds *within* segments — grade A predicts 0.0538 against 0.0514
actual, grade G predicts 0.4752 against 0.4768 — which is a stronger claim than an
aggregate curve, since aggregate calibration can hide offsetting subgroup errors.

The one weak region: the top decile over-predicts by ~1.7pp. The highest-risk loans
are slightly safer than the model claims, and that is exactly where declines happen.

Good calibration allows a real check rather than a decorative plot — the threshold
implied by the cost rates should match the one found by sweeping realised outcomes:

| Term | Theoretical G/(L+G) | Empirical optimum | Gap |
| --- | ---: | ---: | ---: |
| 36 months | 0.316 | 0.345 | +0.029 |
| 60 months | 0.457 | 0.475 | +0.018 |

The small positive bias has a concrete cause: both error costs scale with amount
funded, and loan size correlates with risk (default rate rises from 14.2% in the
smallest amount quintile to 21.0% in the largest), so larger loans pull the
cost-weighted optimum above the unweighted break-even.

### Segment thresholds, and why most of them are illusory

Fitting per-segment thresholds **on the test set** suggests large gains. Fitting the
same thresholds on **out-of-fold training predictions** and scoring on held-out test
data tells a different story:

| Segment | Apparent gain (test-fitted) | Honest gain (OOF-fitted) |
| --- | ---: | ---: |
| Term | +$7.24 | **+$6.60** |
| Grade | +$5.14 | +$0.14 |
| Purpose | +$6.41 | **−$2.02** |

Only the term split survives, and for an economic reason rather than a statistical
one: 60-month loans return 29.6% when repaid against 15.9% for 36-month loans, with
near-identical downside, so their break-even genuinely differs. Grade and purpose
thresholds differed only in where sampling noise placed the bottom of the cost curve.

Out-of-fold fitting is required, not optional. Fitting the global threshold on
*in-sample* training predictions gave 0.380 and cost **$1,006.79** per application on
test — worse than leaving the threshold at 0.5. Out-of-fold fitting gave 0.470 and
$979.15, within $0.32 of the unattainable oracle.

## Limitations

- **Right-censoring in the 60-month stratum.** 16.3% of 2014 60-month loans were
  still Current at snapshot, and those are disproportionately on-schedule payers.
  The observed 60-month default rate of 31.1% is an upper bound; if every censored
  loan repaid it would be 26.0%. The true value is likely near the lower end. The
  36-month stratum (13.7%) is fully resolved and unaffected.
- **Most discriminative power is LendingClub's grade**, as shown above. This model
  refines a funding decision on already-graded loans; it does not replicate
  underwriting from scratch.
- **Cost rates are portfolio averages**, estimated on training rows only. They do not
  vary with borrower characteristics beyond term, and loss given default almost
  certainly does.
- **Single vintage.** No test of stability across origination years, so nothing here
  speaks to drift. A 2015 or 2016 holdout would be the natural next step.
- **Fair lending not formally tested.** `zip_code` was dropped as a proxy for
  protected class and `addr_state` excluded from features, but no disparate-impact
  analysis was run. The segment tables show error concentration, not legal
  compliance.
- **Hyperparameters untuned.** Tuning needs a third split; the marginal AUC was not
  the point.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `accepted_2007_to_2018Q4.csv.gz` from the Kaggle dataset
[`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
and place it at `data/accepted_full.csv.gz`. Data is gitignored: LendingClub's terms
do not grant redistribution rights, and a 374MB blob does not belong in git history.

```bash
python scripts/build_dataset.py       # -> data/loans_2014.parquet  (223,103 loans)
python scripts/run_leakage_audit.py   # -> reports/leakage_audit.md, leakage_demo.md
```

All randomness is seeded with `SEED = 42` (train/test split, both models, K-fold).
The source file is streamed in 200,000-row chunks, so the build runs in under 4GB.

## Repository

```
src/
  data_loading.py    vintage selection, target definition, streaming loader
  leakage_audit.py   152-column classification; raises on unvetted columns
  features.py        missingness indicators, emp_length / term / credit-history parsing
  pipeline.py        LR and LightGBM pipelines; the impute-and-indicate resolution
  costs.py           per-loan cost model, threshold sweep, expected-cost optimisation
  calibration.py     reliability tables and diagrams
  segments.py        segment reports and out-of-fold threshold fitting
scripts/
  build_dataset.py
  run_leakage_audit.py
reports/
  leakage_audit.md   generated, not hand-written — cannot drift from the code
  leakage_demo.md
  figures/calibration.png
notebooks/           exploration only; nothing here is a dependency
```

Every module docstring records the reasoning behind its decisions, including the ones
that turned out to be wrong.