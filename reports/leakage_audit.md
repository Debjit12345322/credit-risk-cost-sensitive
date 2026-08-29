# Leakage Audit

Every column in the source file is classified explicitly. The test applied:

> Would this value exist, with this value, in the lender's system at the
> moment the lend/decline decision is made?

`src/leakage_audit.py` raises on any column it has not seen, so a schema
change fails loudly instead of silently admitting an unvetted field.

## Summary

| Bucket | Columns |
| --- | ---: |
| `TARGET` | 2 |
| `LEAK_PAYMENT` | 13 |
| `LEAK_MONITORING` | 3 |
| `LEAK_OUTCOME` | 21 |
| `EMPTY_OR_CONST` | 35 |
| `IDENTIFIER` | 6 |
| `ANALYSIS_ONLY` | 2 |
| `KEEP` | 70 |
| **Total** | **152** |

## TARGET

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `loan_status` | 100.0 | Raw status the target was derived from. |
| `target` | 100.0 | Derived label: 1 = Charged Off/Default, 0 = Fully Paid. |

## LEAK_PAYMENT

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `pymnt_plan` | 100.0 | Flag for a servicing payment plan; set after origination. |
| `out_prncp` | 100.0 | Outstanding principal; only exists once the loan is being repaid. |
| `out_prncp_inv` | 100.0 | Outstanding principal for investor-funded portion; post-origination. |
| `total_pymnt` | 100.0 | Total payments received to date. Directly reflects repayment behaviour. |
| `total_pymnt_inv` | 100.0 | As total_pymnt, investor portion. |
| `total_rec_prncp` | 100.0 | Principal received to date; a repaid loan has this equal to the loan amount. |
| `total_rec_int` | 100.0 | Interest received to date; accrues only through servicing. |
| `total_rec_late_fee` | 100.0 | Late fees charged; a late fee IS delinquency, i.e. part of the outcome. |
| `recoveries` | 100.0 | Post-charge-off collections. recoveries > 0 has a 100% default rate. |
| `collection_recovery_fee` | 100.0 | Fee on recoveries; same leak as recoveries. |
| `last_pymnt_d` | 99.9 | Date of most recent payment; unknown at origination. |
| `last_pymnt_amnt` | 100.0 | Amount of most recent payment; a large final payment implies early payoff. |
| `next_pymnt_d` | 0.0 | Next scheduled payment; only meaningful for live loans (0% filled here). |

## LEAK_MONITORING

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `last_credit_pull_d` | 100.0 | Date of the most recent credit pull. Sounds like an application field but is refreshed during servicing. |
| `last_fico_range_high` | 100.0 | FICO re-pulled during servicing. See module docstring for the 172-point gap. |
| `last_fico_range_low` | 100.0 | FICO re-pulled during servicing. Measures the outcome, not the risk. |

## LEAK_OUTCOME

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `hardship_type` | 0.1 | Hardship programme detail; only populated for distressed loans. |
| `hardship_reason` | 0.1 | Hardship programme detail; post-origination. |
| `hardship_status` | 0.1 | Hardship programme detail; post-origination. |
| `deferral_term` | 0.1 | Payment deferral granted during servicing. |
| `hardship_amount` | 0.1 | Hardship programme detail; post-origination. |
| `hardship_start_date` | 0.1 | Hardship programme detail; post-origination. |
| `hardship_end_date` | 0.1 | Hardship programme detail; post-origination. |
| `payment_plan_start_date` | 0.1 | Servicing payment plan start; post-origination. |
| `hardship_length` | 0.1 | Hardship programme detail; post-origination. |
| `hardship_dpd` | 0.1 | Days past due at hardship; delinquency is part of the outcome. |
| `hardship_loan_status` | 0.1 | Loan status at hardship; contains the label. |
| `orig_projected_additional_accrued_interest` | 0.1 | Projection made at hardship enrolment; post-origination. |
| `hardship_payoff_balance_amount` | 0.1 | Balance at hardship; post-origination. |
| `hardship_last_payment_amount` | 0.1 | Payment at hardship; post-origination. |
| `debt_settlement_flag` | 100.0 | Settlement only occurs on loans that already defaulted: flag 'Y' -> 100% default rate. |
| `debt_settlement_flag_date` | 1.9 | Date of settlement flag; post-default. |
| `settlement_status` | 1.9 | Settlement programme status; post-default. |
| `settlement_date` | 1.9 | Settlement date; post-default. |
| `settlement_amount` | 1.9 | Negotiated settlement amount; post-default. |
| `settlement_percentage` | 1.9 | Settlement as % of balance; post-default. |
| `settlement_term` | 1.9 | Settlement term; post-default. |

## EMPTY_OR_CONST

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `member_id` | 0.0 | 0% filled (scrubbed in the public release). |
| `policy_code` | 100.0 | Constant 1.0 for all publicly released loans. |
| `application_type` | 100.0 | Constant 'Individual' in 2014 -- joint applications did not exist yet. This is WHY the sec_app_*/joint block below is empty. |
| `annual_inc_joint` | 0.0 | 0% filled; joint product not offered in 2014. |
| `dti_joint` | 0.0 | 0% filled; joint product not offered in 2014. |
| `verification_status_joint` | 0.0 | 0% filled; joint product not offered in 2014. |
| `open_acc_6m` | 0.0 | 0% filled in 2014; this bureau block was added to the schema later. |
| `open_act_il` | 0.0 | 0% filled in 2014; bureau block added later. |
| `open_il_12m` | 0.0 | 0% filled in 2014; bureau block added later. |
| `open_il_24m` | 0.0 | 0% filled in 2014; bureau block added later. |
| `mths_since_rcnt_il` | 0.0 | 0% filled in 2014; bureau block added later. |
| `total_bal_il` | 0.0 | 0% filled in 2014; bureau block added later. |
| `il_util` | 0.0 | 0% filled in 2014; bureau block added later. |
| `open_rv_12m` | 0.0 | 0% filled in 2014; bureau block added later. |
| `open_rv_24m` | 0.0 | 0% filled in 2014; bureau block added later. |
| `max_bal_bc` | 0.0 | 0% filled in 2014; bureau block added later. |
| `all_util` | 0.0 | 0% filled in 2014; bureau block added later. |
| `inq_fi` | 0.0 | 0% filled in 2014; bureau block added later. |
| `total_cu_tl` | 0.0 | 0% filled in 2014; bureau block added later. |
| `inq_last_12m` | 0.0 | 0% filled in 2014; bureau block added later. |
| `revol_bal_joint` | 0.0 | 0% filled; joint product not offered in 2014. |
| `sec_app_fico_range_low` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_fico_range_high` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_earliest_cr_line` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_inq_last_6mths` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_mort_acc` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_open_acc` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_revol_util` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_open_act_il` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_num_rev_accts` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_chargeoff_within_12_mths` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_collections_12_mths_ex_med` | 0.0 | 0% filled; no joint applications in 2014. |
| `sec_app_mths_since_last_major_derog` | 0.0 | 0% filled; no joint applications in 2014. |
| `hardship_flag` | 100.0 | Constant 'N' across all 223,103 rows in 2014; uninformative here. Would be LEAK_OUTCOME in a vintage where it varies. |
| `disbursement_method` | 100.0 | Constant 'Cash' across all 223,103 rows in 2014 -- DirectPay launched later. Chosen at origination so the timing is legitimate, but it carries no variance to learn from in this vintage. |

## IDENTIFIER

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `id` | 100.0 | Row identifier. |
| `emp_title` | 94.3 | Free-text job title, ~200k distinct values. High cardinality and a proxy for protected characteristics. |
| `url` | 100.0 | Link to the listing page; identifier, not a feature. |
| `desc` | 6.8 | Free-text borrower description, only 6.8% filled. Written at application so timing is legitimate, but too sparse to model this weekend. |
| `title` | 100.0 | Free-text borrower-supplied loan title; redundant with the clean `purpose` field. |
| `zip_code` | 100.0 | Dropped on fair-lending grounds: 3-digit ZIP is a strong proxy for race under disparate-impact analysis. |

## ANALYSIS_ONLY

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `issue_d` | 100.0 | Origination date. Known at decision time but constant-by-design within a single vintage; kept for time-based splitting and drift checks. |
| `addr_state` | 100.0 | Geographic proxy for protected class. Retained to test whether model errors concentrate by state; excluded from the feature matrix. |

## KEEP

| Column | Fill % | Rationale |
| --- | ---: | --- |
| `loan_amnt` | 100.0 | Amount requested; known at application. |
| `funded_amnt` | 100.0 | Amount funded; determined at origination, before the first payment. |
| `funded_amnt_inv` | 100.0 | Investor-funded amount; fixed when the listing closes, i.e. at origination. |
| `term` | 100.0 | 36 or 60 months; a term of the contract. |
| `int_rate` | 100.0 | Contract rate set at origination. NOTE: deterministic function of sub_grade -- near-collinear. |
| `installment` | 100.0 | Monthly payment implied by amount/rate/term; fixed at origination. |
| `grade` | 100.0 | LendingClub's own risk grade, assigned before funding. Included per project decision: an investor genuinely observes this when choosing to fund. |
| `sub_grade` | 100.0 | Finer-grained LC risk grade; same rationale and same circularity caveat as grade. |
| `emp_length` | 94.8 | Borrower-stated employment length at application. |
| `home_ownership` | 100.0 | Stated at application. |
| `annual_inc` | 100.0 | Stated income at application (unverified for many loans -- see verification_status). |
| `verification_status` | 100.0 | Whether LC verified stated income; known at underwriting. |
| `purpose` | 100.0 | Borrower-selected loan purpose from a fixed list. |
| `dti` | 100.0 | Debt-to-income at application, computed from bureau data. |
| `delinq_2yrs` | 100.0 | Bureau: delinquencies in the 24 months BEFORE application. |
| `earliest_cr_line` | 100.0 | Bureau: date of first credit line; convert to credit-history length at issue_d. |
| `fico_range_low` | 100.0 | FICO at application. The legitimate counterpart to last_fico_range_low. |
| `fico_range_high` | 100.0 | FICO at application. |
| `inq_last_6mths` | 100.0 | Bureau: credit inquiries in the 6 months before application. |
| `mths_since_last_delinq` | 50.9 | Bureau history. 50.9% filled -- missing means NO PRIOR DELINQUENCY, which is signal, not absence of data. |
| `mths_since_last_record` | 17.9 | Bureau history. 17.9% filled -- missing means no public record. Informative missingness. |
| `open_acc` | 100.0 | Bureau: open credit lines at application. |
| `pub_rec` | 100.0 | Bureau: derogatory public records at application. |
| `revol_bal` | 100.0 | Bureau: revolving balance at application. |
| `revol_util` | 99.9 | Bureau: revolving utilisation at application. |
| `total_acc` | 100.0 | Bureau: total credit lines at application. |
| `initial_list_status` | 100.0 | Whole vs fractional listing; set at listing time. |
| `collections_12_mths_ex_med` | 100.0 | Bureau: collections in the 12 months before application. Backward-looking despite the alarming name. |
| `mths_since_last_major_derog` | 28.5 | Bureau history. 28.5% filled -- missing means no major derogatory event. Informative missingness. |
| `acc_now_delinq` | 100.0 | Bureau: accounts currently delinquent AT APPLICATION. Backward-looking. |
| `tot_coll_amt` | 100.0 | Bureau: total collection amounts at application. |
| `tot_cur_bal` | 100.0 | Bureau: total current balance across accounts at application. |
| `total_rev_hi_lim` | 100.0 | Bureau: total revolving credit limit at application. |
| `acc_open_past_24mths` | 100.0 | Bureau: accounts opened in the 24 months before application. |
| `avg_cur_bal` | 100.0 | Bureau: average current balance at application. |
| `bc_open_to_buy` | 99.0 | Bureau: available bankcard credit at application. |
| `bc_util` | 98.9 | Bureau: bankcard utilisation at application. |
| `chargeoff_within_12_mths` | 100.0 | Bureau: charge-offs on OTHER accounts in the 12 months before application. Not this loan's outcome. |
| `delinq_amnt` | 100.0 | Bureau: amount currently delinquent at application. |
| `mo_sin_old_il_acct` | 96.9 | Bureau: months since oldest instalment account opened. |
| `mo_sin_old_rev_tl_op` | 100.0 | Bureau: months since oldest revolving account opened. |
| `mo_sin_rcnt_rev_tl_op` | 100.0 | Bureau: months since most recent revolving account opened. |
| `mo_sin_rcnt_tl` | 100.0 | Bureau: months since most recent account opened. |
| `mort_acc` | 100.0 | Bureau: number of mortgage accounts at application. |
| `mths_since_recent_bc` | 99.0 | Bureau: months since most recent bankcard opened. |
| `mths_since_recent_bc_dlq` | 26.5 | Bureau history. 26.5% filled -- missing means no bankcard delinquency. Informative missingness. |
| `mths_since_recent_inq` | 91.0 | Bureau: months since most recent inquiry. 91% filled. |
| `mths_since_recent_revol_delinq` | 36.0 | Bureau history. 36.0% filled -- missing means no revolving delinquency. Informative missingness. |
| `num_accts_ever_120_pd` | 100.0 | Bureau: accounts ever 120+ days past due, before application. |
| `num_actv_bc_tl` | 100.0 | Bureau: active bankcard accounts at application. |
| `num_actv_rev_tl` | 100.0 | Bureau: active revolving accounts at application. |
| `num_bc_sats` | 100.0 | Bureau: satisfactory bankcard accounts at application. |
| `num_bc_tl` | 100.0 | Bureau: total bankcard accounts at application. |
| `num_il_tl` | 100.0 | Bureau: total instalment accounts at application. |
| `num_op_rev_tl` | 100.0 | Bureau: open revolving accounts at application. |
| `num_rev_accts` | 100.0 | Bureau: total revolving accounts at application. |
| `num_rev_tl_bal_gt_0` | 100.0 | Bureau: revolving accounts with a balance at application. |
| `num_sats` | 100.0 | Bureau: satisfactory accounts at application. |
| `num_tl_120dpd_2m` | 96.6 | Bureau: accounts 120+ dpd in the 2 months before application. 96.6% filled. |
| `num_tl_30dpd` | 100.0 | Bureau: accounts 30 dpd at application. |
| `num_tl_90g_dpd_24m` | 100.0 | Bureau: accounts 90+ dpd in the 24 months before application. |
| `num_tl_op_past_12m` | 100.0 | Bureau: accounts opened in the 12 months before application. |
| `pct_tl_nvr_dlq` | 100.0 | Bureau: percent of accounts never delinquent, at application. |
| `percent_bc_gt_75` | 98.9 | Bureau: percent of bankcards over 75% utilisation at application. |
| `pub_rec_bankruptcies` | 100.0 | Bureau: bankruptcies on record at application. |
| `tax_liens` | 100.0 | Bureau: tax liens on record at application. |
| `tot_hi_cred_lim` | 100.0 | Bureau: total high credit limit at application. |
| `total_bal_ex_mort` | 100.0 | Bureau: total balance excluding mortgage, at application. |
| `total_bc_limit` | 100.0 | Bureau: total bankcard limit at application. |
| `total_il_high_credit_limit` | 100.0 | Bureau: total instalment high credit limit at application. |
