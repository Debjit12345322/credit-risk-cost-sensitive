"""
Leakage audit for the LendingClub accepted-loans dataset.

THE TEST APPLIED TO EVERY COLUMN
--------------------------------
"Would this value exist, with this value, in the lender's system at the moment
the lend/decline decision is made?"

If the answer is no -- because the field is only populated during servicing,
or is derived from the loan's outcome -- the column is leakage and must be
dropped, regardless of how predictive it looks. In fact the most predictive
columns here are the leaky ones, which is the point.

WORKED EXAMPLE (the one to remember)
------------------------------------
    fico_range_low       (at application)   repaid 694.0 | defaulted 685.5
    last_fico_range_low  (post-origination) repaid 691.6 | defaulted 519.8

Same underlying quantity -- a FICO score -- but the second is re-pulled during
servicing. Its 172-point gap is not predictive power; it is a measurement of an
outcome that has already happened. At decision time that column is empty.

BUCKETS
-------
LEAK_PAYMENT     Populated by loan servicing (payments, balances, recoveries).
LEAK_MONITORING  Credit data re-pulled after origination.
LEAK_OUTCOME     Hardship / settlement programmes. These only exist for loans
                 that already went bad, so they are the label in disguise:
                 debt_settlement_flag == 'Y' has a 100% default rate.
EMPTY_OR_CONST   Structurally unusable in the 2014 vintage. Note this is a
                 DIFFERENT finding from leakage: application_type is 100%
                 'Individual' because LendingClub had not launched joint
                 applications yet, which in turn is why every sec_app_* and
                 joint column is 0% filled.
IDENTIFIER       Row identifiers and free text with no stable signal, plus
                 zip_code which is dropped on fair-lending grounds.
ANALYSIS_ONLY    Kept in the dataframe for segment-level error analysis but
                 deliberately NOT fed to the model. addr_state is a geographic
                 proxy for protected class; it is used to check whether errors
                 are unevenly distributed, not to make predictions.
TARGET           The label, and the raw status it was derived from.
KEEP             Known at origination. Legitimate model input.

DESIGN NOTE
-----------
Every column is classified explicitly. `audit_columns` raises on any column it
has not seen, so a schema change surfaces as a loud failure rather than a
silent inclusion of an unvetted field.
"""

from __future__ import annotations

import pandas as pd

LEAK_PAYMENT = "LEAK_PAYMENT"
LEAK_MONITORING = "LEAK_MONITORING"
LEAK_OUTCOME = "LEAK_OUTCOME"
EMPTY_OR_CONST = "EMPTY_OR_CONST"
IDENTIFIER = "IDENTIFIER"
ANALYSIS_ONLY = "ANALYSIS_ONLY"
TARGET = "TARGET"
KEEP = "KEEP"

DROP_BUCKETS = {
    LEAK_PAYMENT,
    LEAK_MONITORING,
    LEAK_OUTCOME,
    EMPTY_OR_CONST,
    IDENTIFIER,
}

# column -> (bucket, rationale)
COLUMN_AUDIT: dict[str, tuple[str, str]] = {
    # ---------------- target ----------------
    "target": (TARGET, "Derived label: 1 = Charged Off/Default, 0 = Fully Paid."),
    "loan_status": (TARGET, "Raw status the target was derived from."),

    # ---------------- payment / servicing leakage ----------------
    "out_prncp": (LEAK_PAYMENT, "Outstanding principal; only exists once the loan is being repaid."),
    "out_prncp_inv": (LEAK_PAYMENT, "Outstanding principal for investor-funded portion; post-origination."),
    "total_pymnt": (LEAK_PAYMENT, "Total payments received to date. Directly reflects repayment behaviour."),
    "total_pymnt_inv": (LEAK_PAYMENT, "As total_pymnt, investor portion."),
    "total_rec_prncp": (LEAK_PAYMENT, "Principal received to date; a repaid loan has this equal to the loan amount."),
    "total_rec_int": (LEAK_PAYMENT, "Interest received to date; accrues only through servicing."),
    "total_rec_late_fee": (LEAK_PAYMENT, "Late fees charged; a late fee IS delinquency, i.e. part of the outcome."),
    "recoveries": (LEAK_PAYMENT, "Post-charge-off collections. recoveries > 0 has a 100% default rate."),
    "collection_recovery_fee": (LEAK_PAYMENT, "Fee on recoveries; same leak as recoveries."),
    "last_pymnt_d": (LEAK_PAYMENT, "Date of most recent payment; unknown at origination."),
    "last_pymnt_amnt": (LEAK_PAYMENT, "Amount of most recent payment; a large final payment implies early payoff."),
    "next_pymnt_d": (LEAK_PAYMENT, "Next scheduled payment; only meaningful for live loans (0% filled here)."),
    "pymnt_plan": (LEAK_PAYMENT, "Flag for a servicing payment plan; set after origination."),

    # ---------------- post-origination credit monitoring ----------------
    "last_credit_pull_d": (LEAK_MONITORING, "Date of the most recent credit pull. Sounds like an application field but is refreshed during servicing."),
    "last_fico_range_high": (LEAK_MONITORING, "FICO re-pulled during servicing. See module docstring for the 172-point gap."),
    "last_fico_range_low": (LEAK_MONITORING, "FICO re-pulled during servicing. Measures the outcome, not the risk."),

    # ---------------- hardship / settlement (outcome-derived) ----------------
    "hardship_flag": (EMPTY_OR_CONST, "Constant 'N' across all 223,103 rows in 2014; uninformative here. Would be LEAK_OUTCOME in a vintage where it varies."),
    "hardship_type": (LEAK_OUTCOME, "Hardship programme detail; only populated for distressed loans."),
    "hardship_reason": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "hardship_status": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "deferral_term": (LEAK_OUTCOME, "Payment deferral granted during servicing."),
    "hardship_amount": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "hardship_start_date": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "hardship_end_date": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "payment_plan_start_date": (LEAK_OUTCOME, "Servicing payment plan start; post-origination."),
    "hardship_length": (LEAK_OUTCOME, "Hardship programme detail; post-origination."),
    "hardship_dpd": (LEAK_OUTCOME, "Days past due at hardship; delinquency is part of the outcome."),
    "hardship_loan_status": (LEAK_OUTCOME, "Loan status at hardship; contains the label."),
    "orig_projected_additional_accrued_interest": (LEAK_OUTCOME, "Projection made at hardship enrolment; post-origination."),
    "hardship_payoff_balance_amount": (LEAK_OUTCOME, "Balance at hardship; post-origination."),
    "hardship_last_payment_amount": (LEAK_OUTCOME, "Payment at hardship; post-origination."),
    "debt_settlement_flag": (LEAK_OUTCOME, "Settlement only occurs on loans that already defaulted: flag 'Y' -> 100% default rate."),
    "debt_settlement_flag_date": (LEAK_OUTCOME, "Date of settlement flag; post-default."),
    "settlement_status": (LEAK_OUTCOME, "Settlement programme status; post-default."),
    "settlement_date": (LEAK_OUTCOME, "Settlement date; post-default."),
    "settlement_amount": (LEAK_OUTCOME, "Negotiated settlement amount; post-default."),
    "settlement_percentage": (LEAK_OUTCOME, "Settlement as % of balance; post-default."),
    "settlement_term": (LEAK_OUTCOME, "Settlement term; post-default."),

    # ---------------- structurally empty or constant in 2014 ----------------
    "member_id": (EMPTY_OR_CONST, "0% filled (scrubbed in the public release)."),
    "policy_code": (EMPTY_OR_CONST, "Constant 1.0 for all publicly released loans."),
    "application_type": (EMPTY_OR_CONST, "Constant 'Individual' in 2014 -- joint applications did not exist yet. This is WHY the sec_app_*/joint block below is empty."),
    "annual_inc_joint": (EMPTY_OR_CONST, "0% filled; joint product not offered in 2014."),
    "dti_joint": (EMPTY_OR_CONST, "0% filled; joint product not offered in 2014."),
    "verification_status_joint": (EMPTY_OR_CONST, "0% filled; joint product not offered in 2014."),
    "revol_bal_joint": (EMPTY_OR_CONST, "0% filled; joint product not offered in 2014."),
    "open_acc_6m": (EMPTY_OR_CONST, "0% filled in 2014; this bureau block was added to the schema later."),
    "open_act_il": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "open_il_12m": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "open_il_24m": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "mths_since_rcnt_il": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "total_bal_il": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "il_util": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "open_rv_12m": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "open_rv_24m": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "max_bal_bc": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "all_util": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "inq_fi": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "total_cu_tl": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "inq_last_12m": (EMPTY_OR_CONST, "0% filled in 2014; bureau block added later."),
    "sec_app_fico_range_low": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_fico_range_high": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_earliest_cr_line": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_inq_last_6mths": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_mort_acc": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_open_acc": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_revol_util": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_open_act_il": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_num_rev_accts": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_chargeoff_within_12_mths": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_collections_12_mths_ex_med": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),
    "sec_app_mths_since_last_major_derog": (EMPTY_OR_CONST, "0% filled; no joint applications in 2014."),

    # ---------------- identifiers / free text / fair lending ----------------
    "id": (IDENTIFIER, "Row identifier."),
    "url": (IDENTIFIER, "Link to the listing page; identifier, not a feature."),
    "emp_title": (IDENTIFIER, "Free-text job title, ~200k distinct values. High cardinality and a proxy for protected characteristics."),
    "title": (IDENTIFIER, "Free-text borrower-supplied loan title; redundant with the clean `purpose` field."),
    "desc": (IDENTIFIER, "Free-text borrower description, only 6.8% filled. Written at application so timing is legitimate, but too sparse to model this weekend."),
    "zip_code": (IDENTIFIER, "Dropped on fair-lending grounds: 3-digit ZIP is a strong proxy for race under disparate-impact analysis."),

    # ---------------- kept for segment analysis, NOT modelled ----------------
    "addr_state": (ANALYSIS_ONLY, "Geographic proxy for protected class. Retained to test whether model errors concentrate by state; excluded from the feature matrix."),
    "issue_d": (ANALYSIS_ONLY, "Origination date. Known at decision time but constant-by-design within a single vintage; kept for time-based splitting and drift checks."),

    # ---------------- legitimate at-origination features ----------------
    "loan_amnt": (KEEP, "Amount requested; known at application."),
    "funded_amnt": (KEEP, "Amount funded; determined at origination, before the first payment."),
    "funded_amnt_inv": (KEEP, "Investor-funded amount; fixed when the listing closes, i.e. at origination."),
    "term": (KEEP, "36 or 60 months; a term of the contract."),
    "int_rate": (KEEP, "Contract rate set at origination. NOTE: deterministic function of sub_grade -- near-collinear."),
    "installment": (KEEP, "Monthly payment implied by amount/rate/term; fixed at origination."),
    "grade": (KEEP, "LendingClub's own risk grade, assigned before funding. Included per project decision: an investor genuinely observes this when choosing to fund."),
    "sub_grade": (KEEP, "Finer-grained LC risk grade; same rationale and same circularity caveat as grade."),
    "emp_length": (KEEP, "Borrower-stated employment length at application."),
    "home_ownership": (KEEP, "Stated at application."),
    "annual_inc": (KEEP, "Stated income at application (unverified for many loans -- see verification_status)."),
    "verification_status": (KEEP, "Whether LC verified stated income; known at underwriting."),
    "purpose": (KEEP, "Borrower-selected loan purpose from a fixed list."),
    "dti": (KEEP, "Debt-to-income at application, computed from bureau data."),
    "delinq_2yrs": (KEEP, "Bureau: delinquencies in the 24 months BEFORE application."),
    "earliest_cr_line": (KEEP, "Bureau: date of first credit line; convert to credit-history length at issue_d."),
    "fico_range_low": (KEEP, "FICO at application. The legitimate counterpart to last_fico_range_low."),
    "fico_range_high": (KEEP, "FICO at application."),
    "inq_last_6mths": (KEEP, "Bureau: credit inquiries in the 6 months before application."),
    "mths_since_last_delinq": (KEEP, "Bureau history. 50.9% filled -- missing means NO PRIOR DELINQUENCY, which is signal, not absence of data."),
    "mths_since_last_record": (KEEP, "Bureau history. 17.9% filled -- missing means no public record. Informative missingness."),
    "open_acc": (KEEP, "Bureau: open credit lines at application."),
    "pub_rec": (KEEP, "Bureau: derogatory public records at application."),
    "revol_bal": (KEEP, "Bureau: revolving balance at application."),
    "revol_util": (KEEP, "Bureau: revolving utilisation at application."),
    "total_acc": (KEEP, "Bureau: total credit lines at application."),
    "initial_list_status": (KEEP, "Whole vs fractional listing; set at listing time."),
    "collections_12_mths_ex_med": (KEEP, "Bureau: collections in the 12 months before application. Backward-looking despite the alarming name."),
    "mths_since_last_major_derog": (KEEP, "Bureau history. 28.5% filled -- missing means no major derogatory event. Informative missingness."),
    "acc_now_delinq": (KEEP, "Bureau: accounts currently delinquent AT APPLICATION. Backward-looking."),
    "tot_coll_amt": (KEEP, "Bureau: total collection amounts at application."),
    "tot_cur_bal": (KEEP, "Bureau: total current balance across accounts at application."),
    "total_rev_hi_lim": (KEEP, "Bureau: total revolving credit limit at application."),
    "acc_open_past_24mths": (KEEP, "Bureau: accounts opened in the 24 months before application."),
    "avg_cur_bal": (KEEP, "Bureau: average current balance at application."),
    "bc_open_to_buy": (KEEP, "Bureau: available bankcard credit at application."),
    "bc_util": (KEEP, "Bureau: bankcard utilisation at application."),
    "chargeoff_within_12_mths": (KEEP, "Bureau: charge-offs on OTHER accounts in the 12 months before application. Not this loan's outcome."),
    "delinq_amnt": (KEEP, "Bureau: amount currently delinquent at application."),
    "mo_sin_old_il_acct": (KEEP, "Bureau: months since oldest instalment account opened."),
    "mo_sin_old_rev_tl_op": (KEEP, "Bureau: months since oldest revolving account opened."),
    "mo_sin_rcnt_rev_tl_op": (KEEP, "Bureau: months since most recent revolving account opened."),
    "mo_sin_rcnt_tl": (KEEP, "Bureau: months since most recent account opened."),
    "mort_acc": (KEEP, "Bureau: number of mortgage accounts at application."),
    "mths_since_recent_bc": (KEEP, "Bureau: months since most recent bankcard opened."),
    "mths_since_recent_bc_dlq": (KEEP, "Bureau history. 26.5% filled -- missing means no bankcard delinquency. Informative missingness."),
    "mths_since_recent_inq": (KEEP, "Bureau: months since most recent inquiry. 91% filled."),
    "mths_since_recent_revol_delinq": (KEEP, "Bureau history. 36.0% filled -- missing means no revolving delinquency. Informative missingness."),
    "num_accts_ever_120_pd": (KEEP, "Bureau: accounts ever 120+ days past due, before application."),
    "num_actv_bc_tl": (KEEP, "Bureau: active bankcard accounts at application."),
    "num_actv_rev_tl": (KEEP, "Bureau: active revolving accounts at application."),
    "num_bc_sats": (KEEP, "Bureau: satisfactory bankcard accounts at application."),
    "num_bc_tl": (KEEP, "Bureau: total bankcard accounts at application."),
    "num_il_tl": (KEEP, "Bureau: total instalment accounts at application."),
    "num_op_rev_tl": (KEEP, "Bureau: open revolving accounts at application."),
    "num_rev_accts": (KEEP, "Bureau: total revolving accounts at application."),
    "num_rev_tl_bal_gt_0": (KEEP, "Bureau: revolving accounts with a balance at application."),
    "num_sats": (KEEP, "Bureau: satisfactory accounts at application."),
    "num_tl_120dpd_2m": (KEEP, "Bureau: accounts 120+ dpd in the 2 months before application. 96.6% filled."),
    "num_tl_30dpd": (KEEP, "Bureau: accounts 30 dpd at application."),
    "num_tl_90g_dpd_24m": (KEEP, "Bureau: accounts 90+ dpd in the 24 months before application."),
    "num_tl_op_past_12m": (KEEP, "Bureau: accounts opened in the 12 months before application."),
    "pct_tl_nvr_dlq": (KEEP, "Bureau: percent of accounts never delinquent, at application."),
    "percent_bc_gt_75": (KEEP, "Bureau: percent of bankcards over 75% utilisation at application."),
    "pub_rec_bankruptcies": (KEEP, "Bureau: bankruptcies on record at application."),
    "tax_liens": (KEEP, "Bureau: tax liens on record at application."),
    "tot_hi_cred_lim": (KEEP, "Bureau: total high credit limit at application."),
    "total_bal_ex_mort": (KEEP, "Bureau: total balance excluding mortgage, at application."),
    "total_bc_limit": (KEEP, "Bureau: total bankcard limit at application."),
    "total_il_high_credit_limit": (KEEP, "Bureau: total instalment high credit limit at application."),
    "disbursement_method": (EMPTY_OR_CONST, "Constant 'Cash' across all 223,103 rows in 2014 -- DirectPay launched later. Chosen at origination so the timing is legitimate, but it carries no variance to learn from in this vintage."),
}


def audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the audit as a dataframe. Raises if any column is unclassified."""
    unknown = [c for c in df.columns if c not in COLUMN_AUDIT]
    if unknown:
        raise KeyError(
            f"{len(unknown)} column(s) not present in COLUMN_AUDIT and therefore "
            f"never vetted for leakage: {unknown}. Classify them explicitly "
            f"rather than letting them default into the model."
        )

    rows = []
    for col in df.columns:
        bucket, rationale = COLUMN_AUDIT[col]
        rows.append(
            {
                "column": col,
                "bucket": bucket,
                "fill_rate": round(100 * df[col].notna().mean(), 1),
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that may be fed to a model."""
    audit_columns(df)  # fail loudly on unvetted columns
    return [c for c in df.columns if COLUMN_AUDIT[c][0] == KEEP]


def get_analysis_columns(df: pd.DataFrame) -> list[str]:
    """Columns retained for segment-level error analysis but never modelled."""
    return [c for c in df.columns if COLUMN_AUDIT[c][0] == ANALYSIS_ONLY]


LEAK_BUCKETS = {LEAK_PAYMENT, LEAK_MONITORING, LEAK_OUTCOME}


def get_leak_columns(df: pd.DataFrame) -> list[str]:
    """Columns classified as post-origination leakage.

    Used only by the leakage demonstration, which adds these back to show what
    the audit is worth. Never fed to a real model.
    """
    audit_columns(df)
    return [c for c in df.columns if COLUMN_AUDIT[c][0] in LEAK_BUCKETS]
