"""
Regenerate every number the README cites.

    python scripts/run_full_analysis.py            # primary seed + robustness seeds
    python scripts/run_full_analysis.py --quick    # primary seed only

Reads   data/loans_2014.parquet, reports/target_audit.json
Writes  reports/results.md          every table, generated (never hand-edited)
        reports/results.json        key scalars, for programmatic checks
        reports/figures/calibration.png
        reports/figures/cost_curve.png

Every threshold is CHOSEN on out-of-fold predictions for training rows and
EVALUATED once on the held-out test set. The single exception is the
"test-fitted oracle" row, reported only as an unattainable reference point.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.calibration import calibration_summary, calibration_table, plot_calibration
from src.config import (DATASET_FILE, FIGURES_DIR, REPORTS_DIR, ROBUSTNESS_SEEDS, SEED,
                        TARGET_AUDIT_FILE)
from src.costs import (CostRates, estimate_cost_rates, estimate_cost_rates_by_term,
                       optimal_threshold, per_loan_costs, sweep_thresholds,
                       total_cost_of_decisions)
from src.features import MISSINGNESS_INDICATOR_COLS
from src.pipeline import (build_feature_frame, build_lgbm_model, build_lr_pipeline,
                          split_indices, to_lgbm_frame)
from src.reporting import md_table
from src.segments import (apply_segment_thresholds, fit_segment_thresholds,
                          out_of_fold_predictions, segment_report, segment_threshold_cost)


def money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.2f}"


# ============================================================ data

def load():
    df = pd.read_parquet(DATASET_FILE)
    X, numeric, categorical = build_feature_frame(df)
    Xg = to_lgbm_frame(X, categorical)  # cast BEFORE splitting: one shared vocabulary
    meta = pd.DataFrame({
        "target": df["target"].to_numpy(),
        "funded_amnt_inv": df["funded_amnt_inv"].to_numpy(),
        "total_pymnt_inv": df["total_pymnt_inv"].to_numpy(),
        "term_months": X["term_months"].to_numpy(),
        "grade": df["grade"].to_numpy(),
        "purpose": df["purpose"].to_numpy(),
        "annual_inc": df["annual_inc"].to_numpy(),
    })
    return df, X, Xg, numeric, categorical, meta


# ============================================================ per-seed evaluation

def decision_row(name, thresholds, decline, y, fn, fp):
    cost = total_cost_of_decisions(y, decline, fn, fp) / len(y)
    approved = ~decline
    return {"strategy": name, "threshold": thresholds, "cost_per_application": cost,
            "decline_rate": float(decline.mean()),
            "approved_default_rate": float(y[approved].mean()) if approved.any() else np.nan}


def run_seed(seed, X, Xg, numeric, categorical, meta, detailed=False):
    y = meta["target"].to_numpy()
    tr, te = split_indices(meta["target"], seed)
    y_tr, y_te = y[tr], y[te]

    # ---- models, scored on the held-out test set
    lr = build_lr_pipeline(numeric, categorical, seed).fit(X.iloc[tr], y_tr)
    p_lr = lr.predict_proba(X.iloc[te])[:, 1]
    gbm = build_lgbm_model(seed).fit(Xg.iloc[tr], y_tr)
    p_te = gbm.predict_proba(Xg.iloc[te])[:, 1]

    # ---- costs: rates measured on TRAINING rows only
    rates = estimate_cost_rates_by_term(meta.iloc[tr])
    term_tr, term_te = meta["term_months"].to_numpy()[tr], meta["term_months"].to_numpy()[te]
    fn_tr, fp_tr = per_loan_costs(meta["funded_amnt_inv"].to_numpy()[tr], term_tr, rates)
    fn_te, fp_te = per_loan_costs(meta["funded_amnt_inv"].to_numpy()[te], term_te, rates)

    # ---- out-of-fold predictions on training rows: what thresholds are fitted on
    oof = out_of_fold_predictions(lambda: build_lgbm_model(seed), Xg.iloc[tr], y_tr, seed)
    oof_lr = out_of_fold_predictions(lambda: build_lr_pipeline(numeric, categorical, seed),
                                     X.iloc[tr], y_tr, seed)

    t_oof, _ = optimal_threshold(sweep_thresholds(y_tr, oof, fn_tr, fp_tr))
    t_oof_lr, _ = optimal_threshold(sweep_thresholds(y_tr, oof_lr, fn_tr, fp_tr))
    term_t = fit_segment_thresholds(term_tr, y_tr, oof, fn_tr, fp_tr, fallback=t_oof)
    t_oracle, _ = optimal_threshold(sweep_thresholds(y_te, p_te, fn_te, fp_te))

    def cost_of(decline, fn=fn_te, fp=fp_te):
        return total_cost_of_decisions(y_te, decline, fn, fp) / len(y_te)

    # ---- sensitivity: choose a threshold under ASSUMED costs, score it with MEASURED costs
    def assumed_policy_cost(assumed_rates):
        afn, afp = per_loan_costs(meta["funded_amnt_inv"].to_numpy()[tr], term_tr, assumed_rates)
        t, _ = optimal_threshold(sweep_thresholds(y_tr, oof, afn, afp))
        decline = p_te >= t
        return t, cost_of(decline), float(decline.mean())

    sens = {
        "measured": {k: v for k, v in rates.items()},
        "loss 70%, profit measured": {k: CostRates.assumed(0.70, v.profit_rate) for k, v in rates.items()},
        "loss measured, profit 5%": {k: CostRates.assumed(v.loss_rate, 0.05) for k, v in rates.items()},
        "loss 70%, profit 5%": {k: CostRates.assumed(0.70, 0.05) for k in rates},
    }
    sens_results = {name: assumed_policy_cost(r) for name, r in sens.items()}

    def honest_segment_gain(column):
        lab = meta[column].to_numpy()
        cost, _, _ = segment_threshold_cost(lab[tr], y_tr, oof, fn_tr, fp_tr,
                                            lab[te], y_te, p_te, fn_te, fp_te, t_oof)
        return cost_of(p_te >= t_oof) - cost

    scalars = {
        "seed": seed,
        "auc_lr": roc_auc_score(y_te, p_lr),
        "auc_lgbm": roc_auc_score(y_te, p_te),
        "brier_lr": brier_score_loss(y_te, p_lr),
        "brier_lgbm": brier_score_loss(y_te, p_te),
        "threshold_oof": t_oof,
        "cost_fund_all": cost_of(np.zeros(len(y_te), bool)),
        "cost_at_0.5": cost_of(p_te >= 0.5),
        "cost_oof_single": cost_of(p_te >= t_oof),
        "cost_oof_per_term": cost_of(apply_segment_thresholds(term_te, p_te, term_t, t_oof)),
        "cost_oracle": cost_of(p_te >= t_oracle),
        "cost_lr_oof_single": cost_of(p_lr >= t_oof_lr),
        "cost_assumed_70_5": sens_results["loss 70%, profit 5%"][1],
        "cost_assumed_loss70": sens_results["loss 70%, profit measured"][1],
        "cost_assumed_profit5": sens_results["loss measured, profit 5%"][1],
    }
    scalars["gain_tuning_vs_0.5"] = scalars["cost_at_0.5"] - scalars["cost_oof_single"]
    scalars["gain_per_term"] = scalars["cost_oof_single"] - scalars["cost_oof_per_term"]
    scalars["gain_grade_thresholds"] = honest_segment_gain("grade")
    scalars["gain_purpose_thresholds"] = honest_segment_gain("purpose")
    scalars["lr_minus_lgbm_cost"] = scalars["cost_lr_oof_single"] - scalars["cost_oof_single"]
    scalars["ratio_assumed_70_5"] = scalars["cost_assumed_70_5"] / scalars["cost_oof_single"]
    scalars["ratio_loss70"] = scalars["cost_assumed_loss70"] / scalars["cost_oof_single"]
    scalars["ratio_profit5"] = scalars["cost_assumed_profit5"] / scalars["cost_oof_single"]
    print(f"  seed {seed}: AUC LR {scalars['auc_lr']:.4f} LGBM {scalars['auc_lgbm']:.4f} | "
          f"threshold {t_oof:.3f} | {money(scalars['cost_oof_single'])}/app")
    if not detailed:
        return scalars, None

    detail = dict(tr=tr, te=te, y_tr=y_tr, y_te=y_te, p_lr=p_lr, p_te=p_te, oof=oof,
                  rates=rates, fn_tr=fn_tr, fp_tr=fp_tr, fn_te=fn_te, fp_te=fp_te,
                  term_tr=term_tr, term_te=term_te, t_oof=t_oof, t_oof_lr=t_oof_lr,
                  term_t=term_t, t_oracle=t_oracle, sens=sens, sens_results=sens_results,
                  gbm=gbm, cost_of=cost_of)
    return scalars, detail


# ============================================================ report sections

def section_dataset(df, audit):
    lines = ["## 1. Dataset and censoring", ""]
    vt = audit["vintage_status_by_term"]
    lines += [f"2014 vintage: {audit['vintage_rows']:,} loans; {audit['final_rows']:,} with an "
              f"observed outcome; default rate {audit['default_rate']:.4f}.", ""]

    rows = []
    for term in ["36 months", "60 months"]:
        get = lambda s: vt.get(f"{term}|{s}", 0)
        d = get("Charged Off") + get("Default")
        p = get("Fully Paid")
        u = sum(get(s) for s in ["Current", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period"])
        total = d + p + u
        rows.append({"term": term, "loans": total, "unresolved": u,
                     "pct_unresolved": 100 * u / total,
                     "observed_default_rate": 100 * d / (d + p),
                     "if_all_unresolved_repay": 100 * d / total,
                     "if_all_unresolved_default": 100 * (d + u) / total})
    lines += ["Right-censoring bounds. The observed rate uses resolved loans only; the bounds",
              "assume every unresolved loan eventually repays, or eventually defaults.", "",
              md_table(pd.DataFrame(rows), {"loans": ",d", "unresolved": ",d", "pct_unresolved": ".1f",
                                            "observed_default_rate": ".1f",
                                            "if_all_unresolved_repay": ".1f",
                                            "if_all_unresolved_default": ".1f"}), "",
              "Unresolved 60-month loans are mostly `Current` -- on schedule several years in -- so",
              "the true 60-month rate is very likely below the observed figure, toward the lower",
              "bound. This is reasoning about who is censored, not a proof.", ""]

    # matured vs resolved
    issued = pd.to_datetime(df["issue_d"], format="%b-%Y")
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    months = (last.dt.year - issued.dt.year) * 12 + (last.dt.month - issued.dt.month)
    term = df["term"].astype("string").str.extract(r"(\d+)")[0].astype(float)
    m60 = term == 60
    lines += ["Matured vs resolved: months from issue to last recorded payment, 60-month loans.",
              "(For charged-off loans the charge-off itself comes some months after this date.)", "",
              f"- resolved before month 60: {100 * (months[m60] < 60).mean():.1f}%",
              f"- median months to last payment: {months[m60].median():.0f}", ""]
    return lines


def section_missingness(df):
    rows = []
    for c in MISSINGNESS_INDICATOR_COLS:
        miss = df[c].isna()
        ct = pd.crosstab(miss, df["target"])
        rows.append({"column": f"`{c}`", "n_missing": int(miss.sum()),
                     "default_if_missing": 100 * df.loc[miss, "target"].mean(),
                     "default_if_present": 100 * df.loc[~miss, "target"].mean(),
                     "lift_pp": 100 * (df.loc[miss, "target"].mean() - df.loc[~miss, "target"].mean()),
                     "p_value": chi2_contingency(ct)[1]})
    t = pd.DataFrame(rows).sort_values("lift_pp", key=abs, ascending=False)
    return ["## 2. Missingness as signal", "",
            "Computed on all labelled loans, for DESCRIPTION only. Not used to choose features:",
            "every column below gets an indicator, so outcome information never shapes the",
            "feature set. Note the sign flip -- missing delinquency history means good news,",
            "missing employment length means bad news.", "",
            md_table(t, {"n_missing": ",d", "default_if_missing": ".1f", "default_if_present": ".1f",
                         "lift_pp": "+.2f", "p_value": ".1e"}), ""]


def section_models(d, X):
    y_te = d["y_te"]
    rows = []
    for name, p in [("Logistic regression", d["p_lr"]), ("LightGBM", d["p_te"])]:
        s = calibration_summary(y_te, p)
        rows.append({"model": name, "test_auc": roc_auc_score(y_te, p), "brier": s["brier"],
                     "mean_predicted": s["mean_predicted"], "base_rate": s["base_rate"],
                     "max_bin_gap_pp": 100 * s["max_abs_bin_gap"], "top_bin_gap_pp": 100 * s["top_bin_gap"]})
    plot_calibration({"Logistic regression": (y_te, d["p_lr"]), "LightGBM": (y_te, d["p_te"])},
                     FIGURES_DIR / "calibration.png")
    cal = calibration_table(y_te, d["p_te"])
    return ["## 3. Models and calibration", "",
            f"{X.shape[1]} features. Test set: {len(y_te):,} loans (seed {SEED}).", "",
            md_table(pd.DataFrame(rows), {"test_auc": ".4f", "brier": ".4f", "mean_predicted": ".4f",
                                          "base_rate": ".4f", "max_bin_gap_pp": ".2f",
                                          "top_bin_gap_pp": "+.2f"}), "",
            "LightGBM reliability by quantile bin:", "",
            md_table(cal, {"n": ",d", "mean_predicted": ".4f", "observed_rate": ".4f", "gap": "+.4f"}),
            "", "![calibration](figures/calibration.png)", ""]


def section_costs(df, d):
    full = []
    for label, mask in [("Defaulted", df["target"] == 1), ("Fully paid", df["target"] == 0)]:
        g = df[mask]
        net = g["total_pymnt_inv"] - g["funded_amnt_inv"]
        full.append({"outcome": label, "loans": len(g), "mean_funded": money(g["funded_amnt_inv"].mean()),
                     "mean_received": money(g["total_pymnt_inv"].mean()), "mean_net": money(net.mean()),
                     "net_pct_of_funded": 100 * (net / g["funded_amnt_inv"]).mean()})
    rates = d["rates"]
    rt = pd.DataFrame([{"term": f"{t} months", "loss_rate_L": 100 * r.loss_rate,
                        "profit_rate_G": 100 * r.profit_rate, "ratio_L_over_G": r.cost_ratio,
                        "bayes_threshold": r.breakeven_threshold, "train_loans": r.n_defaults + r.n_repaid}
                       for t, r in sorted(rates.items())])
    return ["## 4. Measured costs", "",
            "Realised investor cash flows, all labelled loans (descriptive):", "",
            md_table(pd.DataFrame(full), {"loans": ",d", "net_pct_of_funded": "+.1f"}), "",
            "Rates used for decisions, measured on TRAINING rows only, per term:", "",
            md_table(rt, {"loss_rate_L": ".1f", "profit_rate_G": ".1f", "ratio_L_over_G": ".2f",
                          "bayes_threshold": ".3f", "train_loans": ",d"}), "",
            "Bayes threshold = G/(L+G): decline when the posterior exceeds it.", ""]


def section_decisions(d):
    y, p, fn, fp, cost_of = d["y_te"], d["p_te"], d["fn_te"], d["fp_te"], d["cost_of"]
    gbm = d["gbm"]
    tr_pred = gbm.predict_proba(d["gbm_train_frame"])[:, 1]  # IN-SAMPLE: deliberately wrong
    t_in, _ = optimal_threshold(sweep_thresholds(d["y_tr"], tr_pred, d["fn_tr"], d["fp_tr"]))
    t36, t60 = d["term_t"].get(36), d["term_t"].get(60)

    rows = [
        decision_row("Decline everything", "—", np.ones(len(y), bool), y, fn, fp),
        decision_row("Fund everything", "—", np.zeros(len(y), bool), y, fn, fp),
        decision_row("Fixed 0.5 cutoff", "0.500", p >= 0.5, y, fn, fp),
        decision_row("Fitted on IN-SAMPLE train predictions (wrong)", f"{t_in:.3f}", p >= t_in, y, fn, fp),
        decision_row("**Fitted out-of-fold, single**", f"**{d['t_oof']:.3f}**", p >= d["t_oof"], y, fn, fp),
        decision_row("**Fitted out-of-fold, per term**", f"**{t36:.3f} / {t60:.3f}**",
                     apply_segment_thresholds(d["term_te"], p, d["term_t"], d["t_oof"]), y, fn, fp),
        decision_row("*Test-fitted oracle (unattainable)*", f"*{d['t_oracle']:.3f}*",
                     p >= d["t_oracle"], y, fn, fp),
    ]
    t = pd.DataFrame(rows)
    t["cost_per_application"] = t["cost_per_application"].map(money)
    lr_cost = cost_of(d["p_lr"] >= d["t_oof_lr"])
    gbm_cost = cost_of(p >= d["t_oof"])

    # cost curve figure: the measured-cost policy sits on the flat floor; policies
    # chosen under mis-specified costs land on the steep wall.
    sweep = sweep_thresholds(y, p, fn, fp)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["cost_per_application"], lw=1.8, c="#33415c",
            label="realised cost (measured per-term costs)")
    ax.axvline(0.5, ls=":", c="grey", lw=1, label="fixed 0.5 cutoff")
    markers = [("measured costs", "o", "#2a9d8f"), ("loss 70%, profit measured", "s", "#e9c46a"),
               ("loss measured, profit 5%", "^", "#f4a261"), ("loss 70%, profit 5%", "D", "#e76f51")]
    for name, m, c in markers:
        key = "measured" if name == "measured costs" else name
        t_s, cost_s, _ = d["sens_results"][key]
        ax.plot(t_s, cost_s, m, ms=9, c=c, mec="k",
                label=f"threshold chosen assuming {name}: {t_s:.3f} -> {money(cost_s)}")
    ax.set(xlabel="decline threshold on P(default)", ylabel="regret per application ($)",
           title="What a wrong cost matrix costs (LightGBM, test set, seed 42)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_curve.png", dpi=150)
    plt.close(fig)

    return ["## 5. Decision performance (LightGBM, held-out test set)", "",
            "Cost = regret against a perfect decision: funded defaults cost L x amount, declined",
            "repayments cost G x amount, correct decisions cost zero.", "",
            md_table(t, {"decline_rate": ".1%", "approved_default_rate": ".1%"}), "",
            f"Model choice, priced in dollars: logistic regression with its own OOF-fitted",
            f"threshold ({d['t_oof_lr']:.3f}) costs {money(lr_cost)}/app vs LightGBM {money(gbm_cost)}/app,",
            f"LR minus LightGBM: {money(lr_cost - gbm_cost)} per application. Section 9 shows",
            "the spread of this difference across splits.", "",
            "![cost curve](figures/cost_curve.png)", ""], {"t_in": t_in, "lr_cost": lr_cost,
                                                           "gbm_cost": gbm_cost}


def section_sensitivity(d):
    base = d["sens_results"]["measured"][1]
    rows = []
    for name, (t, cost, decl) in d["sens_results"].items():
        r = d["sens"][name]
        rows.append({"cost assumption used to CHOOSE the threshold": name,
                     "L / G (36mo)": f"{r[36].loss_rate:.0%} / {r[36].profit_rate:.0%}",
                     "chosen_threshold": t, "decline_rate": decl,
                     "true_cost_per_app": money(cost), "vs_measured": f"{cost / base:.2f}x"})
    return ["## 6. Sensitivity: what if the costs were assumed instead of measured?", "",
            "Each row picks a threshold (out-of-fold) under an ASSUMED cost matrix, then scores",
            "that policy on the test set with the MEASURED costs. The alternatives are",
            "illustrative inputs for a stress test, not cited industry figures. Varying one",
            "input at a time shows which assumption drives the damage.", "",
            md_table(pd.DataFrame(rows), {"chosen_threshold": ".3f", "decline_rate": ".1%"}), ""]


def section_theory(d):
    rows = []
    for term, r in sorted(d["rates"].items()):
        rows.append({"term": f"{term} months", "bayes_G_over_L_plus_G": r.breakeven_threshold,
                     "oof_fitted_threshold": d["term_t"][term],
                     "gap": d["term_t"][term] - r.breakeven_threshold})
    te = d["te"]
    amt = pd.Series(d["meta"]["funded_amnt_inv"].to_numpy()[te])
    q = pd.qcut(amt, 5, labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
    qt = pd.DataFrame({"amount_quintile": q, "amt": amt, "y": d["y_te"], "p": d["p_te"]}).groupby(
        "amount_quintile", observed=True).agg(mean_funded=("amt", "mean"), observed_default=("y", "mean"),
                                              mean_predicted=("p", "mean")).reset_index()
    qt["mean_funded"] = qt["mean_funded"].map(money)
    return ["## 7. Theory vs practice: is the Bayes threshold recovered?", "",
            "With calibrated posteriors the fitted threshold should sit near G/(L+G):", "",
            md_table(pd.DataFrame(rows), {"bayes_G_over_L_plus_G": ".3f", "oof_fitted_threshold": ".3f",
                                          "gap": "+.3f"}), "",
            "Remaining gap: both error costs scale with amount funded, and larger loans default",
            "more often, so the cost-weighted optimum shifts relative to the per-loan rule:", "",
            md_table(qt, {"observed_default": ".3f", "mean_predicted": ".3f"}), ""]


def section_segments(d):
    te = d["te"]
    meta = d["meta"]
    decline = d["p_te"] >= d["t_oof"]
    args = (d["y_te"], d["p_te"], d["fn_te"], d["fp_te"], decline)
    income = pd.qcut(meta["annual_inc"].to_numpy()[te], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    fmt = {"n": ",d", "default_rate": ".3f", "mean_predicted": ".3f", "auc": ".4f",
           "decline_rate": ".1%", "cost_per_application": ",.0f"}
    out = ["## 8. Segments (single OOF threshold)", ""]
    for name, labels in [("grade", meta["grade"].to_numpy()[te]),
                         ("term", d["term_te"]), ("income quartile", np.asarray(income))]:
        out += [f"By {name}:", "", md_table(segment_report(labels, *args), fmt), ""]
    out += ["Within-grade AUC is far below overall AUC: most ranking power comes from",
            "re-deriving LendingClub's own grade. Calibration holds inside every segment.", ""]

    # apparent vs honest segment thresholds
    t_glob_test = d["t_oracle"]
    base_apparent = d["cost_of"](d["p_te"] >= t_glob_test)
    base_honest = d["cost_of"](d["p_te"] >= d["t_oof"])
    rows = []
    for seg in ["term", "grade", "purpose"]:
        lab_tr = (d["term_tr"] if seg == "term" else meta[seg].to_numpy()[d["tr"]])
        lab_te = (d["term_te"] if seg == "term" else meta[seg].to_numpy()[te])
        apparent, _, _ = segment_threshold_cost(lab_te, d["y_te"], d["p_te"], d["fn_te"], d["fp_te"],
                                                lab_te, d["y_te"], d["p_te"], d["fn_te"], d["fp_te"],
                                                t_glob_test)
        honest, _, _ = segment_threshold_cost(lab_tr, d["y_tr"], d["oof"], d["fn_tr"], d["fp_tr"],
                                              lab_te, d["y_te"], d["p_te"], d["fn_te"], d["fp_te"],
                                              d["t_oof"])
        rows.append({"segment": seg, "apparent_gain_test_fitted": base_apparent - apparent,
                     "honest_gain_oof_fitted": base_honest - honest})
    t = pd.DataFrame(rows)
    for c in ["apparent_gain_test_fitted", "honest_gain_oof_fitted"]:
        t[c] = t[c].map(lambda v: f"{'+' if v >= 0 else '−'}${abs(v):.2f}")
    out += ["Per-segment thresholds, gain per application vs a single threshold:", "",
            md_table(t), "",
            "Apparent = thresholds fitted and scored on the test set (selects on noise).",
            "Honest = fitted on out-of-fold training predictions, scored on test.", ""]
    return out, rows


def section_robustness(all_scalars):
    frame = pd.DataFrame(all_scalars)
    cols = {"auc_lr": ".4f", "auc_lgbm": ".4f", "threshold_oof": ".3f", "cost_fund_all": ",.2f",
            "cost_at_0.5": ",.2f", "cost_oof_single": ",.2f", "cost_oof_per_term": ",.2f",
            "cost_lr_oof_single": ",.2f", "lr_minus_lgbm_cost": "+,.2f",
            "gain_tuning_vs_0.5": "+,.2f", "gain_per_term": "+,.2f",
            "gain_grade_thresholds": "+,.2f", "gain_purpose_thresholds": "+,.2f",
            "ratio_loss70": ".2f", "ratio_profit5": ".2f", "ratio_assumed_70_5": ".2f"}
    rows = [{"metric": c, "mean": frame[c].mean(), "std": frame[c].std(ddof=1),
             "min": frame[c].min(), "max": frame[c].max(), "_fmt": f} for c, f in cols.items()]
    lines = []
    for r in rows:
        f = r.pop("_fmt")
        for k in ["mean", "std", "min", "max"]:
            r[k] = format(r[k], f)
    seeds = ", ".join(str(s) for s in frame["seed"])
    return ["## 9. Robustness across train/test splits", "",
            f"Seeds: {seeds}. Each seed is a different stratified split, with models, OOF",
            "predictions, cost rates and thresholds all refitted.", "",
            md_table(pd.DataFrame(rows)), "",
            "Gains are per application; positive = the refinement saved money on the test set.",
            "A gain whose sign changes across seeds is not a finding.", "",
            f"LightGBM minus LR AUC per seed: "
            + ", ".join(f"{v:+.4f}" for v in frame["auc_lgbm"] - frame["auc_lr"]) + ".", ""], frame


# ============================================================ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="primary seed only")
    args = parser.parse_args()
    seeds = [SEED] if args.quick else ROBUSTNESS_SEEDS
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, X, Xg, numeric, categorical, meta = load()
    audit = json.loads(TARGET_AUDIT_FILE.read_text())
    print(f"{len(df):,} loans, {X.shape[1]} features. Seeds: {seeds}")

    all_scalars, detail = [], None
    for seed in seeds:
        scalars, d = run_seed(seed, X, Xg, numeric, categorical, meta, detailed=(seed == SEED))
        all_scalars.append(scalars)
        if d is not None:
            detail = d
    detail["meta"] = meta
    detail["gbm_train_frame"] = Xg.iloc[detail["tr"]]

    decisions, dec_extra = section_decisions(detail)
    segments, seg_rows = section_segments(detail)
    robustness, rob_frame = section_robustness(all_scalars)

    lines = ["# Results", "",
             "Generated by `scripts/run_full_analysis.py`. Do not edit by hand.",
             f"Primary seed {SEED}; section 9 repeats the key numbers across {len(seeds)} seed(s).", ""]
    lines += section_dataset(df, audit)
    lines += section_missingness(df)
    lines += section_models(detail, X)
    lines += section_costs(df, detail)
    lines += decisions
    lines += section_sensitivity(detail)
    lines += section_theory(detail)
    lines += segments
    lines += robustness
    (REPORTS_DIR / "results.md").write_text("\n".join(lines))

    primary = next(s for s in all_scalars if s["seed"] == SEED)
    summary = {"primary_seed": primary, "in_sample_threshold": dec_extra["t_in"],
               "per_term_thresholds": {str(k): v for k, v in detail["term_t"].items()},
               "cost_rates": {str(k): {"loss": v.loss_rate, "profit": v.profit_rate}
                              for k, v in detail["rates"].items()},
               "sensitivity": {k: {"threshold": t, "true_cost": c, "decline_rate": dr}
                               for k, (t, c, dr) in detail["sens_results"].items()},
               "segment_thresholds": seg_rows,
               "robustness": {c: {"mean": float(rob_frame[c].mean()), "std": float(rob_frame[c].std(ddof=1))}
                              for c in rob_frame.columns if c != "seed"}}
    (REPORTS_DIR / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote reports/results.md, reports/results.json, reports/figures/*.png")


if __name__ == "__main__":
    main()
