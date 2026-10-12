"""Assemble the self-contained notebook (BDE_ATM_Cash_Management.ipynb) from the repo's src/*.py files.

The code cells are the repo's own scripts split at their '# ----- step -----' markers, so the notebook can never
drift from src/. Run with --execute to also run it and embed the outputs.
"""
import re
import sys
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
URL = "https://github.com/TarunSitaraman/bde"


def chunks(path):
    """Split a script into [(title, code)] at '# ---------- title ----------' marker lines."""
    src = open(path).read()
    src = src.replace("import matplotlib\nmatplotlib.use(\"Agg\")\n", "")
    src = re.sub(r"^spark\.stop\(\)\n?", "", src, flags=re.M)
    src = re.sub(r"(plt\.savefig\([^\n]*\))", r"\1; plt.show()", src)
    parts = re.split(r"^# -{10} (.*?)(?: -{4,})?$", src, flags=re.M)
    out = [("Imports and Spark session", parts[0].strip())]
    for t, c in zip(parts[1::2], parts[2::2]):
        out.append((t.strip(), c.strip()))
    return out


cells = []
cells.append(md(f"""# Predictive ATM Cash Management using Big Data Analytics
**BDE Assignment – Classification + Clustering with Apache Spark (PySpark MLlib)**

This notebook is self-contained: it installs PySpark, generates the dataset, and runs both problems end to end. Every code cell is the same code as in `src/` of the repository ({URL}).

| Part | Contents |
|---|---|
| 0 | Setup and dataset (synthetic, 219,300 daily records) |
| 1 | **Classification** – predict HIGH vs LOW weekly cash demand per ATM |
| 2 | **Clustering** – group ATMs into Low / Medium / High demand tiers |

*Data note:* no public per-ATM transaction dataset was used. The data is simulated (fixed seed 42), so the results demonstrate the method; re-validate on real bank data."""))

cells.append(md("## 0. Setup and dataset"))
cells.append(code('# Colab already has Java; PySpark 4 needs Java 17+.\n%pip install -q "pyspark>=4.0" pyarrow pandas matplotlib\n!java -version 2>&1 | head -1\n\nimport os\nos.makedirs("data", exist_ok=True)\nos.makedirs("results", exist_ok=True)'))
cells.append(md("""### 0.1 Generate the dataset
Each row = one ATM on one day. 300 ATMs x 731 days = 219,300 rows. Demand depends on location type, weekend, payday (day 28–3) and festival effects, a 10% growth trend, noise, and random outages (1.2% of days)."""))
cells.append(code(open("src/generate_data.py").read().strip()))
cells.append(md("""### 0.2 Column dictionary and quick look
| Column | Meaning |
|---|---|
| `atm_id` | ATM identifier |
| `date` | Day of activity |
| `location_type` | Branch / Mall / Transit_Hub / Residential / Highway / Hospital / Rural |
| `total_withdrawal` | Rupees dispensed that day (0 on outage days) |
| `tx_count` | Successful transactions |
| `failed_tx` | Failed transactions |
| `is_weekend`, `is_payday`, `is_festival` | Calendar flags (0/1) |"""))
cells.append(code('''import pandas as pd
raw = pd.read_csv("data/atm_daily.csv", parse_dates=["date"])
print(raw.shape, raw.date.min().date(), "->", raw.date.max().date(), "| missing:", int(raw.isna().sum().sum()),
      "| duplicate (atm,date):", int(raw.duplicated(["atm_id", "date"]).sum()))
print("outage days:", int((raw.tx_count == 0).sum()), f"({(raw.tx_count == 0).mean():.1%})",
      "| failed-tx rate:", f"{raw.failed_tx.sum() / raw.tx_count.sum():.2%}")
display(raw.head())
display(raw.groupby("location_type").agg(atms=("atm_id", "nunique"), avg_daily_withdrawal=("total_withdrawal", "mean"),
                                         avg_daily_tx=("tx_count", "mean")).round(0).sort_values("avg_daily_withdrawal", ascending=False))'''))

# ---------------- Classification ----------------
cells.append(md("""---
# 1. Classification Problem

## 1.1 Problem statement
Predict, at the start of each week, whether an ATM will have **HIGH or LOW cash demand** that week, using only its past transaction/withdrawal patterns plus calendar information known in advance. HIGH = weekly withdrawals above the **training-set median**. Business value: send more or earlier cash to ATMs flagged HIGH (avoid stock-outs) and avoid idle cash at LOW ATMs.

## 1.2 Model and justification
Logistic Regression, Random Forest (150 trees) and Gradient-Boosted Trees (60 iterations) are compared against a naive baseline (“same class as last week”) on a **time-based split** (train: weeks before 2025-07-01, test: later weeks) so the models never see the future. The best model by AUC is selected; when models are close, the simpler, more explainable one is preferred. All lag features use past weeks only (no leakage).

## 1.3 Coding – Spark pipeline (same as `src/classification.py`)"""))
step_notes = {
    "1. Load + weekly aggregation (big-data style group-by on daily records)": "Spark reads the daily CSV and aggregates it to weekly ATM-level rows (complete 7-day weeks only).",
    "2. Features from PAST weeks only (no leakage); calendar features of the target week are known ahead": "Spark window functions create lagged and rolling features; payday/festival counts and week-of-year are known in advance.",
    "3. Time-based split + label threshold learned on TRAIN only": "Train on weeks before 2025-07-01, test on later weeks. The HIGH/LOW threshold is the training median.",
    "4. Feature importance + plots for best model": "Best model is chosen by AUC; feature importance (Random Forest), confusion matrix and comparison plots are saved.",
}
for title, c in chunks("src/classification.py"):
    note = step_notes.get(title, "")
    cells.append(md(f"#### {title}\n{note}"))
    cells.append(code(c))
cells.append(md("## 1.4 Results"))
cells.append(code('''import json
R = json.load(open("results/classification_results.json")); M = R["_meta"]
rows = [(n, R[n]["accuracy"], R[n]["precision"], R[n]["recall"], R[n]["f1"], R[n]["auc"]) for n in ["Logistic Regression", "Random Forest", "Gradient-Boosted Trees"]]
rows.append(("Naive baseline (last week's class)", R["Baseline (last week's class)"]["accuracy"], None, None, R["Baseline (last week's class)"]["f1"], None))
display(pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision (weighted)", "Recall (weighted)", "F1", "AUC"]).round(4))
print("Best model:", M["best_model"], "| label threshold: Rs", f"{M['threshold_inr_per_week']:,.0f}", "per week")
print("Rows – train:", M["train_rows"], "test:", M["test_rows"], "| HIGH share of test:", M["test_high_share"])
print("Confusion matrix:", R[M["best_model"]]["confusion"])
print("HIGH-class precision:", round(M["high_class_precision"], 4), "recall:", round(M["high_class_recall"], 4))
print("\\nTop Random-Forest features:")
for n, v in M["rf_feature_importance"][:6]: print(f"  {n:<14}{v:.3f}")'''))
cells.append(md("""## 1.5 Inference
- Logistic Regression is the best model (about 95.8% accuracy, AUC about 0.994), versus 90.6% for the naive “same as last week” rule – a gain of about 5 points, mostly from catching ATMs whose demand crosses the threshold in payday and festival weeks.
- Recent demand dominates: the 4-week rolling mean, last week’s withdrawals and transactions and the week before account for about 88% of Random-Forest importance; payday days come next, then location.
- The costly error is predicting LOW when demand is actually HIGH (stock-out): about 4.3% of high-demand weeks. Lowering the decision threshold trades extra false alarms for fewer misses.
- The three models are within about one point because the simulated demand is multiplicative and persistent; real data may favour boosted trees. Accuracy on real data will probably be lower.

## 1.6 URL of implementation
""" + URL + " (`src/classification.py`, results in `results/`)"))

# ---------------- Clustering ----------------
cells.append(md("""---
# 2. Clustering Problem

## 2.1 Problem statement
Group the 300 ATMs into **Low-, Medium- and High-demand** categories purely from their withdrawal behaviour (no labels), so replenishment frequency and cash limits can be set per tier instead of per machine.

## 2.2 Model and justification
**K-Means with k = 3** on log-transformed, standardised volume features (average daily withdrawal, 90th-percentile daily withdrawal, average daily transactions).
- Unsupervised, distance-based and scalable on Spark MLlib.
- k = 3 matches the required Low/Medium/High tiers. k is also examined for 2–7 with the elbow (WSSSE) and silhouette; k = 2 scores a higher silhouette but cannot express a medium tier.
- Demand is right-skewed, so a log transform is applied. Without it, K-Means produced tiers of 193 / 102 / 5 ATMs.
- A Gaussian Mixture is run for comparison. Other behavioural metrics (variability, weekend ratio, payday uplift, failure rate) only *profile* the tiers; they do not form them.

## 2.3 Coding – Spark pipeline (same as `src/clustering.py`)"""))
cl_notes = {
    "ATM-level behavioural profile (one row per ATM)": "One behavioural profile per ATM, computed with a Spark `groupBy` over all 731 days. A log transform is applied to the volume features before scaling.",
    "Choose k: elbow (WSSSE) + silhouette": "K-Means is trained for k = 2…7; WSSSE and silhouette are recorded.",
    "Final model: K-Means k=3 (business requirement: Low / Medium / High) + alternatives": "Final K-Means (k=3), compared with a Gaussian Mixture. Clusters are ranked by mean withdrawal and named Low / Medium / High, then profiled.",
}
for title, c in chunks("src/clustering.py"):
    cells.append(md(f"#### {title}\n{cl_notes.get(title, '')}"))
    cells.append(code(c))
cells.append(md("## 2.4 Results"))
cells.append(code('''K = json.load(open("results/clustering_results.json"))
display(pd.DataFrame(K["k_selection"]).T.rename_axis("k").round(3))
print("Algorithm comparison (silhouette):", {k: round(v, 3) for k, v in K["algorithm_comparison_silhouette"].items()})
display(pd.read_csv("results/cluster_profile.csv", index_col="demand_tier").round(2))
display(pd.read_csv("results/cluster_location_mix.csv", index_col="location_type"))'''))
cells.append(md("""## 2.5 Inference
- Final K-Means (k = 3) has silhouette about 0.67 and splits the 300 ATMs into Low (81), Medium (114) and High (105) tiers.
- The High tier withdraws about 4.2x the cash of the Low tier (about Rs 406k vs Rs 96k per day), so one replenishment schedule would starve High ATMs or over-stock Low ones.
- Tiers follow location: Malls, Transit hubs and Branches are mostly High; Residential mostly Medium; Rural, Highway and Hospital mostly Low – so new sites can be pre-assigned a tier.
- Low-demand ATMs show the strongest payday uplift (about 1.35x), so they still need month-end top-ups.
- Suggested policy: High – daily monitoring; Medium – every 2–3 days; Low – weekly plus payday top-up; the Part 1 classifier then adjusts each ATM weekly.
- Caveat: demand is a continuum, so tier boundaries are partly a business convention.

## 2.6 URL of implementation
""" + URL + " (`src/clustering.py`, assignments in `results/atm_clusters.csv`)"))

cells.append(md("---\n## 3. Conclusion and reproducibility\nBoth BDE problems are implemented with PySpark MLlib: the classifier predicts next-week HIGH/LOW demand and the clustering groups ATMs into demand tiers; together they give short-term prediction and long-term segmentation for ATM cash management. Seeds are fixed (data 42; models 7 and 11), so re-running reproduces the numbers above."))
cells.append(code("spark.stop()"))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                 "language_info": {"name": "python"}, "colab": {"provenance": []}})
if "--execute" in sys.argv:
    from nbconvert.preprocessors import ExecutePreprocessor
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "BDE_ATM_Cash_Management.ipynb")
print("wrote BDE_ATM_Cash_Management.ipynb with", len(cells), "cells")
