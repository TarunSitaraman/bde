# Predictive ATM Cash Management using Big Data Analytics

BDE assignment: **classification** (high vs low weekly cash demand) and **clustering** (Low/Medium/High ATM tiers) on Apache Spark (PySpark MLlib).

| Path | Purpose |
|---|---|
| `src/generate_data.py` | Synthetic data: 300 ATMs x 2 years = 219,300 daily records (seed 42) |
| `src/classification.py` | Weekly features, LR / Random Forest / GBT, time-based split, metrics |
| `src/clustering.py` | ATM profiles, K-Means (k selection by elbow + silhouette), tier profiling |
| `src/build_docs.py` | Builds the report and slides from `results/*.json` |
| `results/` | Metrics (JSON), plots, per-ATM cluster assignments |
| `docs/` | `BDE_Assignment_Report.docx`, `BDE_Assignment_Presentation.pptx` |

```
pip install -r requirements.txt   # needs Java 17+
python src/generate_data.py && python src/classification.py && python src/clustering.py && python src/build_docs.py
```
Headline results: best classifier Logistic Regression 95.8% accuracy / AUC 0.994 (naive baseline 90.6%); K-Means k=3 tiers of 81/114/105 ATMs, silhouette 0.67.
Data is synthetic (no public per-ATM dataset); re-validate on real bank data.
