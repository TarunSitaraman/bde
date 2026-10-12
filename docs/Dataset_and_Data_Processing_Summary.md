# Dataset and Data Processing Summary

Project: Predictive ATM Cash Management using Big Data Analytics (PySpark MLlib).
All figures below come from `data/atm_daily.csv` and `results/*.json`.

## 1. Dataset

**Source.** Synthetic. No public per-ATM transaction dataset was used, so `src/generate_data.py` (fixed seed 42) simulates one. Conclusions describe the method on this data, not a real bank's ATMs.

| Item | Value |
|---|---|
| Rows | 219,300 daily records (300 ATMs x 731 days) |
| Period | 2024-01-01 to 2025-12-31 |
| Columns | 9 |
| Missing values / duplicate (ATM, date) rows | 0 / 0 |
| Total cash dispensed | about ₹5,335 crore |
| Mean daily withdrawal per ATM | ₹243,299 (median ₹201,666, max ₹2,163,707) |
| Mean daily transactions per ATM | 74 (max 689) |
| Failed-transaction rate | 2.58% |
| Outage days (zero transactions) | 2,622 (1.2% of rows) |
| Weekend / payday / festival day share | 28.5% / 21.2% / 6.8% |

**Columns**

| Column | Meaning |
|---|---|
| `atm_id` | ATM0001 to ATM0300 |
| `date` | Day of activity |
| `location_type` | Branch, Mall, Transit_Hub, Residential, Highway, Hospital, Rural |
| `total_withdrawal` | Rupees dispensed that day |
| `tx_count` | Successful transactions |
| `failed_tx` | Failed transactions |
| `is_weekend`, `is_payday`, `is_festival` | Calendar flags (payday = day 1-3 or 28+; festival = 3 days before to 1 day after 10 listed dates) |

**ATM mix and average daily withdrawal by location**

| Location | ATMs | Avg daily withdrawal |
|---|---|---|
| Transit_Hub | 22 | ₹513,118 |
| Mall | 35 | ₹378,772 |
| Branch | 72 | ₹305,177 |
| Residential | 84 | ₹192,414 |
| Hospital | 26 | ₹143,882 |
| Highway | 30 | ₹141,372 |
| Rural | 31 | ₹75,047 |

**How the data is simulated.** Each ATM gets a location-type base demand, scaled by its own log-normal factor (sigma 0.35). Daily demand is that base multiplied by:
- a weekend factor and a payday factor, both specific to the location type;
- a 1.35x festival uplift;
- a 10% growth trend over the two years;
- log-normal day-to-day noise (sigma 0.18).

Transaction count is withdrawal divided by an average ticket of about ₹3,200 (about 25% higher at transit hubs). Failures are binomial at an ATM-specific rate of around 2.5%. Random outages (1.2% of days) zero out that day's cash and transactions.

## 2. Data processing: classification (weekly demand)

1. **Load.** Spark reads the CSV with the schema inferred.
2. **Weekly aggregation.** Group by ATM, location and week (Monday start). Each week gets withdrawals (`wd`), transactions, failed transactions, payday days, festival days, weekend withdrawals and outage days.
3. **Complete weeks only.** Filter to weeks with all 7 days. This drops the final partial week (29-31 Dec 2025), leaving 104 full weeks.
4. **Features from past weeks only (Spark window functions, so no leakage):**
   - withdrawals 1 and 2 weeks back;
   - the 4-week rolling mean and standard deviation of withdrawals;
   - last week's transaction count, average ticket size, failure rate, weekend share and outage days.
5. **Features known in advance for the target week.** These are payday days, festival days, and week-of-year as sine and cosine.
6. **Drop warm-up rows.** `dropna()` removes the first 2 weeks of each ATM, which lack history. The result is 30,600 ATM-week rows (300 ATMs x 102 weeks).
7. **Time-based split at 2025-07-01.**
   - Train: 23,100 rows (77 weeks).
   - Test: 7,500 rows (25 weeks).
   - The test period is later than the train period, mimicking deployment.
8. **Label.** HIGH = weekly withdrawal above the training-set median of ₹1,432,552, so no future information is used. HIGH is 53.1% of test rows, because demand grows over time.
9. **Model input.** The location type is string-indexed and one-hot encoded and combined with 13 numeric features. Features are scaled for Logistic Regression.

## 3. Data processing: clustering (ATM profiles)

1. **One row per ATM over all 731 days (300 rows).** Computed: average, 90th-percentile and variability (CV) of daily withdrawal; average daily transactions; average ticket; weekend ratio; payday uplift; peak-to-mean; failure rate.
2. **Clustering features** are only three volume measures: average daily withdrawal, 90th-percentile daily withdrawal and average daily transactions.
3. **Log transform (`log1p`)**, then standardisation (mean 0, std 1). Demand is right-skewed. Without the log step, K-Means gave tiers of 193 / 102 / 5 ATMs.
4. **The other profile metrics** are not used to form the clusters. They only describe the tiers afterwards.
5. **Tier naming.** Clusters are ranked by mean withdrawal and named Low, Medium, High. This gives 81, 114 and 105 ATMs.

## 4. Headline outputs

- Classification: Logistic Regression 95.8% accuracy, AUC 0.994. Random Forest 95.1% and Gradient-Boosted Trees 95.1%. The naive "same class as last week" baseline scores 90.6%.
- Clustering: K-Means k=3, silhouette 0.670 (k=2 is 0.732, chosen against because three tiers are required). Gaussian Mixture k=3 scores 0.395.

## 5. Limitations

- Synthetic data with a clean multiplicative structure, so real-world accuracy will probably be lower.
- Demand is a continuum, so tier boundaries are partly a business convention.
- Spark's BisectingKMeans returned a single cluster on this data and version, so it is not part of the comparison.
