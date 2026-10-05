"""Generate a synthetic ATM daily-transaction dataset (no public ATM dataset is bundled).

300 ATMs x 2 years (2024-01-01 .. 2025-12-31) of daily records, with realistic structure:
location-specific base demand, day-of-week, payday and festival spikes, random outages.
"""
import numpy as np
import pandas as pd

SEED = 42
N_ATMS = 300
START, END = "2024-01-01", "2025-12-31"
LOC = {  # base daily withdrawal (INR), weekend multiplier, payday sensitivity, prob
    "Branch":      (260_000, 0.80, 1.25, 0.20),
    "Mall":        (320_000, 1.35, 1.15, 0.12),
    "Transit_Hub": (420_000, 1.05, 1.10, 0.10),
    "Residential": (150_000, 1.10, 1.45, 0.28),
    "Highway":     (110_000, 1.20, 1.05, 0.10),
    "Hospital":    (130_000, 1.00, 1.10, 0.08),
    "Rural":       (60_000,  0.95, 1.60, 0.12),
}
FESTIVALS = pd.to_datetime(["2024-03-25", "2024-08-15", "2024-10-12", "2024-11-01", "2024-12-25",
                            "2025-03-14", "2025-08-15", "2025-10-02", "2025-10-20", "2025-12-25"])


def main(out="data/atm_daily.csv"):
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    names, probs = list(LOC), np.array([v[3] for v in LOC.values()])
    types = rng.choice(names, N_ATMS, p=probs / probs.sum())
    atm_scale = rng.lognormal(0, 0.35, N_ATMS)
    # festival window: 3 days before to 1 day after
    fest = np.zeros(len(dates), bool)
    for f in FESTIVALS:
        fest |= (dates >= f - pd.Timedelta(days=3)) & (dates <= f + pd.Timedelta(days=1))
    dow = dates.dayofweek.values
    weekend = dow >= 5
    dom = dates.day.values
    payday = (dom <= 3) | (dom >= 28)
    trend = 1 + 0.10 * np.arange(len(dates)) / len(dates)  # 10% growth over two years
    rows = []
    for i in range(N_ATMS):
        base, wk, pay, _ = LOC[types[i]]
        mult = np.where(weekend, wk, 1.0) * np.where(payday, pay, 1.0) * np.where(fest, 1.35, 1.0) * trend
        mu = base * atm_scale[i] * mult
        withdrawal = mu * rng.lognormal(0, 0.18, len(dates))
        avg_amt = rng.normal(3200, 250) * (1 + 0.25 * (types[i] == "Transit_Hub")) * rng.lognormal(0, 0.06, len(dates))
        tx = np.maximum(1, (withdrawal / avg_amt)).astype(int)
        failed = rng.binomial(tx, np.clip(rng.normal(0.025, 0.01), 0.005, 0.06))
        outage = rng.random(len(dates)) < 0.012  # ATM down: no cash dispensed
        withdrawal = np.where(outage, 0.0, withdrawal)
        tx = np.where(outage, 0, tx)
        rows.append(pd.DataFrame({
            "atm_id": f"ATM{i+1:04d}", "date": dates, "location_type": types[i],
            "total_withdrawal": withdrawal.round(0), "tx_count": tx, "failed_tx": np.where(outage, 0, failed),
            "is_weekend": weekend.astype(int), "is_payday": payday.astype(int), "is_festival": fest.astype(int)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df):,} rows, {df.atm_id.nunique()} ATMs")


if __name__ == "__main__":
    main()
