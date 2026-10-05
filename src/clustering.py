"""Clustering: group ATMs into Low / Medium / High demand tiers from withdrawal behaviour (PySpark MLlib)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.master("local[*]").appName("ATM-clustering").config("spark.sql.shuffle.partitions", "8").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
daily = spark.read.csv("data/atm_daily.csv", header=True, inferSchema=True)

# ---------- ATM-level behavioural profile (one row per ATM) ----------
live = daily.filter("tx_count > 0")
prof = (daily.groupBy("atm_id", "location_type").agg(
    F.avg("total_withdrawal").alias("avg_daily_wd"),
    F.percentile_approx("total_withdrawal", 0.9).alias("p90_daily_wd"),
    (F.stddev("total_withdrawal") / F.avg("total_withdrawal")).alias("cv_wd"),
    F.avg("tx_count").alias("avg_daily_tx"),
    (F.sum("total_withdrawal") / F.sum("tx_count")).alias("avg_txn_amount"),
    (F.avg(F.when(F.col("is_weekend") == 1, F.col("total_withdrawal"))) / F.avg(F.when(F.col("is_weekend") == 0, F.col("total_withdrawal")))).alias("weekend_ratio"),
    (F.avg(F.when(F.col("is_payday") == 1, F.col("total_withdrawal"))) / F.avg(F.when(F.col("is_payday") == 0, F.col("total_withdrawal")))).alias("payday_uplift"),
    (F.max("total_withdrawal") / F.avg("total_withdrawal")).alias("peak_to_mean"),
    (F.sum("failed_tx") / F.sum("tx_count")).alias("fail_rate"),
)).cache()
cols = ["avg_daily_wd", "p90_daily_wd", "avg_daily_tx", "cv_wd", "avg_txn_amount", "weekend_ratio", "payday_uplift", "peak_to_mean", "fail_rate"]
cluster_cols = ["avg_daily_wd", "p90_daily_wd", "avg_daily_tx"]   # volume features drive the tiers; others are only used to profile them
# demand is right-skewed (a few very busy ATMs) -> log-transform the volume features before scaling
for c in cluster_cols:
    prof = prof.withColumn("log_" + c, F.log1p(c))
vec = VectorAssembler(inputCols=["log_" + c for c in cluster_cols], outputCol="raw").transform(prof)
sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(vec)
data = sc.transform(vec).cache()

# ---------- Choose k: elbow (WSSSE) + silhouette ----------
ev = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
sel = {}
for k in range(2, 8):
    m = KMeans(k=k, seed=11, maxIter=100, initMode="k-means||").fit(data)
    sel[k] = {"wssse": m.summary.trainingCost, "silhouette": ev.evaluate(m.transform(data))}
    print("k", k, sel[k])

# ---------- Final model: K-Means k=3 (business requirement: Low / Medium / High) + alternatives ----------
km = KMeans(k=3, seed=11, maxIter=100).fit(data)
alt = {"K-Means (k=3)": ev.evaluate(km.transform(data)),
       "Gaussian Mixture (k=3)": ev.evaluate(GaussianMixture(k=3, seed=11).fit(data).transform(data).select("features", "prediction"))}
print(alt)
out = km.transform(data)
# rank clusters by mean daily withdrawal -> Low / Medium / High
order = [r["prediction"] for r in out.groupBy("prediction").agg(F.avg("avg_daily_wd").alias("m")).orderBy("m").collect()]
tier = {c: t for c, t in zip(order, ["Low", "Medium", "High"])}
tier_udf = F.udf(lambda c: tier[c])
out = out.withColumn("demand_tier", tier_udf("prediction"))
summary = (out.groupBy("demand_tier").agg(F.count("*").alias("n_atms"), *[F.round(F.avg(c), 2).alias(c) for c in cols]).toPandas()
           .set_index("demand_tier").loc[["Low", "Medium", "High"]])
mix = out.groupBy("location_type", "demand_tier").count().toPandas().pivot(index="location_type", columns="demand_tier", values="count").fillna(0).astype(int)[["Low", "Medium", "High"]]
print(summary.T); print(mix)

pdf = out.select("atm_id", "location_type", "demand_tier", *cols).toPandas()
pdf.to_csv("results/atm_clusters.csv", index=False)
summary.to_csv("results/cluster_profile.csv"); mix.to_csv("results/cluster_location_mix.csv")

# PCA for 2-D visualisation
pca = PCA(k=2, inputCol="features", outputCol="pc").fit(data)
pc = pca.transform(out).select("demand_tier", "pc").toPandas()
xy = np.array([r.toArray() for r in pc["pc"]])
json.dump({"k_selection": sel, "algorithm_comparison_silhouette": alt, "profile": summary.reset_index().to_dict("records"),
           "location_mix": mix.reset_index().to_dict("records"), "pca_explained_variance": pca.explainedVariance.toArray().tolist(),
           "n_atms": len(pdf)}, open("results/clustering_results.json", "w"), indent=2)

colors = {"Low": "#2da44e", "Medium": "#bf8700", "High": "#cf222e"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ks = list(sel)
ax[0].plot(ks, [sel[k]["wssse"] for k in ks], "o-", color="#1f6feb"); ax[0].set_title("Elbow: within-cluster SSE vs k"); ax[0].set_xlabel("k")
ax[1].plot(ks, [sel[k]["silhouette"] for k in ks], "o-", color="#8250df"); ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k")
for t in colors:
    m = (pc["demand_tier"] == t).values
    ax[2].scatter(xy[m, 0], xy[m, 1], s=18, c=colors[t], label=f"{t} ({m.sum()})", alpha=.8)
ax[2].legend(); ax[2].set_title("ATM clusters (PCA projection)"); ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")
plt.tight_layout(); plt.savefig("results/clustering_plots.png", dpi=160)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
bx = [pdf.loc[pdf.demand_tier == t, "avg_daily_wd"] / 1e3 for t in colors]
b = ax.boxplot(bx, tick_labels=list(colors), patch_artist=True)
for p, c in zip(b["boxes"], colors.values()): p.set_facecolor(c)
ax.set_ylabel("Average daily withdrawal (₹ '000)"); ax.set_title("Withdrawal level by demand tier")
plt.tight_layout(); plt.savefig("results/clustering_boxplot.png", dpi=160)
spark.stop()
