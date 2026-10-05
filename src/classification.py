"""Classification: will an ATM have HIGH or LOW cash demand next week? (PySpark MLlib)"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.master("local[*]").appName("ATM-classification").config("spark.sql.shuffle.partitions", "8").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ---------- 1. Load + weekly aggregation (big-data style group-by on daily records) ----------
daily = spark.read.csv("data/atm_daily.csv", header=True, inferSchema=True)
weekly = (daily.withColumn("week", F.date_trunc("week", "date"))
          .groupBy("atm_id", "location_type", "week")
          .agg(F.sum("total_withdrawal").alias("wd"), F.sum("tx_count").alias("tx"), F.sum("failed_tx").alias("failed"),
               F.sum("is_payday").alias("payday_days"), F.sum("is_festival").alias("festival_days"),
               F.sum(F.when(F.col("is_weekend") == 1, F.col("total_withdrawal")).otherwise(0)).alias("wd_weekend"),
               F.sum(F.when(F.col("tx_count") == 0, 1).otherwise(0)).alias("outage_days"),
               F.count("*").alias("n_days"))
          .filter("n_days = 7"))

# ---------- 2. Features from PAST weeks only (no leakage); calendar features of the target week are known ahead ----------
w = Window.partitionBy("atm_id").orderBy("week")
w4 = w.rowsBetween(-4, -1)
feat = (weekly
        .withColumn("lag1_wd", F.lag("wd", 1).over(w)).withColumn("lag2_wd", F.lag("wd", 2).over(w))
        .withColumn("lag1_tx", F.lag("tx", 1).over(w))
        .withColumn("roll4_wd", F.avg("wd").over(w4)).withColumn("roll4_std", F.stddev("wd").over(w4))
        .withColumn("lag1_avg_amt", F.lag(F.col("wd") / F.greatest(F.col("tx"), F.lit(1)), 1).over(w))
        .withColumn("lag1_fail_rate", F.lag(F.col("failed") / F.greatest(F.col("tx"), F.lit(1)), 1).over(w))
        .withColumn("lag1_weekend_share", F.lag(F.col("wd_weekend") / F.greatest(F.col("wd"), F.lit(1)), 1).over(w))
        .withColumn("lag1_outage_days", F.lag("outage_days", 1).over(w))
        .withColumn("woy_sin", F.sin(2 * np.pi * F.weekofyear("week") / 52)).withColumn("woy_cos", F.cos(2 * np.pi * F.weekofyear("week") / 52))
        .dropna())

# ---------- 3. Time-based split + label threshold learned on TRAIN only ----------
cut = "2025-07-01"
train_raw, test_raw = feat.filter(F.col("week") < cut), feat.filter(F.col("week") >= cut)
thr = train_raw.approxQuantile("wd", [0.5], 0.001)[0]          # high demand = above median weekly withdrawal
label = lambda d: d.withColumn("label", (F.col("wd") > thr).cast("double"))
train, test = label(train_raw), label(test_raw)

num = ["lag1_wd", "lag2_wd", "lag1_tx", "roll4_wd", "roll4_std", "lag1_avg_amt", "lag1_fail_rate", "lag1_weekend_share",
       "lag1_outage_days", "payday_days", "festival_days", "woy_sin", "woy_cos"]
stages = [StringIndexer(inputCol="location_type", outputCol="loc_idx"),
          OneHotEncoder(inputCols=["loc_idx"], outputCols=["loc_vec"]),
          VectorAssembler(inputCols=num + ["loc_vec"], outputCol="raw")]

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=100),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=150, maxDepth=8, seed=7),
    "Gradient-Boosted Trees": GBTClassifier(featuresCol="features", labelCol="label", maxIter=60, maxDepth=5, seed=7),
}
auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
ev = lambda m: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=m)

results, fitted = {}, {}
for name, clf in models.items():
    extra = [StandardScaler(inputCol="raw", outputCol="features")] if name == "Logistic Regression" else []
    if name != "Logistic Regression":
        extra = [VectorAssembler(inputCols=["raw"], outputCol="features")]
    pm = Pipeline(stages=stages + extra + [clf]).fit(train)
    pred = pm.transform(test)
    cm = pred.groupBy("label", "prediction").count().toPandas()
    g = lambda l, p: int(cm[(cm.label == l) & (cm.prediction == p)]["count"].sum())
    results[name] = {"accuracy": ev("accuracy").evaluate(pred), "precision": ev("weightedPrecision").evaluate(pred),
                     "recall": ev("weightedRecall").evaluate(pred), "f1": ev("f1").evaluate(pred), "auc": auc.evaluate(pred),
                     "confusion": {"TN": g(0, 0), "FP": g(0, 1), "FN": g(1, 0), "TP": g(1, 1)}}
    fitted[name] = pm
    print(name, {k: round(v, 4) for k, v in results[name].items() if k != "confusion"}, results[name]["confusion"])

# Baseline: "same class as last week" (naive persistence) to show the model adds value
base = test.withColumn("prediction", (F.col("lag1_wd") > thr).cast("double"))
results["Baseline (last week's class)"] = {"accuracy": ev("accuracy").evaluate(base), "f1": ev("f1").evaluate(base)}
print("Baseline", results["Baseline (last week's class)"])

# ---------- 4. Feature importance + plots for best model ----------
best = max((k for k in models), key=lambda k: results[k]["auc"])
rf = fitted["Random Forest"]
names = num + [f"loc_{c}" for c in rf.stages[0].labels[:-1]]
imp = sorted(zip(names, rf.stages[-1].featureImportances.toArray()), key=lambda t: -t[1])
hc = results[best]["confusion"]
results["_meta"] = {"best_model": best, "high_class_recall": hc["TP"] / (hc["TP"] + hc["FN"]), "high_class_precision": hc["TP"] / (hc["TP"] + hc["FP"]), "threshold_inr_per_week": thr, "train_rows": train.count(), "test_rows": test.count(),
                    "weekly_rows": feat.count(), "daily_rows": daily.count(), "split": cut,
                    "test_high_share": test.agg(F.avg("label")).first()[0],
                    "rf_feature_importance": [(n, float(v)) for n, v in imp]}
json.dump(results, open("results/classification_results.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
top = imp[:10][::-1]
ax[0].barh([t[0] for t in top], [t[1] for t in top], color="#1f6feb"); ax[0].set_title("Random Forest – top 10 feature importances")
cmv = results[best]["confusion"]
M = np.array([[cmv["TN"], cmv["FP"]], [cmv["FN"], cmv["TP"]]])
ax[1].imshow(M, cmap="Blues"); ax[1].set_xticks([0, 1], ["Low", "High"]); ax[1].set_yticks([0, 1], ["Low", "High"])
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title(f"{best} – confusion matrix (test)")
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, f"{M[i, j]:,}", ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black", fontsize=13)
plt.tight_layout(); plt.savefig("results/classification_plots.png", dpi=160)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ms = list(models) ; x = np.arange(len(ms))
for k, (m, c) in enumerate([("accuracy", "#1f6feb"), ("f1", "#2da44e"), ("auc", "#bf8700")]):
    ax.bar(x + (k - 1) * 0.26, [results[n][m] for n in ms], 0.26, label=m.upper() if m != "accuracy" else "Accuracy", color=c)
ax.set_xticks(x, ms); ax.set_ylim(0.5, 1.0); ax.legend(); ax.set_title("Model comparison on held-out future weeks")
plt.tight_layout(); plt.savefig("results/classification_model_comparison.png", dpi=160)
spark.stop()
