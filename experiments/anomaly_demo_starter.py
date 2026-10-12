"""
Starter experiment: "autoencoder" vs a classical baseline for anomaly detection.

This is YOUR hands-on section for the article — run it, look at the plots,
tweak the TODOs, and write up what YOU observe. The figures land in ../assets/.

Setup:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
Run:    .venv/bin/python anomaly_demo_starter.py
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display needed; figures are saved to files
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)

# ---------------------------------------------------------------- 1. Data
# Two "normal" blobs + scattered anomalies. Train on normal data only,
# exactly like the unsupervised setup described in the survey.
n_normal = 1000
blob1 = rng.normal(loc=[0, 0], scale=0.8, size=(n_normal // 2, 2))
blob2 = rng.normal(loc=[5, 5], scale=0.8, size=(n_normal // 2, 2))
X_normal = np.vstack([blob1, blob2])

n_anom = 60
X_anom = rng.uniform(low=-4, high=9, size=(n_anom, 2))

X_train = X_normal
X_test = np.vstack([X_normal, X_anom])
y_test = np.array([0] * len(X_normal) + [1] * len(X_anom))  # 1 = anomaly

# ------------------------------------------------- 2. Classical baseline
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)
iso_scores = -iso.score_samples(X_test)  # higher = more anomalous

# ------------------------------------------------- 3. Autoencoder-style
# An MLP trained to reconstruct its own input. The bottleneck forces it to
# learn "normal"; anomalies reconstruct poorly -> high error = anomaly score.
ae = MLPRegressor(hidden_layer_sizes=(8, 2, 8), max_iter=500, random_state=42)
ae.fit(X_train, X_train)
ae_scores = ((X_test - ae.predict(X_test)) ** 2).mean(axis=1)

# ---------------------------------------------------------------- 4. Metrics
for name, scores in [("IsolationForest", iso_scores), ("Autoencoder", ae_scores)]:
    print(f"{name:16s} ROC-AUC: {roc_auc_score(y_test, scores):.3f}")

# ---------------------------------------------------------------- 5. Plots
fig, ax = plt.subplots()
ax.scatter(X_normal[:, 0], X_normal[:, 1], s=8, alpha=0.5, label="normal")
ax.scatter(X_anom[:, 0], X_anom[:, 1], s=20, alpha=0.9, label="anomaly")
ax.legend()
ax.set_title("Synthetic data: normal blobs + anomalies")
fig.savefig("../assets/fig1_data.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, scores, title in zip(axes,
                            [iso_scores, ae_scores],
                            ["IsolationForest scores", "Autoencoder recon. error"]):
    a.hist(scores[y_test == 0], bins=40, alpha=0.6, label="normal")
    a.hist(scores[y_test == 1], bins=40, alpha=0.6, label="anomaly")
    a.set_title(title)
    a.legend()
fig.suptitle("Score distributions: how well do the two methods separate?")
fig.savefig("../assets/fig2_scores.png", dpi=150)
print("Saved ../assets/fig1_data.png and ../assets/fig2_scores.png")

# --------------------------------------------------------------------------
# TODOs for you (this is the "my own two cents" part of your article):
#  1. Bottleneck size: try hidden_layer_sizes=(2,) vs (16, 4, 16). How does
#     ROC-AUC change? What does that say about reconstruction-based methods?
#  2. Contamination: try IsolationForest(contamination=0.01) and (0.10).
#  3. Hard case: move the anomaly box inside the blobs (low=-1, high=6) so
#     anomalies overlap normal data. Which method degrades more, and why?
#  4. Write 3-4 sentences on what surprised you. That goes in your article.
