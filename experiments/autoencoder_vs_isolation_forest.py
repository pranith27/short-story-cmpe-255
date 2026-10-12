"""
Anomaly detection comparison: autoencoder vs Isolation Forest.

Trains on normal data only (unsupervised setting, as described in the survey)
and compares a reconstruction-based deep learning approach (MLP autoencoder)
against a classical baseline (Isolation Forest) on synthetic data with
injected anomalies.

Setup:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
Run:    .venv/bin/python autoencoder_vs_isolation_forest.py

Figures are saved to ../assets/.
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: figures are written to files
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

rng = np.random.RandomState(42)

# ------------------------------------------------------------ 1. Dataset
# Two Gaussian "normal" blobs plus uniformly scattered anomalies.
# The model sees only normal data during training.
N_NORMAL = 1000
blob_a = rng.normal(loc=[0, 0], scale=0.8, size=(N_NORMAL // 2, 2))
blob_b = rng.normal(loc=[5, 5], scale=0.8, size=(N_NORMAL // 2, 2))
X_normal = np.vstack([blob_a, blob_b])

N_ANOMALY = 60
X_anomaly = rng.uniform(low=-4, high=9, size=(N_ANOMALY, 2))

X_train = X_normal
X_test = np.vstack([X_normal, X_anomaly])
y_test = np.array([0] * len(X_normal) + [1] * len(X_anomaly))  # 1 = anomaly

# ------------------------------------------- 2. Classical baseline method
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(X_train)
iso_scores = -isolation_forest.score_samples(X_test)  # higher = more anomalous

# --------------------------------------- 3. Reconstruction-based method
# MLP trained to reconstruct its own input. The narrow bottleneck forces it
# to model "normal"; anomalies reconstruct poorly, so reconstruction error
# serves as the anomaly score.
autoencoder = MLPRegressor(hidden_layer_sizes=(8, 2, 8),
                           max_iter=500, random_state=42)
autoencoder.fit(X_train, X_train)
ae_scores = ((X_test - autoencoder.predict(X_test)) ** 2).mean(axis=1)

# ------------------------------------------------------------ 4. Metrics
for name, scores in [("IsolationForest", iso_scores),
                     ("Autoencoder", ae_scores)]:
    print(f"{name:16s} ROC-AUC: {roc_auc_score(y_test, scores):.3f}")

# ------------------------------------------------------------ 5. Figures
fig, ax = plt.subplots()
ax.scatter(X_normal[:, 0], X_normal[:, 1], s=8, alpha=0.5, label="normal")
ax.scatter(X_anomaly[:, 0], X_anomaly[:, 1], s=20, alpha=0.9, label="anomaly")
ax.legend()
ax.set_title("Synthetic data: normal clusters with injected anomalies")
fig.savefig("../assets/anomaly_data_distribution.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, scores, title in zip(axes,
                            [iso_scores, ae_scores],
                            ["IsolationForest anomaly scores",
                             "Autoencoder reconstruction error"]):
    a.hist(scores[y_test == 0], bins=40, alpha=0.6, label="normal")
    a.hist(scores[y_test == 1], bins=40, alpha=0.6, label="anomaly")
    a.set_title(title)
    a.legend()
fig.suptitle("Score distributions on held-out data")
fig.savefig("../assets/anomaly_score_distributions.png", dpi=150)
print("Saved ../assets/anomaly_data_distribution.png")
print("Saved ../assets/anomaly_score_distributions.png")

# ------------------------------------------------- 6. Experiment notes
# Parameters worth varying when extending this comparison:
#  * Bottleneck width: hidden_layer_sizes=(2,) vs (16, 4, 16) — effect of
#    compression strength on reconstruction-based detection quality.
#  * Contamination: IsolationForest(contamination=0.01) vs (0.10).
#  * Overlap: draw anomalies from low=-1, high=6 so they overlap the normal
#    clusters; observe which method degrades more and why.
