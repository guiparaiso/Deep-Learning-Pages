# Exercise 2 — Non-Linearity in Higher Dimensions
# Requirements: numpy, matplotlib, scikit-learn, seaborn (optional)
# pip install numpy matplotlib scikit-learn seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

np.random.seed(0)
OUT_DIR = "ex2_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Parameters from the assignment ----------------
# Class A
mu_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0]
])

# Class B
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5]
])

n_A = 500
n_B = 500

# ---------------- Generate data ----------------
X_A = np.random.multivariate_normal(mean=mu_A, cov=Sigma_A, size=n_A)
X_B = np.random.multivariate_normal(mean=mu_B, cov=Sigma_B, size=n_B)

X = np.vstack([X_A, X_B])
y = np.hstack([np.zeros(n_A, dtype=int), np.ones(n_B, dtype=int)])  # 0 = A, 1 = B

# ---------------- PCA projection to 2D for visualization ----------------
scaler = StandardScaler()
Xs = scaler.fit_transform(X)   # standardize before PCA (recommended)

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(Xs)

# Save explained variance info
explained = pca.explained_variance_ratio_

# Scatter plot of PCA projection
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=palette, s=30, edgecolor='k', alpha=0.9)
plt.title(f"PCA projection (2D) — Class A vs B  (explained var: {explained[0]:.2f}, {explained[1]:.2f})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(["Class A","Class B"])
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_scatter.png"), dpi=200)
plt.show()

# ---------------- Train/Test split and classifiers ----------------
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=1, stratify=y)

# Linear classifier (logistic)
clf_lin = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=0)
clf_lin.fit(X_train, y_train)
y_pred_lin = clf_lin.predict(X_test)
acc_lin = accuracy_score(y_test, y_pred_lin)

# Non-linear classifier (MLP with tanh)
clf_mlp = MLPClassifier(hidden_layer_sizes=(50,25), activation='tanh', max_iter=2000, random_state=2)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print("Logistic Regression (linear) test accuracy: ", acc_lin)
print("MLP (tanh) test accuracy: ", acc_mlp)
print("\nClassification report (MLP):")
print(classification_report(y_test, y_pred_mlp, digits=4))

# ---------------- Decision region visualization (in PCA 2D space) ----------------
# We'll create a classifier pipeline that maps grid points through the same scaler+PCA and then uses classifier
from sklearn.pipeline import make_pipeline

# pipeline for linear in original space -> project to PCA 2D for plotting
pipe_lin = make_pipeline(clf_lin, )
pipe_mlp = make_pipeline(clf_mlp, )

# Create grid in PCA space
x_min, x_max = X_pca[:,0].min() - 1.0, X_pca[:,0].max() + 1.0
y_min, y_max = X_pca[:,1].min() - 1.0, X_pca[:,1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
grid_pca = np.c_[xx.ravel(), yy.ravel()]

# To classify grid points we must map them back to original standardized 5D space.
# Approx: use PCA.components_ and mean_ to reconstruct approximate original standardized coordinates:
# Xs ≈ (X_pca) @ pca.components_ + pca.mean_
grid_Xs_approx = grid_pca @ pca.components_ + pca.mean_  # shape (n_points, 5)
# Now predict with classifiers trained on Xs
Z_lin = clf_lin.predict(grid_Xs_approx).reshape(xx.shape)
Z_mlp = clf_mlp.predict(grid_Xs_approx).reshape(xx.shape)

# Plot decision regions (linear)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z_lin, alpha=0.25, cmap=plt.get_cmap("coolwarm"))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=palette, s=30, edgecolor='k', alpha=0.9)
plt.title(f"Decision Regions — Logistic Regression (test acc={acc_lin:.3f})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(["Class A","Class B"])
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_lin_pca.png"), dpi=200)
plt.show()

# Plot decision regions (MLP)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z_mlp, alpha=0.25, cmap=plt.get_cmap("coolwarm"))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=palette, s=30, edgecolor='k', alpha=0.9)
plt.title(f"Decision Regions — MLP tanh (test acc={acc_mlp:.3f})")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(["Class A","Class B"])
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_mlp_pca.png"), dpi=200)
plt.show()

# ---------------- Save a short summary ----------------
summary = {
    "explained_variance_ratio": explained.tolist(),
    "acc_logistic": float(acc_lin),
    "acc_mlp": float(acc_mlp)
}
import json
with open(os.path.join(OUT_DIR, "summary_ex2.json"), "w") as f:
    json.dump(summary, f, indent=2)

