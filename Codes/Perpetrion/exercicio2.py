import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

# --- Parâmetros do enunciado ---
mu_A = np.array([0, 0, 0, 0, 0], dtype=float)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0]
], dtype=float)

mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float)
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5]
], dtype=float)

n_per_class = 500

# --- Geração dos dados 5D ---
X_A = rng.multivariate_normal(mean=mu_A, cov=Sigma_A, size=n_per_class)
X_B = rng.multivariate_normal(mean=mu_B, cov=Sigma_B, size=n_per_class)
X = np.vstack([X_A, X_B])
y = np.array([0]*n_per_class + [1]*n_per_class)  # 0 = A, 1 = B

# --- PCA (com padronização) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_

print(f"Variância explicada: PC1={evr[0]:.3f}, PC2={evr[1]:.3f} (total={evr.sum():.3f})")

# --- Plot 2D ---
plt.figure(figsize=(7,6))
plt.scatter(X_2d[y==0, 0], X_2d[y==0, 1], s=12, alpha=0.7, label='Classe A')
plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], s=12, alpha=0.7, label='Classe B')
plt.title('Ex.2 — PCA 5D→2D (A vs B)')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
