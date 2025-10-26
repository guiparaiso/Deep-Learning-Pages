import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)

# --- PARAMETERS (anisotropic: std is a length-2 vector) ---
params = {
    0: {"mean": np.array([2.0, 3.0]),  "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]),  "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]),  "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}
samples_per_class = 100

# --- GENERATE SAMPLES ---
records = []
for label, cfg in params.items():
    mean = cfg["mean"]
    std = cfg["std"]  # length-2 -> diagonal (anisotropic)
    pts = rng.normal(loc=mean, scale=std, size=(samples_per_class, 2))
    for p in pts:
        records.append({"x1": float(p[0]), "x2": float(p[1]), "label": int(label)})
df = pd.DataFrame(records)

# --- SAVE CSV & PREVIEW ---
csv_path = "synthetic_gaussians.csv"
df.to_csv(csv_path, index=False)
print("CSV salvo em:", csv_path)
print(df.head())

# --- PLOT DATA ---
plt.figure(figsize=(6,6))
for label in sorted(df['label'].unique()):
    sub = df[df['label'] == label]
    plt.scatter(sub['x1'], sub['x2'], label=f'class {label}', alpha=0.7, s=20)
plt.legend()
plt.title('Synthetic Gaussian dataset (2D, 4 classes, diagonal cov)')
plt.xlabel('x1'); plt.ylabel('x2'); plt.grid(True); plt.tight_layout()
plt.savefig("synthetic_gaussians_plot.png", dpi=120)

plt.show()

# --- ANALYSE & DRAW (fronteiras lineares aproximadas) ---
# Modelo linear one-vs-rest só para desenhar fronteiras de decisão
# X = df[['x1','x2']].values
# y = df['label'].values
# clf = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr'))
# clf.fit(X, y)

# # grade para contorno das fronteiras
# xx, yy = np.meshgrid(
#     np.linspace(df['x1'].min()-1, df['x1'].max()+1, 400),
#     np.linspace(df['x2'].min()-1, df['x2'].max()+1, 400)
# )
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# # sobrepõe fronteiras (linhas de mudança de classe)
# plt.contour(xx, yy, Z, levels=sorted(df['label'].unique()), linewidths=1)
# plt.show()
