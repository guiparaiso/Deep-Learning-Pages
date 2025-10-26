# Exercício 1 — Exploring Class Separability in 2D
# Requisitos: numpy, matplotlib, scikit-learn, (opcional: pandas)
# Ex.: pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ---------- Configurações ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUT_DIR = "figs_ex1"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Parâmetros do dataset (4 classes em 2D) ----------
# Se preferir, substitua 'means' e 'covs' pelos parâmetros exatos do enunciado.
means = [
    [2, 3],   # classe 0
    [5, 6],   # classe 1
    [8, 1],   # classe 2
    [15, 4]    # classe 3
]

covs = [
    [0.8,2.5],    # classe 0: compacto
    [1.2,1.9],    # classe 1: um pouco elíptico
    [0.9,0.9],# classe 2: pequena rotação
    [0.5,2.0]     # classe 3: mais disperso
]

n_per_class = 100  # total 400

# ---------- Gerar dados ----------
X_list = []
y_list = []
for i in range(4):
    # transformar [σx, σy] em matriz diagonal [[σx², 0], [0, σy²]]
    cov_matrix = np.diag(np.array(covs[i]) ** 2)
    Xc = np.random.multivariate_normal(mean=means[i], cov=cov_matrix, size=n_per_class)
    X_list.append(Xc)
    y_list.append(np.full(n_per_class, i))

X = np.vstack(X_list)   # (400,2)
y = np.concatenate(y_list)

# embaralhar
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# ---------- Plot: scatter 2D ----------
plt.figure(figsize=(7,6))
scatter = plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap="tab10", edgecolor='k', alpha=0.9)
plt.title("Exercise 1 — 2D scatter of 4 classes (400 samples)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_4classes.png"), dpi=200)
plt.show()

# ---------- Pré-processamento: padronizar (importante para MLP) ----------
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25,
                                                    random_state=1, stratify=y)

# ---------- Treinar classificador linear (multinomial logistic) ----------
clf_lin = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, random_state=RANDOM_SEED)
clf_lin.fit(X_train, y_train)
y_pred_lin = clf_lin.predict(X_test)
acc_lin = accuracy_score(y_test, y_pred_lin)

# ---------- Treinar MLP com ativação tanh ----------
clf_mlp = MLPClassifier(hidden_layer_sizes=(20,10), activation='tanh', max_iter=2000, random_state=2)
clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"Accuracy (Logistic Regression - linear): {acc_lin:.3f}")
print(f"Accuracy (MLP tanh - non-linear): {acc_mlp:.3f}\n")
print("Classification Report (MLP):")
print(classification_report(y_test, y_pred_mlp, digits=3))

# ---------- Função para plotar regiões de decisão no espaço original ----------
def plot_decision_regions(clf, scaler, X_orig, y, title, fname=None, cmap_name="tab10"):
    # grid no espaço original (não padronizado)
    x_min, x_max = X_orig[:,0].min() - 1.0, X_orig[:,0].max() + 1.0
    y_min, y_max = X_orig[:,1].min() - 1.0, X_orig[:,1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # transformar para o mesmo espaço que o classificador espera
    grid_s = scaler.transform(grid)
    Z = clf.predict(grid_s)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap=plt.get_cmap(cmap_name))
    plt.scatter(X_orig[:,0], X_orig[:,1], c=y, s=30, cmap=plt.get_cmap(cmap_name), edgecolor='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    plt.show()

# Plot regiões de decisão (Logistic)
plot_decision_regions(
    clf_lin, scaler, X, y,
    f"Decision Regions — Logistic Regression (test acc={acc_lin:.3f})",
    fname=os.path.join(OUT_DIR, "decision_lin.png")
)

# Plot decision regions (MLP)
plot_decision_regions(
    clf_mlp, scaler, X, y,
    f"Decision Regions — MLP tanh (test acc={acc_mlp:.3f})",
    fname=os.path.join(OUT_DIR, "decision_mlp.png")
)


# ---------- Plot ilustrativo: retas possíveis (manual) ----------
plt.figure(figsize=(7,6))
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap="tab10", edgecolor='k')
# duas retas ilustrativas
plt.plot([1.5,1.5], [X[:,1].min()-1, X[:,1].max()+1], linestyle='--', linewidth=2)  # x = 1.5
plt.plot([X[:,0].min()-1, X[:,0].max()+1], [1.5,1.5], linestyle='--', linewidth=2)  # y = 1.5
plt.title("Scatter with illustrative lines (possible linear boundaries)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.grid(alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_with_lines.png"), dpi=200)
plt.show()

# ---------- Resumo (para relatório) ----------
summary = {
    "X_shape": X.shape,
    "y_shape": y.shape,
    "means_used": means,
    "covariances_used": covs,
    "logistic_acc": float(acc_lin),
    "mlp_acc": float(acc_mlp)
}

# imprimir resumo
print("\nResumo dos experimentos:")
for k,v in summary.items():
    print(f" - {k}: {v}")

# (Opcional) salvar resumo em arquivo
import json
with open(os.path.join(OUT_DIR, "summary_ex1.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ---------------- FIM ----------------
