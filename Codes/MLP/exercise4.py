import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import the SAME MLP class (reusing code from Exercise 2 and 3!)
from mlp import MLP

"""
Exercise 4: Multi-Class Classification with Deeper MLP
- Same dataset as Exercise 3 (1500 samples, 3 classes, 4 features)
- Deep architecture with AT LEAST 2 hidden layers
- Focus: Demonstrating deeper architecture performance
"""

print("="*70)
print("EXERCISE 4: DEEP MLP (2+ HIDDEN LAYERS)")
print("="*70)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC DATA (Same as Exercise 3)
# ============================================================================
print("\nSTEP 1: Generating synthetic dataset (same as Exercise 3)...")

# Class 0: 2 clusters (500 samples)
X_class0, y_class0 = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=2,
    n_classes=1,
    random_state=42,
    flip_y=0.05,
    class_sep=1.2
)
y_class0 = np.zeros(len(y_class0))

# Class 1: 3 clusters (500 samples)
X_class1, y_class1 = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=3,
    n_classes=1,
    random_state=24,
    flip_y=0.05,
    class_sep=1.2
)
y_class1 = np.ones(len(y_class1))

# Class 2: 4 clusters (500 samples)
X_class2, y_class2 = make_classification(
    n_samples=500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_clusters_per_class=4,
    n_classes=1,
    random_state=99,
    flip_y=0.05,
    class_sep=1.2
)
y_class2 = np.full(len(y_class2), 2)

# Combine datasets
X = np.vstack([X_class0, X_class1, X_class2])
y = np.hstack([y_class0, y_class1, y_class2])

# Shuffle
shuffle_idx = np.random.RandomState(42).permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Dataset shape: {X.shape}")
print(f"Total samples: {len(y)}")
print(f"Class 0: {np.sum(y == 0)} samples (2 clusters)")
print(f"Class 1: {np.sum(y == 1)} samples (3 clusters)")
print(f"Class 2: {np.sum(y == 2)} samples (4 clusters)")

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================
print("\nSTEP 2: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# Visualize data
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
colors = ['blue', 'red', 'green']
class_names = ['Class 0 (2 clusters)', 'Class 1 (3 clusters)', 'Class 2 (4 clusters)']
for i, (color, name) in enumerate(zip(colors, class_names)):
    mask = y_train == i
    plt.scatter(X_train[mask, 0], X_train[mask, 1], 
               c=color, label=name, alpha=0.6, edgecolors='k', s=30)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data Distribution (First 2 Features)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i, (color, name) in enumerate(zip(colors, class_names)):
    mask = y_test == i
    plt.scatter(X_test[mask, 0], X_test[mask, 1], 
               c=color, label=f'Class {i}', alpha=0.6, edgecolors='k', s=30)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data Distribution (First 2 Features)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise4_data.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise4_data.png")
plt.show()

# ============================================================================
# STEP 3: BUILD AND TRAIN DEEP MLP
# ============================================================================
print("\nSTEP 3: Building and training DEEP MLP...")
print("NOTE: Using the SAME MLP class, but with DEEPER architecture!\n")

# Deep MLP Architecture:
# - Input layer: 4 features
# - Hidden layer 1: 32 neurons
# - Hidden layer 2: 16 neurons
# - Hidden layer 3: 8 neurons  <- ADDED THIRD HIDDEN LAYER!
# - Output layer: 3 neurons
mlp_deep = MLP(
    layer_sizes=[4, 32, 16, 8, 3],  # 3 hidden layers!
    activation='tanh',
    learning_rate=0.01
)

print(f"Deep MLP Architecture: {mlp_deep.layer_sizes}")
print(f"Number of hidden layers: {len(mlp_deep.layer_sizes) - 2}")
print(f"Activation function: {mlp_deep.activation}")
print(f"Learning rate: {mlp_deep.lr}")

# Count parameters
total_params = sum(w.size for w in mlp_deep.weights) + sum(b.size for b in mlp_deep.biases)
print(f"Total parameters: {total_params}")

# Show layer-by-layer parameter count
print("\nLayer-by-layer breakdown:")
for i in range(len(mlp_deep.layer_sizes) - 1):
    in_size = mlp_deep.layer_sizes[i]
    out_size = mlp_deep.layer_sizes[i + 1]
    layer_params = (in_size * out_size) + out_size
    print(f"  Layer {i+1}: [{in_size} → {out_size}] = {layer_params} parameters")

# Train the model
print("\nTraining Deep MLP...")
mlp_deep.train(X_train, y_train, epochs=400, batch_size=32, verbose=True)

print("\n✓ Training completed!")

# ============================================================================
# STEP 4: TRAIN SHALLOW MLP FOR COMPARISON
# ============================================================================
print("\nSTEP 4: Training shallow MLP for comparison...")

# Shallow MLP (same as Exercise 3)
mlp_shallow = MLP(
    layer_sizes=[4, 16, 8, 3],  # 2 hidden layers
    activation='tanh',
    learning_rate=0.01
)

print(f"Shallow MLP Architecture: {mlp_shallow.layer_sizes}")
print(f"Number of hidden layers: {len(mlp_shallow.layer_sizes) - 2}")

# Train
mlp_shallow.train(X_train, y_train, epochs=400, batch_size=32, verbose=False)
print("✓ Shallow MLP training completed!")

# ============================================================================
# STEP 5: EVALUATE BOTH MODELS
# ============================================================================
print("\nSTEP 5: Evaluating both models...")

# Deep MLP predictions
y_pred_train_deep = mlp_deep.predict(X_train)
y_pred_test_deep = mlp_deep.predict(X_test)
train_acc_deep = accuracy_score(y_train, y_pred_train_deep)
test_acc_deep = accuracy_score(y_test, y_pred_test_deep)

# Shallow MLP predictions
y_pred_train_shallow = mlp_shallow.predict(X_train)
y_pred_test_shallow = mlp_shallow.predict(X_test)
train_acc_shallow = accuracy_score(y_train, y_pred_train_shallow)
test_acc_shallow = accuracy_score(y_test, y_pred_test_shallow)

print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"\nDEEP MLP [4, 32, 16, 8, 3]:")
print(f"  Training Accuracy: {train_acc_deep:.4f} ({train_acc_deep*100:.2f}%)")
print(f"  Test Accuracy:     {test_acc_deep:.4f} ({test_acc_deep*100:.2f}%)")
print(f"  Final Loss:        {mlp_deep.loss_history[-1]:.6f}")
print(f"  Parameters:        {sum(w.size for w in mlp_deep.weights) + sum(b.size for b in mlp_deep.biases)}")

print(f"\nSHALLOW MLP [4, 16, 8, 3]:")
print(f"  Training Accuracy: {train_acc_shallow:.4f} ({train_acc_shallow*100:.2f}%)")
print(f"  Test Accuracy:     {test_acc_shallow:.4f} ({test_acc_shallow*100:.2f}%)")
print(f"  Final Loss:        {mlp_shallow.loss_history[-1]:.6f}")
print(f"  Parameters:        {sum(w.size for w in mlp_shallow.weights) + sum(b.size for b in mlp_shallow.biases)}")

print(f"\nIMPROVEMENT:")
print(f"  Test Accuracy: {(test_acc_deep - test_acc_shallow)*100:+.2f}%")
print("="*70)

# Confusion matrices
cm_deep = confusion_matrix(y_test, y_pred_test_deep)
cm_shallow = confusion_matrix(y_test, y_pred_test_shallow)

print("\nConfusion Matrix - DEEP MLP:")
print(cm_deep)

print("\nConfusion Matrix - SHALLOW MLP:")
print(cm_shallow)

# Classification reports
print("\nClassification Report - DEEP MLP:")
print(classification_report(y_test, y_pred_test_deep, 
                          target_names=['Class 0', 'Class 1', 'Class 2']))

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\nSTEP 6: Creating visualizations...")

# Plot 1: Compare training loss curves
plt.figure(figsize=(12, 6))
plt.plot(mlp_deep.loss_history, label='Deep MLP [4,32,16,8,3]', linewidth=2)
plt.plot(mlp_shallow.loss_history, label='Shallow MLP [4,16,8,3]', linewidth=2, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Comparison: Deep vs Shallow MLP', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exercise4_loss_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise4_loss_comparison.png")
plt.show()

# Plot 2: Deep MLP Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Deep MLP
im1 = axes[0].imshow(cm_deep, cmap='Blues', interpolation='nearest')
axes[0].figure.colorbar(im1, ax=axes[0])
axes[0].set(xticks=np.arange(3), yticks=np.arange(3),
           xticklabels=['Class 0', 'Class 1', 'Class 2'],
           yticklabels=['Class 0', 'Class 1', 'Class 2'],
           ylabel='True Label', xlabel='Predicted Label',
           title=f'Deep MLP - Test Accuracy: {test_acc_deep*100:.2f}%')

thresh1 = cm_deep.max() / 2.
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, format(cm_deep[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_deep[i, j] > thresh1 else "black",
                    fontsize=16, fontweight='bold')

# Shallow MLP
im2 = axes[1].imshow(cm_shallow, cmap='Oranges', interpolation='nearest')
axes[1].figure.colorbar(im2, ax=axes[1])
axes[1].set(xticks=np.arange(3), yticks=np.arange(3),
           xticklabels=['Class 0', 'Class 1', 'Class 2'],
           yticklabels=['Class 0', 'Class 1', 'Class 2'],
           ylabel='True Label', xlabel='Predicted Label',
           title=f'Shallow MLP - Test Accuracy: {test_acc_shallow*100:.2f}%')

thresh2 = cm_shallow.max() / 2.
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, format(cm_shallow[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_shallow[i, j] > thresh2 else "black",
                    fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('exercise4_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise4_confusion_matrices.png")
plt.show()

# Plot 3: Predictions comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ground truth
for i, color in enumerate(colors):
    mask = y_test == i
    axes[0].scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=f'Class {i}', alpha=0.6, edgecolors='k', s=50)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Ground Truth')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Deep MLP predictions
for i, color in enumerate(colors):
    mask = y_pred_test_deep == i
    axes[1].scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=f'Class {i}', alpha=0.6, edgecolors='k', s=50)
misclass_deep = y_test != y_pred_test_deep
if np.any(misclass_deep):
    axes[1].scatter(X_test[misclass_deep, 0], X_test[misclass_deep, 1],
                   c='yellow', marker='x', s=200, linewidths=3,
                   label=f'Errors ({np.sum(misclass_deep)})', zorder=5)
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title(f'Deep MLP ({test_acc_deep*100:.1f}% acc)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Shallow MLP predictions
for i, color in enumerate(colors):
    mask = y_pred_test_shallow == i
    axes[2].scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=f'Class {i}', alpha=0.6, edgecolors='k', s=50)
misclass_shallow = y_test != y_pred_test_shallow
if np.any(misclass_shallow):
    axes[2].scatter(X_test[misclass_shallow, 0], X_test[misclass_shallow, 1],
                   c='orange', marker='x', s=200, linewidths=3,
                   label=f'Errors ({np.sum(misclass_shallow)})', zorder=5)
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
axes[2].set_title(f'Shallow MLP ({test_acc_shallow*100:.1f}% acc)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise4_predictions_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise4_predictions_comparison.png")
plt.show()

# Plot 4: Architecture visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Deep architecture
deep_layers = mlp_deep.layer_sizes
deep_y = 0.7
for i, size in enumerate(deep_layers):
    x = i / (len(deep_layers) - 1)
    circle = plt.Circle((x, deep_y), 0.05, color='steelblue', alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, deep_y - 0.12, f'{size}\nneurons', 
           ha='center', fontsize=10, fontweight='bold')
    if i < len(deep_layers) - 1:
        ax.arrow(x + 0.06, deep_y, 
                (1/(len(deep_layers)-1)) - 0.12, 0,
                head_width=0.03, head_length=0.03, 
                fc='steelblue', ec='steelblue', alpha=0.5)

ax.text(0.5, deep_y + 0.15, 'DEEP MLP', ha='center', fontsize=14, fontweight='bold')

# Shallow architecture  
shallow_layers = mlp_shallow.layer_sizes
shallow_y = 0.3
for i, size in enumerate(shallow_layers):
    x = i / (len(shallow_layers) - 1)
    circle = plt.Circle((x, shallow_y), 0.05, color='coral', alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, shallow_y - 0.12, f'{size}\nneurons', 
           ha='center', fontsize=10, fontweight='bold')
    if i < len(shallow_layers) - 1:
        ax.arrow(x + 0.06, shallow_y, 
                (1/(len(shallow_layers)-1)) - 0.12, 0,
                head_width=0.03, head_length=0.03, 
                fc='coral', ec='coral', alpha=0.5)

ax.text(0.5, shallow_y + 0.15, 'SHALLOW MLP', ha='center', fontsize=14, fontweight='bold')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Architecture Comparison', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('exercise4_architecture.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise4_architecture.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - EXERCISE 4")
print("="*70)
print(f"\nDataset: Same as Exercise 3 (1500 samples, 3 classes, 4 features)")
print(f"\nDEEP MLP:")
print(f"  Architecture: {mlp_deep.layer_sizes}")
print(f"  Hidden layers: 3")
print(f"  Test accuracy: {test_acc_deep*100:.2f}%")
print(f"  Parameters: {sum(w.size for w in mlp_deep.weights) + sum(b.size for b in mlp_deep.biases)}")
print(f"\nSHALLOW MLP:")
print(f"  Architecture: {mlp_shallow.layer_sizes}")
print(f"  Hidden layers: 2")
print(f"  Test accuracy: {test_acc_shallow*100:.2f}%")
print(f"  Parameters: {sum(w.size for w in mlp_shallow.weights) + sum(b.size for b in mlp_shallow.biases)}")
print(f"\n✓ Deep architecture demonstrated with 3 hidden layers!")
print("="*70)