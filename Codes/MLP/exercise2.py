import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import the MLP class
from mlp import MLP

"""
Exercise 2: Binary Classification with Synthetic Data
- 1000 samples
- 2 classes
- 1 cluster for class 0, 2 clusters for class 1
- 2 features for visualization
"""

print("="*70)
print("EXERCISE 2: BINARY CLASSIFICATION")
print("="*70)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC DATA
# ============================================================================
print("\nSTEP 1: Generating synthetic dataset...")

# Strategy: Generate each class separately with different n_clusters_per_class
# Class 0: 1 cluster (400 samples)
X_class0, y_class0 = make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=1,
    random_state=42,
    flip_y=0.05,
    class_sep=1.5
)
y_class0 = np.zeros(len(y_class0))

# Class 1: 2 clusters (600 samples)
X_class1, y_class1 = make_classification(
    n_samples=600,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=2,
    n_classes=1,
    random_state=24,
    flip_y=0.05,
    class_sep=1.5
)
y_class1 = np.ones(len(y_class1))

# Combine datasets
X = np.vstack([X_class0, X_class1])
y = np.hstack([y_class0, y_class1])

# Shuffle
shuffle_idx = np.random.RandomState(42).permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Dataset shape: {X.shape}")
print(f"Total samples: {len(y)}")
print(f"Class 0: {np.sum(y == 0)} samples")
print(f"Class 1: {np.sum(y == 1)} samples")

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================
print("\nSTEP 2: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Visualize data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
           c='blue', label='Class 0 (1 cluster)', alpha=0.6, edgecolors='k', s=30)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
           c='red', label='Class 1 (2 clusters)', alpha=0.6, edgecolors='k', s=30)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], 
           c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=30)
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], 
           c='red', label='Class 1', alpha=0.6, edgecolors='k', s=30)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise2_data.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise2_data.png")
plt.show()

# ============================================================================
# STEP 3: BUILD AND TRAIN MLP
# ============================================================================
print("\nSTEP 3: Building and training MLP...")

# MLP Architecture:
# - Input layer: 2 features
# - Hidden layer 1: 8 neurons
# - Hidden layer 2: 4 neurons
# - Output layer: 1 neuron (binary classification)
mlp = MLP(
    layer_sizes=[2, 8, 4, 1],
    activation='tanh',
    learning_rate=0.01
)

print(f"MLP Architecture: {mlp.layer_sizes}")
print(f"Activation function: {mlp.activation}")
print(f"Learning rate: {mlp.lr}")
print(f"Number of parameters: {sum(w.size for w in mlp.weights) + sum(b.size for b in mlp.biases)}")

# Train the model
print("\nTraining...")
mlp.train(X_train, y_train, epochs=200, batch_size=32, verbose=True)

print("\n✓ Training completed!")

# ============================================================================
# STEP 4: EVALUATE ON TEST SET
# ============================================================================
print("\nSTEP 4: Evaluating on test set...")

# Make predictions
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1']))

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\nSTEP 5: Creating visualizations...")

# Plot 1: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_history, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exercise2_loss.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise2_loss.png")
plt.show()

# Plot 2: Decision Boundary
print("\nGenerating decision boundary...")
mlp.plot_decision_boundary(X_test, y_test, resolution=0.02)
plt.savefig('exercise2_decision_boundary.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise2_decision_boundary.png")

# Plot 3: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

# Labels
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Class 0', 'Class 1'],
       yticklabels=['Class 0', 'Class 1'],
       ylabel='True Label',
       xlabel='Predicted Label',
       title='Confusion Matrix')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig('exercise2_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise2_confusion_matrix.png")
plt.show()

# Plot 4: Prediction Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ground truth
axes[0].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], 
               c='blue', label='Class 0', alpha=0.6, edgecolors='k', s=50)
axes[0].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], 
               c='red', label='Class 1', alpha=0.6, edgecolors='k', s=50)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Ground Truth')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predictions
axes[1].scatter(X_test[y_pred_test==0, 0], X_test[y_pred_test==0, 1], 
               c='blue', label='Predicted Class 0', alpha=0.6, edgecolors='k', s=50)
axes[1].scatter(X_test[y_pred_test==1, 0], X_test[y_pred_test==1, 1], 
               c='red', label='Predicted Class 1', alpha=0.6, edgecolors='k', s=50)

# Highlight misclassified points
misclassified = y_test != y_pred_test
if np.any(misclassified):
    axes[1].scatter(X_test[misclassified, 0], X_test[misclassified, 1],
                   c='yellow', marker='x', s=200, linewidths=3,
                   label='Misclassified', zorder=5)

axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].set_title('MLP Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise2_predictions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise2_predictions.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - EXERCISE 2")
print("="*70)
print(f"Dataset: 1000 samples (400 class 0, 600 class 1)")
print(f"Features: 2")
print(f"Architecture: {mlp.layer_sizes}")
print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Final training loss: {mlp.loss_history[-1]:.6f}")
print("="*70)