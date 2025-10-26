import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import the SAME MLP class from Exercise 2 (no modifications!)
from mlp import MLP

"""
Exercise 3: Multi-Class Classification with Synthetic Data
- 1500 samples
- 3 classes
- 4 features
- 2 clusters for class 0, 3 for class 1, 4 for class 2
"""

print("="*70)
print("EXERCISE 3: MULTI-CLASS CLASSIFICATION")
print("="*70)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC DATA
# ============================================================================
print("\nSTEP 1: Generating synthetic dataset...")

# Strategy: Generate each class separately with different n_clusters_per_class
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

# Visualize data (using first 2 features for 2D plot)
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
plt.savefig('exercise3_data.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_data.png")
plt.show()

# Visualize all feature pairs
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

feature_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
for idx, (i, j) in enumerate(feature_pairs):
    for class_idx, color in enumerate(colors):
        mask = y_train == class_idx
        axes[idx].scatter(X_train[mask, i], X_train[mask, j],
                         c=color, alpha=0.5, edgecolors='k', s=20)
    axes[idx].set_xlabel(f'Feature {i+1}')
    axes[idx].set_ylabel(f'Feature {j+1}')
    axes[idx].set_title(f'Features {i+1} vs {j+1}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise3_feature_pairs.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_feature_pairs.png")
plt.show()

# ============================================================================
# STEP 3: BUILD AND TRAIN MLP (REUSING SAME CODE FROM EXERCISE 2!)
# ============================================================================
print("\nSTEP 3: Building and training MLP...")
print("NOTE: Using the EXACT SAME MLP class from Exercise 2!")
print("      Only changing hyperparameters for multi-class classification.\n")

# MLP Architecture:
# - Input layer: 4 features
# - Hidden layer 1: 16 neurons
# - Hidden layer 2: 8 neurons
# - Output layer: 3 neurons (one for each class)
mlp = MLP(
    layer_sizes=[4, 16, 8, 3],  # Changed: input=4, output=3
    activation='tanh',           # Same activation
    learning_rate=0.01           # Same learning rate
)

print(f"MLP Architecture: {mlp.layer_sizes}")
print(f"Activation function: {mlp.activation}")
print(f"Learning rate: {mlp.lr}")
print(f"Number of parameters: {sum(w.size for w in mlp.weights) + sum(b.size for b in mlp.biases)}")

# Train the model
print("\nTraining...")
mlp.train(X_train, y_train, epochs=300, batch_size=32, verbose=True)

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
print(classification_report(y_test, y_pred_test, 
                          target_names=['Class 0', 'Class 1', 'Class 2']))

# Per-class accuracy
for i in range(3):
    mask = y_test == i
    class_acc = accuracy_score(y_test[mask], y_pred_test[mask])
    print(f"Class {i} accuracy: {class_acc:.4f} ({class_acc*100:.2f}%)")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\nSTEP 5: Creating visualizations...")

# Plot 1: Training Loss
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_history, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time (Multi-Class)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('exercise3_loss.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_loss.png")
plt.show()

# Plot 2: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.figure.colorbar(im, ax=ax)

# Labels
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Class 0', 'Class 1', 'Class 2'],
       yticklabels=['Class 0', 'Class 1', 'Class 2'],
       ylabel='True Label',
       xlabel='Predicted Label',
       title='Confusion Matrix (Multi-Class)')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig('exercise3_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_confusion_matrix.png")
plt.show()

# Plot 3: Prediction scatter (first 2 features)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

# Predictions
for i, color in enumerate(colors):
    mask = y_pred_test == i
    axes[1].scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=f'Predicted Class {i}', alpha=0.6, edgecolors='k', s=50)

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
plt.savefig('exercise3_predictions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_predictions.png")
plt.show()

# Plot 4: Class probabilities for test samples
print("\nGenerating probability distributions...")
y_proba = mlp.predict_proba(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for class_idx in range(3):
    # Get samples from this class
    mask = y_test == class_idx
    proba_class = y_proba[mask]
    
    # Plot histogram for each output neuron
    for output_idx in range(3):
        axes[class_idx].hist(proba_class[:, output_idx], bins=20, alpha=0.6,
                            label=f'Output {output_idx}', edgecolor='black')
    
    axes[class_idx].set_xlabel('Probability')
    axes[class_idx].set_ylabel('Frequency')
    axes[class_idx].set_title(f'Probability Distribution for True Class {class_idx}')
    axes[class_idx].legend()
    axes[class_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise3_probabilities.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exercise3_probabilities.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY - EXERCISE 3")
print("="*70)
print(f"Dataset: 1500 samples (500 per class)")
print(f"Features: 4")
print(f"Classes: 3")
print(f"Clusters per class: [2, 3, 4]")
print(f"Architecture: {mlp.layer_sizes}")
print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Final training loss: {mlp.loss_history[-1]:.6f}")
print("\n✓ REUSED THE SAME MLP CLASS FROM EXERCISE 2 (Extra point earned!)")
print("="*70)