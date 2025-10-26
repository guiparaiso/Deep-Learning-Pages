# Exercise 2: Binary Classification with Synthetic Data

## Objective

Implement an MLP from scratch (using only NumPy) to perform binary classification on a synthetic dataset with the following specifications:

- **1000 samples**
- **2 classes**
- **2 features** (for visualization)
- **1 cluster for Class 0**
- **2 clusters for Class 1**

---

## Dataset Generation

### Strategy

Since `make_classification` applies the same number of clusters to all classes by default, we generate each class separately and then combine them:

```python
from sklearn.datasets import make_classification
import numpy as np

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

# Combine and shuffle
X = np.vstack([X_class0, X_class1])
y = np.hstack([y_class0, y_class1])
shuffle_idx = np.random.RandomState(42).permutation(len(y))
X = X[shuffle_idx]
y = y[shuffle_idx]
```

### Dataset Characteristics

- **Total samples**: 1000
- **Class 0**: 400 samples (1 cluster)
- **Class 1**: 600 samples (2 clusters)
- **Features**: 2 (fully informative)
- **Train/Test split**: 80/20 (800/200 samples)

### Data Visualization

![Data Distribution](exercise2_data.png)

The visualization shows the distinctive pattern of Class 0 (blue) forming a single compact cluster, while Class 1 (red) forms two separate clusters, making this a challenging binary classification problem.

---

## MLP Architecture

### Network Design

```python
from mlp import MLP

mlp = MLP(
    layer_sizes=[2, 8, 4, 1],
    activation='tanh',
    learning_rate=0.01
)
```

**Architecture breakdown:**

| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| Input | - | 2 | - |
| Hidden 1 | Dense | 8 | tanh |
| Hidden 2 | Dense | 4 | tanh |
| Output | Dense | 1 | tanh |

**Total parameters**: 
- Hidden 1: $(2 \times 8) + 8 = 24$ parameters
- Hidden 2: $(8 \times 4) + 4 = 36$ parameters
- Output: $(4 \times 1) + 1 = 5$ parameters
- **Total**: 65 parameters

### Design Rationale

1. **Two hidden layers**: Provides sufficient capacity to learn the non-linear decision boundary required to separate the 2 clusters of Class 1 from the single cluster of Class 0
2. **Decreasing neuron count** (8 → 4 → 1): Funnel architecture that progressively abstracts features
3. **tanh activation**: Allows for negative values and has stronger gradients than sigmoid
4. **Single output neuron**: Binary classification using threshold at 0.5

---

## Training Process

### Hyperparameters

```python
mlp.train(
    X_train, 
    y_train, 
    epochs=200, 
    batch_size=32, 
    verbose=True
)
```

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Epochs | 200 | Sufficient for convergence |
| Batch size | 32 | Mini-batch gradient descent for stability |
| Learning rate | 0.01 | Conservative rate to avoid overshooting |
| Loss function | MSE | Mean Squared Error for binary output |

### Training Loss Curve

![Training Loss](exercise2_loss.png)

The loss curve shows:
- **Rapid initial descent**: Quick learning in early epochs
- **Smooth convergence**: Stable training without oscillations
- **Final loss**: ~0.01, indicating good fit

---

## Results

### Performance Metrics

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **Accuracy** | 99.00% | 98.50% |
| **Loss** | 0.0089 | - |

### Confusion Matrix

![Confusion Matrix](exercise2_confusion_matrix.png)

**Test Set Confusion Matrix:**

|  | Predicted 0 | Predicted 1 |
|---|-------------|-------------|
| **Actual 0** | 78 | 2 |
| **Actual 1** | 1 | 119 |

**Analysis:**
- **True Negatives (TN)**: 78
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 119
- **Precision (Class 1)**: 119/(119+2) = 98.35%
- **Recall (Class 1)**: 119/(119+1) = 99.17%

### Classification Report

```
              precision    recall  f1-score   support

     Class 0       0.99      0.97      0.98        80
     Class 1       0.98      0.99      0.99       120

    accuracy                           0.98       200
   macro avg       0.99      0.98      0.98       200
weighted avg       0.99      0.98      0.98       200
```

### Decision Boundary

![Decision Boundary](exercise2_decision_boundary.png)

The decision boundary visualization demonstrates:
- **Non-linear separation**: The MLP successfully learned a complex boundary
- **Cluster separation**: Correctly distinguishes the single cluster of Class 0 from the two clusters of Class 1
- **Generalization**: Clean boundaries without excessive overfitting

### Prediction Visualization

![Predictions Comparison](exercise2_predictions.png)

**Left**: Ground truth labels  
**Right**: MLP predictions with misclassified points marked with yellow X

Only 3 misclassified samples out of 200 test samples, demonstrating excellent generalization.

---

## Implementation Details

### Key Components

The MLP implementation (from `mlp.py`) includes:

1. **Forward Pass**:
   ```python
   def forward(self, X):
       activations = [X]
       for i in range(self.num_layers - 1):
           z = self.weights[i] @ activations[-1] + self.biases[i]
           a = self._activation_function(z)
           activations.append(a)
       return activations
   ```

2. **Backward Pass** (Backpropagation):
   ```python
   def backward(self, X, y, activations):
       # Compute output gradient
       delta = -(y - y_pred) * activation_derivative(y_pred)
       
       # Backpropagate through layers
       for i in reversed(range(self.num_layers - 1)):
           gradients_w[i] = delta @ activations[i].T
           gradients_b[i] = np.sum(delta, axis=1)
           delta = weights[i].T @ delta * activation_derivative(activations[i])
   ```

3. **Parameter Update** (Gradient Descent):
   ```python
   for i in range(self.num_layers - 1):
       self.weights[i] -= learning_rate * gradients_w[i]
       self.biases[i] -= learning_rate * gradients_b[i]
   ```

### Loss Function

Mean Squared Error (MSE):
$$L = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Where:
- $m$ = number of samples
- $y_i$ = true label
- $\hat{y}_i$ = predicted output

---

## Code Structure

```
code/
├── mlp.py              # Reusable MLP class
└── exercise2.py        # Binary classification implementation

images/
├── exercise2_data.png
├── exercise2_loss.png
├── exercise2_decision_boundary.png
├── exercise2_confusion_matrix.png
└── exercise2_predictions.png
```

---

## Observations and Analysis

### Strengths

1. **High accuracy**: 98.5% on test set demonstrates excellent learning
2. **Good generalization**: Minimal gap between train (99%) and test (98.5%) accuracy
3. **Robust to complexity**: Successfully handles the 2-cluster pattern in Class 1
4. **Clean boundaries**: Decision boundary is smooth and not overfitted

### Potential Improvements

1. **Regularization**: Could add L2 regularization to prevent any overfitting
2. **Early stopping**: Monitor validation loss to stop training optimally
3. **Learning rate scheduling**: Decrease learning rate over time for finer convergence
4. **Data augmentation**: Could add noise to make the model more robust

### Challenges Addressed

1. **Imbalanced clusters**: Different number of clusters per class
2. **Non-linear separation**: Requires complex decision boundary
3. **From scratch implementation**: No high-level frameworks used (only NumPy)

---

## Conclusion

The MLP successfully learned to classify the binary dataset with **98.5% test accuracy**. The implementation demonstrates:

✅ **Correct forward and backward propagation**  
✅ **Effective training with mini-batch gradient descent**  
✅ **Strong generalization to unseen data**  
✅ **Clean, reusable code structure**  

The model effectively handles the challenging scenario of different cluster patterns per class, validating the MLP architecture and training approach.