# Exercise 1: Manual Calculation of MLP Steps

## Problem Statement

Consider a simple MLP with:
- **2 input features**
- **1 hidden layer** with 2 neurons
- **1 output neuron**
- **Activation function**: hyperbolic tangent (tanh) for both layers
- **Loss function**: Mean Squared Error (MSE)

### Given Values

**Input and Output:**
- $\mathbf{x} = [0.5, -0.2]$
- $y = 1.0$

**Hidden Layer Parameters:**
- $\mathbf{W}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix}$
- $\mathbf{b}^{(1)} = [0.1, -0.2]$

**Output Layer Parameters:**
- $\mathbf{W}^{(2)} = [0.5, -0.3]$
- $b^{(2)} = 0.2$

**Learning Rate:**
- $\eta = 0.3$

---

## Solution

### Step 1: Forward Pass

#### 1.1 Hidden Layer Pre-activations

Calculate $\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$

$$
\mathbf{z}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}
$$

For the first neuron:
$$
z_1^{(1)} = (0.3 \times 0.5) + (-0.1 \times -0.2) + 0.1 = 0.15 + 0.02 + 0.1 = 0.27
$$

For the second neuron:
$$
z_2^{(1)} = (0.2 \times 0.5) + (0.4 \times -0.2) + (-0.2) = 0.1 - 0.08 - 0.2 = -0.18
$$

**Result:**
$$
\mathbf{z}^{(1)} = [0.2700, -0.1800]
$$

#### 1.2 Hidden Layer Activations

Apply tanh activation: $\mathbf{h}^{(1)} = \tanh(\mathbf{z}^{(1)})$

$$
h_1^{(1)} = \tanh(0.27) = 0.2636
$$

$$
h_2^{(1)} = \tanh(-0.18) = -0.1781
$$

**Result:**
$$
\mathbf{h}^{(1)} = [0.2636, -0.1781]
$$

#### 1.3 Output Pre-activation

Calculate $u^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + b^{(2)}$

$$
u^{(2)} = [0.5, -0.3] \begin{bmatrix} 0.2636 \\ -0.1781 \end{bmatrix} + 0.2
$$

$$
u^{(2)} = (0.5 \times 0.2636) + (-0.3 \times -0.1781) + 0.2
$$

$$
u^{(2)} = 0.1318 + 0.0534 + 0.2 = 0.3852
$$

**Result:**
$$
u^{(2)} = 0.3852
$$

#### 1.4 Final Output

Apply tanh activation: $\hat{y} = \tanh(u^{(2)})$

$$
\hat{y} = \tanh(0.3852) = 0.3672
$$

**Result:**
$$
\hat{y} = 0.3672
$$

---

### Step 2: Loss Calculation

Calculate MSE loss: $L = (y - \hat{y})^2$

$$
L = (1.0 - 0.3672)^2 = (0.6328)^2 = 0.4004
$$

**Result:**
$$
L = 0.4004
$$

---

### Step 3: Backward Pass (Backpropagation)

#### 3.1 Gradient of Loss w.r.t. Output

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(1.0 - 0.3672) = -2(0.6328) = -1.2655
$$

**Result:**
$$
\frac{\partial L}{\partial \hat{y}} = -1.2655
$$

#### 3.2 Gradient w.r.t. Output Pre-activation

Using the tanh derivative: $\frac{d}{du}\tanh(u) = 1 - \tanh^2(u)$

$$
\frac{\partial \hat{y}}{\partial u^{(2)}} = 1 - \hat{y}^2 = 1 - (0.3672)^2 = 1 - 0.1349 = 0.8651
$$

Apply chain rule:
$$
\frac{\partial L}{\partial u^{(2)}} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial u^{(2)}} = -1.2655 \times 0.8651 = -1.0948
$$

**Result:**
$$
\frac{\partial L}{\partial u^{(2)}} = -1.0948
$$

#### 3.3 Gradients for Output Layer

**Gradient w.r.t. output weights:**
$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial u^{(2)}} \times \mathbf{h}^{(1)} = -1.0948 \times [0.2636, -0.1781]
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = [-0.2886, 0.1950]
$$

**Gradient w.r.t. output bias:**
$$
\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial u^{(2)}} = -1.0948
$$

#### 3.4 Propagate to Hidden Layer

$$
\frac{\partial L}{\partial \mathbf{h}^{(1)}} = \frac{\partial L}{\partial u^{(2)}} \times \mathbf{W}^{(2)} = -1.0948 \times [0.5, -0.3]
$$

$$
\frac{\partial L}{\partial \mathbf{h}^{(1)}} = [-0.5474, 0.3284]
$$

#### 3.5 Gradient w.r.t. Hidden Pre-activations

Calculate tanh derivative for hidden layer:
$$
\frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}} = 1 - (\mathbf{h}^{(1)})^2 = [1 - 0.2636^2, 1 - (-0.1781)^2]
$$

$$
\frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}} = [0.9305, 0.9683]
$$

Apply chain rule:
$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(1)}} \odot \frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}}
$$

$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5474, 0.3284] \odot [0.9305, 0.9683]
$$

$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5094, 0.3180]
$$

#### 3.6 Gradients for Hidden Layer

**Gradient w.r.t. hidden weights (outer product):**
$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} \otimes \mathbf{x}
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \begin{bmatrix} -0.5094 \\ 0.3180 \end{bmatrix} \otimes \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}
$$

**Gradient w.r.t. hidden bias:**
$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5094, 0.3180]
$$

---

### Step 4: Parameter Update

Using gradient descent: $\theta_{\text{new}} = \theta_{\text{old}} - \eta \frac{\partial L}{\partial \theta}$

#### 4.1 Update Output Layer

**Update output weights:**
$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5, -0.3] - 0.3 \times [-0.2886, 0.1950]
$$

$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5, -0.3] - [-0.0866, 0.0585]
$$

$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5866, -0.3585]
$$

**Update output bias:**
$$
b^{(2)}_{\text{new}} = 0.2 - 0.3 \times (-1.0948) = 0.2 + 0.3284 = 0.5284
$$

#### 4.2 Update Hidden Layer

**Update hidden weights:**
$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} - 0.3 \times \begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}
$$

$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} - \begin{bmatrix} -0.0764 & 0.0306 \\ 0.0477 & -0.0191 \end{bmatrix}
$$

$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3764 & -0.1306 \\ 0.1523 & 0.4191 \end{bmatrix}
$$

**Update hidden bias:**
$$
\mathbf{b}^{(1)}_{\text{new}} = [0.1, -0.2] - 0.3 \times [-0.5094, 0.3180]
$$

$$
\mathbf{b}^{(1)}_{\text{new}} = [0.1, -0.2] - [-0.1528, 0.0954]
$$

$$
\mathbf{b}^{(1)}_{\text{new}} = [0.2528, -0.2954]
$$

---

## Summary of Results

### Forward Pass
| Variable | Value |
|----------|-------|
| $\mathbf{z}^{(1)}$ | $[0.2700, -0.1800]$ |
| $\mathbf{h}^{(1)}$ | $[0.2636, -0.1781]$ |
| $u^{(2)}$ | $0.3852$ |
| $\hat{y}$ | $0.3672$ |

### Loss
$$L = 0.4004$$

### Gradients
| Parameter | Gradient |
|-----------|----------|
| $\frac{\partial L}{\partial \mathbf{W}^{(2)}}$ | $[-0.2886, 0.1950]$ |
| $\frac{\partial L}{\partial b^{(2)}}$ | $-1.0948$ |
| $\frac{\partial L}{\partial \mathbf{W}^{(1)}}$ | $\begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}$ |
| $\frac{\partial L}{\partial \mathbf{b}^{(1)}}$ | $[-0.5094, 0.3180]$ |

### Updated Parameters
| Parameter | New Value |
|-----------|-----------|
| $\mathbf{W}^{(2)}_{\text{new}}$ | $[0.5866, -0.3585]$ |
| $b^{(2)}_{\text{new}}$ | $0.5284$ |
| $\mathbf{W}^{(1)}_{\text{new}}$ | $\begin{bmatrix} 0.3764 & -0.1306 \\ 0.1523 & 0.4191 \end{bmatrix}$ |
| $\mathbf{b}^{(1)}_{\text{new}}$ | $[0.2528, -0.2954]$ |

---

## Code Verification

The calculations above were verified using Python/NumPy:

```python
import numpy as np

# Given values
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# Forward pass
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
loss = (y - y_hat)**2

# Backward pass
dL_dyhat = -2 * (y - y_hat)
dL_du2 = dL_dyhat * (1 - y_hat**2)
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1**2)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

# Parameter update
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1

print(f"Loss: {loss:.4f}")
print(f"Updated W2: {W2_new}")
print(f"Updated b2: {b2_new:.4f}")
```

All numerical values match the manual calculations with precision to 4 decimal places.# Exercise 1: Manual Calculation of MLP Steps

## Problem Statement

Consider a simple MLP with:
- **2 input features**
- **1 hidden layer** with 2 neurons
- **1 output neuron**
- **Activation function**: hyperbolic tangent (tanh) for both layers
- **Loss function**: Mean Squared Error (MSE)

### Given Values

**Input and Output:**
- $\mathbf{x} = [0.5, -0.2]$
- $y = 1.0$

**Hidden Layer Parameters:**
- $\mathbf{W}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix}$
- $\mathbf{b}^{(1)} = [0.1, -0.2]$

**Output Layer Parameters:**
- $\mathbf{W}^{(2)} = [0.5, -0.3]$
- $b^{(2)} = 0.2$

**Learning Rate:**
- $\eta = 0.3$

---

## Solution

### Step 1: Forward Pass

#### 1.1 Hidden Layer Pre-activations

Calculate $\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$

$$
\mathbf{z}^{(1)} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}
$$

For the first neuron:
$$
z_1^{(1)} = (0.3 \times 0.5) + (-0.1 \times -0.2) + 0.1 = 0.15 + 0.02 + 0.1 = 0.27
$$

For the second neuron:
$$
z_2^{(1)} = (0.2 \times 0.5) + (0.4 \times -0.2) + (-0.2) = 0.1 - 0.08 - 0.2 = -0.18
$$

**Result:**
$$
\mathbf{z}^{(1)} = [0.2700, -0.1800]
$$

#### 1.2 Hidden Layer Activations

Apply tanh activation: $\mathbf{h}^{(1)} = \tanh(\mathbf{z}^{(1)})$

$$
h_1^{(1)} = \tanh(0.27) = 0.2636
$$

$$
h_2^{(1)} = \tanh(-0.18) = -0.1781
$$

**Result:**
$$
\mathbf{h}^{(1)} = [0.2636, -0.1781]
$$

#### 1.3 Output Pre-activation

Calculate $u^{(2)} = \mathbf{W}^{(2)} \mathbf{h}^{(1)} + b^{(2)}$

$$
u^{(2)} = [0.5, -0.3] \begin{bmatrix} 0.2636 \\ -0.1781 \end{bmatrix} + 0.2
$$

$$
u^{(2)} = (0.5 \times 0.2636) + (-0.3 \times -0.1781) + 0.2
$$

$$
u^{(2)} = 0.1318 + 0.0534 + 0.2 = 0.3852
$$

**Result:**
$$
u^{(2)} = 0.3852
$$

#### 1.4 Final Output

Apply tanh activation: $\hat{y} = \tanh(u^{(2)})$

$$
\hat{y} = \tanh(0.3852) = 0.3672
$$

**Result:**
$$
\hat{y} = 0.3672
$$

---

### Step 2: Loss Calculation

Calculate MSE loss: $L = (y - \hat{y})^2$

$$
L = (1.0 - 0.3672)^2 = (0.6328)^2 = 0.4004
$$

**Result:**
$$
L = 0.4004
$$

---

### Step 3: Backward Pass (Backpropagation)

#### 3.1 Gradient of Loss w.r.t. Output

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2(1.0 - 0.3672) = -2(0.6328) = -1.2655
$$

**Result:**
$$
\frac{\partial L}{\partial \hat{y}} = -1.2655
$$

#### 3.2 Gradient w.r.t. Output Pre-activation

Using the tanh derivative: $\frac{d}{du}\tanh(u) = 1 - \tanh^2(u)$

$$
\frac{\partial \hat{y}}{\partial u^{(2)}} = 1 - \hat{y}^2 = 1 - (0.3672)^2 = 1 - 0.1349 = 0.8651
$$

Apply chain rule:
$$
\frac{\partial L}{\partial u^{(2)}} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial u^{(2)}} = -1.2655 \times 0.8651 = -1.0948
$$

**Result:**
$$
\frac{\partial L}{\partial u^{(2)}} = -1.0948
$$

#### 3.3 Gradients for Output Layer

**Gradient w.r.t. output weights:**
$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial u^{(2)}} \times \mathbf{h}^{(1)} = -1.0948 \times [0.2636, -0.1781]
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(2)}} = [-0.2886, 0.1950]
$$

**Gradient w.r.t. output bias:**
$$
\frac{\partial L}{\partial b^{(2)}} = \frac{\partial L}{\partial u^{(2)}} = -1.0948
$$

#### 3.4 Propagate to Hidden Layer

$$
\frac{\partial L}{\partial \mathbf{h}^{(1)}} = \frac{\partial L}{\partial u^{(2)}} \times \mathbf{W}^{(2)} = -1.0948 \times [0.5, -0.3]
$$

$$
\frac{\partial L}{\partial \mathbf{h}^{(1)}} = [-0.5474, 0.3284]
$$

#### 3.5 Gradient w.r.t. Hidden Pre-activations

Calculate tanh derivative for hidden layer:
$$
\frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}} = 1 - (\mathbf{h}^{(1)})^2 = [1 - 0.2636^2, 1 - (-0.1781)^2]
$$

$$
\frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}} = [0.9305, 0.9683]
$$

Apply chain rule:
$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(1)}} \odot \frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{z}^{(1)}}
$$

$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5474, 0.3284] \odot [0.9305, 0.9683]
$$

$$
\frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5094, 0.3180]
$$

#### 3.6 Gradients for Hidden Layer

**Gradient w.r.t. hidden weights (outer product):**
$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} \otimes \mathbf{x}
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \begin{bmatrix} -0.5094 \\ 0.3180 \end{bmatrix} \otimes \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix}
$$

$$
\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}
$$

**Gradient w.r.t. hidden bias:**
$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = [-0.5094, 0.3180]
$$

---

### Step 4: Parameter Update

Using gradient descent: $\theta_{\text{new}} = \theta_{\text{old}} - \eta \frac{\partial L}{\partial \theta}$

#### 4.1 Update Output Layer

**Update output weights:**
$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5, -0.3] - 0.3 \times [-0.2886, 0.1950]
$$

$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5, -0.3] - [-0.0866, 0.0585]
$$

$$
\mathbf{W}^{(2)}_{\text{new}} = [0.5866, -0.3585]
$$

**Update output bias:**
$$
b^{(2)}_{\text{new}} = 0.2 - 0.3 \times (-1.0948) = 0.2 + 0.3284 = 0.5284
$$

#### 4.2 Update Hidden Layer

**Update hidden weights:**
$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} - 0.3 \times \begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}
$$

$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{bmatrix} - \begin{bmatrix} -0.0764 & 0.0306 \\ 0.0477 & -0.0191 \end{bmatrix}
$$

$$
\mathbf{W}^{(1)}_{\text{new}} = \begin{bmatrix} 0.3764 & -0.1306 \\ 0.1523 & 0.4191 \end{bmatrix}
$$

**Update hidden bias:**
$$
\mathbf{b}^{(1)}_{\text{new}} = [0.1, -0.2] - 0.3 \times [-0.5094, 0.3180]
$$

$$
\mathbf{b}^{(1)}_{\text{new}} = [0.1, -0.2] - [-0.1528, 0.0954]
$$

$$
\mathbf{b}^{(1)}_{\text{new}} = [0.2528, -0.2954]
$$

---

## Summary of Results

### Forward Pass
| Variable | Value |
|----------|-------|
| $\mathbf{z}^{(1)}$ | $[0.2700, -0.1800]$ |
| $\mathbf{h}^{(1)}$ | $[0.2636, -0.1781]$ |
| $u^{(2)}$ | $0.3852$ |
| $\hat{y}$ | $0.3672$ |

### Loss
$$L = 0.4004$$

### Gradients
| Parameter | Gradient |
|-----------|----------|
| $\frac{\partial L}{\partial \mathbf{W}^{(2)}}$ | $[-0.2886, 0.1950]$ |
| $\frac{\partial L}{\partial b^{(2)}}$ | $-1.0948$ |
| $\frac{\partial L}{\partial \mathbf{W}^{(1)}}$ | $\begin{bmatrix} -0.2547 & 0.1019 \\ 0.1590 & -0.0636 \end{bmatrix}$ |
| $\frac{\partial L}{\partial \mathbf{b}^{(1)}}$ | $[-0.5094, 0.3180]$ |

### Updated Parameters
| Parameter | New Value |
|-----------|-----------|
| $\mathbf{W}^{(2)}_{\text{new}}$ | $[0.5866, -0.3585]$ |
| $b^{(2)}_{\text{new}}$ | $0.5284$ |
| $\mathbf{W}^{(1)}_{\text{new}}$ | $\begin{bmatrix} 0.3764 & -0.1306 \\ 0.1523 & 0.4191 \end{bmatrix}$ |
| $\mathbf{b}^{(1)}_{\text{new}}$ | $[0.2528, -0.2954]$ |

---

## Code Verification

The calculations above were verified using Python/NumPy:

```python
import numpy as np

# Given values
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# Forward pass
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
loss = (y - y_hat)**2

# Backward pass
dL_dyhat = -2 * (y - y_hat)
dL_du2 = dL_dyhat * (1 - y_hat**2)
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1 - h1**2)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1

# Parameter update
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1

print(f"Loss: {loss:.4f}")
print(f"Updated W2: {W2_new}")
print(f"Updated b2: {b2_new:.4f}")
```

All numerical values match the manual calculations with precision to 4 decimal places.