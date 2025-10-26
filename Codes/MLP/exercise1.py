import numpy as np

"""
Exercise 1: Manual Calculation of MLP Steps
"""

# Given values
x = np.array([0.5, -0.2])
y = 1.0

# Hidden layer weights and biases
W1 = np.array([[0.3, -0.1], 
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])

# Output layer weights and bias
W2 = np.array([0.5, -0.3])
b2 = 0.2

# Learning rate
eta = 0.3

print("="*70)
print("EXERCISE 1: MANUAL MLP CALCULATION")
print("="*70)

# ============================================================================
# STEP 1: FORWARD PASS
# ============================================================================
print("\n" + "="*70)
print("STEP 1: FORWARD PASS")
print("="*70)

# Hidden layer pre-activations: z1 = W1 @ x + b1
z1 = W1 @ x + b1
print(f"\n1.1) Hidden layer pre-activations (z^(1)):")
print(f"z^(1) = W^(1) @ x + b^(1)")
print(f"z^(1) = {W1} @ {x} + {b1}")
print(f"z^(1) = [{W1[0,0]*x[0] + W1[0,1]*x[1]}, {W1[1,0]*x[0] + W1[1,1]*x[1]}] + {b1}")
print(f"z^(1) = [{0.3*0.5 + (-0.1)*(-0.2)}, {0.2*0.5 + 0.4*(-0.2)}] + {b1}")
print(f"z^(1) = [{0.15 + 0.02}, {0.1 - 0.08}] + {b1}")
print(f"z^(1) = [0.17, 0.02] + [0.1, -0.2]")
print(f"z^(1) = {z1}")

# Hidden layer activations: h1 = tanh(z1)
h1 = np.tanh(z1)
print(f"\n1.2) Hidden layer activations (h^(1)):")
print(f"h^(1) = tanh(z^(1))")
print(f"h^(1) = tanh({z1})")
print(f"h^(1) = {h1}")

# Output pre-activation: u2 = W2 @ h1 + b2
u2 = W2 @ h1 + b2
print(f"\n1.3) Output pre-activation (u^(2)):")
print(f"u^(2) = W^(2) @ h^(1) + b^(2)")
print(f"u^(2) = {W2} @ {h1} + {b2}")
print(f"u^(2) = {W2[0]*h1[0] + W2[1]*h1[1]} + {b2}")
print(f"u^(2) = {W2[0]*h1[0]} + {W2[1]*h1[1]} + {b2}")
print(f"u^(2) = {u2}")

# Final output: y_hat = tanh(u2)
y_hat = np.tanh(u2)
print(f"\n1.4) Final output (ŷ):")
print(f"ŷ = tanh(u^(2))")
print(f"ŷ = tanh({u2})")
print(f"ŷ = {y_hat}")

# ============================================================================
# STEP 2: LOSS CALCULATION
# ============================================================================
print("\n" + "="*70)
print("STEP 2: LOSS CALCULATION")
print("="*70)

loss = (y - y_hat)**2
print(f"\nMSE Loss: L = (y - ŷ)²")
print(f"L = ({y} - {y_hat})²")
print(f"L = ({y - y_hat})²")
print(f"L = {loss}")

# ============================================================================
# STEP 3: BACKWARD PASS (BACKPROPAGATION)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: BACKWARD PASS (BACKPROPAGATION)")
print("="*70)

# 3.1) Gradient of loss w.r.t. y_hat
dL_dyhat = -2 * (y - y_hat)
print(f"\n3.1) ∂L/∂ŷ:")
print(f"∂L/∂ŷ = -2(y - ŷ)")
print(f"∂L/∂ŷ = -2({y} - {y_hat})")
print(f"∂L/∂ŷ = -2({y - y_hat})")
print(f"∂L/∂ŷ = {dL_dyhat}")

# 3.2) Gradient w.r.t. u2 (using tanh derivative)
# d/du tanh(u) = 1 - tanh²(u) = 1 - y_hat²
dtanh_u2 = 1 - y_hat**2
dL_du2 = dL_dyhat * dtanh_u2
print(f"\n3.2) ∂L/∂u^(2):")
print(f"∂tanh(u^(2))/∂u^(2) = 1 - tanh²(u^(2)) = 1 - ŷ²")
print(f"∂tanh(u^(2))/∂u^(2) = 1 - {y_hat}²")
print(f"∂tanh(u^(2))/∂u^(2) = 1 - {y_hat**2}")
print(f"∂tanh(u^(2))/∂u^(2) = {dtanh_u2}")
print(f"\n∂L/∂u^(2) = ∂L/∂ŷ × ∂ŷ/∂u^(2)")
print(f"∂L/∂u^(2) = {dL_dyhat} × {dtanh_u2}")
print(f"∂L/∂u^(2) = {dL_du2}")

# 3.3) Gradients for output layer
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
print(f"\n3.3) Gradients for output layer:")
print(f"∂L/∂W^(2) = ∂L/∂u^(2) × h^(1)")
print(f"∂L/∂W^(2) = {dL_du2} × {h1}")
print(f"∂L/∂W^(2) = {dL_dW2}")
print(f"\n∂L/∂b^(2) = ∂L/∂u^(2)")
print(f"∂L/∂b^(2) = {dL_db2}")

# 3.4) Propagate to hidden layer
dL_dh1 = dL_du2 * W2
print(f"\n3.4) Propagate to hidden layer:")
print(f"∂L/∂h^(1) = ∂L/∂u^(2) × W^(2)")
print(f"∂L/∂h^(1) = {dL_du2} × {W2}")
print(f"∂L/∂h^(1) = {dL_dh1}")

# 3.5) Gradient w.r.t. z1
dtanh_z1 = 1 - h1**2
dL_dz1 = dL_dh1 * dtanh_z1
print(f"\n3.5) ∂L/∂z^(1):")
print(f"∂tanh(z^(1))/∂z^(1) = 1 - tanh²(z^(1)) = 1 - h^(1)²")
print(f"∂tanh(z^(1))/∂z^(1) = 1 - {h1}²")
print(f"∂tanh(z^(1))/∂z^(1) = {dtanh_z1}")
print(f"\n∂L/∂z^(1) = ∂L/∂h^(1) × ∂h^(1)/∂z^(1)")
print(f"∂L/∂z^(1) = {dL_dh1} × {dtanh_z1}")
print(f"∂L/∂z^(1) = {dL_dz1}")

# 3.6) Gradients for hidden layer
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
print(f"\n3.6) Gradients for hidden layer:")
print(f"∂L/∂W^(1) = ∂L/∂z^(1) ⊗ x (outer product)")
print(f"∂L/∂W^(1) = {dL_dz1.reshape(-1,1)} ⊗ {x}")
print(f"∂L/∂W^(1) =")
print(dL_dW1)
print(f"\n∂L/∂b^(1) = ∂L/∂z^(1)")
print(f"∂L/∂b^(1) = {dL_db1}")

# ============================================================================
# STEP 4: PARAMETER UPDATE
# ============================================================================
print("\n" + "="*70)
print("STEP 4: PARAMETER UPDATE (Gradient Descent)")
print("="*70)
print(f"\nLearning rate η = {eta}")

# Update output layer
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
print(f"\n4.1) Update output layer:")
print(f"W^(2)_new = W^(2) - η × ∂L/∂W^(2)")
print(f"W^(2)_new = {W2} - {eta} × {dL_dW2}")
print(f"W^(2)_new = {W2} - {eta * dL_dW2}")
print(f"W^(2)_new = {W2_new}")
print(f"\nb^(2)_new = b^(2) - η × ∂L/∂b^(2)")
print(f"b^(2)_new = {b2} - {eta} × {dL_db2}")
print(f"b^(2)_new = {b2} - {eta * dL_db2}")
print(f"b^(2)_new = {b2_new}")

# Update hidden layer
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
print(f"\n4.2) Update hidden layer:")
print(f"W^(1)_new = W^(1) - η × ∂L/∂W^(1)")
print(f"W^(1)_new =")
print(W1)
print(f"- {eta} ×")
print(dL_dW1)
print(f"=")
print(W1_new)
print(f"\nb^(1)_new = b^(1) - η × ∂L/∂b^(1)")
print(f"b^(1)_new = {b1} - {eta} × {dL_db1}")
print(f"b^(1)_new = {b1} - {eta * dL_db1}")
print(f"b^(1)_new = {b1_new}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(f"\nForward Pass:")
print(f"  z^(1) = {z1}")
print(f"  h^(1) = {h1}")
print(f"  u^(2) = {u2}")
print(f"  ŷ = {y_hat}")
print(f"\nLoss: {loss}")
print(f"\nGradients:")
print(f"  ∂L/∂W^(2) = {dL_dW2}")
print(f"  ∂L/∂b^(2) = {dL_db2}")
print(f"  ∂L/∂W^(1) =\n{dL_dW1}")
print(f"  ∂L/∂b^(1) = {dL_db1}")
print(f"\nUpdated Parameters:")
print(f"  W^(2)_new = {W2_new}")
print(f"  b^(2)_new = {b2_new}")
print(f"  W^(1)_new =\n{W1_new}")
print(f"  b^(1)_new = {b1_new}")
print("="*70)