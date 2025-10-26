import numpy as np
import matplotlib.pyplot as plt

class MLP:
    """
    Multi-Layer Perceptron implementation from scratch.
    Reusable for binary and multi-class classification.
    """
    
    def __init__(self, layer_sizes, activation='tanh', learning_rate=0.01):
        """
        Initialize MLP with given architecture.
        
        Parameters:
        -----------
        layer_sizes : list
            Number of neurons in each layer [input, hidden1, ..., output]
            Example: [2, 4, 3] means 2 inputs, 4 hidden neurons, 3 outputs
        activation : str
            Activation function: 'tanh', 'sigmoid', or 'relu'
        learning_rate : float
            Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.uniform(-limit, limit, (layer_sizes[i+1], layer_sizes[i]))
            b = np.zeros((layer_sizes[i+1], 1))
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Store training history
        self.loss_history = []
    
    def _activation_function(self, z):
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, a):
        """Compute derivative of activation function."""
        if self.activation == 'tanh':
            return 1 - a**2
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'relu':
            return (a > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _softmax(self, z):
        """Softmax for output layer (multi-class)."""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : array-like, shape (n_features, n_samples)
            Input data
            
        Returns:
        --------
        activations : list
            All layer activations including input
        pre_activations : list
            All layer pre-activations (z values)
        """
        activations = [X]
        pre_activations = []
        
        for i in range(self.num_layers - 1):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            pre_activations.append(z)
            
            # Last layer: use softmax for multi-class, activation for others
            if i == self.num_layers - 2:
                if self.layer_sizes[-1] > 1:  # Multi-class
                    a = self._softmax(z)
                else:  # Binary or regression
                    a = self._activation_function(z)
            else:
                a = self._activation_function(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def backward(self, X, y, activations, pre_activations):
        """
        Backward pass (backpropagation).
        
        Parameters:
        -----------
        X : array-like, shape (n_features, n_samples)
            Input data
        y : array-like, shape (n_outputs, n_samples)
            True labels (one-hot encoded for multi-class)
        activations : list
            Forward pass activations
        pre_activations : list
            Forward pass pre-activations
            
        Returns:
        --------
        gradients_w : list
            Gradients for weights
        gradients_b : list
            Gradients for biases
        """
        m = X.shape[1]  # Number of samples
        gradients_w = [None] * (self.num_layers - 1)
        gradients_b = [None] * (self.num_layers - 1)
        
        # Output layer gradient
        if self.layer_sizes[-1] > 1:  # Multi-class (softmax + cross-entropy)
            delta = activations[-1] - y
        else:  # Binary (sigmoid/tanh + MSE/BCE)
            y_pred = activations[-1]
            delta = -(y - y_pred) * self._activation_derivative(y_pred)
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            gradients_w[i] = (delta @ activations[i].T) / m
            gradients_b[i] = np.sum(delta, axis=1, keepdims=True) / m
            
            if i > 0:
                delta = (self.weights[i].T @ delta) * self._activation_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def _compute_loss(self, y_true, y_pred):
        """Compute loss (MSE for binary, cross-entropy for multi-class)."""
        m = y_true.shape[1]
        
        if self.layer_sizes[-1] > 1:  # Multi-class cross-entropy
            loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        else:  # Binary MSE
            loss = np.sum((y_true - y_pred)**2) / m
        
        return loss
    
    def train(self, X, y, epochs=100, batch_size=None, verbose=True):
        """
        Train the MLP using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            Target labels
        epochs : int
            Number of training epochs
        batch_size : int or None
            Mini-batch size (None = full batch)
        verbose : bool
            Print progress
        """
        # Transpose and prepare data
        X = X.T  # (n_features, n_samples)
        
        # Prepare labels
        if self.layer_sizes[-1] > 1:  # Multi-class
            n_classes = self.layer_sizes[-1]
            y_encoded = np.zeros((n_classes, X.shape[1]))
            for i, label in enumerate(y):
                y_encoded[int(label), i] = 1
            y = y_encoded
        else:  # Binary
            y = y.reshape(1, -1)
        
        n_samples = X.shape[1]
        if batch_size is None:
            batch_size = n_samples
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            y_shuffled = y[:, indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                
                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                
                # Compute loss
                loss = self._compute_loss(y_batch, activations[-1])
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, pre_activations)
                
                # Update parameters
                for i in range(self.num_layers - 1):
                    self.weights[i] -= self.lr * gradients_w[i]
                    self.biases[i] -= self.lr * gradients_b[i]
            
            # Average loss
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : array-like
            Predicted class labels
        """
        X = X.T
        activations, _ = self.forward(X)
        output = activations[-1]
        
        if self.layer_sizes[-1] > 1:  # Multi-class
            return np.argmax(output, axis=0)
        else:  # Binary
            return (output > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        X = X.T
        activations, _ = self.forward(X)
        return activations[-1].T
    
    def score(self, X, y):
        """Compute accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_loss(self):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()
    
    def plot_decision_boundary(self, X, y, resolution=0.02):
        """
        Plot decision boundary (only for 2D input features).
        """
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D input!")
            return
        
        # Create meshgrid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                            np.arange(y_min, y_max, resolution))
        
        # Predict on meshgrid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=1)
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                            edgecolors='black', s=50, linewidth=1.5)
        
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.title('MLP Decision Boundary', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Class')
        plt.grid(True, alpha=0.3)