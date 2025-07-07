import numpy as np
import matplotlib.pyplot as plt

# --- Neural Network Core Classes ---

"""layer_dense class"""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # He (Kaiming) Initialization for ReLU
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons)) # Biases typically start at 0

    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values for the previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def normalization(self):
        """to implement - currently not used"""
        pass

"""Loss classes"""
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        self.data_loss = np.mean(sample_losses)
        return self.data_loss

class loss_mse(Loss):
    """Mean Squared Error Loss"""
    def forward(self, y_pred, y_true):
        self.diff = (y_pred - y_true)
        self.output = self.diff**2
        return self.output

    def backward(self, dvalues=None, y_true=None): # dvalues/y_true parameters are placeholders for clarity
                                                 # as this specific backward uses self.diff directly
        # Gradient of MSE with respect to y_pred
        self.dinputs = 2 * self.diff / np.size(self.output)
        return self.dinputs

class loss_CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy Loss"""
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: # Sparse labels (integers)
            correct_conf = y_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # One-hot encoded labels
            correct_conf = np.sum(y_clipped * y_true, axis=1)
        
        self.output = -np.log(correct_conf)
        return self.output

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs

"""Activation classes"""
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 # Zero gradient where input values were negative
        return self.dinputs
 
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        # This standalone Softmax backward is typically not used directly
        # when combined with CCE Loss for numerical stability.
        # Its implementation would be complex and context-dependent.
        # For simplicity, we return self if it's not the combined loss.
        return self

class Activation_SoftMax_crosscategorical_loss(Loss):
    """Combined Softmax activation and Categorical Cross-Entropy loss for stability."""
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output # Store softmax output for potential use or logging
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 # Combined gradient for Softmax+CCE
        self.dinputs = self.dinputs / samples
        return self.dinputs # Crucial: Must return the gradient

"""Optimizers"""
class optimizer_SGD:
    """SGD Optimizer with learning rate decay and momentum."""
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update(self):
        self.iterations += 1

"""Sequential Model Class"""
class Sequential:
    optimizers = {
        'sgd': optimizer_SGD,
    }
    loss_func = {
        'mse': loss_mse,
        'cce': loss_CategoricalCrossEntropy,
        'softmax_cce': Activation_SoftMax_crosscategorical_loss # For convenience
    }

    def __init__(self, layers: list):
        self.layers = layers
        self.loss = None
        self.optimizer = None
        self.metrics = None # Not implemented for metrics calculation yet, but compiled for.

    def compile_net(self, optimizer, loss, metrics: list = None, learning_rate=1.0):
        if loss == 'softmax_cce': # Use the combined loss for softmax + cce
            self.loss = self.loss_func['softmax_cce']()
        else:
            self.loss = self.loss_func[loss]()

        self.optimizer = self.optimizers[optimizer](learning_rate=learning_rate)
        self.metrics = metrics

    def fit(self, input_data, output_data, epochs=5, batch_size=-1, shuffle_batch=False):
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before calling fit().")

        num_samples = input_data.shape[0]
        
        # Determine effective batch size for num_batches calculation
        effective_batch_size = num_samples if batch_size == -1 else batch_size
        num_batches_per_epoch = (num_samples + effective_batch_size - 1) // effective_batch_size
        
        print(f"Starting training for {epochs} epochs with batch size {effective_batch_size}")

        for epoch in range(epochs):
            self.loss_avg = 0.0 # Reset average loss for the current epoch
            batch_count_this_epoch = 0

            # Create a new generator for batches for each epoch.
            batches_generator = self._create_batches(input_data, output_data,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle_batch)

            for X_batch, y_true_batch in batches_generator:
                # 1. Forward Pass
                y_pred_batch = self._forward(X_batch)

                # 2. Calculate Loss
                batch_loss = self.loss.calculate(y_pred_batch, y_true_batch)
                self.loss_avg += batch_loss
                batch_count_this_epoch += 1

                # 3. Backward Pass & Optimizer Update
                self._backward(y_true_batch) # y_true_batch is needed by loss.backward

            # Print epoch summary
            if batch_count_this_epoch > 0:
                print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {self.loss_avg / batch_count_this_epoch:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - No batches processed (check input data/batch size).")

        print("Training complete.")

    def evaluate(self, input_data, output_data):
        y_pred = self._forward(input_data)
        final_loss = self.loss.calculate(y_pred, output_data)
        print(f"Evaluation Loss: {final_loss:.6f}")
        return final_loss

    def predict(self, input_data):
        return self._forward(input_data)
    
    def get_loss(self, input_data, output_data): # Legacy/helper, prefer evaluate
        print("Use .evaluate(input_data, output_data) for model loss calculation.")
        return self.evaluate(input_data, output_data)

    def _forward(self, input_data):
        current_output = input_data
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def _backward(self, y_true_batch):
        # Start the backward pass from the loss function.
        # The loss function's backward method must compute self.dinputs.
        self.loss.backward(self.loss.output, y_true_batch) # Pass `dvalues` and `y_true` as expected by loss.backward

        self.optimizer.pre_update()

        current_dvalues = self.loss.dinputs # Get the gradient from the loss w.r.t. the network's final output

        for layer in reversed(self.layers):
            layer.backward(current_dvalues) # Pass the current gradients to the layer's backward
            current_dvalues = layer.dinputs # Update gradients for the next layer in the chain

            if hasattr(layer, 'weights'):
                self.optimizer.update(layer)

        self.optimizer.post_update()

    def _create_batches(self, input_data, output_data, batch_size=1, shuffle=False):
        """
        Divides the rows of NumPy arrays into batches. This is a generator function.
        """
        num_samples = input_data.shape[0]

        if num_samples != output_data.shape[0]:
            raise ValueError("input_data and output_data must have the same number of samples.")
        if not isinstance(batch_size, int) or (batch_size <= 0 and batch_size != -1):
            raise ValueError("batch_size must be a positive integer or -1.")

        if batch_size == -1:
            batch_size = num_samples

        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch = input_data[batch_indices]
            y_batch = output_data[batch_indices]
            yield x_batch, y_batch


# --- Main Execution / Training Script ---

if __name__ == "__main__":
    """gemini test client"""
    # Define the range for training (e.g., 0 to 4*pi for 2 full cycles)
    # This is crucial for the network to generalize beyond a single cycle.
    train_range_start = 0
    train_range_end = 4 * np.pi # Train on 2 full cycles
    num_train_samples = 500 # More samples for smoother learning over a wider range

    time_train_original_range = np.linspace(train_range_start, train_range_end, num_train_samples, endpoint=True).reshape(-1, 1)
    sine_wave_train = np.sin(time_train_original_range)

    # Scale the input training data to [-1, 1]
    # This mapping is based on the *training data's* range
    time_train_scaled = (time_train_original_range - (train_range_start + train_range_end) / 2) / ((train_range_end - train_range_start) / 2)

    # Model Definition
    model = Sequential([
        Layer_Dense(1, 100),       # Input (1) -> Hidden (100)
        Activation_ReLU(),
        Layer_Dense(100, 100),     # Hidden (100) -> Hidden (100)
        Activation_ReLU(),
        Layer_Dense(100, 1)        # Hidden (100) -> Output (1) - No activation for regression
    ])

    # Model Compilation
    model.compile_net(
        optimizer="sgd",
        loss='mse',
        learning_rate=0.001 # Start with a good learning rate. You might fine-tune this.
    )

    # Model Training
    print("\n--- Training Model ---")
    model.fit(time_train_scaled, sine_wave_train, epochs=2000, batch_size=32, shuffle_batch=True) # Increased epochs significantly

    # --- Prediction Data ---
    # Define the range for prediction (can be wider than training if desired, but expect extrapolation issues)
    # For a good plot, let's predict over the same range as trained or slightly wider.
    predict_range_start = 0
    predict_range_end = 4 * np.pi # Predict over the same 2 cycles
    num_predict_samples = 500 # More points for a smooth prediction curve

    x_predict_original_range = np.linspace(predict_range_start, predict_range_end, num_predict_samples, endpoint=True).reshape(-1, 1)
    y_true_predict = np.sin(x_predict_original_range)

    # Crucial: Apply the SAME scaling transformation as training to prediction inputs.
    # The network expects scaled inputs.
    x_predict_scaled = (x_predict_original_range - (train_range_start + train_range_end) / 2) / ((train_range_end - train_range_start) / 2)


    # Make Predictions
    print("\n--- Making Predictions ---")
    y_predicted = model.predict(x_predict_scaled)

    # Plotting Results
    plt.figure(figsize=(12, 7))
    plt.plot(x_predict_original_range, y_true_predict, label='True Sine Wave', color='blue', linestyle='--')
    plt.plot(x_predict_original_range, y_predicted, label='Predicted Sine Wave', color='red')

    # Plot only a subset of training points if num_train_samples is very large for clarity
    plt.scatter(time_train_original_range[::5], sine_wave_train[::5], label='Training Data Points (sampled)', color='green', s=10, alpha=0.5)

    plt.title('Sine Wave Prediction by Neural Network')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1.1) # Ensure consistent y-axis limits
    plt.tight_layout()
    plt.show()

    # Evaluate final loss on the training data
    print("\n--- Final Evaluation on Training Data ---")
    model.evaluate(time_train_scaled, sine_wave_train)