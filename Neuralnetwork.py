import numpy as np
import matplotlib.pyplot as plt
# If you have Neuralnetwork.py, you'd typically import it like:
# import Neuralnetwork as nn

# --- Neural Network Core Classes (full code for context) ---

"""layer_dense class"""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

    def normalization(self):
        pass

"""Loss classes"""
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        self.data_loss = np.mean(sample_losses)
        return self.data_loss

class loss_mse(Loss):
    def forward(self, y_pred, y_true):
        self.diff = (y_pred - y_true)
        self.output = self.diff**2
        return self.output

    def backward(self, dvalues=None, y_true=None):
        self.dinputs = 2 * self.diff / np.size(self.output)
        return self.dinputs

class loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_conf = y_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
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
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
 
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        return self # Not typically used alone for backward in a classification context

class Activation_SoftMax_crosscategorical_loss(Loss):
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

"""Optimizers"""
class optimizer_SGD:
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

class optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1**(self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2**(self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1

"""Sequential Model Class"""
class Sequential:
    optimizers = {
        'sgd': optimizer_SGD,
        'adam': optimizer_Adam,
    }
    loss_func = {
        'mse': loss_mse,
        'cce': loss_CategoricalCrossEntropy,
        'softmax_cce': Activation_SoftMax_crosscategorical_loss
    }

    def __init__(self, layers: list):
        self.layers = layers
        self.loss = None
        self.optimizer = None
        self.metrics = None # `metrics` is a list, currently used for nothing.

    def compile_net(self, optimizer, loss, metrics: list = None, learning_rate=1.0):
        if loss == 'softmax_cce':
            self.loss = self.loss_func['softmax_cce']()
        else:
            self.loss = self.loss_func[loss]()

        self.optimizer = self.optimizers[optimizer](learning_rate=learning_rate)
        self.metrics = metrics # Placeholder, not actively used in training loop yet

    def fit(self, input_data, output_data, epochs=5, batch_size=-1, shuffle_batch=False):
        if self.loss is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled before calling fit().")

        num_samples = input_data.shape[0]
        effective_batch_size = num_samples if batch_size == -1 else batch_size
        
        print(f"Starting training for {epochs} epochs with batch size {effective_batch_size}")

        for epoch in range(epochs):
            self.loss_avg = 0.0
            correct_predictions_epoch = 0 # Added for accuracy
            total_samples_epoch = 0       # Added for accuracy
            batch_count_this_epoch = 0 
            batches_generator = self._create_batches(input_data, output_data,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle_batch)

            for X_batch, y_true_batch in batches_generator:
                # 1. Forward Pass
                y_pred_batch = self._forward(X_batch)

                # 2. Calculate Loss
                batch_loss = self.loss.calculate(y_pred_batch, y_true_batch)
                self.loss_avg += batch_loss
                
                batch_count_this_epoch+=1
                # 3. Calculate Training Accuracy for the batch (if applicable)
                # This logic assumes classification problems where y_pred_batch are probabilities/logits
                # and y_true_batch are one-hot or sparse labels.
                if hasattr(self.loss, 'activation') and isinstance(self.loss.activation, Activation_Softmax):
                    # For classification, we use argmax on the predictions
                    predicted_classes = np.argmax(y_pred_batch, axis=1)
                    
                    # Convert true labels to sparse if they are one-hot encoded
                    if len(y_true_batch.shape) == 2:
                        true_classes = np.argmax(y_true_batch, axis=1)
                    else:
                        true_classes = y_true_batch # Assume sparse if not 2D

                    correct_predictions_batch = np.sum(predicted_classes == true_classes)
                    correct_predictions_epoch += correct_predictions_batch
                    total_samples_epoch += len(X_batch)

                # 4. Backward Pass & Optimizer Update
                self._backward(y_true_batch)
            
            # Print epoch summary
            if batch_count_this_epoch > 0: # Check if any batches were processed
                epoch_loss_avg = self.loss_avg / batch_count_this_epoch
                log_string = f"Epoch {epoch + 1}/{epochs} - Avg Loss: {epoch_loss_avg:.6f}"
                
                if total_samples_epoch > 0: # Only add accuracy if it was calculated
                    epoch_accuracy = correct_predictions_epoch / total_samples_epoch
                    log_string += f", Train Accuracy: {epoch_accuracy:.4f}"
                print(log_string)
            else:
                print(f"Epoch {epoch + 1}/{epochs} - No batches processed (check input data/batch size).")

        print("Training complete.")

    def evaluate(self, input_data, output_data):
        y_pred = self._forward(input_data)
        final_loss = self.loss.calculate(y_pred, output_data)
        
        # Also calculate accuracy for evaluation if applicable
        accuracy = 0.0
        if hasattr(self.loss, 'activation') and isinstance(self.loss.activation, Activation_Softmax):
            predicted_classes = np.argmax(y_pred, axis=1)
            if len(output_data.shape) == 2:
                true_classes = np.argmax(output_data, axis=1)
            else:
                true_classes = output_data
            accuracy = np.mean(predicted_classes == true_classes)
            print(f"Evaluation Loss: {final_loss:.6f}, Evaluation Accuracy: {accuracy:.4f}")
        else:
            print(f"Evaluation Loss: {final_loss:.6f}")
        return final_loss, accuracy # Return both for potential use

    def predict(self, input_data):
        return self._forward(input_data)
    
    def get_loss(self, input_data, output_data):
        print("Use .evaluate(input_data, output_data) for model loss calculation and metrics.")
        return self.evaluate(input_data, output_data)

    def _forward(self, input_data):
        current_output = input_data
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output

    def _backward(self, y_true_batch):
        self.loss.backward(self.loss.output, y_true_batch)
        self.optimizer.pre_update()
        current_dvalues = self.loss.dinputs

        for layer in reversed(self.layers):
            layer.backward(current_dvalues)
            current_dvalues = layer.dinputs
            if hasattr(layer, 'weights'):
                self.optimizer.update(layer)
        self.optimizer.post_update()

    def _create_batches(self, input_data, output_data, batch_size=1, shuffle=False):
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
        optimizer="adam",
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