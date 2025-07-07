import numpy as np
import matplotlib.pyplot as plt
import Neuralnetwork as nn # Assuming your classes are in Neuralnetwork.py
# Try to import MNIST from Keras datasets
try:
    from tensorflow.keras.datasets import mnist
except ImportError:
    print("TensorFlow/Keras not found. MNIST dataset loading will be skipped.")
    print("Please install TensorFlow (pip install tensorflow) or provide MNIST data manually.")
    mnist = None # Set mnist to None if import fails

def load_and_preprocess_mnist():
    if mnist is None:
        print("Cannot load MNIST. Exiting.")
        return None, None, None, None

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Original x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"Original x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Preprocess images: Flatten and Normalize
    # Flatten images from (28, 28) to (784,)
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

    # Normalize pixel values to [0, 1]
    x_train_normalized = x_train_flat / 255.0
    x_test_normalized = x_test_flat / 255.0

    # Preprocess labels: One-hot encode
    # Number of classes for MNIST is 10 (digits 0-9)
    num_classes = 10
    y_train_one_hot = np.eye(num_classes)[y_train]
    y_test_one_hot = np.eye(num_classes)[y_test]

    print(f"Processed x_train shape: {x_train_normalized.shape}, y_train shape: {y_train_one_hot.shape}")
    print(f"Processed x_test shape: {x_test_normalized.shape}, y_test shape: {y_test_one_hot.shape}")

    return x_train_normalized, y_train_one_hot, x_test_normalized, y_test_one_hot

if __name__ == "__main__":
    x_train, y_train_one_hot, x_test, y_test_one_hot = load_and_preprocess_mnist()

    if x_train is None:
        exit("MNIST data not loaded. Please address the TensorFlow installation or data loading issue.")

    # --- 1. Define the Network Architecture for MNIST ---
    # Input layer: 784 neurons (28x28 pixels flattened)
    # Hidden layers: Use ReLU activations
    # Output layer: 10 neurons (for 10 digits), Softmax activation handled by combined loss
    
    print("\n--- Defining Neural Network Architecture ---")
    mnist_model = nn.Sequential([
        nn.Layer_Dense(784, 128),      # Input (784) -> Hidden (128)
        nn.Activation_ReLU(),
        nn.Layer_Dense(128, 128),      # Hidden (128) -> Hidden (128)
        nn.Activation_ReLU(),
        nn.Layer_Dense(128, 10)        # Hidden (128) -> Output (10 classes)
    ])

    # --- 2. Compile the Model ---
    # Use Adam optimizer and the combined Softmax Cross-Entropy loss
    print("\n--- Compiling Model ---")
    mnist_model.compile_net(
        optimizer="adam",
        loss='softmax_cce', # Use the combined Softmax + CCE loss
        learning_rate=0.001 # Adam's default LR is a good starting point
    )

    # --- 3. Train the Model ---
    print("\n--- Training Model on MNIST ---")
    # Training for more epochs and with a reasonable batch size is crucial for MNIST
    mnist_model.fit(x_train, y_train_one_hot, epochs=10, batch_size=64, shuffle_batch=True)

    # --- 4. Evaluate the Model on Test Data ---
    print("\n--- Evaluating Model on Test Data ---")
    # Calculate loss on test set
    test_loss = mnist_model.evaluate(x_test, y_test_one_hot)

    # Calculate Accuracy manually (since your Sequential class doesn't have built-in metrics)
    print("\n--- Calculating Test Accuracy ---")
    predictions = mnist_model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test_one_hot, axis=1) # Convert one-hot to sparse for comparison

    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- 5. Optional: Visualize a few predictions ---
    print("\n--- Visualizing Sample Predictions ---")
    num_samples_to_show = 5
    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(15, 3))

    for i in range(num_samples_to_show):
        # Get a random test sample
        idx = np.random.randint(0, len(x_test))
        sample_image = x_test[idx].reshape(28, 28)
        true_label = true_labels[idx]
        predicted_label = predicted_labels[idx]

        axes[i].imshow(sample_image, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()