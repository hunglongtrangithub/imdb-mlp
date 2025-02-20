import tensorflow as tf
import numpy as np

from src.utils import char_level_tokenizer, texts_to_bow, one_hot_encode

tf.random.set_seed(1234)
np.random.seed(1234)


# -------------------------------
# Original MLP Class Definition
# -------------------------------
class MLP(object):
    def __init__(
        self,
        size_input,
        size_hidden1,
        size_hidden2,
        size_output,
        activation="relu",
        device=None,
    ):
        """
        size_input: int, size of input layer
        size_hidden1: int, size of the 1st hidden layer
        size_hidden2: int, size of the 2nd hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None.
        """
        self.size_input = size_input
        self.size_hidden1 = size_hidden1
        self.size_hidden2 = size_hidden2
        self.size_output = size_output
        self.device = device
        self.activation = activation

        # Initialize weights and biases for first hidden layer
        self.W1 = tf.Variable(
            tf.random.normal([self.size_input, self.size_hidden1], stddev=0.1)
        )
        self.b1 = tf.Variable(tf.zeros([1, self.size_hidden1]))

        # Initialize weights and biases for second hidden layer
        self.W2 = tf.Variable(
            tf.random.normal([self.size_hidden1, self.size_hidden2], stddev=0.1)
        )
        self.b2 = tf.Variable(tf.zeros([1, self.size_hidden2]))

        # Initialize weights and biases for output layer
        self.W3 = tf.Variable(
            tf.random.normal([self.size_hidden2, self.size_output], stddev=0.1)
        )
        self.b3 = tf.Variable(tf.zeros([1, self.size_output]))

        # List of variables to update during backpropagation
        self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]

    def forward(self, X):
        """
        Forward pass.
        X: Tensor, inputs.
        """
        if self.device is not None:
            with tf.device("gpu:0" if self.device == "gpu" else "cpu"):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)
        return self.y

    def loss(self, y_pred, y_true):
        """
        Computes the loss between predicted and true outputs.
        y_pred: Tensor of shape (batch_size, size_output)
        y_true: Tensor of shape (batch_size, size_output)
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_x = cce(y_true_tf, y_pred_tf)
        return loss_x

    def backward(self, X_train, y_train):
        """
        Backward pass: compute gradients of the loss with respect to the variables.
        """
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        grads = tape.gradient(current_loss, self.variables)
        return grads

    def activate(self, x):
        """
        Activation function.
        x: Tensor.
        """
        if self.activation == "relu":
            return tf.nn.relu(x)
        elif self.activation == "tanh":
            return tf.nn.tanh(x)
        elif self.activation == "leaky_relu":
            return tf.nn.leaky_relu(x)
        else:
            raise ValueError("Invalid activation function.")

    def compute_output(self, X):
        """
        Custom method to compute the output tensor during the forward pass.
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        # First hidden layer
        h1 = tf.matmul(X_tf, self.W1) + self.b1
        z1 = self.activate(h1)
        # Second hidden layer
        h2 = tf.matmul(z1, self.W2) + self.b2
        z2 = self.activate(h2)
        # Output layer (logits)
        output = tf.matmul(z2, self.W3) + self.b3
        return output


import tensorflow as tf
from dataclasses import dataclass
from typing import List, Literal, Union
import numpy as np


@dataclass
class MLPConfig:
    """Configuration class for MLP hyperparameters"""

    # Architecture
    input_size: int
    hidden_layers: List[int]  # List of neurons per hidden layer
    output_size: int

    # Training
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 10

    # Optimizer
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    optimizer_params: dict = None

    # Activation
    activation: Literal["relu", "tanh", "leaky_relu"] = "relu"
    activation_params: dict = None

    # Device
    device: Union[str, None] = None

    def __post_init__(self):
        # Set default optimizer params if none provided
        if self.optimizer_params is None:
            self.optimizer_params = {}

        # Set default activation params if none provided
        if self.activation_params is None:
            self.activation_params = {}
            if self.activation == "leaky_relu":
                self.activation_params["alpha"] = 0.01


class ConfigurableMLP(object):
    def __init__(self, config: MLPConfig):
        """
        Initialize MLP with configurable architecture and hyperparameters.

        Args:
            config: MLPConfig object containing all hyperparameters
        """
        self.config = config

        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []

        # Input to first hidden layer
        self.weights.append(
            tf.Variable(
                tf.random.normal(
                    [config.input_size, config.hidden_layers[0]], stddev=0.1
                )
            )
        )
        self.biases.append(tf.Variable(tf.zeros([1, config.hidden_layers[0]])))

        # Hidden layers
        for i in range(len(config.hidden_layers) - 1):
            self.weights.append(
                tf.Variable(
                    tf.random.normal(
                        [config.hidden_layers[i], config.hidden_layers[i + 1]],
                        stddev=0.1,
                    )
                )
            )
            self.biases.append(tf.Variable(tf.zeros([1, config.hidden_layers[i + 1]])))

        # Last hidden layer to output
        self.weights.append(
            tf.Variable(
                tf.random.normal(
                    [config.hidden_layers[-1], config.output_size], stddev=0.1
                )
            )
        )
        self.biases.append(tf.Variable(tf.zeros([1, config.output_size])))

        # List of all variables for the optimizer
        self.variables = []
        for w, b in zip(self.weights, self.biases):
            self.variables.extend([w, b])

        # Set up optimizer
        self.optimizer = self._get_optimizer()

    def _get_activation(self, x):
        """Get activation function based on config"""
        if self.config.activation == "relu":
            return tf.nn.relu(x)
        elif self.config.activation == "tanh":
            return tf.nn.tanh(x)
        elif self.config.activation == "leaky_relu":
            alpha = self.config.activation_params.get("alpha", 0.01)
            return tf.nn.leaky_relu(x, alpha=alpha)
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")

    def _get_optimizer(self):
        """Get optimizer based on config"""
        if self.config.optimizer == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate, **self.config.optimizer_params
            )
        elif self.config.optimizer == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=self.config.learning_rate, **self.config.optimizer_params
            )
        elif self.config.optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=self.config.learning_rate, **self.config.optimizer_params
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def forward(self, X):
        """Forward pass with configurable device"""
        if self.config.device is not None:
            with tf.device("gpu:0" if self.config.device == "gpu" else "cpu"):
                return self.compute_output(X)
        return self.compute_output(X)

    def compute_output(self, X):
        """Forward computation through all layers"""
        X_tf = tf.cast(X, dtype=tf.float32)
        current_input = X_tf

        # Process through all layers except the last
        for i in range(len(self.weights) - 1):
            h = tf.matmul(current_input, self.weights[i]) + self.biases[i]
            current_input = self._get_activation(h)

        # Last layer (no activation for logits)
        output = tf.matmul(current_input, self.weights[-1]) + self.biases[-1]
        return output

    def loss(self, y_pred, y_true):
        """Compute loss between predicted and true outputs"""
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        return cce(y_true_tf, y_pred_tf)

    def backward(self, X_train, y_train):
        """Compute gradients using gradient tape"""
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
        return tape.gradient(current_loss, self.variables)


# Example usage:
def create_model_with_config(input_size: int, output_size: int) -> ConfigurableMLP:
    """Create a model with specific configuration"""
    config = MLPConfig(
        input_size=input_size,
        hidden_layers=[128, 64, 32],  # Three hidden layers
        output_size=output_size,
        learning_rate=0.001,
        batch_size=128,
        epochs=10,
        optimizer="adam",
        optimizer_params={"beta_1": 0.9, "beta_2": 0.999},
        activation="relu",
        device=None,
    )
    return ConfigurableMLP(config)


# Alternative configuration examples:
def get_small_model_config(input_size: int, output_size: int) -> MLPConfig:
    """Configuration for a smaller model"""
    return MLPConfig(
        input_size=input_size,
        hidden_layers=[64, 32],
        output_size=output_size,
        learning_rate=0.01,
        batch_size=64,
        optimizer="sgd",
        activation="tanh",
    )


def get_large_model_config(input_size: int, output_size: int) -> MLPConfig:
    """Configuration for a larger model"""
    return MLPConfig(
        input_size=input_size,
        hidden_layers=[256, 128, 64, 32],
        output_size=output_size,
        learning_rate=0.0005,
        batch_size=256,
        optimizer="rmsprop",
        activation="leaky_relu",
        activation_params={"alpha": 0.02},
    )


# -------------------------------
# Example Usage for IMDB Classification
# -------------------------------
def test_mlp():
    # Example IMDB reviews (In practice, load your dataset here)
    texts = [
        "I loved this movie! It was fantastic.",
        "The film was terrible and boring.",
    ]
    # One-hot encoded labels for 2 classes (e.g., positive: [0,1], negative: [1,0])
    labels = np.array([[0, 1], [1, 0]])

    # Create and fit a character-level tokenizer
    tokenizer = char_level_tokenizer(texts)

    # Convert texts to bag-of-characters representation
    X = texts_to_bow(tokenizer, texts)
    print("Input shape:", X.shape)

    # Set model hyperparameters.
    # The input size is equal to the dimension of the bag-of-characters vector.
    size_input = X.shape[1]
    size_hidden1 = 64
    size_hidden2 = 32

    size_output = 2

    # Instantiate the MLP model.
    model = MLP(size_input, size_hidden1, size_hidden2, size_output, device=None)

    # Define an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Training loop (for demonstration purposes; adjust epochs and batch size as needed)
    epochs = 10
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = model.forward(X)
        # Compute loss
        current_loss = model.loss(predictions, labels)
        # Backward pass: compute gradients
        grads = model.backward(X, labels)
        # Update weights manually using the optimizer.
        optimizer.apply_gradients(zip(grads, model.variables))
        print(f"Epoch {epoch+1}, Loss: {current_loss.numpy()}")

    # Testing the model on a new review.
    new_text = ["An amazing film with a thrilling plot."]
    X_new = texts_to_bow(tokenizer, new_text)
    logits = model.forward(X_new)
    probabilities = tf.nn.softmax(logits)
    print("Predicted probabilities:", probabilities.numpy())


if __name__ == "__main__":
    test_mlp()
