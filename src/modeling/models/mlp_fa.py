import tensorflow as tf


class MLP_FA(object):
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

        # Create fixed random feedback matrices for feedback alignment:
        # B3: used to propagate the error from the output layer to the second hidden layer.
        # It replaces the use of W3^T. Its shape is (size_output, size_hidden2).
        self.B3 = tf.Variable(
            tf.random.normal([self.size_output, self.size_hidden2]), trainable=False
        )

        # B2: used to propagate the error from the second hidden layer to the first hidden layer.
        # Its shape is (size_hidden2, size_hidden1).
        self.B2 = tf.Variable(
            tf.random.normal([self.size_hidden2, self.size_hidden1]), trainable=False
        )

        # Define variables to be updated during training
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
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        """
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_x = cce(y_true_tf, y_pred_tf)
        return loss_x

    def activate(self, x):
        """
        Activation function.
        x: Tensor, input to the activation function.
        """
        if self.activation == "relu":
            return tf.nn.relu(x)
        elif self.activation == "tanh":
            return tf.nn.tanh(x)
        elif self.activation == "leaky_relu":
            return tf.nn.leaky_relu(x)
        else:
            raise ValueError(f"Activation function {self.activation} not supported.")

    def backward(self, X_train, y_train):
        """
        Backward pass using feedback alignment.
        Computes gradients manually using fixed random feedback matrices.
        X_train: Input data (numpy array)
        y_train: One-hot encoded labels (numpy array)
        Returns: List of gradients corresponding to [dW1, dW2, dW3, db1, db2, db3]
        """
        # Cast input to float32 tensor
        X_tf = tf.cast(X_train, tf.float32)

        # --- Forward Pass ---
        # First hidden layer
        h1 = tf.matmul(X_tf, self.W1) + self.b1
        a1 = self.activate(h1)
        # Second hidden layer
        h2 = tf.matmul(a1, self.W2) + self.b2
        a2 = self.activate(h2)
        # Output layer (logits)
        logits = tf.matmul(a2, self.W3) + self.b3
        # Softmax predictions
        y_pred = tf.nn.softmax(logits)

        # --- Compute Output Error ---
        # For cross-entropy with softmax, the derivative is (y_pred - y_true)
        delta3 = y_pred - tf.cast(y_train, tf.float32)  # shape: (batch, size_output)
        batch_size = tf.cast(tf.shape(X_tf)[0], tf.float32)

        # --- Gradients for Output Layer ---
        dW3 = tf.matmul(tf.transpose(a2), delta3) / batch_size
        db3 = tf.reduce_mean(delta3, axis=0, keepdims=True)

        # --- Feedback Alignment for Second Hidden Layer ---
        # Instead of delta2 = (delta3 dot W3^T) * ReLU'(h2), use a fixed random matrix B3.
        relu_grad_h2 = tf.cast(h2 > 0, tf.float32)
        # delta3 has shape (batch, size_output) and B3 has shape (size_output, size_hidden2)
        delta2 = (
            tf.matmul(delta3, self.B3) * relu_grad_h2
        )  # shape: (batch, size_hidden2)

        dW2 = tf.matmul(tf.transpose(a1), delta2) / batch_size
        db2 = tf.reduce_mean(delta2, axis=0, keepdims=True)

        # --- Feedback Alignment for First Hidden Layer ---
        # Instead of delta1 = (delta2 dot W2^T) * ReLU'(h1), use a fixed random matrix B2.
        relu_grad_h1 = tf.cast(h1 > 0, tf.float32)
        # delta2 has shape (batch, size_hidden2) and B2 has shape (size_hidden2, size_hidden1)
        delta1 = (
            tf.matmul(delta2, self.B2) * relu_grad_h1
        )  # shape: (batch, size_hidden1)

        dW1 = tf.matmul(tf.transpose(X_tf), delta1) / batch_size
        db1 = tf.reduce_mean(delta1, axis=0, keepdims=True)

        return [dW1, dW2, dW3, db1, db2, db3]

    def compute_output(self, X):
        """
        Custom method to obtain output tensor during the forward pass.
        """
        X_tf = tf.cast(X, dtype=tf.float32)
        h1 = tf.matmul(X_tf, self.W1) + self.b1
        z1 = self.activate(h1)
        h2 = tf.matmul(z1, self.W2) + self.b2
        z2 = self.activate(h2)
        output = tf.matmul(z2, self.W3) + self.b3
        return output
