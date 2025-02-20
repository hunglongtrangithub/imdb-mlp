import tensorflow as tf


class MLP_rnd(object):
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
        size_hidden3: int, size of the 3rd hidden layer (not used in compute_output here)
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
        # self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        self.variables = [self.W3, self.b3]

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

    def activate(self, X):
        """
        Activation function.
        """
        if self.activation == "relu":
            return tf.nn.relu(X)
        elif self.activation == "leaky_relu":
            return tf.nn.leaky_relu(X)
        elif self.activation == "tanh":
            return tf.nn.tanh(X)
        else:
            raise ValueError("Unknown activation function")

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
