import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from src.utils import char_level_tokenizer, texts_to_bow, one_hot_encode


class IMDBTrainer:
    def __init__(self, model_class, batch_size=128, epochs=10, learning_rate=0.001):
        self.model_class = model_class
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def load_data(self):
        """Load and prepare IMDB dataset"""
        print("Loading IMDB dataset...")
        (ds_train, ds_test), _ = tfds.load(
            "imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True
        )

        # Process training data
        train_texts, train_labels = self._process_dataset(ds_train)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )

        # Process test data
        test_texts, test_labels = self._process_dataset(ds_test)

        print(
            f"Train samples: {len(train_texts)}, "
            f"Validation samples: {len(val_texts)}, "
            f"Test samples: {len(test_texts)}"
        )

        return (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels),
        )

    def _process_dataset(self, dataset):
        """Convert dataset to lists of texts and labels"""
        texts, labels = [], []
        for text, label in tfds.as_numpy(dataset):
            texts.append(text.decode("utf-8"))
            labels.append(label)
        return texts, np.array(labels)

    def preprocess_data(
        self, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    ):
        """Tokenize and vectorize the data"""
        tokenizer = char_level_tokenizer(train_texts)
        print("Tokenizer vocabulary size:", len(tokenizer.word_index) + 1)

        # Convert texts to BOW representation
        X_train = texts_to_bow(tokenizer, train_texts)
        X_val = texts_to_bow(tokenizer, val_texts)
        X_test = texts_to_bow(tokenizer, test_texts)

        # Convert labels to one-hot encoding
        y_train = one_hot_encode(train_labels)
        y_val = one_hot_encode(val_labels)
        y_test = one_hot_encode(test_labels)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def setup_model(self, input_size):
        """Initialize model and optimizer"""
        model = self.model_class(
            size_input=input_size,
            size_hidden1=128,
            size_hidden2=64,
            size_hidden3=32,  # Placeholder (not used in forward pass)
            size_output=2,
            device=None,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        return model, optimizer

    def train_epoch(self, model, optimizer, X_train, y_train):
        """Train for one epoch"""
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        num_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
        epoch_loss = 0

        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, X_train.shape[0])
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            predictions = model.forward(X_batch)
            loss_value = model.loss(predictions, y_batch)
            grads = model.backward(X_batch, y_batch)
            optimizer.apply_gradients(zip(grads, model.variables))
            epoch_loss += loss_value.numpy() * (end - start)

        return epoch_loss / X_train.shape[0]

    def evaluate(self, model, X, y):
        """Evaluate model performance"""
        logits = model.forward(X)
        loss = model.loss(logits, y).numpy()
        preds = np.argmax(logits.numpy(), axis=1)
        true_vals = np.argmax(y, axis=1)

        return {
            "loss": loss,
            "accuracy": np.mean(preds == true_vals),
            "precision": precision_score(true_vals, preds),
            "recall": recall_score(true_vals, preds),
        }

    def train(self):
        """Main training loop"""
        # Load and preprocess data
        train_data, val_data, test_data = self.load_data()
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.preprocess_data(
            *train_data, *val_data, *test_data
        )

        # Setup model
        model, optimizer = self.setup_model(X_train.shape[1])

        # Training loop
        print("\nStarting training...\n")
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(model, optimizer, X_train, y_train)
            val_metrics = self.evaluate(model, X_val, y_val)

            print(
                f"Epoch {epoch+1:02d} | "
                f"Training Loss: {epoch_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f}"
            )

        # Final evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(model, X_test, y_test)
        print(
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
            f"Test Precision: {test_metrics['precision']:.4f} | "
            f"Test Recall: {test_metrics['recall']:.4f}"
        )

        return model, test_metrics


# Usage example:
def train_imdb(model_class):
    trainer = IMDBTrainer(model_class)
    model, metrics = trainer.train()
    return model, metrics
