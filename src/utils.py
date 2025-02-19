import tensorflow as tf
import numpy as np
import json


class CustomEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types to make them JSON serializable."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()  # Convert arrays to lists
        elif isinstance(
            o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)
        ):
            return int(o)  # Convert NumPy integers to Python int
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)  # Convert NumPy floats to Python float
        elif isinstance(o, (np.bool_)):
            return bool(o)  # Convert NumPy boolean to Python bool
        if isinstance(o, set):
            return list(o)  # Convert sets to lists
        else:
            return super().default(o)  # Default behavior for other types


def save_to_json(data, filename):
    """Save data to a JSON file using the custom encoder."""
    with open(filename, "w") as f:
        json.dump(data, f, cls=CustomEncoder, indent=4)


def char_level_tokenizer(texts, num_words=None):
    """
    Create and fit a character-level tokenizer.

    Args:
        texts (list of str): List of texts.
        num_words (int or None): Maximum number of tokens to keep.

    Returns:
        tokenizer: A fitted Tokenizer instance.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, char_level=True, lower=True
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_bow(tokenizer, texts):
    """
    Convert texts to a bag-of-characters representation.

    Args:
        tokenizer: A fitted character-level Tokenizer.
        texts (list of str): List of texts.

    Returns:
        Numpy array representing the binary bag-of-characters for each text.
    """
    # texts_to_matrix with mode 'binary' produces a fixed-length binary vector per text.
    matrix = tokenizer.texts_to_matrix(texts, mode="binary")
    return matrix


def one_hot_encode(labels, num_classes=2):
    """
    Convert numeric labels to one-hot encoded vectors.
    """
    return np.eye(num_classes)[labels]


if __name__ == "__main__":
    # Sample texts
    texts = ["hello", "world", "hello world"]

    # Test char_level_tokenizer
    tokenizer = char_level_tokenizer(texts)
    print("Tokenizer word index:", tokenizer.word_index)

    # Test texts_to_bow
    bow_matrix = texts_to_bow(tokenizer, texts)
    print("Bag-of-characters matrix:\n", bow_matrix)

    # Sample labels
    labels = [0, 1, 0]

    # Test one_hot_encode
    one_hot_labels = one_hot_encode(labels, num_classes=2)
    print("One-hot encoded labels:\n", one_hot_labels)
