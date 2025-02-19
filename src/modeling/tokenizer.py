from itertools import islice
from loguru import logger
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
from src.utils import save_to_json


def load_imdb_dataset() -> List[str]:
    """Load IMDB dataset and return list of text samples."""
    print("Loading IMDB dataset...")
    (ds_train, ds_test), _ = tfds.load(
        "imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True
    )

    # NOTE: Combine train and test texts
    texts = []
    for dataset in [ds_train, ds_test]:
        for text, _ in tfds.as_numpy(dataset):
            texts.append(text.decode("utf-8"))

    print(f"Loaded {len(texts)} text samples")
    return texts


def create_char_tokenizer(
    texts: List[str], num_words: int = None
) -> Tuple[Tokenizer, Dict]:
    """Create and fit a character-level tokenizer."""
    # Create character-level tokenizer
    char_tokenizer = Tokenizer(num_words=num_words, char_level=True)
    char_tokenizer.fit_on_texts(texts)
    logger.info(
        f"char_tokenizer.word_index: {dict(islice(char_tokenizer.word_index.items(), 10))}..."
    )
    # Analyze character-level statistics
    stats = {
        "vocab_size": len(char_tokenizer.word_index) + 1,
        "unique_characters": set(char_tokenizer.word_index.keys()),
        "char_frequencies": Counter("".join(texts)),
    }

    return char_tokenizer, stats


def create_word_tokenizer(
    texts: List[str], num_words: int = None
) -> Tuple[Tokenizer, Dict]:
    """Create and fit a word-level tokenizer."""
    # Create word-level tokenizer
    word_tokenizer = Tokenizer(num_words=num_words, char_level=False)
    word_tokenizer.fit_on_texts(texts)
    logger.info(
        f"word_tokenizer.word_index: {dict(islice(word_tokenizer.word_index.items(), 10))}..."
    )
    # Analyze word-level statistics
    all_words = " ".join(texts).split()
    stats = {
        "vocab_size": len(word_tokenizer.word_index) + 1,
        "unique_words": set(word_tokenizer.word_index.keys()),
        "word_frequencies": Counter(all_words),
    }

    return word_tokenizer, stats


def analyze_sequence_lengths(
    texts: List[str], char_tokenizer: Tokenizer, word_tokenizer: Tokenizer
) -> Dict:
    """Analyze sequence lengths for both tokenization approaches."""
    # Get sequences
    # NOTE: process all texts in the list
    char_sequences = char_tokenizer.texts_to_sequences(texts)
    word_sequences = word_tokenizer.texts_to_sequences(texts)

    # Calculate lengths
    char_lengths = [len(seq) for seq in char_sequences]
    word_lengths = [len(seq) for seq in word_sequences]

    return {
        "char_lengths": {
            "mean": np.mean(char_lengths),
            "median": np.median(char_lengths),
            "min": np.min(char_lengths),
            "max": np.max(char_lengths),
        },
        "word_lengths": {
            "mean": np.mean(word_lengths),
            "median": np.median(word_lengths),
            "min": np.min(word_lengths),
            "max": np.max(word_lengths),
        },
    }


def visualize_comparisons(char_stats: Dict, word_stats: Dict, length_stats: Dict):
    """Create visualizations comparing the tokenization approaches."""
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Top characters frequency
    char_freq = pd.DataFrame.from_dict(
        dict(
            sorted(
                char_stats["char_frequencies"].items(), key=lambda x: x[1], reverse=True
            )[:20]
        ),
        orient="index",
        columns=["frequency"],
    )
    char_freq.plot(kind="bar", ax=axs[0, 0], title="Top 20 Character Frequencies")
    axs[0, 0].set_xlabel("Character")
    axs[0, 0].set_ylabel("Frequency")

    # 2. Top words frequency
    word_freq = pd.DataFrame.from_dict(
        dict(
            sorted(
                word_stats["word_frequencies"].items(), key=lambda x: x[1], reverse=True
            )[:20]
        ),
        orient="index",
        columns=["frequency"],
    )
    word_freq.plot(kind="bar", ax=axs[0, 1], title="Top 20 Word Frequencies")
    axs[0, 1].set_xlabel("Word")
    axs[0, 1].set_ylabel("Frequency")

    # 3. Sequence length distributions
    axs[1, 0].hist(
        length_stats["char_lengths"].values(),
        bins=50,
        alpha=0.5,
        label="Character-level",
    )
    axs[1, 0].hist(
        length_stats["word_lengths"].values(), bins=50, alpha=0.5, label="Word-level"
    )
    axs[1, 0].set_title("Sequence Length Distributions")
    axs[1, 0].set_xlabel("Sequence Length")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].legend()

    # 4. Vocabulary size comparison
    vocab_sizes = {
        "Character-level": char_stats["vocab_size"],
        "Word-level": word_stats["vocab_size"],
    }
    axs[1, 1].bar(vocab_sizes.keys(), vocab_sizes.values())
    axs[1, 1].set_title("Vocabulary Size Comparison")
    axs[1, 1].set_ylabel("Size")

    plt.tight_layout()
    return fig


def main():
    import json
    from pathlib import Path

    # Load dataset
    texts = load_imdb_dataset()

    # Create tokenizers
    print("\nCreating tokenizers...")
    char_tokenizer, char_stats = create_char_tokenizer(texts)
    word_tokenizer, word_stats = create_word_tokenizer(texts)

    # Analyze sequence lengths
    print("\nAnalyzing sequence lengths...")
    length_stats = analyze_sequence_lengths(texts, char_tokenizer, word_tokenizer)

    # Print comparison
    print("\nTokenization Comparison:")
    print("-" * 50)
    print(f"Character-level vocabulary size: {char_stats['vocab_size']}")
    print(f"Word-level vocabulary size: {word_stats['vocab_size']}")
    print("\nSequence Length Statistics:")
    print("Character-level:")
    for key, value in length_stats["char_lengths"].items():
        print(f"  {key}: {value:.2f}")
    print("\nWord-level:")
    for key, value in length_stats["word_lengths"].items():
        print(f"  {key}: {value:.2f}")

    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_comparisons(char_stats, word_stats, length_stats)

    results = {
        "char_stats": char_stats,
        "word_stats": word_stats,
        "length_stats": length_stats,
    }

    # save the stats to json file, and figure to png file
    stats_file = (
        Path(__file__).resolve().parents[2] / "reports" / "tokenizer_stats.json"
    )
    save_to_json(results, stats_file)
    print(f"Results saved to {stats_file}")

    fig_file = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "figures"
        / "tokenizer_stats.png"
    )
    fig.savefig(fig_file)
    print(f"Figure saved to {fig_file}")


if __name__ == "__main__":
    main()
