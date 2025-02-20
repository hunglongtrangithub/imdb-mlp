import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
import numpy as np
import random
from loguru import logger

from src.modeling.trainer import IMDBExperimentConfig, IMDBTrainer
from src.modeling.models.mlp import MLP

hyperparam_results_json = (
    Path(__file__).resolve().parents[2] / "reports" / "hyperparam_results.json"
)

# Load the results
with open(hyperparam_results_json, "r") as f:
    results = json.load(f)

def train_and_test(model_class, model_name):
    training_results = []
    for token_level in ["word_level", "char_level"]:
        token_level_trainer_config = results[token_level]["best_config"]
        print("Best char level config:")
        print(json.dumps(token_level_trainer_config, indent=2))
        char_level_trainer = IMDBTrainer(IMDBExperimentConfig(**token_level_trainer_config), model_class=model_class)

        (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels),
        ) = IMDBTrainer.load_data()
        logger.debug(f"test_labels: {test_labels}")
        logger.info("Preprocessing data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = IMDBTrainer.preprocess_data(
            token_level == "char_level",
            train_texts,
            val_texts,
            test_texts,
            train_labels,
            val_labels,
            test_labels,
        )
        logger.info("Data preprocessed. Starting training and testing")

        for i in range(3):
            random.seed(i)
            np.random.seed(i)
            tf.random.set_seed(i)
            _, test_metrics = char_level_trainer.train(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

            # append results to dataframe with the corresponding random seed and config
            training_results.append(
                {
                    "token_level": token_level,
                    "random_seed": i,
                    "loss": float(test_metrics["loss"]),
                    "accuracy": float(test_metrics["accuracy"]),
                },
            )

    # Compute mean and std for each token level
    results_df = pd.DataFrame(training_results)
    stats_df = results_df.groupby(["token_level"]).agg({"loss": "mean", "accuracy": "mean"})
    # Save the results
    results_df.to_csv(
        Path(__file__).resolve().parents[2] / "reports" / f"{model_name}_results.csv",
        index=False,
    )
    stats_df.to_csv(
        Path(__file__).resolve().parents[2] / "reports" / f"{model_name}_stats.csv",
        index=False,
    )


if __name__ == "__main__":
    train_and_test(MLP, "best_model")