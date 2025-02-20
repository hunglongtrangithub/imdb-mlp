from typing import List, Dict, Any, Tuple
from itertools import product
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.utils import save_to_json
from src.modeling.trainer import IMDBTrainer, IMDBExperimentConfig


class ExperimentTracker:
    """Tracks and logs experimental results"""

    def __init__(self, base_path: str = "experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.experiment_log = []

    def log_experiment(self, config: IMDBExperimentConfig, results: Dict[str, float]):
        """Log a single experiment run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment = {
            "timestamp": timestamp,
            "config": config.__dict__,
            "results": results,
        }
        self.experiment_log.append(experiment)

        # Save to file
        exp_path = self.base_path / f"experiment_{timestamp}.json"
        save_to_json(experiment, exp_path)

    def get_results_df(self) -> pd.DataFrame:
        """Convert experiment log to DataFrame"""
        records = []
        for exp in self.experiment_log:
            record = {**exp["config"], **exp["results"]}
            record["timestamp"] = exp["timestamp"]
            records.append(record)
        return pd.DataFrame(records)


class HyperParameterOptimizer:
    def __init__(self, search_space: Dict[str, List[Any]], num_trials: int = 3):
        self.search_space = search_space
        self.num_trials = num_trials
        self.tracker = ExperimentTracker()

    def grid_search(self, train_fn, X_train, y_train, X_val, y_val):
        """Perform grid search over hyperparameter space"""
        # Generate all combinations
        keys, values = zip(*self.search_space.items())
        configurations = [dict(zip(keys, v)) for v in product(*values)]

        for config_dict in configurations:
            for seed in range(self.num_trials):
                # Create experiment config
                config = IMDBExperimentConfig(**config_dict, random_seed=seed)

                # Run experiment
                results = self._run_experiment(
                    config, train_fn, X_train, y_train, X_val, y_val
                )

                # Log results
                self.tracker.log_experiment(config, results)

    def random_search(self, train_fn, X_train, y_train, X_val, y_val, num_configs: int):
        """Perform random search over hyperparameter space"""
        random.seed(0)
        for _ in range(num_configs):
            # Randomly sample configuration
            config_dict = {
                key: random.choice(values) for key, values in self.search_space.items()
            }

            for seed in range(self.num_trials):
                config = IMDBExperimentConfig(**config_dict, random_seed=seed)

                results = self._run_experiment(
                    config, train_fn, X_train, y_train, X_val, y_val
                )

                self.tracker.log_experiment(config, results)

    def _run_experiment(
        self, config: IMDBExperimentConfig, train_fn, X_train, y_train, X_val, y_val
    ) -> Dict[str, float]:
        """Run a single experiment with given configuration"""
        # Set random seeds
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        logger.info(f"Running experiment with config: {config.__dict__}")
        # Train model and get results
        return train_fn(config, X_train, y_train, X_val, y_val)

    def get_best_config(
        self, metric: str = "val_accuracy"
    ) -> Tuple[IMDBExperimentConfig, Dict[str, float]]:
        """Get the best performing configuration"""
        df = self.tracker.get_results_df()

        # Group by configuration (excluding seed and timestamp) and compute statistics
        config_cols = [
            col
            for col in df.columns
            if col
            not in ["random_seed", "timestamp"]
            + list(df.filter(regex="^(train|val)_").columns)
        ]

        stats_df = (
            df.groupby(config_cols)
            .agg({metric: ["mean", "std", "count"]})
            .reset_index()
        )

        # Find best configuration
        best_idx = stats_df[metric]["mean"].idxmax()
        # Extract plain values from pandas Series
        best_config_dict = {
            col: stats_df.loc[best_idx, col].item() for col in config_cols
        }
        logger.info(f"Best configuration: {best_config_dict}")

        return (
            IMDBExperimentConfig(**best_config_dict, random_seed=0),
            {
                f"{metric}_mean": stats_df.loc[best_idx, (metric, "mean")],
                f"{metric}_std": stats_df.loc[best_idx, (metric, "std")],
                f"{metric}_trials": stats_df.loc[best_idx, (metric, "count")],
            },
        )

    def plot_results(self, metric: str = "val_accuracy"):
        """Plot experimental results"""
        df = self.tracker.get_results_df()

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Learning rate vs metric
        sns.boxplot(data=df, x="learning_rate", y=metric, ax=axs[0, 0])
        axs[0, 0].set_title(f"Learning Rate vs {metric}")

        # 3. Optimizer vs metric
        sns.boxplot(data=df, x="optimizer", y=metric, ax=axs[1, 0])
        axs[1, 0].set_title(f"Optimizer vs {metric}")

        # 4. Activation vs metric
        sns.boxplot(data=df, x="activation", y=metric, ax=axs[1, 1])
        axs[1, 1].set_title(f"Activation Function vs {metric}")

        plt.tight_layout()
        return fig


def train_imdb_with_config(
    config: IMDBExperimentConfig, X_train, y_train, X_val, y_val
):
    """Training function wrapper for hyperparameter optimization"""
    # Initialize trainer with config
    trainer = IMDBTrainer(config)

    # Train the model and get metrics
    _, metrics = trainer.train(X_train, y_train, X_val, y_val)

    # Return metrics in the format expected by HyperParameterOptimizer
    return {
        "val_loss": metrics["loss"],
        "val_accuracy": metrics["accuracy"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
    }


def main():
    # add logger to file
    logger.add("hyperparam.log")
    results = {}
    # Define search space for IMDB experiments
    search_space = {
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [32, 64, 128],
        "epochs": [5, 10],
        "hidden_size1": [128, 256, 512],
        "hidden_size2": [128, 256, 512],
        "optimizer": ["adam", "sgd", "rmsprop"],
        "activation": ["relu", "tanh", "leaky_relu"],
    }
    results["search_space"] = search_space
    logger.info(f"Search space: {search_space}")
    for token_level in ["char_level", "word_level"]:
        logger.info(f"Token level: {token_level}")
        sub_results = {}
        # Initialize optimizer with fewer trials for demonstration
        optimizer = HyperParameterOptimizer(search_space)

        # Load and preprocess data once
        is_char_level = token_level == "char_level"
        (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels),
        ) = IMDBTrainer.load_data()
        logger.debug(f"test_labels: {test_labels}")
        logger.info("Preprocessing data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
            IMDBTrainer.preprocess_data(
                is_char_level,
                train_texts,
                val_texts,
                test_texts,
                train_labels,
                val_labels,
                test_labels,
            )
        )
        logger.info(
            "Data preprocessed. Starting hyperparameter optimization: random search"
        )
        # Run random search
        optimizer.random_search(
            train_fn=lambda config, X_train, y_train, X_val, y_val: train_imdb_with_config(
                config, X_train, y_train, X_val, y_val
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_configs=5,  # Number of random configurations to try
        )
        logger.info("Random search completed.")

        # # Run grid search
        # logger.info(
        #     "Data preprocessed. Starting hyperparameter optimization: grid search"
        # )
        # optimizer.grid_search(
        #     train_fn=lambda config, X_train, y_train, X_val, y_val: train_imdb_with_config(
        #         config, X_train, y_train, X_val, y_val
        #     ),
        #     X_train=X_train,
        #     y_train=y_train,
        #     X_val=X_val,
        #     y_val=y_val,
        # )
        # logger.info("Grid search completed.")

        # Get and print best configuration
        best_config, best_stats = optimizer.get_best_config(metric="val_accuracy")
        print("Best configuration:", best_config)
        print("Best validation performance:", best_stats)
        sub_results["best_config"] = best_config.__dict__
        sub_results["best_stats"] = best_stats

        # Plot results
        optimizer.plot_results(metric="val_accuracy")
        logger.info("Training final model with best configuration...")
        # Train final model with best configuration
        final_trainer = IMDBTrainer(best_config)
        final_model, test_metrics = final_trainer.train(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        print("\nFinal Test Set Performance:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        sub_results["test_metrics"] = test_metrics

        results[token_level] = sub_results

    logger.info("Saving results to file...")
    # Save results
    results_file = (
        Path(__file__).resolve().parents[2] / "reports" / "hyperparam_results.json"
    )
    save_to_json(results, results_file)


if __name__ == "__main__":
    main()
