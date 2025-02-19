import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from itertools import product
import pandas as pd
from datetime import datetime
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import save_to_json


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""

    learning_rate: float
    num_hidden_layers: int
    hidden_sizes: List[int]
    batch_size: int
    optimizer: str
    activation: str
    tokenization: str  # 'char' or 'word'
    random_seed: int


class ExperimentTracker:
    """Tracks and logs experimental results"""

    def __init__(self, base_path: str = "experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.experiment_log = []

    def log_experiment(self, config: ExperimentConfig, results: Dict[str, float]):
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
                config = ExperimentConfig(**config_dict, random_seed=seed)

                # Run experiment
                results = self._run_experiment(
                    config, train_fn, X_train, y_train, X_val, y_val
                )

                # Log results
                self.tracker.log_experiment(config, results)

    def random_search(self, train_fn, X_train, y_train, X_val, y_val, num_configs: int):
        """Perform random search over hyperparameter space"""
        for _ in range(num_configs):
            # Randomly sample configuration
            config_dict = {
                key: random.choice(values) for key, values in self.search_space.items()
            }

            for seed in range(self.num_trials):
                config = ExperimentConfig(**config_dict, random_seed=seed)

                results = self._run_experiment(
                    config, train_fn, X_train, y_train, X_val, y_val
                )

                self.tracker.log_experiment(config, results)

    def _run_experiment(
        self, config: ExperimentConfig, train_fn, X_train, y_train, X_val, y_val
    ) -> Dict[str, float]:
        """Run a single experiment with given configuration"""
        # Set random seeds
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        # Train model and get results
        return train_fn(config, X_train, y_train, X_val, y_val)

    def get_best_config(
        self, metric: str = "val_accuracy"
    ) -> Tuple[ExperimentConfig, Dict[str, float]]:
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
        best_config_dict = {col: stats_df.loc[best_idx, col] for col in config_cols}

        # Get all results for this configuration
        best_results = df[
            df[config_cols].apply(
                lambda x: all(x == best_config_dict.get(i) for i, v in x.items()),
                axis=1,
            )
        ]

        return (
            ExperimentConfig(**best_config_dict, random_seed=0),
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

        # 2. Hidden layers vs metric
        sns.boxplot(data=df, x="num_hidden_layers", y=metric, ax=axs[0, 1])
        axs[0, 1].set_title(f"Number of Hidden Layers vs {metric}")

        # 3. Optimizer vs metric
        sns.boxplot(data=df, x="optimizer", y=metric, ax=axs[1, 0])
        axs[1, 0].set_title(f"Optimizer vs {metric}")

        # 4. Activation vs metric
        sns.boxplot(data=df, x="activation", y=metric, ax=axs[1, 1])
        axs[1, 1].set_title(f"Activation Function vs {metric}")

        plt.tight_layout()
        return fig


# Example usage:
def main():
    # Define search space
    search_space = {
        "learning_rate": [0.001, 0.0005, 0.0001],
        "num_hidden_layers": [1, 2, 3],
        "hidden_sizes": [128, 256, 512],
        "batch_size": [32, 64, 128],
        "optimizer": ["adam", "sgd", "rmsprop"],
        "activation": ["relu", "tanh", "leaky_relu"],
        "tokenization": ["char", "word"],
    }

    # Initialize optimizer
    optimizer = HyperParameterOptimizer(search_space, num_trials=3)

    # Define training function
    def train_and_evaluate(config, X_train, y_train, X_val, y_val):
        # Create and train model using config...
        # Return metrics
        return {
            "train_loss": 0.5,
            "train_accuracy": 0.85,
            "val_loss": 0.6,
            "val_accuracy": 0.82,
        }

    # Perform grid search
    optimizer.grid_search(
        train_and_evaluate,
        X_train=None,  # Add your data
        y_train=None,
        X_val=None,
        y_val=None,
    )

    # Get best configuration
    best_config, best_stats = optimizer.get_best_config()
    print("Best configuration:", best_config)
    print("Performance statistics:", best_stats)

    # Plot results
    fig = optimizer.plot_results()
    plt.show()


if __name__ == "__main__":
    main()
