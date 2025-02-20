from pathlib import Path
import pandas as pd

results_file = Path(__file__).resolve().parent / "reports" / "hyperparam_results.csv"

# Load the results
df = pd.read_csv(results_file)

metric = "val_accuracy"
# Group by configuration (excluding seed and timestamp) and compute statistics
config_cols = [
    col
    for col in df.columns
    if col
    not in ["random_seed", "timestamp"] + list(df.filter(regex="^(train|val)_").columns)
]
stats_df = df.groupby(config_cols).agg({metric: ["mean", "std", "count"]}).reset_index()
print(stats_df)
# Find best configuration
best_idx = stats_df[metric]["mean"].idxmax()  # type: ignore
# Extract plain values from pandas Series
best_config_dict = {
    col: stats_df.loc[best_idx, col].item() for col in config_cols
}
print(f"Best configuration: {best_config_dict}")

# Sort by mean validation accuracy
stats_df = stats_df.sort_values((metric, "mean"), ascending=False)

# Save the results
stats_df.to_csv(results_file.parent / "hyperparam_stats.csv", index=False)

