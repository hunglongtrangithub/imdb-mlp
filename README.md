# Neural Network Charater-level vs Word-level Tokenization Analysis

This project investigates the impact of tokenization methods (character-level vs. word-level) and hyperparameter optimization on neural network performance for sentiment analysis using the IMDB reviews dataset. The study compares the effectiveness of different tokenization approaches, identifies optimal hyperparameters, and evaluates model performance.

All training, testing, and tokenization results are stored in the `reports/` directory. The detailed report is available in the `NLP.01.docx` file. The project is developed using `.py` files for scalability and maintainability, with all code located in the `src/` directory.

---

## Key Findings

- **Tokenization Impact**: Word-level tokenization significantly outperformed character-level tokenization, achieving **86.32% test accuracy** compared to **56.79%** for character-level models.
- **Optimal Hyperparameters**: Both tokenization methods converged to the same optimal hyperparameters:
  - Learning Rate: 0.001
  - Batch Size: 32
  - Optimizer: RMSprop
  - Activation: LeakyReLU
  - Hidden Layers: 256 → 128 neurons
- **Model Stability**: Word-level models demonstrated greater stability, with a standard deviation of **0.16%** in validation accuracy compared to **1.38%** for character-level models.
- **Generalization**: Word-level models showed better generalization, with test accuracy closely matching validation accuracy.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- [UV](https://github.com/astral-sh/uv) for dependency management

### Install Required Libraries

Use `uv` to install dependencies:

```bash
uv sync
```

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## Project Structure

```
.
├── reports/                  # Contains all experimental results and the final report
│   └── NLP.01.docx           # Detailed report of the experiment
├── src/                      # Source code directory
│   ├── modeling/             # Tokenization and hyperparameter optimization code
│   │   ├── tokenizer.py      # Tokenization implementation
│   │   └── hyperparam.py     # Hyperparameter optimization code
│   ├── scripts/              # Scripts for training and testing models
│   │   ├── test_best_model.py # Train and test the best-performing model
│   │   └── test_random_model.py # Train and test random baseline models
│   └── utils/                # Utility functions and helpers
└── README.md                 # Project overview and instructions
```

---

## Running the Code

### 1. Tokenization

Run the tokenization script to preprocess the dataset:

```bash
python -m src.modeling.tokenizer
```

### 2. Hyperparameter Optimization

Perform hyperparameter optimization using random search:

```bash
python -m src.modeling.hyperparam
```

### 3. Train and Test Models

- Train and test the best-performing model:
  ```bash
  python -m src.scripts.test_best_model
  ```
- Train and test random baseline models:
  ```bash
  python -m src.scripts.test_random_model
  ```

---

## Results

### Tokenization Comparison

- **Character-Level Tokenization**:
  - Vocabulary Size: 163 unique characters
  - Mean Sequence Length: 1,309.43 characters
- **Word-Level Tokenization**:
  - Vocabulary Size: 124,253 unique words
  - Mean Sequence Length: 235.03 words

### Model Performance

| Metric         | Word-Level Model | Character-Level Model |
| -------------- | ---------------- | --------------------- |
| Test Accuracy  | 86.32%           | 56.79%                |
| Test Precision | 86.74%           | 67.03%                |
| Test Recall    | 85.74%           | 26.74%                |
| Test Loss      | 0.8503           | 0.6742                |

### Robustness Analysis

- Word-level models showed consistent performance across 27 trials, with a standard deviation of **0.16%** in validation accuracy.
- Character-level models exhibited higher variance, with a standard deviation of **1.38%**.

---

## Discussion

### Key Insights

1. **Tokenization Choice**: Word-level tokenization is more effective for sentiment analysis, as it captures meaningful semantic features at the word level.
2. **Hyperparameter Sensitivity**: The optimal hyperparameters were consistent across tokenization methods, suggesting robust model design.
3. **Model Generalization**: Word-level models generalized better to unseen data, with minimal performance drop between validation and test sets.

### Limitations

- Limited exploration of deeper architectures for character-level models.
- Fixed epoch count (5) may have restricted the potential of some configurations.
- Random search covered only 10 configurations due to computational constraints.

---

## Future Work

- Explore hybrid tokenization approaches combining character-level and word-level features.
- Investigate the impact of pre-trained embeddings (e.g., Word2Vec, GloVe) on model performance.
- Conduct more extensive hyperparameter optimization using grid search or Bayesian methods.
- Experiment with advanced architectures, such as transformers or attention mechanisms.

---

## License

This project is licensed under the MIT License.

