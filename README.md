# Neural Network Tokenization and Hyperparameter Optimization Analysis
All training/testing/tokenizing results are in the `reports/` directory. The report is the `NLP.01.docx` file. I don't use the .ipynb file because it is too difficult to develop into large, complicated projects. I use the .py file to develop the project. All of my code is in the `src/` directory.

## Install the required libraries
use uv to install:
```bash
uv sync
```
activate the virtual environment:
```bash
source .venv/bin/activate
```

# Run the code
tokenizer:
```bash
python -m src.modeling.tokenizer
```
hyperparameter optimization:
```bash
python -m src.modeling.hyperparam
```
train and test best models / random models:
```bash
python -m src.scripts.test_best_model
```
```bash
python -m src.scripts.test_random_model
```