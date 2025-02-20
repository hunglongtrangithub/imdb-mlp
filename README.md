# Assignment 2 Todos

## Overview

- **Objective:**  
  Modify your model’s text preprocessing by changing from character-level tokenization to word-level tokenization. Compare the performance of both tokenization methods. Additionally, perform hyper-parameter optimization by experimenting with various settings (learning rate, hidden layers, hidden sizes, batch sizes, optimizers, and activation functions) and report your findings.

## 1. Initial Setup

- [x] **Set Random Seeds:**
      Ensure reproducibility by setting seeds for all random number generators (e.g., Python’s `random`, NumPy, TensorFlow/PyTorch).
- [x] **Prepare the Environment:**
  - Create a new or update an existing Jupyter Notebook.
  - Ensure that all necessary libraries (e.g., NumPy, pandas, TensorFlow/PyTorch, matplotlib, etc.) are installed.
- [x] **Version Control:**  
       Initialize a Git repository (if not already done) and commit your initial setup.

## 2. Data Preprocessing

- [x] **Load Dataset:**  
       Load your dataset into the notebook.
- [x] **Tokenization:**
  - **Character-Level Tokenization:**
    - Tokenize the text data at the character level.
    - Save and log the processed data.
  - **Word-Level Tokenization:**
    - Modify the tokenization process to tokenize the text by words.
    - Save and log the processed data.
- [x] **Comparison:**
  - Create a section in your notebook to compare the two tokenization approaches.
  - Visualize or tabulate differences in vocabulary size, sequence lengths, and other relevant metrics.

## 3. Model Architecture

- [x] **Define the Model:**  
       Develop a model (or models) that can handle both tokenization types. Include the following adjustable hyper-parameters:
  - Learning rate
  - Number of hidden layers
  - Hidden sizes (neurons per layer)
  - Batch sizes
  - Optimizers (e.g., Adam, SGD, RMSProp)
  - Activation functions (e.g., ReLU, Tanh, LeakyReLU)

## 4. Hyper-Parameter Optimization

- [x] **Experiment Setup:**  
       For each hyper-parameter configuration, perform at least 3 different tests to ensure robustness.
- [x] **Grid/Random Search:**  
       Set up a search over the following hyper-parameter ranges (example values provided):
  - **Learning Rate:** `[0.001, 0.0005, 0.0001]`
  - **Hidden Layers:** `[1, 2, 3]`
  - **Hidden Sizes:** `[128, 256, 512]`
  - **Batch Sizes:** `[32, 64, 128]`
  - **Optimizers:** `[Adam, SGD, RMSProp]`
  - **Activation Functions:** `[ReLU, Tanh, LeakyReLU]`
- [x] **Logging:**  
       Record the results (accuracy, loss, etc.) for each configuration in tables or charts.

## 5. Model Training and Evaluation

- [x] **Training with Each Configuration:**  
       Run experiments for both tokenization approaches with each set of hyper-parameters:
  - Train the model at least 3 times per configuration (keeping the seed constant at this stage).
  - Log training and validation performance.
- [x] **Identify the Best Model:**  
       Select the best performing configuration based on validation metrics (e.g., accuracy).

## 6. Final Experiments

- [x] **Robustness Check:**  
       Once the best model is identified:
  - Re-run the experiments at least 3 times with different random seeds.
  - Record the performance (accuracy) for each run.
- [x] **Statistical Reporting:**
  - Compute the **mean accuracy** and **standard error** across these runs.
  - Include these statistics in your report.

## 7. Documentation and Reporting

- [x] **Jupyter Notebook:**
  - Ensure that your notebook is well-commented and clearly documents each step.
  - Include code cells for setting seeds, data preprocessing, model building, training, evaluation, and visualization.
- [ ] **Detailed Report (Word Document):**  
       Prepare a report that includes:
  - **Introduction:** Objectives and overview of the work.
  - **Methodology:** Detailed explanation of tokenization changes and hyper-parameter optimization strategy.
  - **Experiments and Results:**
    - Comparison between character-level and word-level tokenization.
    - Tables/graphs for hyper-parameter experiments.
    - Final model performance with mean accuracy and standard error.
  - **Discussion:** Analysis of results, challenges encountered, and insights.
  - **Conclusion:** Summarize the key findings.
- [ ] **Submission:**
  - Submit your Jupyter Notebook.
  - Submit your Word document report.
  - Ensure that both files are included in your repository or submission package.

## 8. Final Checklist

- [x] All experiments have at least 3 different tests.
- [x] Random seeds are set before any experiment.
- [x] Hyper-parameter optimization covers changes in learning rate, hidden layers, hidden sizes, batch sizes, optimizers, and activation functions.
- [x] The best model’s performance is verified with experiments on different seeds.
- [x] Best model should be compared with random model shown above.
- [x] The report clearly documents the methodology, experiments, results, and final conclusions.
- [ ] If experiments are shown with deeper MLP_FA with best settings (Extra credits -- 2 points)

---

> **Note:**  
> Keep thorough logs and document any observations during your experiments. Clear documentation is key to reproducibility and understanding your results.

## Submission Details:

### Tokenizer Word Level vs Character Level

Logs:

```
Loaded 50000 text samples

Creating tokenizers...
2025-02-19 09:37:41.983 | INFO     | __main__:create_char_tokenizer:37 - char_tokenizer.word_index: {' ': 1, 'e': 2, 't': 3, 'a': 4, 'i': 5, 'o': 6, 's': 7, 'n': 8, 'r': 9, 'h': 10}...
2025-02-19 09:37:53.565 | INFO     | __main__:create_word_tokenizer:57 - word_tokenizer.word_index: {'the': 1, 'and': 2, 'a': 3, 'of': 4, 'to': 5, 'is': 6, 'br': 7, 'in': 8, 'it': 9, 'i': 10}...

Analyzing sequence lengths...

Tokenization Comparison:
--------------------------------------------------
Character-level vocabulary size: 163
Word-level vocabulary size: 124253

Sequence Length Statistics:
Character-level:
  mean: 1309.43
  median: 970.00
  min: 32.00
  max: 13704.00

Word-level:
  mean: 235.03
  median: 176.00
  min: 6.00
  max: 2493.00

Creating visualizations...
Results saved to /home/80026129/PROJECTS/learning/NLP_01/reports/tokenizer_stats.json
Figure saved to /home/80026129/PROJECTS/learning/NLP_01/reports/figures/tokenizer_stats.png
```
