==the MLP class definitions at the first and second code cells are the same.==

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
- [ ] **Tokenization:**
  - **Character-Level Tokenization:**
    - Tokenize the text data at the character level.
    - Save and log the processed data.
  - **Word-Level Tokenization:**
    - Modify the tokenization process to tokenize the text by words.
    - Save and log the processed data.
- [ ] **Comparison:**
  - Create a section in your notebook to compare the two tokenization approaches.
  - Visualize or tabulate differences in vocabulary size, sequence lengths, and other relevant metrics.

## 3. Model Architecture

- [ ] **Define the Model:**  
       Develop a model (or models) that can handle both tokenization types. Include the following adjustable hyper-parameters:
  - Learning rate
  - Number of hidden layers
  - Hidden sizes (neurons per layer)
  - Batch sizes
  - Optimizers (e.g., Adam, SGD, RMSProp)
  - Activation functions (e.g., ReLU, Tanh, LeakyReLU)

## 4. Hyper-Parameter Optimization

- [ ] **Experiment Setup:**  
       For each hyper-parameter configuration, perform at least 3 different tests to ensure robustness.
- [ ] **Grid/Random Search:**  
       Set up a search over the following hyper-parameter ranges (example values provided):
  - **Learning Rate:** `[0.001, 0.0005, 0.0001]`
  - **Hidden Layers:** `[1, 2, 3]`
  - **Hidden Sizes:** `[128, 256, 512]`
  - **Batch Sizes:** `[32, 64, 128]`
  - **Optimizers:** `[Adam, SGD, RMSProp]`
  - **Activation Functions:** `[ReLU, Tanh, LeakyReLU]`
- [ ] **Logging:**  
       Record the results (accuracy, loss, etc.) for each configuration in tables or charts.

## 5. Model Training and Evaluation

- [ ] **Training with Each Configuration:**  
       Run experiments for both tokenization approaches with each set of hyper-parameters:
  - Train the model at least 3 times per configuration (keeping the seed constant at this stage).
  - Log training and validation performance.
- [ ] **Identify the Best Model:**  
       Select the best performing configuration based on validation metrics (e.g., accuracy).

## 6. Final Experiments

- [ ] **Robustness Check:**  
       Once the best model is identified:
  - Re-run the experiments at least 3 times with different random seeds.
  - Record the performance (accuracy) for each run.
- [ ] **Statistical Reporting:**
  - Compute the **mean accuracy** and **standard error** across these runs.
  - Include these statistics in your report.

## 7. Documentation and Reporting

- [ ] **Jupyter Notebook:**
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

- [ ] All experiments have at least 3 different tests.
- [ ] Random seeds are set before any experiment.
- [ ] Hyper-parameter optimization covers changes in learning rate, hidden layers, hidden sizes, batch sizes, optimizers, and activation functions.
- [ ] The best model’s performance is verified with experiments on different seeds.
- [ ] Best model should be compared with random model shown above.
- [ ] The report clearly documents the methodology, experiments, results, and final conclusions.
- [ ] If experiments are shown with deeper MLP_FA with best settings (Extra credits -- 2 points)

---

> **Note:**  
> Keep thorough logs and document any observations during your experiments. Clear documentation is key to reproducibility and understanding your results.
