# A Multi-Dataset Benchmark of Tabular Self-Attention Residual Networks and Strong Baseline

Name: Krrish Chhabra
Section: D
Subject: Deep Learning
Registration Number: 2430010396

## Title
A Multi-Dataset Benchmark of Tabular Self-Attention Residual Networks and Strong Baseline

## Research Objective
This project evaluates whether a tabular self-attention residual neural
network can provide competitive performance against classical machine
learning models across multiple tabular datasets.

## Research Question
Under what conditions does Tab-ResNet-Attn provide benefits compared
with classical tree-based models and a standard MLP?

## Datasets
1. Adult Census Income
2. Bank Marketing
3. Default of Credit Card Clients
4. Covertype

The Adult dataset predicts income above $50K. It does not directly
measure charitable donor behavior.

## Proposed Model
Tab-ResNet-Attn combines:
- Feature tokenization
- Multi-head self-attention
- Residual processing
- Feed-forward neural layers
- A task-specific prediction head

## Baselines
- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost
- Multilayer Perceptron

Not every baseline is available for every dataset or experiment.

## Evaluation
Classification metrics may include:
- Accuracy
- Precision
- Recall
- F1-score
- F0.5-score
- ROC-AUC
- Average Precision
- Macro-F1
- Macro Recall

## Reproducibility
See:
- docs/methodology.md
- docs/dataset_sources.md

Experiment outputs are stored under results/.

## Important
Results depend on the dataset split, preprocessing, hyperparameters,
random seeds, and available hardware.

Do not treat preliminary or unmatched experiments as definitive
comparisons.
