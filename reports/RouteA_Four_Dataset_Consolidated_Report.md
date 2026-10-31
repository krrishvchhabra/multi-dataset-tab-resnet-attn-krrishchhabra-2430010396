# Route A: Four-Dataset Consolidated Report

## Project
Finding Donors for CharityML Using Supervised Learning:
A Tabular Self-Attention Residual Network (Tab-ResNet-Attn) Approach

## Research objective
Evaluate the performance of a tabular self-attention residual neural
network against classical machine learning models on multiple tabular
classification datasets.

## Datasets
1. Adult Census Income
2. Bank Marketing
3. Default of Credit Card Clients
4. Covertype

## Results

### Adult Census Income
Best reported classical model:
XGBoost, ROC-AUC 0.928278 ± 0.002141.

Tab-ResNet-Attn:
ROC-AUC 0.909446 ± 0.002351.
F0.5 0.711897 ± 0.004986.

The proposed model outperformed the reported MLP result but did not
outperform the strongest tree-based models.

### Bank Marketing
Best reported random-split model:
HistGradientBoosting, ROC-AUC 0.8044.

Tab-ResNet-Attn:
ROC-AUC 0.7935.
F0.5 0.4841.

Temporal split experiments showed substantial shifts in positive-class
prevalence and model performance. Temporal and random-split results
must be reported separately.

### Default of Credit Card Clients
Best reported model:
CatBoost, ROC-AUC 0.782824 ± 0.009955.

Tab-ResNet-Attn:
ROC-AUC 0.7065 ± 0.0141.

The proposed model did not outperform the strongest classical
tree-based baselines on this dataset.

### Covertype
Three-seed compute-conscious benchmark:
20,000 training rows, 30,000 test rows.

Random Forest:
Accuracy 0.8267.
Macro-F1 0.7022.
Macro ROC-AUC 0.9735.

XGBoost:
Accuracy 0.7963.
Macro-F1 0.7089.
Macro ROC-AUC 0.9671.

MLP:
Accuracy 0.7221.
Macro-F1 0.4125.
Macro ROC-AUC 0.9264.

Tab-ResNet-Attn:
Accuracy 0.7145.
Macro-F1 0.4851.
Macro ROC-AUC 0.9302.

Tab-ResNet-Attn improved over the MLP in macro-F1, macro recall,
and macro ROC-AUC but did not outperform Random Forest or XGBoost.

## Overall interpretation
Across the currently reported experiments, the proposed neural
architecture improves over the MLP baseline in several cases but
does not consistently outperform strong classical tree-based models.

This suggests that self-attention and residual processing alone do not
guarantee improved performance on tabular data.

## Limitations
- Hyperparameter tuning may not be equally exhaustive across models.
- Compute-conscious Covertype results use subsets of the dataset.
- Different dataset splits and evaluation protocols must not be treated
  as directly comparable.
- Reported performance is dependent on preprocessing and split design.
- Adult income prediction is not equivalent to direct donor prediction.

## Future work
- Improve hyperparameter tuning consistency.
- Evaluate calibration and decision thresholds.
- Add confidence intervals and statistical comparisons.
- Investigate feature attribution and model explainability.
- Evaluate additional datasets and split protocols.
- Test the model under compute-matched conditions.
