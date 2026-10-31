# Research Methodology

## Objective
Evaluate whether Tab-ResNet-Attn provides competitive results
compared with classical machine learning models on tabular datasets.

## Experimental Design
For each dataset:
1. Load and validate the dataset.
2. Identify the target variable.
3. Separate features and target.
4. Create train, validation, and test partitions.
5. Fit preprocessing on training data only.
6. Train each model using the same partition.
7. Evaluate on held-out test data.
8. Repeat with multiple random seeds where possible.
9. Report mean and standard deviation.

## Metrics
For binary classification:
- Accuracy
- Precision
- Recall
- F1
- F0.5
- ROC-AUC
- Average Precision

For multiclass classification:
- Accuracy
- Macro-F1
- Macro Recall
- Macro ROC-AUC when supported

## Leakage Prevention
Do not fit scalers, encoders, or imputers on the test partition.

For Bank Marketing, the duration feature should be excluded because
it is not known before a marketing call and may leak outcome information.

## Reporting
Clearly distinguish:
- Random split experiments
- Temporal split experiments
- Preliminary exploratory runs
- Matched benchmark runs

Do not compare results from different split protocols as though
they were directly equivalent.
