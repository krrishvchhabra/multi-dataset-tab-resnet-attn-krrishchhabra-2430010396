"""
Bank Marketing benchmark.

Expected data file:
data/bank-additional-full.csv

Target:
y

The duration feature is excluded to avoid outcome leakage.
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


DATA_PATH = "data/bank-additional-full.csv"
OUTPUT_DIR = "results/bank_marketing"
SEEDS = [42, 52, 62]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
    )

    y = (
        df["y"]
        .astype(str)
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    X = df.drop(columns=["y"])

    # Exclude duration from a realistic predictive setting.
    if "duration" in X.columns:
        X = X.drop(columns=["duration"])

    numeric = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="median"
                )),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                )),
            ]),
            categorical,
        ),
    ])

    rows = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=seed,
                stratify=y,
            )
        )

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
            ),
            "HistGradientBoosting": (
                HistGradientBoostingClassifier(
                    random_state=seed,
                )
            ),
        }

        for name, model in models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            pipeline.fit(X_train, y_train)

            pred = pipeline.predict(X_test)
            prob = pipeline.predict_proba(X_test)[:, 1]

            rows.append({
                "dataset": "Bank Marketing",
                "model": name,
                "seed": seed,
                "accuracy": accuracy_score(
                    y_test, pred
                ),
                "f1": f1_score(
                    y_test, pred
                ),
                "roc_auc": roc_auc_score(
                    y_test, prob
                ),
            })

    results = pd.DataFrame(rows)

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "benchmark_runs.csv",
        ),
        index=False,
    )

    summary = (
        results.groupby("model")
        .agg(["mean", "std"])
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "benchmark_summary.csv",
        )
    )

    print(summary)


if __name__ == "__main__":
    main()
