"""
Default of Credit Card Clients benchmark.

Expected data file:
data/default_of_credit_card_clients.xls

Target:
default payment next month
"""

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


DATA_PATH = "data/default_of_credit_card_clients.xls"
OUTPUT_DIR = "results/credit_card"
SEEDS = [42, 52, 62]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_PATH, header=1)

    target_candidates = [
        "default payment next month",
        "default.payment.next.month",
    ]

    target = next(
        (
            c for c in target_candidates
            if c in df.columns
        ),
        None,
    )

    if target is None:
        raise ValueError(
            "Could not identify the target column."
        )

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Remove ID column if present.
    for col in X.columns:
        if str(col).strip().lower() == "id":
            X = X.drop(columns=[col])

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
            "LogisticRegression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=seed,
                n_jobs=4,
            ),
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            rows.append({
                "dataset": "Credit Card Default",
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
                "average_precision": (
                    average_precision_score(
                        y_test, prob
                    )
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
