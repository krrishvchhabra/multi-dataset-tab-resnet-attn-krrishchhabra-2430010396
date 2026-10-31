import numpy as np
from sklearn.preprocessing import StandardScaler


class NumericPreprocessor:
    """
    Standardize numeric features.

    Fit only on training data to avoid data leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X_train):
        self.scaler.fit(X_train)
        return self

    def transform(self, X):
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)


def encode_labels(y):
    """
    Convert labels into consecutive integer class IDs.
    """
    classes, encoded = np.unique(y, return_inverse=True)
    return encoded.astype(np.int64), classes
