"""
Model training functions.
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def train_svm(X_train, y_train, kernel='rbf', C=1.0):
    """
    Scales features (StandardScaler) and trains an SVM classifier.
    Returns (model, scaler) -- both are needed at prediction time,
    since new data must go through the same scaler before predict().
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel=kernel, C=C, probability=True, random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    return model, scaler