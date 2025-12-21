import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

from feature_selection import select_top_features
from preprocessing import scale_train_test

def train():
    # Load dataset
    data = load_breast_cancer(as_frame = True)
    df = data.frame

    X = df.drop('target', axis = 1)
    y = df['target']

    # Feature Selection
    top_features = select_top_features(X, y, k = 10)
    X_reduced = X[top_features]

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size = 0.2, random_state= 42)

    # Scaling
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    # Train model
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train_scaled, y_train)

    # Save artifacts
    joblib.dump(model, "models/reduced_LR_model.pkl")
    joblib.dump(scaler, 'models/reduced_LR_scaler.pkl')
    joblib.dump(top_features, 'models/reduced_LR_columns.pkl')

    print("Training complete. Model, scaler and columns saved.")

if __name__ == "__main__":
    train()