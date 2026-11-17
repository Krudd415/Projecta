"""Train a predictive model and save it to models/."""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_model(data_path="data/dataset.csv", target_col="target"):
    if not os.path.exists(data_path):
        print(f"File '{data_path}' not found. Creating a quick dummy dataset to demonstrate...")
        # Create a simple sample dataset if none exists yet
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target":   [0,  0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample data saved to {data_path}")
    else:
        df = pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
