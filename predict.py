"""Load the trained model and make predictions."""

import os
import joblib
import pandas as pd


def make_predictions(input_data="data/new_data.csv", model_path="models/model.joblib"):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found! Run 'python train.py' first.")
        return

    model = joblib.load(model_path)

    if not os.path.exists(input_data):
        print(f"Input file '{input_data}' not found. Using sample row for demonstration:")
        df = pd.DataFrame({
            "feature1": [3, 8],
            "feature2": [8, 3]
        })
    else:
        df = pd.read_csv(input_data)

    predictions = model.predict(df)
    df["prediction"] = predictions

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df.drop(columns=["prediction"]))
        if probs.shape[1] == 2:
            df["probability"] = probs[:, 1]

    print("--- Predictions ---")
    print(df)

    output_path = "data/predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    make_predictions()
