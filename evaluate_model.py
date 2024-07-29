import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the model
model_path = "models/random_forest_lightcurve_classifier.pkl"
model = joblib.load(model_path)

# Paths to validation CSV files
validation_confirmed_lc_path = "data/validation_lightcurves/confirmed_validation"
validation_false_lc_path = "data/validation_lightcurves/false_validation"


def mean_prediction(predictions, threshold=0.3):
    return np.mean(predictions) > threshold


def aggregate_predictions_mean(filepath, model, threshold=0.3):
    df = pd.read_csv(filepath)
    df = df.iloc[:, :13]
    df = df.drop(["QUALITY", "ORBITID"], axis=1)
    df = df.rename(
        columns={
            "KSPSAP_FLUX": "DET_FLUX",
            "KSPSAP_FLUX_ERR": "DET_FLUX_ERR",
            "KSPSAP_FLUX_SML": "DET_FLUX_SML",
            "KSPSAP_FLUX_LAG": "DET_FLUX_LAG",
        }
    )
    df = df.dropna()
    df = df[~df.isin([np.inf]).any(axis=1)]
    probabilities = model.predict_proba(df)[:, 1]
    final_prediction = mean_prediction(probabilities, threshold)
    return final_prediction


def evaluate_model_mean(
    validation_confirmed_path, validation_false_path, model, threshold=0.3
):
    true_labels = []
    predicted_labels = []

    for file in os.listdir(validation_confirmed_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_confirmed_path, file)
            true_labels.append(1)
            predicted_labels.append(
                aggregate_predictions_mean(filepath, model, threshold)
            )

    for file in os.listdir(validation_false_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_false_path, file)
            true_labels.append(0)
            predicted_labels.append(
                aggregate_predictions_mean(filepath, model, threshold)
            )

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


evaluate_model_mean(
    validation_confirmed_lc_path, validation_false_lc_path, model, threshold=0.3
)
