import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from custom_classifier_definitions import MajorityVotingClassifier, FeatureEngineeringClassifier

# Load the model
raw_data_model_path = "models/random_forest_lightcurve_classifier.pkl"
raw_data_model = joblib.load(raw_data_model_path)

feature_engineering_model_path = "models/feature_engineered_rf_model.pkl"
feature_engineering_model = joblib.load(feature_engineering_model_path)


# Paths to validation CSV files
validation_confirmed_lc_path = "data/validation_lightcurves/confirmed_validation"
validation_false_lc_path = "data/validation_lightcurves/false_validation"


def evaluate_model_engineered_features(
    validation_confirmed_path, validation_false_path, model
):
    true_labels = []
    predicted_labels = []

    for file in os.listdir(validation_confirmed_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_confirmed_path, file)
            true_labels.append(1)
            predicted_labels.append(model.predict(filepath))

    for file in os.listdir(validation_false_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_false_path, file)
            true_labels.append(0)
            predicted_labels.append(model.predict(filepath))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy for Engineered Feature Model: {accuracy}")
    print(f"Precision for Engineered Feature Model: {precision}")
    print(f"Recall for Engineered Feature Model: {recall}")
    print(f"F1 Score for Engineered Feature Model: {f1}")


def evaluate_model_majority(validation_confirmed_path, validation_false_path, model):
    true_labels = []
    predicted_labels = []

    for file in os.listdir(validation_confirmed_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_confirmed_path, file)
            true_labels.append(1)
            predicted_labels.append(model.predict(filepath))

    for file in os.listdir(validation_false_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_false_path, file)
            true_labels.append(0)
            predicted_labels.append(model.predict(filepath))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy for Majority Vote Model: {accuracy}")
    print(f"Precision for Majority Vote Model: {precision}")
    print(f"Recall for Majority Vote Model: {recall}")
    print(f"F1 Score for Majority Vote Model: {f1}")


evaluate_model_majority(
    validation_confirmed_lc_path, validation_false_lc_path, raw_data_model
)


print("*" * 50)
evaluate_model_engineered_features(
    validation_confirmed_lc_path, validation_false_lc_path, feature_engineering_model
)
