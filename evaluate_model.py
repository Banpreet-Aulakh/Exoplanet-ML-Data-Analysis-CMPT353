import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the model
raw_data_model_path = "models/random_forest_lightcurve_classifier.pkl"
raw_data_model = joblib.load(raw_data_model_path)

feature_engineering_model_path = "models/feature_engineered_rf_model.pkl"
feature_engineering_model = joblib.load(feature_engineering_model_path)


# Paths to validation CSV files
validation_confirmed_lc_path = "data/validation_lightcurves/confirmed_validation"
validation_false_lc_path = "data/validation_lightcurves/false_validation"


def mean_prediction(predictions, threshold=0.5):
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

    print(f"Accuracy for Raw Data Model: {accuracy}")
    print(f"Precision for Raw Data Model: {precision}")
    print(f"Recall for Raw Data Model: {recall}")
    print(f"F1 Score for Raw Data Model: {f1}")


def evaluate_model_engineered_features(
    validation_confirmed_path, validation_false_path, model
):
    def get_csv_data(file_path):
        df = pd.read_csv(file_path)
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
        return df

    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def engineer_features(df):

        df = remove_outliers(df, "SAP_FLUX")
        df = remove_outliers(df, "SAP_X")
        df = remove_outliers(df, "SAP_Y")
        df = remove_outliers(df, "SAP_BKG")
        df = remove_outliers(df, "DET_FLUX_ERR")
        df = remove_outliers(df, "SAP_BKG_ERR")

        features = {}

        features["mean_sap_flux"] = df["SAP_FLUX"].mean()
        features["median_sap_flux"] = df["SAP_FLUX"].median()

        features["std_sap_flux"] = df["SAP_FLUX"].std()
        features["var_sap_flux"] = df["SAP_FLUX"].var()
        features["range_sap_flux"] = df["SAP_FLUX"].max() - df["SAP_FLUX"].min()
        features["iqr_sap_flux"] = df["SAP_FLUX"].quantile(0.75) - df[
            "SAP_FLUX"
        ].quantile(0.25)

        features["skew_sap_flux"] = df["SAP_FLUX"].skew()
        features["kurt_sap_flux"] = df["SAP_FLUX"].kurt()

        features["10th_percentile_sap_flux"] = df["SAP_FLUX"].quantile(0.1)
        features["90th_percentile_sap_flux"] = df["SAP_FLUX"].quantile(0.9)

        features["mean_sap_x"] = df["SAP_X"].mean()
        features["std_sap_x"] = df["SAP_X"].std()
        features["mean_sap_y"] = df["SAP_Y"].mean()
        features["std_sap_y"] = df["SAP_Y"].std()

        features["mean_sap_bkg"] = df["SAP_BKG"].mean()
        features["std_sap_bkg"] = df["SAP_BKG"].std()

        features["mean_det_flux_err"] = df["DET_FLUX_ERR"].mean()
        features["std_det_flux_err"] = df["DET_FLUX_ERR"].std()
        features["mean_sap_bkg_err"] = df["SAP_BKG_ERR"].mean()
        features["std_sap_bkg_err"] = df["SAP_BKG_ERR"].std()

        return features

    def aggregate_predictions_engineered(filemath, model):
        df = get_csv_data(filepath)
        features = engineer_features(df)
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        return prediction[0]

    true_labels = []
    predicted_labels = []

    for file in os.listdir(validation_confirmed_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_confirmed_path, file)
            true_labels.append(1)
            predicted_labels.append(aggregate_predictions_engineered(filepath, model))

    for file in os.listdir(validation_false_path):
        if file.endswith(".csv"):
            filepath = os.path.join(validation_false_path, file)
            true_labels.append(0)
            predicted_labels.append(aggregate_predictions_engineered(filepath, model))

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Accuracy for Engineered Feature Model: {accuracy}")
    print(f"Precision for Engineered Feature Model: {precision}")
    print(f"Recall for Engineered Feature Model: {recall}")
    print(f"F1 Score for Engineered Feature Model: {f1}")


evaluate_model_engineered_features(
    validation_confirmed_lc_path, validation_false_lc_path, feature_engineering_model
)

evaluate_model_mean(
    validation_confirmed_lc_path,
    validation_false_lc_path,
    raw_data_model,
    threshold=0.3,
)
