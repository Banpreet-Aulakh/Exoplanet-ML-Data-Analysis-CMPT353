import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Data Loading
def load_data(dir):
    data = []
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            filepath = os.path.join(dir, file)
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
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Feature Engineering
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[column] >= (q1 - 1.5 * iqr)) & (df[column] <= (q3 + 1.5 * iqr))]
    return df

def engineer_features(df):
    df = remove_outliers(df, 'SAP_FLUX')
    df = remove_outliers(df, 'SAP_X')
    df = remove_outliers(df, 'SAP_Y')
    df = remove_outliers(df, 'SAP_BKG')
    df = remove_outliers(df, 'DET_FLUX_ERR')
    df = remove_outliers(df, 'SAP_BKG_ERR')

    features = {
        'mean_sap_flux': df['SAP_FLUX'].mean(),
        'median_sap_flux': df['SAP_FLUX'].median(),
        'std_sap_flux': df['SAP_FLUX'].std(),
        'var_sap_flux': df['SAP_FLUX'].var(),
        'range_sap_flux': df['SAP_FLUX'].max() - df['SAP_FLUX'].min(),
        'iqr_sap_flux': df['SAP_FLUX'].quantile(0.75) - df['SAP_FLUX'].quantile(0.25),
        'skew_sap_flux': df['SAP_FLUX'].skew(),
        'kurt_sap_flux': df['SAP_FLUX'].kurt(),
        '10th_percentile_sap_flux': df['SAP_FLUX'].quantile(0.1),
        '90th_percentile_sap_flux': df['SAP_FLUX'].quantile(0.9),
        'mean_sap_x': df['SAP_X'].mean(),
        'std_sap_x': df['SAP_X'].std(),
        'mean_sap_y': df['SAP_Y'].mean(),
        'std_sap_y': df['SAP_Y'].std(),
        'mean_sap_bkg': df['SAP_BKG'].mean(),
        'std_sap_bkg': df['SAP_BKG'].std(),
        'mean_det_flux_err': df['DET_FLUX_ERR'].mean(),
        'std_det_flux_err': df['DET_FLUX_ERR'].std(),
        'mean_sap_bkg_err': df['SAP_BKG_ERR'].mean(),
        'std_sap_bkg_err': df['SAP_BKG_ERR'].std()
    }
    
    return pd.DataFrame([features])

def engineer_features_from_files(dir, target_label):
    features_list = []
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            filepath = os.path.join(dir, file)
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
            features = engineer_features(df)
            features['target'] = target_label
            features_list.append(features)
    return pd.concat(features_list, ignore_index=True).dropna().reset_index(drop=True)

# Model Evaluation
def evaluate_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Raw Model Training and Evaluation
def train_raw_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    evaluate_model(y_test, y_pred)
    return rf

# Engineered Model Training and Evaluation
def train_engineered_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    evaluate_model(y_test, y_pred)
    return rf

# Hyperparameter Tuning for Engineered Model
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 150, 180, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters: ", grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    return best_grid

# Pipeline for Feature Selection and Model Training
def pipeline_feature_selection(X_train, y_train, X_test, y_test):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('classification', RandomForestClassifier(n_estimators=180, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    return pipeline

# Main function
if __name__ == "__main__":
    confirmed_lc_path = "data/confirmed_lightcurves"
    false_lc_path = "data/false_lightcurves"
    
    # Load and preprocess data
    print("Loading data...")
    confirmed_lc = load_data(confirmed_lc_path)
    false_lc = load_data(false_lc_path)

    # Combine the data
    print("Combining data...")
    confirmed_lc['target'] = 1
    false_lc['target'] = 0
    data = pd.concat([confirmed_lc, false_lc], ignore_index=True)

    # Raw data training
    print("Training raw model...")
    X_raw = data.drop('target', axis=1)
    y_raw = data['target']
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    raw_model = train_raw_model(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
    
    # Feature Engineering from files
    print("Feature engineering...")
    confirmed_features = engineer_features_from_files(confirmed_lc_path, 1)
    false_features = engineer_features_from_files(false_lc_path, 0)
    feature_data = pd.concat([confirmed_features, false_features], ignore_index=True)
    
    print("Feature Data Shape:", feature_data.shape)
    print("Feature Data Head:\n", feature_data.head())
    
    X_eng = feature_data.drop('target', axis=1)
    y_eng = feature_data['target']
    
    # Split data into training and test sets
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y_eng, test_size=0.2, random_state=42)
    
    print("Training data size:", X_train_eng.shape, y_train_eng.shape)
    print("Test data size:", X_test_eng.shape, y_test_eng.shape)
    
    # Train and evaluate model on engineered features
    print("Training engineered model...")
    engineered_model = train_engineered_model(X_train_eng, y_train_eng, X_test_eng, y_test_eng)
    
    # Hyperparameter tuning on engineered features
    print("Hyperparameter tuning...")
    best_engineered_model = hyperparameter_tuning(X_train_eng, y_train_eng)

    # Save the best engineered model
    if not os.path.exists('models'):
        print("Creating models directory...")
        os.makedirs('models')
    
    print("Saving models...")
    joblib.dump(best_engineered_model, 'models/feature_engineered_rf_model.pkl')
    joblib.dump(raw_model, 'models/random_forest_lightcurve_classifier.pkl')
    # Feature selection pipeline on engineered features not completed
    # pipeline_model = pipeline_feature_selection(X_train_eng, y_train_eng, X_test_eng, y_test_eng)
    
    # Save the pipeline model not completed
    # joblib.dump(pipeline_model, 'models/pipeline_rf_model.pkl')
