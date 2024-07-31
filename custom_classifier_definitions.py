import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

class MajorityVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        # No training needed for this model
        return self

    def predict(self, X):
        processed_X = self._preprocess(X)
        raw_predictions = self.base_classifier.predict(processed_X)
        # return array because it's good for sklearn compatibility
        return np.array([self.majority_vote(raw_predictions)])
    
    def majority_vote(self, predictions):
        return 1 if sum(predictions) > len(predictions) / 2 else 0
    
    def _preprocess(self, filepath):
        # check if being given file or dataframe
        if isinstance(filepath, str):
            df = pd.read_csv(filepath)
        else:
            df = filepath
        
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
    
class FeatureEngineeringClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        # No training needed for this model
        return self

    def predict(self, X):
        processed_X = self._preprocess(X)
        return self.base_classifier.predict(processed_X)
    
    def _preprocess(self, filepath):
        if isinstance(filepath, str):
            df = pd.read_csv(filepath)
        else:
            df = filepath
        
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
        features = self._engineer_features(df)
        return features

    def _engineer_features(self, df):
        df = self._remove_outliers(df, 'SAP_FLUX')
        df = self._remove_outliers(df, 'SAP_X')
        df = self._remove_outliers(df, 'SAP_Y')
        df = self._remove_outliers(df, 'SAP_BKG')
        df = self._remove_outliers(df, 'DET_FLUX_ERR')
        df = self._remove_outliers(df, 'SAP_BKG_ERR')

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

    def _remove_outliers(self, df, column):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[column] >= (q1 - 1.5 * iqr)) & (df[column] <= (q3 + 1.5 * iqr))]
        return df

