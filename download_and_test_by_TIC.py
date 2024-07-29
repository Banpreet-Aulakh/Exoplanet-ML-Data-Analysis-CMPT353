import os
import time
import pandas as pd
from astroquery.mast import Observations
from astropy.io import fits
from astroquery.exceptions import RemoteServiceError
import joblib
import gdown
import numpy as np

# Link to the model
link_to_model = "https://drive.google.com/file/d/1392bI0iI4CIff_0FBrpIY1JVzquyO6xW/view?usp=drive_link"
csv_dir = "random_TIC_for_testing"
model_dir = "models"


# Download the model from the link
def download_model_from_google_drive(link, save_dir, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output = os.path.join(save_dir, model_name)
    gdown.download(link, output, quiet=False)
    return output



def process_and_save(tic_id_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    retry = 3
    for tries in range(retry):
        try:
            print("Processing and Saving Data Start...")
            for tic_id in tic_id_list:
                try:
                    print(f"Creating Observation Table for TIC ID: {tic_id}")
                    obs_table = Observations.query_criteria(
                        provenance_name="QLP", target_name=tic_id
                    )
                    data_products = Observations.get_product_list(obs_table)
                    print("Attempting to download...")
                    download_lc = Observations.download_products(data_products)
                    for product in download_lc["Local Path"]:
                        if product is not None:
                            try:
                                with fits.open(product) as hdulist:
                                    lc_data = hdulist[1].data
                                    lc_df = pd.DataFrame(lc_data)
                                    tic_id = hdulist[0].header["TICID"]
                                    lc_df.to_csv(
                                        os.path.join(
                                            save_dir, f"{tic_id}_lightcurve.csv"
                                        ),
                                        index=False,
                                    )
                            except (OSError, TypeError) as e:
                                print(f"Error processing file {product}: {e}")
                except RemoteServiceError as e:
                    print(f"Error: {e}. Retrying in 3 seconds...")
                    time.sleep(3)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
        except Exception as e:
            print(f"Error fetching data on attempt {tries + 1}/{retry}: {e}")
            time.sleep(5)  # Wait before retrying
            continue
        break  # break if successful



def prepare_data(save_dir):
    print("Preparing Data for Model Testing...")
    data = []
    for file in os.listdir(save_dir):
        if file.endswith(".csv"):
            filepath = os.path.join(save_dir, file)
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
    return data

def majority_vote(predictions):
    return np.bincount(predictions).argmax()

def mean_prediction(predictions, threshold=0.5):
    return np.mean(predictions) > threshold


def main():
    # Update the TIC ID list with the TIC IDs you want to test
    tic_id_list = ["391115637"]

    # check if model exists in the directory and download if not
    if not os.path.exists(model_dir):
        print("Downloading Model...")
        model_path = download_model_from_google_drive(link_to_model, model_dir, "random_forest_lightcurve_classifier.pkl")
    else:
        model_path = os.path.join(model_dir, "random_forest_lightcurve_classifier.pkl")
        print("Model Already Exists")
    
    if (not os.path.exists(csv_dir)) or (len(tic_id_list) > 0):
        print("Creating Random TIC ID for Testing...")
        process_and_save(tic_id_list, csv_dir)
    
    data = prepare_data(csv_dir)
    print("Data Preparation Complete")

    # Load the model
    model = joblib.load(os.path.join(model_path))

    # Test the model
    print("Testing Model...")
    for i in range(len(data)):
        prediction = model.predict(data[i])
        print("CSV File: ", os.listdir(csv_dir)[i])
        print(f"Prediction for TIC ID {i}: {prediction}")
        print(f"Majority Vote: {majority_vote(prediction)}")
        print(f"Mean Prediction: {mean_prediction(prediction)}")
    print("Model Testing Complete")




if __name__ == "__main__":
    main()

