import os
import time
import pandas as pd
from astroquery.mast import Observations
from astropy.io import fits
from astroquery.exceptions import RemoteServiceError
import gdown
import numpy as np

# Link to the model

confirmed_TIC_dir = "data/validation_lightcurves/confirmed_validation"
confirmed_TIC_list_dir = "data/confirmed_planets.csv"

def process_and_save_TICs(tic_id_list, save_dir):
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

def main():

    # randomly download 10 confirmed TICs from the confirmed_TIC_list_dir after the first 2000 (those are in training set)
    confirmed_TIC_list = pd.read_csv(confirmed_TIC_list_dir)
    confirmed_TIC_list = confirmed_TIC_list.iloc[2000:]
    confirmed_TIC_list = confirmed_TIC_list.sample(30)

    tic_id_list = confirmed_TIC_list["TIC ID"].values

    process_and_save_TICs(tic_id_list, confirmed_TIC_dir)
    




if __name__ == "__main__":
    main()

