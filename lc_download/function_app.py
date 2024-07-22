import logging
import azure.functions as func
import os
import pandas as pd
import io
import lightkurve as lk
from lightkurve.search import search_lightcurve
from azure.storage.blob import BlobServiceClient
import warnings
from astropy.units import UnitsWarning

app = func.FunctionApp()

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UnitsWarning)
warnings.filterwarnings('ignore', message='.*cadences will be ignored due to the quality mask.*')

class IgnoreCadenceWarningFilter(logging.Filter):
    def filter(self, record):
        return 'cadences will be ignored due to the quality mask' not in record.getMessage()

# Apply custom warning filter to logger
logging.getLogger().addFilter(IgnoreCadenceWarningFilter())

# Suppress warnings from lightkurve
logging.getLogger('lightkurve').setLevel(logging.ERROR)

# Azure Blob Storage configuration EDIT THIS TO YOUR CONTAINER NAME
container_name = "lightcurves"
connection_string = os.environ.get("AzureWebJobsStorage", "")
if not connection_string:
    logging.error("AzureWebJobsStorage environment variable is not set.")
    raise ValueError("AzureWebJobsStorage environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Read CSV files from Blob Storage
def read_csv_from_blob(blob_client):
    try:
        blob_data = blob_client.download_blob().readall()
        return pd.read_csv(io.StringIO(blob_data.decode()))
    except Exception as e:
        logging.error(f"Failed to read CSV from blob: {e}")
        raise

# Function to process light curves and upload
def download_and_upload_lightcurve(tic_id, label):
    try:
        logging.info(f"Processing TIC ID: {tic_id}")
        lcf_collection = search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC").download_all()
        if not lcf_collection:
            logging.warning(f"No light curve found for TIC {tic_id}")
            return
        
        stitched_lc = lcf_collection.stitch()
        stitched_lc = stitched_lc.remove_nans().remove_outliers()
        df = stitched_lc.to_pandas()
        csv_data = df.to_csv(index=False)
        blob_name = f"{label}/{tic_id}.csv"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(csv_data.encode(), overwrite=True)
        logging.info(f"Uploaded CSV for TIC {tic_id} to {blob_name}")
    except Exception as e:
        logging.error(f"Error processing TIC {tic_id}: {e}")

@app.route(route="LightcurveDownloader", auth_level=func.AuthLevel.FUNCTION)
def LightcurveDownloader(req: func.HttpRequest) -> func.HttpResponse:
    try:
        confirmed_blob_client = container_client.get_blob_client("confirmed_planets.csv")
        false_positives_blob_client = container_client.get_blob_client("false_positives.csv")

        confirmed_planets = read_csv_from_blob(confirmed_blob_client)
        false_positives = read_csv_from_blob(false_positives_blob_client)

        confirmed_planet_tic_ids = confirmed_planets["TIC ID"].values
        false_positive_tic_ids = false_positives["TIC ID"].values

        # Process all TIC IDs
        for tic_id in confirmed_planet_tic_ids:
            download_and_upload_lightcurve(tic_id, "confirmed_exoplanets")

        for tic_id in false_positive_tic_ids:
            download_and_upload_lightcurve(tic_id, "false_positives")

        return func.HttpResponse("Processing complete.")
    except Exception as e:
        logging.error(f"Function execution failed: {e}")
        return func.HttpResponse(f"Function execution failed: {e}", status_code=500)

# Apply custom warning filter to logger
logging.getLogger().addFilter(IgnoreCadenceWarningFilter())