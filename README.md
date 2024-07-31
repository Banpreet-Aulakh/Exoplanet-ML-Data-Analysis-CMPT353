
# Exoplanet ML Data Analysis using Light Curves

This project aims to analyze telescope light curve data to identify potential exoplanets. We will analyze and explore our dataset using transit photometry techniques and then use machine learning to detect and classify the light curves and periodic dips in a star's brightness caused by a planet passing in front of it. Data from NASA's Transiting Exoplanet Survey Satellite (TESS) will be used to train a machine-learning model to sort through light curve data and classify whether a planet is orbiting a star.

## Table of Contents
- [Project Description](#project-description)
- [Data](#data)
- [Methods](#methods)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Team](#team)
- [References and Acknowledgements](#references-and-acknowledgements)

## Project Description
This project focuses on creating a model to classify false positive planets and confirmed planets from TESS space telescope light curves. The primary goal is to develop two classifiers: an LSTM-based classifier (work done by Jordan Clough) and a stats-based Random Forest classifier (work done by Banpreet Aulakh).

## Data
- The exoplanet candidate and confirmed exoplanet data was gathered via the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/index.html).
- Individual TIC Data to process for the model was gathered from the [MAST Archive](https://archive.stsci.edu/) using their TESS data products.
- Two main CSV files were created: `confirmed_planets.csv` and `false_positives.csv`.
- The TIC IDs from the CSV files were used to download the lightcurve data from MAST using AstroQuery and saved as `{TIC_ID}_lightcurves.csv` in their respective directories

## Methods
### Data Collection and Processing
- **lightcurve_downloader.ipynb**: Downloads light curves in batches of 150 using astroquery and saves them to folders, ensuring data balance between false positives and confirmed planets.
- **data_exploration.ipynb**: Processes and extracts TICs, dispositions, and TOIs from the NASA TESS project candidates CSV from the exoplanet archive.
- **download_and_test_by_TIC.py**: Downloads a TIC using astroquery and tests it using the mean aggregate model.
- **download_random_confirmed_TICs.py**: Downloads additional confirmed planets' TICs and saves them to the validation data folder.

### Model Training
- **lightcurve_randforest.ipynb**: Processes the data, trains a Random Forest model using raw light curve data, and saves the model using Joblib.
- **lightcurve_randforest_feature_engineering.ipynb**: Similar to the previous notebook but includes feature engineering to improve model performance.

### Model Evaluation
- **evaluate_model.py**: Evaluates both models using a validation dataset consisting of 60 random files from the false and confirmed light curve directories. The validation files must be downloaded and separated manually.

## Results
The performance of the models is summarized below:
- **Stats-based Classifier (First Model)**:
  - Initial high accuracy on row classification: 99.7%
  - Final performance: 
    - Accuracy: 0.75
    - Precision: 0.76
    - Recall: 0.73
    - F1 Score: 0.75
- **Stats-based Classifier with Feature Engineering (Second Model)**:
  - Final performance after feature engineering and optimization:
    - Accuracy: 0.62
    - Precision: 0.61
    - Recall: 0.63
    - F1 Score: 0.62

## Usage
### Training the Models
1. Download the data folder and replace it with the data folder from [here](https://drive.google.com/file/d/1aaVGC6HPTHWbxfuUbTZPnbUmz3ApxLam/view?usp=drive_link) in the cloned directory
   *Alternatively you can use the lightcurve downloader.ipynb and manually download the files in individual batches to suit your needs*
2. Train the Random Forest models by running `train_statsmodels.py`.

### Evaluating the Models
1. Ensure you have the data in the correct directories. (It should be if you ran `train_statsmodels.py` above)
2. Run `evaluate_model.py` to evaluate both stats-based classifier models using the validation dataset.
   *You can also test it on the other validation lightcurves, but those only contain confirmed data, no false positives*

## Dependencies
- lightkurve
- AstroQuery
- Numpy
- Pandas
- Joblib
- shutil
- zip
- sklearn
- AstroPy

## Team
- Banpreet Aulakh
- Jordan Clough

## References and Acknowledgements
This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program.
