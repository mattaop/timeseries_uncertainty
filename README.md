# Forecasting Uncertainty in Neural Networks With Dropout
Release version 1.0: 29.01.2020

### Requirements
Dependencies necessary to run the code:
```bash
numpy
pandas
sklearn
tensorflow
keras
pydataset
pyyaml
statsmodels
tqdm
keras-rectified-adam
keras-lookahead
seaborn
pmdarima
```

Installation can be done manually or by creating an environment, activate it and install requirements:
```bash
conda create -n [name_of_environment] python=3.6
conda activate [name_of_environment]
pip install -r requirements.txt
```

### Folder structure
This folder structure required to run the project without any modifications:

```bash
|---- data  # Folder for data sets
|     |----- raw  # Datasets requiered to train the models
|            |---- AirPassengers.csv
|            |---- avocado.csv
|            |---- OsloTemperature.csv
|---- src
|     |---- modeling  # Main modules to run
|           |---- sliding_window_airpassengers.py  # Module for AirPassengers data set
|           |---- sliding_window_oslo_temperature.py  # Module for Oslo Temperature data set
|           |---- sliding_window_avocado.py  # Module for Avocado Price data set
|     |---- networks  # Neural Network strucutres
|           |---- cnn.py
|           |---- lstm.py
|           |---- rnn.py
|           |---- train_model.py
|     |---- preparation  # Load config and data
|           |---- load_data.py
|           |---- config
|                 |---- airpassengers.yml
|                 |---- avocado.yml
|                 |---- oslo.yml
|                 |---- open_config.py
|     |---- processing  # Preprocess data
|           |---- split_data.py
|     |---- utility  # Compute and plot results
|           |---- compute_coverage.py
|           |---- plot_forecast.py
|---- .gitignore
|---- README.md
|---- requirements.txt
```

### Data sets
In order to run the code, one have to download the data sets, and place it according to the following folder structure and name conventions:
```bash
|---- data  # Folder for data sets
|     |----- raw  # Datasets requiered to train the models
|            |---- AirPassengers.csv
|            |---- avocado.csv
|            |---- OsloTemperature.csv
``` 

The data sets can be found and downloaded here:

AirPassengers data set: https://www.kaggle.com/chirag19/air-passengers

Oslo Temperature data set: https://wiki.math.ntnu.no/lib/exe/fetch.php?tok=5deb8a&media=https%3A%2F%2Fwww.math.ntnu.no%2Femner%2FTMA4285%2F2019h%2Fpdf%2Fdata.xlsx

Avocado Price data set: https://www.kaggle.com/neuromusic/avocado-prices

### Run code
The following files have to be ran in order to do the experiments:
```bash
src.modeling.sliding_window_airpassengers.py
src.modeling.sliding_window_oslo_temperature.py
src.modeling.sliding_window_avocado.oy
```
At default one network type (RNN) and baseline models are ran, but this can be changed in the main function at the bottom of each fil. Changing networks and other configurations can be done in config files:
```bash
src.preparation.config.airpassengers.yml
src.preparation.config.oslo.yml
src.preparation.config.avocado.yml
```
