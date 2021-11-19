# Finance ML Project
Author: Ezra Schiff <ezra.schiff@gmail.com>

## Overview
This project aims to predict which tickers are going to perform best when held for
n quarters (default 4) based on quarterly financial data gathered from Yahoo Finance's API. 

## Requirements
* [Docker](https://www.docker.com/products/docker-desktop)
* Python 3.7+
* [Poetry](https://python-poetry.org/docs/)
* [Gradle](https://docs.gradle.org/current/userguide/installation.html)
* (Tilt) - still a work in progress. Don't yet have serving infrastructure set up.

## Background
Data used by this project is a combination of data from StockPup (now defunct)
and Yahoo Finance. I've been running a script quarterly to pull in new quarterly
data and make new predictions.

## Functionality
This project builds a Docker image and executes functionality inside
of the container. We use poetry for package management. The current model being trained is
an LGBMRegressor, although I've tried to make the pipeline configurable to allow training with multiple 
different types of models.

### Training
Run `./gradlew train` to train a new model using existing data and hyperparams. This will
also spit out stocks predicted to appreciate the most. A pickled version of the model is saved
to the `/models` directory.

#### Hyperparameters
Hyperparameters can be passed in using `./gradlew train -Pargs="HYPERPARAM=value OTHER=value"`

Options include (default in parenthesis):
- N_QUARTERS_OUT_TO_PREDICT - num quarters out to predict performance (4)
- NUM_QUARTERS_FOR_TRAINING - num quarters back to train on (16)
- N_STOCKS_TO_BUY - num stocks to buy per quarter (7)
- LEARNING_RATE - learning rate (0.1)
- EXTRACT_OUTLIERS - whether to extract outliers (False)
- TEST_SIZE - test size (0.2)
- ALLOCATION_FN - allocation function of stocks purchased ('equal' or 'fibonacci' - 'equal' is default)
- ADJUST_FOR_CURRENT_PRICE - whether to adjust the prediction by the current price 
(eg: if the price is predicted to appreciate by 10%, but the current price is already 10% greater 
than it was last quarter then the predicted appreciation will be 0)

### Updating the Database
#### Update Quarterly Data
Run `./gradlew update_db` to pull new data from Yahoo Finance and update the 
quarterly database.

Warning: This can take a LONG time to run as it makes an api query per ticker!

#### Add new Tickers
Run `./gradlew update_db -Pargs="NEW_TICKER_1 NEW_TICKER_2 ..."`

## Development
### Debugging inside a container
Run `./gradlew debug` will build the image and run a container while also generating a `debug`
executable file that you can run to enter an interactive terminal inside the container.

### Testing
Run `./gradlew test` to run unit tests

### Backtesting & Hyperparameter Tuning
Run `jupyter notebook` and open `finance_ml/notebooks/Backtesting.ipynb`
