# Finance ML Project
Author: Ezra Schiff <ezra.schiff@gmail.com>

## Overview
This project aims to predict which tickers are going to perform best
n quarters (default 4 quarters) out based on quarterly financial data
scraped from Yahoo Finance's API. 

## Requirements
* Docker
* Python 3.7+
* Poetry

## Background
Data used by this project is a combination of data from StockPup (now defunct)
and Yahoo Finance. I've been running a script quarterly to pull in new quarterly
data and make new predictions.

## Functionality
This project builds a Docker image and executes functionality inside
of the container. We use poetry for package management.

### Training
Run `./gradlew train` to train a new model using existing data and hyperparams. This will
also spit out stocks predicted to appreciate the most.

#### Hyperparameters
Hyperparameters can be passed in using `./gradlew train -Pargs="HYPERPARAM=value OTHER=value"`

Options include (default in parenthesis):
- N_QUARTERS_OUT_TO_PREDICT - num quarters out to predict performance (4)
- NUM_QUARTERS_FOR_TRAINING - num quarters back to train on (12)
- N_STOCKS_TO_BUY - num stocks to buy per quarter (7)
- LEARNING_RATE - learning rate (0.1)
- EXTRACT_OUTLIERS - whether to extract outliers (False)
- TEST_SIZE - test size (0.2)
- ALLOCATION_FN - allocation function of stocks purchased ('equal' or 'fibonacci' - 'equal' is default)
- ADJUST_FOR_CURRENT_PRICE - whether to adjust the prediction by the current price 
(eg: if the price is predicted to appreciate by 10%, but the current price is already 10% greater 
than it was last quarter then the predicted appreciation will be 0)

### Testing
Run `./gradlew test` to run unit tests

### Updating the Database
#### Update Quarterly Data
Run `./gradlew update_db` to pull new data from Yahoo Finance and update the 
quarterly database.

Warning: This can take a LONG time to run as it makes an api query per ticker!

#### Add new Tickers
Run `./gradlew update_db -Pargs="NEW_TICKER_1 NEW_TICKER_2 ..."`
