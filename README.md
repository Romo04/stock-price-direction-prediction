# Stock Price Direction Prediction

A machine learning project that predicts the next-day direction (Up/Down) of stock prices using technical indicators and ensemble methods.

## Overview

This project implements and compares two machine learning approaches for stock price direction prediction:
- **Random Forest Classifier** (51.69% accuracy on AAPL test set)
- **XGBoost Classifier** (50.95% accuracy on AAPL test set)

The accuracy does not seem that good, but from all i know this is decent when it comes to stock market predictions.

## Dataset

- **Source**: [Kaggle - Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- **Stock**: Apple Inc. (AAPL) (Used for model training)
- **Period**: September 7, 1984 to November 10, 2017
- **Samples**: 8,364 trading days (8,165 after feature engineering)
- **Features**: OHLCV (Open, High, Low, Close, Volume) data

## Features

The model uses 18 technical indicators engineered from raw price data, they are divided into these categories:

**Price-based:** Daily Return, Price Range, Price Change
**Trend:** Distance from SMA (5, 10, 20, 50 days), 50/200 Golden Cross
**Momentum:** RSI, MACD, MACD Signal, MACD Histogram
**Volatility:** Bollinger Band Position
**Volume:** Volume Ratio vs 20-day average
**Lagged:** Returns from 1, 2, 3, and 5 days ago

## Project Structure
stock-price-direction-prediction/
    README.md               
    feature_list.json       # List of features used
    features.py             # Feature engineering function
    pedrict.py              # Prediction/inference script
    train.py                # Model training script
    rf_stock_model.pkl      # Saved Random Forest model
    xgb_stock_model.pkl     # Saved XGBoost model
    model_comparison.png    # Training results visualization
    
## Installation

## Clone the repository
git clone

## Install dependencies
pip install pandas numpy matplotlib scikit-learn xgboost kagglehub

# Download dataset (automatic on first run)
# Or manually from: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

## Usage

## Training Models
**Train the models by running:**
python train.py

## This will:
    Load and preprocess the data
    Engineer 18 technical features
    Train both models
    Evaluate on test set (2012-2017)
    Save models and visualizations

## Making Predictions
**You can predict on new data (a different dataset) as long as it has the same structure as the stocks that the kaggle datasets use:**
python predict.py amzn.us.txt rf

## Evaluate with accuracy:
python predict.py msft.us.txt rf eval

## Or other models (using your own model shouldn't be hard at all to add):
python predict.py amzn.us.txt xgb eval

## Results

## Model performance (accuracy):
Random Forest: 51.69%
XGBoost: 50.95%
Baseline (Random): 50%

## Findings
1) **Feature Importance:** The most predictive features were:
    Return_Lag_3 (7.24%)
    Price_Change (6.78%)
    Return_Lag_5 (6.63%)
This suggests short-term momentum effects exist in AAPL stock, at least i think.

2) **Cross-Stock Generalization:** When tested on other stocks (A.US, MSFT), accuracy drops to ~51%, indicating the model learns some stock-specific patterns.
3) **Market Regimes:** Rolling 30-day accuracy varies significantly (30% to 70%), suggesting the model performs better in certain market conditions.

## Methodology

## Data Preprocessing
    Train/Test Split: Time-based split at January 1, 2012 (roughly a 80/20 split)
    Feature Scaling: Not required for tree-based models
    Missing Values: Dropped 199 rows (warm-up period for indicators)

## Model Configuration
Random Forest:
    200 estimators
    Max depth: 10
    Min samples split: 30
    Min samples leaf: 15

XGBoost:
    200 estimators
    Max depth: 4 (shallower to prevent overfitting)
    Learning rate: 0.1
    Subsample: 0.8

## Limitations
1) **Survivorship Bias:** Dataset only includes stocks that survived until 2017
2) **No Transaction Costs**: Real trading includes fees (roughly 0.1% per trade) that would reduce profitability
3) **Market Efficiency**: 51% accuracy may not be profitable after costs
4) **Temporal Validity**: Model trained on 1984-2014 data.

## Future Improvements
I will probably not touch this again, but there's a bunch of stuff that could be improved like using LSTM, optimizations different types of validation, etc, etc.

## References
Dataset: Kaggle - borismarjanovic/price-volume-data-for-all-us-stocks-etfs
Google my beloved.

## Note/Disclaimer
This project is for educational and research purposes only. It does not constitute financial advice. Stock market prediction is inherently uncertain, and past performance does not guarantee future results. Do not use this model for actual trading without extensive validation and risk management (Although, it would be funny).
