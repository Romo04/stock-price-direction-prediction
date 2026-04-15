"""
STOCK PRICE DIRECTION MODEL PREDICTION
Predicts whether the stock price will go up or down.
Dataset: borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""
import joblib
import json
import pandas as pd
import numpy as np
import sys
import os
import kagglehub
from features import create_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def predict_stock(stock_file, model_name='rf', evaluate=False):
    print(f"=== STOCK PREDICTION - Using {model_name.upper()} Model ===")
    
    # Load model
    if model_name == 'rf':
        model = joblib.load("rf_stock_model.pkl")
    else:
        model = joblib.load("xgb_stock_model.pkl")
    
    with open("feature_list.json", "r") as f:
        feature_cols = json.load(f)
    
    # Load data
    df = pd.read_csv(stock_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    stock_name = os.path.basename(stock_file).replace('.us.txt', '').upper()
    print(f"Loaded: {stock_file}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Feature engineering
    data = create_features(df)
    # There's some sort of warning without .copy()
    data_clean = data.dropna().copy()
    
    # --- Smart Eval Logic ---
    if evaluate:
        # Check if this is the same stock we trained on (AAPL)
        is_same_stock = (stock_name == 'AAPL')
        
        if is_same_stock:
            # Only test on data the model hasn't seen
            # Model was trained on AAPL up to 2012-01-01
            eval_start = "2012-01-01"
            data_clean = data_clean[data_clean.index >= eval_start]
            print(f"\nSame as training stock - testing only on {eval_start} onwards")
        else:
            print(f"\nDifferent stock - testing on all available data")
        
        print(f"Evaluation samples: {len(data_clean)}")
        
        # Check if we have Target column for evaluation
        if 'Target' not in data_clean.columns:
            print("ERROR: Cannot evaluate - no Target column found")
            return None
    
    if len(data_clean) == 0:
        print("ERROR: No data after filtering")
        return None
    
    # Make predictions
    X = data_clean[feature_cols]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Store results
    data_clean['Prediction'] = predictions
    data_clean['Confidence_UP'] = probabilities
    
    # Display last 5 predictions
    print(f"\n--- LAST 5 PREDICTIONS ---")
    print(f"{'Date':<12} {'Price':>8} {'Pred':>6} {'Conf':>6} {'Signal':<12}")
    print("-" * 55)
    
    for idx, row in data_clean.tail(5).iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        price = f"${row['Close']:.2f}"
        pred = "UP" if row['Prediction'] == 1 else "DOWN"
        conf = f"{row['Confidence_UP']:.0%}"
        
        if row['Confidence_UP'] > 0.6:
            signal = "STRONG BUY"
        elif row['Confidence_UP'] < 0.4:
            signal = "STRONG SELL"
        elif row['Prediction'] == 1:
            signal = "WEAK BUY"
        else:
            signal = "WEAK SELL"
        
        print(f"{date_str} {price:>8} {pred:>6} {conf:>6} {signal:<12}")
    
    # Evaluation mode
    if evaluate:
        y_true = data_clean['Target']
        y_pred = predictions
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n--- EVALUATION RESULTS ---")
        print(f"Stock: {stock_name}")
        print(f"Test period: {data_clean.index[0].date()} to {data_clean.index[-1].date()}")
        print(f"Samples: {len(data_clean)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Baseline (random): 50.00%")
        
        if is_same_stock:
            print(f"Note: This is the TEST SET (post-2012 only)")
        else:
            print(f"Note: This is CROSS-STOCK generalization")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Pred Down  Pred Up")
        print(f"Actual Down       {cm[0,0]:4d}     {cm[0,1]:4d}")
        print(f"Actual Up         {cm[1,0]:4d}     {cm[1,1]:4d}")
        
        return accuracy
    
    # Final Prediction
    else:
        print(f"\n--- PREDICTION SUMMARY ---")
        print(f"Total predictions: {len(predictions)}")
        print(f"Bullish: {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
        print(f"Bearish: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
        
        strong = ((probabilities > 0.6) | (probabilities < 0.4)).sum()
        print(f"High confidence: {strong} ({strong/len(predictions):.1%})")
        
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":    
    # Simple command line interface
    # Usage: python predict.py [stock_file] [model] [eval]
    # Example: python predict.py aapl.us.txt rf eval
    path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
    
    stock_file = path + "/Data/Stocks/" + sys.argv[1] if len(sys.argv) > 1 else path + "/Data/Stocks/aapl.us.txt"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "rf"
    evaluate_mode = "eval" in sys.argv
    
    predict_stock(stock_file, model_name, evaluate_mode)