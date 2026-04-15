"""
STOCK PRICE DIRECTION MODEL TRAINING
Trains and saves RF + XGBoost models
Dataset: borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from features import create_features

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Train model and return metrics - avoids code duplication."""
    print(f"\n=== Training {model_name} ===")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return model, predictions, probabilities, accuracy

def plot_results(y_test, rf_pred, rf_prob, xgb_pred, importance_df, feature_cols):
    """Create comparison plots for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Feature Importance (from RF)
    ax1 = axes[0, 0]
    top_10 = importance_df.head(10)
    ax1.barh(top_10['feature'], top_10['importance'], color='steelblue')
    ax1.set_xlabel('Importance')
    ax1.set_title('Top 10 Feature Importances (RF)')
    ax1.invert_yaxis()
    
    # Plot 2: Model Comparison (Confusion Matrix style)
    ax2 = axes[0, 1]
    cm_rf = confusion_matrix(y_test, rf_pred)
    cm_xgb = confusion_matrix(y_test, xgb_pred)
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x - width/2, [cm_rf[0,0], cm_rf[1,1]], width, label='RF Correct', color='blue', alpha=0.7)
    ax2.bar(x + width/2, [cm_xgb[0,0], cm_xgb[1,1]], width, label='XGB Correct', color='orange', alpha=0.7)
    ax2.set_ylabel('Correct Predictions')
    ax2.set_title('Model Comparison: Correct Predictions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Down', 'Up'])
    ax2.legend()
    
    # Plot 3: RF Confidence Distribution
    ax3 = axes[1, 0]
    ax3.hist(rf_prob[y_test == 0], bins=25, alpha=0.6, label='Actual Down', color='red', density=True)
    ax3.hist(rf_prob[y_test == 1], bins=25, alpha=0.6, label='Actual Up', color='green', density=True)
    ax3.axvline(x=0.5, color='black', linestyle='--')
    ax3.set_xlabel('Predicted Probability of UP')
    ax3.set_title('RF Prediction Confidence')
    ax3.legend()
    
    # Plot 4: Rolling Accuracy Comparison
    ax4 = axes[1, 1]
    results_df = pd.DataFrame({
        'actual': y_test, 
        'rf_correct': y_test == rf_pred,
        'xgb_correct': y_test == xgb_pred
    })
    results_df['rf_rolling'] = results_df['rf_correct'].rolling(30).mean()
    results_df['xgb_rolling'] = results_df['xgb_correct'].rolling(30).mean()
    
    ax4.plot(results_df.index, results_df['rf_rolling'], label='RF (30-day)', color='blue')
    ax4.plot(results_df.index, results_df['xgb_rolling'], label='XGB (30-day)', color='orange')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Rolling Accuracy Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

def main():
    print("=== STOCK PRICE PREDICTION - MODEL TRAINING ===")
    
    # --- Load Data ---
    path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
    df = pd.read_csv(f"{path}/Data/Stocks/aapl.us.txt")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    print(f"Loaded {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    
    # --- Feature Engineering ---
    df_features = create_features(df)
    df_clean = df_features.dropna()
    print(f"After cleaning: {len(df_clean)} rows")
    
    # --- Setup ---
    feature_cols = [
        'Daily_Return', 'Price_Range', 'Price_Change',
        'Dist_SMA_5', 'Dist_SMA_10', 'Dist_SMA_20', 'Dist_SMA_50',
        'Trend_50_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Position', 'Volume_Ratio',
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_5'
    ]
    
    # --- Split ---
    split_date = "2012-01-01"
    train_data = df_clean[:split_date]
    test_data = df_clean[split_date:]
    
    X_train, y_train = train_data[feature_cols], train_data["Target"]
    X_test, y_test = test_data[feature_cols], test_data["Target"]
    
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    print(f"Class balance: {y_train.mean():.1%} Up")
    
    # --- Define Models ---
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=30,
            min_samples_leaf=15, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    }
    
    # --- Train & Evaluate ---
    results = {}
    for name, model in models.items():
        trained_model, pred, prob, acc = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, name
        )
        results[name] = {
            'model': trained_model,
            'predictions': pred,
            'probabilities': prob,
            'accuracy': acc
        }
    
    # --- Feature Importance (from RF) ---
    rf_model = results['Random Forest']['model']
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n=== Top 10 Features ===")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:<20}: {row['importance']:.4f}")
    
    # --- Save Models ---
    joblib.dump(results['Random Forest']['model'], "rf_stock_model.pkl")
    joblib.dump(results['XGBoost']['model'], "xgb_stock_model.pkl")
    
    with open("feature_list.json", "w") as f:
        json.dump(feature_cols, f)
    
    print("\nModels saved: rf_stock_model.pkl, xgb_stock_model.pkl")
    
    # --- Visualization ---
    plot_results(
        y_test,
        results['Random Forest']['predictions'],
        results['Random Forest']['probabilities'],
        results['XGBoost']['predictions'],
        importance_df,
        feature_cols
    )
    
    # --- Summary ---
    print("Final Results:")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<15}: {data['accuracy']*100:.2f}% accuracy")
    print(f"Baseline       : 50.00% (random)")

if __name__ == "__main__":
    main()