"""
Python file with method to create technical indicators on a dataframe.
"""

def create_features(df):
    """
    Create technical indicators for stock prediction.
    This returns a dataframe with features + Target column.
    """
    df = df.copy()
    
    # Price-based features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Price_Change'] = df['Close'] - df['Open']
    
    # Moving averages and distances
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        # Distance from SMA (normalized: how far is price from MA?)
        df[f'Dist_SMA_{window}'] = (df['Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']
    
    # Long-term trend
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Trend_50_200'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    
    # Momentum: RSI (0-100, >70 overbought, <30 oversold)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Momentum: MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands (Volatility indicator)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    # Position within bands (0 = at lower, 1 = at upper, 0.5 = middle)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Lagged returns (past performance)
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
    
    # TARGET: 1 if tomorrow's Close > today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df