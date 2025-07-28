# predict.py

# ==============================================================================
# LIBRARIES & SETUP
# ==============================================================================
import argparse
import pandas as pd
import numpy as np

# --- WORKAROUND FOR 'NaN' ImportError ---
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
# --- END WORKAROUND ---

import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from datetime import date, timedelta


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
def get_data_and_features(ticker, start_date, end_date):
    """Downloads data and engineers features."""
    print(f"\n[INFO] Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print(f"[ERROR] No data found for ticker {ticker}. Exiting.")
        return None
    print("[INFO] Download complete.")

    # --- ROBUST FIX for Column Names ---
    new_cols = []
    for col in data.columns:
        if isinstance(col, tuple):
            new_cols.append(col[0].lower())
        else:
            new_cols.append(str(col).lower())
    data.columns = new_cols

    # --- Feature Engineering ---
    # Calculate daily return first as it's needed for other features
    data['daily_return'] = data['close'].pct_change()

    # Short-term indicators
    print("[INFO] Engineering short-term features (RSI, MACD, EMAs)...")
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.ema(length=50, append=True)
    data.ta.ema(length=200, append=True)

    # --- FIX: Standardize column names to lowercase BEFORE using them ---
    data.columns = [col.lower() for col in data.columns]

    # --- NEW: Long-Term Indicators ---
    print("[INFO] Engineering long-term features...")
    # Yearly momentum
    data['feat_yearly_return'] = data['close'].pct_change(252)
    # Yearly volatility
    data['feat_volatility_yearly'] = data['daily_return'].rolling(window=252).std() * np.sqrt(252)
    # Distance from long-term average (as a percentage)
    data['feat_dist_from_ema200'] = (data['close'] - data['ema_200']) / data['ema_200']

    return data


def predict_stock_direction(ticker, duration):
    """
    Trains a model to predict stock direction over a specified duration.

    Args:
        ticker (str): The stock ticker symbol.
        duration (str): 'day', 'week', 'month', or 'year'.
    """
    try:
        # --- 1. Setup Prediction Horizon ---
        duration_map = {'day': 1, 'week': 5, 'month': 21, 'year': 252}
        horizon = duration_map.get(duration, 1)

        start_date = '2015-01-01'
        end_date = date.today()

        data = get_data_and_features(ticker, start_date, end_date)
        if data is None:
            return

        # --- 2. Create the Target Variable ---
        data['target'] = (data['close'].shift(-horizon) > data['close']).astype(int)

        data.dropna(inplace=True)

        if data.empty:
            print("[ERROR] Not enough data after feature creation. Try an earlier start date.")
            return

        # --- 3. Train the Model ---
        # Updated feature list to include new long-term indicators
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd_12_26_9', 'ema_50', 'ema_200',
            'feat_yearly_return', 'feat_volatility_yearly', 'feat_dist_from_ema200'
        ]

        # Ensure all features exist in the dataframe
        features = [f for f in features if f in data.columns]

        X = data[features]
        y = data['target']

        X_train, y_train = X.iloc[:-1], y.iloc[:-1]
        X_predict = X.iloc[-1].values.reshape(1, -1)

        print(f"[INFO] Training model on {len(X_train)} data points...")

        # Increased bias to make model more "optimistic"
        up_bias_ratio = 2.0
        custom_class_weight = {0: 1, 1: up_bias_ratio}

        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=50,
            random_state=42,
            class_weight=custom_class_weight,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("[INFO] Model training complete.")

        # --- 4. Make and Display Prediction ---
        prediction = model.predict(X_predict)
        prediction_proba = model.predict_proba(X_predict)

        direction = "UP" if prediction[0] == 1 else "DOWN"
        confidence = prediction_proba[0][prediction[0]]

        print("\n" + "=" * 60)
        print(f"PREDICTION FOR {ticker.upper()} OVER THE NEXT {duration.upper()}")
        print(f"   (Prediction generated on: {end_date.strftime('%Y-%m-%d')})")
        print("-" * 60)
        print(f"The model predicts the price will go {direction} with {confidence:.2%} confidence.")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")


# ==============================================================================
# --- MAIN EXECUTION (CLI) ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A CLI tool to predict stock price direction using a Random Forest model.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help="Stock ticker symbol to predict (e.g., AAPL, GOOG, NVDA)."
    )

    parser.add_argument(
        '--duration',
        type=str,
        default='day',
        choices=['day', 'week', 'month', 'year'],
        help="The prediction duration.\n"
             " 'day':   Predict direction for the next trading day.\n"
             " 'week':  Predict direction over the next 5 trading days.\n"
             " 'month': Predict direction over the next 21 trading days.\n"
             " 'year':  Predict direction over the next 252 trading days."
    )

    args = parser.parse_args()

    predict_stock_direction(args.ticker, args.duration)
