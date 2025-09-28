# app.py
"""
Flask backend for Stock Prediction System
Endpoints:
 - GET  /health
 - GET  /validate?symbol=TSLA
 - GET  /history?symbol=TSLA&start=2020-01-01&end=2024-01-01
 - POST /train  (json: {symbol, start, end, model_type ('linear'|'rf'|'lstm'), epochs (for lstm), force})
 - GET  /predict?symbol=TSLA&model_type=rf&days=30
 - GET  /visualize?symbol=TSLA&model_type=rf&days=60
Notes:
 - Models saved under models/<symbol>_<model_type>/
 - Plots served as PNG images
"""

import os
import io
import json
import datetime as dt
from typing import Tuple

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Optional tensorflow import for LSTM
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# --- Configuration ---
MODEL_DIR = "models"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow cross-origin for frontend usage


# -----------------------
# Helper functions
# -----------------------
def fetch_history(symbol: str, start: str = None, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV using yfinance.
    start, end should be 'YYYY-MM-DD' or None.
    """
    if start is None:
        start = "2015-01-01"
    if end is None:
        end = dt.datetime.now().strftime("%Y-%m-%d")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)
    if df.empty:
        raise ValueError(f"No data for symbol {symbol} in given range.")
    df.reset_index(inplace=True)
    # Ensure consistent columns: Date, Open, High, Low, Close, Volume
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()


def prepare_series(df: pd.DataFrame) -> pd.Series:
    """Return close price series indexed by date."""
    s = df[['Date','Close']].copy()
    s.set_index('Date', inplace=True)
    s.index = pd.to_datetime(s.index)
    return s['Close']


def get_model_paths(symbol: str, model_type: str) -> Tuple[str, str]:
    """
    Return (model_path, scaler_path) for a given symbol and model_type
    """
    base = os.path.join(MODEL_DIR, f"{symbol.upper()}_{model_type}")
    os.makedirs(base, exist_ok=True)
    if model_type == 'lstm':
        return os.path.join(base, "lstm_model.h5"), os.path.join(base, "scaler.gz")
    else:
        return os.path.join(base, f"{model_type}_model.joblib"), os.path.join(base, "scaler.gz")


def train_sklearn_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported sklearn model type")
    model.fit(X_train, y_train)
    return model


def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def create_supervised(series: np.ndarray, window: int = 10):
    """
    Convert 1D series into supervised learning X, y using sliding window
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    y = np.array(y)
    return X, y


# -----------------------
# Endpoints
# -----------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tensorflow_available": TF_AVAILABLE})


@app.route("/validate", methods=["GET"])
def validate_symbol():
    """
    Quick check if yfinance can fetch the ticker info
    /validate?symbol=TSLA
    """
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol query param required"}), 400
    try:
        t = yf.Ticker(symbol)
        info = t.info  # This may raise or be empty for invalid tickers
        # pick a few informative fields if present
        brief = {
            "symbol": symbol,
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }
        return jsonify({"valid": True, "info": brief})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


@app.route("/history", methods=["GET"])
def history():
    """
    Return historical OHLCV as JSON
    /history?symbol=TSLA&start=2020-01-01&end=2024-01-01
    """
    symbol = request.args.get("symbol", "").strip().upper()
    start = request.args.get("start", None)
    end = request.args.get("end", None)
    if not symbol:
        return jsonify({"error": "symbol param required"}), 400
    try:
        df = fetch_history(symbol, start, end)
        # convert dates to isoformat strings
        records = df.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("Date"), (pd.Timestamp, dt.date, dt.datetime)):
                r["Date"] = pd.to_datetime(r["Date"]).strftime("%Y-%m-%d")
        return jsonify({"symbol": symbol, "history": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/train", methods=["POST"])
def train():
    """
    Train model endpoint.
    JSON body: {
      "symbol": "TSLA",
      "start": "2015-01-01",
      "end": "2024-01-01",
      "model_type": "rf" | "linear" | "lstm",
      "window": 20,
      "epochs": 20,         # for LSTM
      "force": false
    }
    """
    payload = request.get_json(force=True)
    symbol = payload.get("symbol", "").strip().upper()
    model_type = payload.get("model_type", "rf").lower()
    start = payload.get("start", None)
    end = payload.get("end", None)
    window = int(payload.get("window", 20))
    epochs = int(payload.get("epochs", 20))
    force = bool(payload.get("force", False))

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    model_path, scaler_path = get_model_paths(symbol, model_type)
    try:
        df = fetch_history(symbol, start, end)
        series = prepare_series(df).values.reshape(-1, 1).astype(float)
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series).flatten()

        # Save scaler
        joblib.dump(scaler, scaler_path)

        if model_type in ['linear', 'rf']:
            X, y = create_supervised(series_scaled, window=window)
            # reshape X to 2D for sklearn
            X2 = X.reshape(X.shape[0], -1)
            model = train_sklearn_model(X2, y, model_type)
            joblib.dump(model, model_path)
            return jsonify({"status": "trained", "symbol": symbol, "model_type": model_type, "model_path": model_path})
        elif model_type == 'lstm':
            if not TF_AVAILABLE:
                return jsonify({"error": "TensorFlow not available on server. Install tensorflow to train LSTM."}), 500
            X, y = create_supervised(series_scaled, window=window)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X, y, epochs=epochs, batch_size=32, callbacks=[es], verbose=1)
            model.save(model_path)
            return jsonify({"status": "trained", "symbol": symbol, "model_type": model_type, "model_path": model_path})
        else:
            return jsonify({"error": f"unsupported model_type {model_type}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["GET"])
def predict():
    """
    Generate predictions.
    Query params:
     - symbol (required)
     - model_type (linear|rf|lstm) default rf
     - days (int) how many future days to predict (default 30)
     - window (int) window size used in training/prediction (default 20)
    Returns JSON with predicted dates & prices (and optionally last_history)
    """
    symbol = request.args.get("symbol", "").strip().upper()
    model_type = request.args.get("model_type", "rf").lower()
    days = int(request.args.get("days", 30))
    window = int(request.args.get("window", 20))

    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    model_path, scaler_path = get_model_paths(symbol, model_type)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return jsonify({"error": "model or scaler not found. Train first via /train"}), 400

    try:
        # get recent history for seed values
        hist_df = fetch_history(symbol, None, None)
        close_series = prepare_series(hist_df).values.reshape(-1, 1).astype(float)
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(close_series).flatten()

        # create seed from last 'window' points
        seed = list(scaled[-window:])

        preds_scaled = []
        if model_type in ['linear', 'rf']:
            model = joblib.load(model_path)
            for _ in range(days):
                X_in = np.array(seed[-window:]).reshape(1, -1)  # shape (1, window)
                yhat = model.predict(X_in)[0]
                preds_scaled.append(float(yhat))
                seed.append(float(yhat))
            preds_scaled = np.array(preds_scaled)
        elif model_type == 'lstm':
            if not TF_AVAILABLE:
                return jsonify({"error": "TensorFlow not available on server. Cannot load LSTM model."}), 500
            model = load_model(model_path)
            for _ in range(days):
                X_in = np.array(seed[-window:]).reshape(1, window, 1)
                yhat = model.predict(X_in, verbose=0)[0,0]
                preds_scaled.append(float(yhat))
                seed.append(float(yhat))
            preds_scaled = np.array(preds_scaled)
        else:
            return jsonify({"error": f"unsupported model_type {model_type}"}), 400

        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        # prepare dates for predictions (next business days)
        last_date = pd.to_datetime(hist_df['Date'].iloc[-1])
        future_dates = []
        cur = last_date
        while len(future_dates) < days:
            cur = cur + pd.Timedelta(days=1)
            # skip weekends (optional)
            if cur.weekday() >= 5:
                continue
            future_dates.append(cur.strftime("%Y-%m-%d"))

        predictions = [{"date": d, "predicted_close": float(p)} for d,p in zip(future_dates, preds.tolist())]

        # include last N history points for frontend convenience
        last_history = hist_df[['Date','Close']].tail(200).copy()
        last_history['Date'] = last_history['Date'].dt.strftime("%Y-%m-%d")
        history_records = last_history.to_dict(orient='records')

        return jsonify({"symbol": symbol, "model_type": model_type, "predictions": predictions, "last_history": history_records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualize", methods=["GET"])
def visualize():
    """
    Returns a PNG plot combining historical close price and predictions.
    Query params:
     - symbol (required)
     - model_type (rf|linear|lstm)
     - days (future days to predict)
     - window (int)
    """
    symbol = request.args.get("symbol", "").strip().upper()
    model_type = request.args.get("model_type", "rf").lower()
    days = int(request.args.get("days", 30))
    window = int(request.args.get("window", 20))

    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    # call predict() internally to reuse logic
    with app.test_request_context(f"/predict?symbol={symbol}&model_type={model_type}&days={days}&window={window}"):
        resp = predict()
    if isinstance(resp, tuple):
        data, status = resp
        return jsonify({"error": "prediction failed", "details": data}), status
    data = resp.get_json()

    if 'predictions' not in data:
        return jsonify({"error": "prediction failed", "details": data}), 500

    # build plot
    last_history = pd.DataFrame(data['last_history'])
    last_history['Date'] = pd.to_datetime(last_history['Date'])
    preds = pd.DataFrame(data['predictions'])
    preds['date'] = pd.to_datetime(preds['date'])

    plt.figure(figsize=(10,5))
    plt.plot(last_history['Date'], last_history['Close'], label='History')
    plt.plot(preds['date'], preds['predicted_close'], linestyle='--', marker='o', label='Predicted')
    plt.title(f"{symbol} - History + {len(preds)} day prediction ({model_type})")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return send_file(buf, mimetype='image/png', as_attachment=False, download_name=f"{symbol}_{model_type}_viz.png")


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    # Default for local dev; for prod use gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)

