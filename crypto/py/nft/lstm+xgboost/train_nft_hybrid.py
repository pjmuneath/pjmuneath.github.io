import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# load preprocessed nft data and sort by date
df = pd.read_csv("nft_volume_preprocessed.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values("date", inplace=True)

features = ['volume', 'volume_ma3', 'volume_diff', 'volatility', 'sin_day', 'cos_day', 'is_holiday', 'lag1', 'lag2', 'lag3', 'lag7']
look_back = 120
n_future = 30

# load the pre-fitted scaler and scale the data
scaler = joblib.load("scaler.pkl")
data_scaled = scaler.transform(df[features].values)

def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+n_future, 0])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(data_scaled, look_back, n_future)

meta_samples = 100 if X_seq.shape[0] >= 100 else X_seq.shape[0]
meta_X = X_seq[-meta_samples:]
meta_y = y_seq[-meta_samples:]

# lstm
lstm_model = tf.keras.models.load_model("lstm_nft_model_seq2seq.h5")

lstm_preds = []
for i in range(meta_X.shape[0]):
    pred = lstm_model.predict(meta_X[i:i+1], verbose=0).flatten()
    lstm_preds.append(pred)
lstm_preds = np.array(lstm_preds)

meta_X_flat = meta_X.reshape(meta_X.shape[0], -1)

# xgboost
n_samples = X_seq.shape[0]
X_flat = X_seq.reshape(n_samples, -1)
xgb_model = MultiOutputRegressor(
    XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
)
xgb_model.fit(X_flat, y_seq)
xgb_preds = xgb_model.predict(meta_X_flat)

meta_features = np.hstack([lstm_preds, xgb_preds])
meta_targets = meta_y

meta_model = MultiOutputRegressor(LinearRegression())
meta_model.fit(meta_features, meta_targets)

last_seq = data_scaled[-look_back:].reshape(1, look_back, len(features))
lstm_forecast = lstm_model.predict(last_seq, verbose=0).flatten()
last_seq_flat = data_scaled[-look_back:].reshape(1, -1)
xgb_forecast = xgb_model.predict(last_seq_flat).flatten()
hybrid_features = np.hstack([lstm_forecast, xgb_forecast]).reshape(1, -1)
hybrid_forecast_log = meta_model.predict(hybrid_features).flatten()

def invert_transform(values, scaler, features, index=0):
    dummy = np.zeros((len(values), len(features)))
    dummy[:, index] = values
    inv = scaler.inverse_transform(dummy)[:, index]
    return np.expm1(inv)

hybrid_forecast = invert_transform(hybrid_forecast_log, scaler, features, index=0)

# plot
plt.figure(figsize=(12,6))
plt.plot(df['date'], np.expm1(df['volume']), label="Historical NFT Volume", alpha=0.7)
last_date = df['date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future, freq="D")
plt.plot(future_dates, hybrid_forecast, label="Hybrid Forecast", linestyle="dashed", color="green")
plt.yscale("log")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.title("Daily NFT Market Volume LSTM + XGBoost 30-Day Forecast")
plt.legend()
plt.show()
