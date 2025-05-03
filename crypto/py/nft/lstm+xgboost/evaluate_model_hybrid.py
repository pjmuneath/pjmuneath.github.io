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

# load preprocessed data and sequences
df = pd.read_csv("nft_volume_preprocessed.csv")
df['date'] = pd.to_datetime(df['date'])
df.sort_values("date", inplace=True)

# load sequences (assumed to be created previously)
X_seq = np.load("X.npy")
y_seq = np.load("y.npy")

# use the last 100 sequences for meta-model training/evaluation
meta_samples = 100 if X_seq.shape[0] >= 100 else X_seq.shape[0]
meta_X = X_seq[-meta_samples:]
meta_y = y_seq[-meta_samples:]

# load the pre-trained LSTM model for NFT volume forecasting
lstm_model = tf.keras.models.load_model("lstm_nft_model_seq2seq.h5")

# get LSTM predictions on meta samples
lstm_preds = []
for i in range(meta_X.shape[0]):
    pred = lstm_model.predict(meta_X[i:i+1], verbose=0).flatten()
    lstm_preds.append(pred)
lstm_preds = np.array(lstm_preds)  # shape: (meta_samples, n_future)

# prepare XGBoost model
meta_X_flat = meta_X.reshape(meta_X.shape[0], -1)
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
xgb_preds = xgb_model.predict(meta_X_flat)  # shape: (meta_samples, n_future)

# combine predictions from LSTM and XGBoost for the meta-model
meta_features = np.hstack([lstm_preds, xgb_preds])
meta_targets = meta_y

# train a linear regression meta-model to combine predictions
meta_model = MultiOutputRegressor(LinearRegression())
meta_model.fit(meta_features, meta_targets)

# forecast the next n_future days using the most recent sequence
last_seq = np.copy(X_seq[-1:])  # shape: (1, look_back, n_features)
lstm_forecast = lstm_model.predict(last_seq, verbose=0).flatten()  # shape: (n_future,)
last_seq_flat = X_seq[-1:].reshape(1, -1)
xgb_forecast = xgb_model.predict(last_seq_flat).flatten()  # shape: (n_future,)
hybrid_features = np.hstack([lstm_forecast, xgb_forecast]).reshape(1, -1)
hybrid_forecast_log = meta_model.predict(hybrid_features).flatten()

# function to invert scaling
def invert_transform(values, scaler, features, index=0):
    n_expected = scaler.n_features_in_
    dummy = np.zeros((len(values), n_expected))
    dummy[:, index] = values
    inv = scaler.inverse_transform(dummy)[:, index]
    return np.expm1(inv)

# load the pre-fitted scaler
scaler = joblib.load("scaler.pkl")
features_list = ['volume', 'volume_diff', 'volatility', 'sin_day', 'cos_day', 'is_holiday', 'lag1', 'lag2', 'lag3', 'lag7']

# Invert transformation for the hybrid forecast (evaluate only on the last meta sample)
hybrid_forecast = invert_transform(hybrid_forecast_log, scaler, features_list, index=0)
actuals_log = meta_targets[-1]
actuals = invert_transform(actuals_log, scaler, features_list, index=0)

mae_val = mean_absolute_error(actuals, hybrid_forecast)
rmse_val = np.sqrt(mean_squared_error(actuals, hybrid_forecast))
mape_val = np.mean(np.abs((actuals - hybrid_forecast) / (actuals + 1e-8))) * 100

print("Hybrid Model Evaluation on Last Meta Sample:")
print(f"MAE: {mae_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAPE: {mape_val:.2f}%")

n_future_steps = meta_y.shape[1]
plt.figure(figsize=(12,6))
plt.plot(range(1, n_future_steps+1), actuals, label="Actual NFT Volume", marker="o")
plt.plot(range(1, n_future_steps+1), hybrid_forecast, label="Hybrid Forecast", marker="x", linestyle="--")
plt.xlabel("Forecast Horizon (Days)")
plt.ylabel("NFT Volume")
plt.title("LSTM + XGBoost Model Forecast vs Actual (Last Meta Sample)")
plt.legend()
plt.show()
