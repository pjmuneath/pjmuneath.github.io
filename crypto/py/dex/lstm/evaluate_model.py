import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load the trained model and scaler
model = tf.keras.models.load_model("lstm_dex_model_seq2seq.h5")
scaler = joblib.load("scaler.pkl")

# load and prepare the preprocessed dataset
df = pd.read_csv("dex_volume_preprocessed.csv")
df['date'] = pd.to_datetime(df['date'])
features = [
    'volume',
    'volume_diff',
    'volatility',
    'sin_day',
    'cos_day',
    'is_holiday',
    'lag1',
    'lag2',
    'lag3'
]

look_back = 120
n_future = 30

# create a pseudo-test set from the last (look_back + n_future) days of historical data
test_data = df.iloc[-(look_back + n_future):].copy()
data_scaled = scaler.transform(test_data[features].values)

# prepare the input sequence from the test set (first look_back days)
input_seq = data_scaled[:look_back].reshape(1, look_back, len(features))

# actual future values (in log space) for the next n_future days from the test set
actual_log = data_scaled[look_back:look_back + n_future, 0]  # 'volume' is at index 0

# model prediction for the test period
predicted_log = model.predict(input_seq, verbose=0).flatten()

# function to invert scaling and reverse the log transformation
def invert_transform(values, scaler, features, index=0):
    dummy = np.zeros((len(values), len(features)))
    dummy[:, index] = values
    inv = scaler.inverse_transform(dummy)[:, index]
    return np.expm1(inv)

# invert both predictions and actual values to original volume scale
predicted = invert_transform(predicted_log, scaler, features, index=0)
actual = invert_transform(actual_log, scaler, features, index=0)

# calculate error metrics
mae_val = mean_absolute_error(actual, predicted)
rmse_val = np.sqrt(mean_squared_error(actual, predicted))
mape_val = np.mean(np.abs((actual - predicted) / actual)) * 100

print("evaluation on historical test period:")
print(f"mae: {mae_val:.2f}")
print(f"rmse: {rmse_val:.2f}")
print(f"mape: {mape_val:.2f}%")
