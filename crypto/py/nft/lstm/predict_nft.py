import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# load the trained model and scaler
model = tf.keras.models.load_model("lstm_nft_model_seq2seq.h5")
scaler = joblib.load("scaler.pkl")

# load and prepare the preprocessed dataset
df = pd.read_csv("nft_volume_preprocessed.csv")
df['date'] = pd.to_datetime(df['date'])

look_back = 120
n_future = 30
features = [
    'volume',
    'volume_ma3',
    'volume_diff',
    'volatility',
    'sin_day',
    'cos_day',
    'is_holiday',
    'lag1',
    'lag2',
    'lag3',
    'lag7'
]

# scale features using the saved scaler
data_scaled = scaler.transform(df[features].values)

# prepare the input sequence using the last look_back days
input_seq = data_scaled[-look_back:].reshape(1, look_back, len(features))

# predict the next n_future days (predictions are in the log-transformed domain)
predicted_log = model.predict(input_seq, verbose=0).flatten()

# function to invert scaling and reverse the log transformation
def invert_transform(values, scaler, features, index=0):
    dummy = np.zeros((len(values), len(features)))
    dummy[:, index] = values
    inv = scaler.inverse_transform(dummy)[:, index]
    return np.expm1(inv)

# get predictions on the original volume scale
predicted = invert_transform(predicted_log, scaler, features, index=0)

# create a dataframe for predicted future dates and volumes
last_date = df['date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_future)
pred_df = pd.DataFrame({
    'date': future_dates,
    'predicted_volume': predicted
})

# plot historical and predicted volumes (applying inverse log transform on historical data)
plt.figure(figsize=(12,6))
plt.plot(df['date'], np.expm1(df['volume']), label="historical volume", alpha=0.7)
plt.plot(pred_df['date'], pred_df['predicted_volume'], label="predicted volume", linestyle='dashed', color="orange")
plt.yscale("log")
plt.xlabel("date")
plt.ylabel("volume")
plt.title("Daily NFT Market Volume LSTM 30-Day Forecast")
plt.legend()
plt.show()
