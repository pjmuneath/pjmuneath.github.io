import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load preprocessed data
df = pd.read_csv("dex_volume_preprocessed.csv")
df['date'] = pd.to_datetime(df['date'])

# define feature list
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

# define parameters
look_back = 120  # number of past days used as input
n_future = 30    # number of days to forecast

# load the pre-fitted scaler and scale data
scaler = joblib.load("scaler.pkl")
data_scaled = scaler.transform(df[features].values)

# create sequences using a sliding window
def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i + look_back])
        # predict the 'volume' feature (index 0) for the next n_future days
        y.append(data[i + look_back: i + look_back + n_future, 0])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(data_scaled, look_back, n_future)

# xgboost expects 2d tabular data, so flatten the time dimension
n_samples = X_seq.shape[0]
X_flat = X_seq.reshape(n_samples, -1)  # shape becomes (n_samples, look_back * number_of_features)

# split data into training and testing sets (no shuffling, to preserve time order)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_seq, test_size=0.2, shuffle=False)

# build and train the xgboost model using multioutputregressor for multi-step forecasting
xgb_model = MultiOutputRegressor(
    XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
)
xgb_model.fit(X_train, y_train)

# evaluate the model on the test set
y_pred = xgb_model.predict(X_test)
mae_val = mean_absolute_error(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
mape_val = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("xgboost model evaluation on test set:")
print(f"mae: {mae_val:.2f}")
print(f"rmse: {rmse_val:.2f}")
print(f"mape: {mape_val:.2f}%")

# forecast the next 30 days using the most recent data
last_input = data_scaled[-look_back:].reshape(1, -1)
xgb_forecast_log = xgb_model.predict(last_input).flatten()

# function to invert scaling and reverse the log transformation
def invert_transform(values, scaler, features, index=0):
    dummy = np.zeros((len(values), len(features)))
    dummy[:, index] = values
    inv = scaler.inverse_transform(dummy)[:, index]
    return np.expm1(inv)

# invert transformation for the forecast
xgb_forecast = invert_transform(xgb_forecast_log, scaler, features, index=0)

# visualization
plt.figure(figsize=(12, 6))
plt.plot(df['date'], np.expm1(df['volume']), label="historical volume", alpha=0.7)
last_date = df['date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_future)
plt.plot(future_dates, xgb_forecast, label="xgboost forecast", linestyle='dashed', color="orange")
plt.yscale("log")
plt.xlabel("date")
plt.ylabel("volume")
plt.title("Daily DEX Market Volume XGBoost 30-Day Forecast")
plt.legend()
plt.show()