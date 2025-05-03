import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

# load the scaler and scale the data
scaler = joblib.load("scaler.pkl")
data_scaled = scaler.transform(df[features].values)

# function to create sequences for multistep forecasting
def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i + look_back])
        # predict the 'volume' feature (index 0) for the next n_future days
        y.append(data[i + look_back: i + look_back + n_future, 0])
    return np.array(X), np.array(y)

# create sequences from the scaled data
X_seq, y_seq = create_sequences(data_scaled, look_back, n_future)

# split data into training and test sets (preserving time order)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# flatten the time dimension
n_samples_train = X_train.shape[0]
X_train_flat = X_train.reshape(n_samples_train, -1)

# train xgboost model on training set
xgb_model = MultiOutputRegressor(
    XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
)
xgb_model.fit(X_train_flat, y_train)

# load the pre-trained lstm model
lstm_model = tf.keras.models.load_model("lstm_dex_model_seq2seq.h5")

# create meta training data from the last part of the training set
meta_train_size = min(100, X_train.shape[0])
meta_X_train = X_train[-meta_train_size:]
meta_y_train = y_train[-meta_train_size:]

# get base model predictions (lstm and xgboost) on meta training samples
lstm_meta_preds = []
xgb_meta_preds = []
for i in range(meta_train_size):
    sample = meta_X_train[i:i+1]
    # get lstm prediction
    pred_lstm = lstm_model.predict(sample, verbose=0).flatten()
    lstm_meta_preds.append(pred_lstm)
    # flatten sample for xgboost input
    sample_flat = sample.reshape(1, -1)
    pred_xgb = xgb_model.predict(sample_flat).flatten()
    xgb_meta_preds.append(pred_xgb)
lstm_meta_preds = np.array(lstm_meta_preds)
xgb_meta_preds = np.array(xgb_meta_preds)

# combine base model predictions to form meta features
meta_features_train = np.hstack([lstm_meta_preds, xgb_meta_preds])

# train a linear regression metamodel (wrapped in multioutputregressor) to combine predictions
meta_model = MultiOutputRegressor(LinearRegression())
meta_model.fit(meta_features_train, meta_y_train)

# evaluate the hybrid model on the test set
n_samples_test = X_test.shape[0]
lstm_test_preds = []
xgb_test_preds = []
for i in range(n_samples_test):
    sample = X_test[i:i+1]
    pred_lstm = lstm_model.predict(sample, verbose=0).flatten()
    lstm_test_preds.append(pred_lstm)
    sample_flat = sample.reshape(1, -1)
    pred_xgb = xgb_model.predict(sample_flat).flatten()
    xgb_test_preds.append(pred_xgb)
lstm_test_preds = np.array(lstm_test_preds)
xgb_test_preds = np.array(xgb_test_preds)
meta_features_test = np.hstack([lstm_test_preds, xgb_test_preds])

# use metamodel to generate hybrid predictions on test set
hybrid_test_preds = meta_model.predict(meta_features_test)

# calculate evaluation metrics on test set for hybrid model
mae_val = mean_absolute_error(y_test, hybrid_test_preds)
rmse_val = np.sqrt(mean_squared_error(y_test, hybrid_test_preds))
mape_val = np.mean(np.abs((y_test - hybrid_test_preds) / y_test)) * 100

print("hybrid model evaluation on test set:")
print(f"mae: {mae_val:.2f}")
print(f"rmse: {rmse_val:.2f}")
print(f"mape: {mape_val:.2f}%")

# visualization
plt.figure(figsize=(12,6))
plt.plot(y_test[0], label="actual forecast")
plt.plot(hybrid_test_preds[0], label="hybrid forecast", linestyle='dashed', color="green")
plt.xlabel("day")
plt.ylabel("scaled volume")
plt.title("LSTM + XGBoost Model Forecast vs Actual (Test Sample)")
plt.legend()
plt.show()
