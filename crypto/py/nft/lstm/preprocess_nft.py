import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

# load the dataset
df = pd.read_csv("nft_volume_data.csv")

# rename columns and convert date to datetime
df.rename(columns={'day': 'date', 'total_volume': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

# apply log transformation to stabilize variance
df['volume'] = np.log1p(df['volume'])

# add a short-term moving average (3-day) to smooth noise
df['volume_ma3'] = df['volume'].rolling(window=3).mean()

# compute daily difference and 7-day volatility
df['volume_diff'] = df['volume'].diff()
df['volatility'] = df['volume'].rolling(window=7).std()

# add lag features to capture short-term dependencies
df['lag1'] = df['volume'].shift(1)
df['lag2'] = df['volume'].shift(2)
df['lag3'] = df['volume'].shift(3)
df['lag7'] = df['volume'].shift(7)

# encode day of week as sine and cosine (to capture cyclical patterns)
df['day_of_week'] = df['date'].dt.dayofweek
df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# is holiday flag
holiday_list = [
    '2021-01-01','2021-12-25','2022-01-01','2022-12-25',
    '2023-01-01','2023-12-25','2024-01-01','2024-12-25','2025-01-01'
]
holiday_dates = set(pd.to_datetime(holiday_list).date)
df['is_holiday'] = df['date'].dt.date.apply(lambda d: 1 if d in holiday_dates else 0)

# drop rows with nan values (from rolling and lag computations)
df = df.dropna()

# define parameters for sequence creation
look_back = 120   # use past 120 days as input
n_future = 30     # forecast the next 30 days

# define feature columns
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

# scale all features together using robustscaler to reduce the impact of outliers
scaler = RobustScaler()
data_scaled = scaler.fit_transform(df[features].values)
joblib.dump(scaler, "scaler.pkl")

# function to create sequences for multi-step forecasting
def create_sequences(data, look_back, n_future):
    X, y = [], []
    for i in range(len(data) - look_back - n_future + 1):
        X.append(data[i:i + look_back])
        # we predict the 'volume' (index 0) for the next n_future days
        y.append(data[i + look_back: i + look_back + n_future, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, look_back, n_future)
np.save("X.npy", X)
np.save("y.npy", y)

# save the preprocessed dataset for reference
df.to_csv("nft_volume_preprocessed.csv", index=False)
print("data preprocessing complete.")
