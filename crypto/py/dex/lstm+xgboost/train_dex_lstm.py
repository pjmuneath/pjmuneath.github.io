import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt

# load the preprocessed data and sequences
df = pd.read_csv("dex_volume_preprocessed.csv")
X = np.load("X.npy")
y = np.load("y.npy")

# perform a time-based split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

look_back = 120
n_future = 30
n_features = X.shape[2]

# bidirectional lstm
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(look_back, n_features)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(n_future, activation='linear'))

# compile with an adam optimizer and huber loss
model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(delta=1.0))

# train the model
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# save the model
model.save("lstm_dex_model_seq2seq.h5")

# plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

print("training complete and model saved.")
