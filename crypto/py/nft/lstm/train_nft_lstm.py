import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# load the preprocessed data and sequences
df = pd.read_csv("nft_volume_preprocessed.csv")
X = np.load("X.npy")
y = np.load("y.npy")

# perform a time-based split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

look_back = 120
n_future = 30
n_features = X.shape[2]

model = Sequential()
# first bidirectional lstm layer
model.add(Bidirectional(
    LSTM(32, return_sequences=True, kernel_regularizer=l2(0.005)),
    input_shape=(look_back, n_features)
))
model.add(Dropout(0.6))
model.add(BatchNormalization())

# second bidirectional lstm layer
model.add(Bidirectional(
    LSTM(16, return_sequences=False, kernel_regularizer=l2(0.005))
))
model.add(Dropout(0.6))
model.add(BatchNormalization())

# output dense layer for forecasting n_future days
model.add(Dense(n_future, activation='linear'))

# compile the model with adam optimizer and huber loss
model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(delta=1.0))

# set up callbacks for early stopping and reducing lr on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# save the model
model.save("lstm_nft_model_seq2seq.h5")

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
