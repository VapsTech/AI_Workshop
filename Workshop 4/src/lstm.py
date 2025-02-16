import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

def lstm_train_predict(x_train, y_train, x_test):
    # Scale the data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)

    Y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))

    # Reshape the data
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Create the Neural Network Model
    model = Sequential()
    model.add(LSTM(100, activation='sigmoid', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit/Train the model
    model.fit(X_train_scaled, Y_train_scaled, epochs=50, batch_size=32)

    # Predictions
    predictions_scaled = model.predict(X_test_scaled)
    # Inverse the scaling to Orignal Values
    predictions = scaler_Y.inverse_transform(predictions_scaled)

    return predictions