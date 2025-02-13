import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

#1) IMPORTING DATA 
data = pd.read_csv('Workshop 4/data/stocks_data.csv')

df = pd.DataFrame(data)

features = ['open', 'high', 'low', 'volume', 'return', 'rolling_mean', 'rolling_std']
target = 'close'

X = df[features]
Y = df[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, Y_test))

model.predict(X_test_scaled)

