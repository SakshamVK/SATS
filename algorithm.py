import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv('traffic_data.csv')

# Filter numeric columns
numeric_cols = ['car_count', 'waiting_time']
numeric_data = data[numeric_cols]

# Check for NaN values
if numeric_data.isnull().values.any():
    raise ValueError("NaN values present in numeric data. Please handle or remove them.")

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(numeric_data)

# Prepare data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])  # Using only car_count for Y
    return np.array(X), np.array(Y)

look_back = 3
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 2)))  # Input shape adjusted to (look_back, 2)
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, Y, epochs=10, batch_size=2, verbose=2)

# Predict
predictions = model.predict(X)
predictions = np.reshape(predictions, (-1, 1))  # Reshape predictions to match scaler's expectations
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], 1)))))[:, 0]  # Inverse transform with dummy second column

# Evaluate
import matplotlib.pyplot as plt

plt.plot(data['car_count'], label='True Data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
