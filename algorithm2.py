import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the collected data
data = pd.read_csv('traffic_data.csv')

# Prepare the data for training
X = data[['car_count', 'waiting_time']].values
y = data['waiting_time'].values  # Assuming waiting_time as the target for simplicity

# Define and train a simple neural network model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2, batch_size=32)

# Save the trained model
model.save('traffic_light_model.h5')
