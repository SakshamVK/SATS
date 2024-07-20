import traci
import pandas as pd
import time

# Define the path to your SUMO executable and configuration file
sumoCmd = [
    "C:\\Sumo\\bin\\sumo",
    "-c", "D:\\Saksham\\signal syncronization\\configuration.sumo.cfg"
]

# Start SUMO simulation with TraCI
traci.start(sumoCmd)

def control_traffic(step):
    # Here, you would implement your ML algorithm to adjust traffic lights based on real-time data
    # For demonstration purposes, this function will just print the current step
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import joblib

    # Load and preprocess data
    data = pd.read_csv('traffic_data.csv')

    # Filter numeric columns
    numeric_cols = ['car_count', 'waiting_time']
    lane_ids = data['lane_id'].unique()
    scalers = {}

    # Normalize data per lane
    for lane_id in lane_ids:
        lane_data = data[data['lane_id'] == lane_id][numeric_cols]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scalers[lane_id] = scaler
        data.loc[data['lane_id'] == lane_id, numeric_cols] = scaler.fit_transform(lane_data)

    # Save the scalers for later use
    joblib.dump(scalers, 'scalers.pkl')

    # Prepare data for LSTM
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, 1])  # Using waiting_time for Y (predicting green light duration)
        return np.array(X), np.array(Y)

    look_back = 3
    X_list, Y_list = [], []

    # Create dataset for each lane
    for lane_id in lane_ids:
        lane_data = data[data['lane_id'] == lane_id][numeric_cols].values
        if len(lane_data) > look_back:
            X_lane, Y_lane = create_dataset(lane_data, look_back)
            X_list.append(X_lane)
            Y_list.append(Y_lane)
        else:
            print(f"Not enough data for lane {lane_id}")

    # Combine datasets from all lanes
    if X_list and Y_list:
        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 2)))  # Input shape adjusted to (look_back, 2)
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, Y, epochs=10, batch_size=32, verbose=2)

        # Save the trained model
        model.save('traffic_light_model.h5')

        # Predict for each lane
        predictions_list = []

        for lane_id in lane_ids:
            lane_data = data[data['lane_id'] == lane_id][numeric_cols].values
            if len(lane_data) > look_back:
                X_lane, _ = create_dataset(lane_data, look_back)
                X_lane = np.reshape(X_lane, (X_lane.shape[0], X_lane.shape[1], X_lane.shape[2]))

                predictions = model.predict(X_lane)
                predictions = np.reshape(predictions, (-1, 1))  # Reshape predictions to match scaler's expectations
                scaler = scalers[lane_id]
                predictions = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 1)), predictions)))[:, 1]  # Inverse transform with dummy first column
                predictions_list.append(predictions)
            else:
                print(f"Not enough data for lane {lane_id} to make predictions")

        # Combine predictions for visualization
        if predictions_list:
            all_predictions = np.concatenate(predictions_list)
            print(all_predictions)

            # Evaluate and plot results for each lane
            plt.figure(figsize=(15, 10))

            for lane_id in lane_ids:
                lane_data = data[data['lane_id'] == lane_id]
                true_data = lane_data['waiting_time'].values[look_back+1:]  # Adjust for look_back
                predictions = all_predictions[:len(true_data)]
                all_predictions = all_predictions[len(true_data):]

                plt.plot(true_data, label=f'True Data for lane {lane_id}')
                plt.plot(predictions, label=f'Predictions for lane {lane_id}')

            plt.legend()
            plt.show()
    else:
        print("Not enough data to train the model.")


# Initialize data collection
data = []

step = 0
while step < 50:
    traci.simulationStep()  # Advance the simulation by one step
    
    # Fetch data for each lane
    lane_ids = traci.lane.getIDList()  # Get all lane IDs
    for lane_id in lane_ids:
        car_count = traci.lane.getLastStepVehicleNumber(lane_id)
        waiting_time = traci.lane.getWaitingTime(lane_id)
        data.append([step, lane_id, car_count, waiting_time])
    
    # Control traffic based on the collected data
    control_traffic(step)
    
    step += 1
    time.sleep(0.1)  # Add a short delay to simulate real-time control

# Close TraCI connection
traci.close()

# Save data to a CSV file
df = pd.DataFrame(data, columns=['step', 'lane_id', 'car_count', 'waiting_time'])
df.to_csv('traffic_data.csv', index=False)
