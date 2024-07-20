# traffic_light_simulation.py

import traci
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load your trained ML model and scalers
model = load_model('traffic_light_model.h5')
scalers = joblib.load('scalers.pkl')

# Define the function to collect data and make predictions
def get_traffic_data(lane_ids):
    data = []
    for lane_id in lane_ids:
        car_count = traci.lane.getLastStepVehicleNumber(lane_id)
        waiting_time = traci.lane.getWaitingTime(lane_id)
        data.append([car_count, waiting_time])
    return np.array(data)

def make_predictions(data, scalers, look_back=3):
    predictions = []
    for lane_id, lane_data in zip(scalers.keys(), data):
        scaler = scalers[lane_id]
        scaled_data = scaler.transform(lane_data)
        X = []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back), :])
        X = np.array(X)
        if len(X) == 0:
            predictions.append(0)  # Append a default value if not enough data
        else:
            prediction = model.predict(X)
            prediction = np.reshape(prediction, (-1, 1))  # Reshape prediction to match scaler's expectations
            prediction = scaler.inverse_transform(np.hstack((np.zeros((prediction.shape[0], 1)), prediction)))[:, 1]  # Inverse transform with dummy first column
            predictions.append(prediction[-1])  # Append the last prediction for the current lane
    return predictions

# Start SUMO simulation with TraCI
sumoCmd = ["sumo", "-c", "configuration.sumo.cfg"]
traci.start(sumoCmd)

# Dynamically get all lane IDs and traffic light IDs
lane_ids = traci.lane.getIDList()
traffic_light_ids = traci.trafficlight.getIDList()
step = 0
look_back = 3
traffic_data = []

while step < 50:
    traci.simulationStep()

    current_data = get_traffic_data(lane_ids)
    traffic_data.append(current_data)

    if len(traffic_data) >= look_back:
        predictions = make_predictions(np.array(traffic_data[-look_back:]), scalers, look_back)

        if len(predictions) < len(traffic_light_ids):
            predictions.extend([0] * (len(traffic_light_ids) - len(predictions)))  # Extend predictions to match traffic lights

        for i, traffic_light_id in enumerate(traffic_light_ids):
            green_time = predictions[i]
            print(f"Predicted green light time for {traffic_light_id}: {green_time:.2f} seconds")
            traci.trafficlight.setPhaseDuration(traffic_light_id, green_time)

    step += 1

traci.close()
