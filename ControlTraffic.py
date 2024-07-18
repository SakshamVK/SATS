import traci
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained ML model
model = load_model('traffic_light_model.h5')

# Define the function to collect data and make predictions
def get_traffic_data(lane_ids):
    data = []
    for lane_id in lane_ids:
        car_count = traci.lane.getLastStepVehicleNumber(lane_id)
        waiting_time = traci.lane.getWaitingTime(lane_id)
        data.append([car_count, waiting_time])
    return np.array(data)

def make_predictions(data, scaler, look_back=3):
    scaled_data = scaler.transform(data)
    X = []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), :])
    X = np.array(X)
    predictions = model.predict(X)
    return predictions

# Start SUMO simulation with TraCI
sumoCmd = ["sumo", "-c", "your_config_file.sumocfg"]
traci.start(sumoCmd)

# Define the lane IDs you want to control
lane_ids = ['lane1', 'lane2', 'lane3', 'lane4']

# Assume scaler is already fitted from your training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Assuming you have saved the scaler from your training process
# scaler = joblib.load('scaler.pkl')

step = 0
look_back = 3
traffic_data = []

while step < 1000:
    traci.simulationStep()

    # Collect data from SUMO
    current_data = get_traffic_data(lane_ids)
    traffic_data.append(current_data)
    
    if len(traffic_data) >= look_back:
        # Make predictions using the ML model
        predictions = make_predictions(np.array(traffic_data[-look_back:]), scaler)
        
        # Adjust traffic signals based on predictions
        for i, lane_id in enumerate(lane_ids):
            green_time = predictions[i][0]
            # Here, you would set the green light duration for the traffic light controlling this lane
            traci.trafficlight.setPhaseDuration('traffic_light_id', green_time)
    
    step += 1

traci.close()
