import traci
import pandas as pd

# Define the path to your SUMO executable and configuration file
sumoCmd = [
    "C:\\Sumo\\bin\\sumo",
    "-c", "D:\\Saksham\\signal syncronization\\configuration.sumo.cfg"
]

# Start SUMO simulation with TraCI on the default port 8813
traci.start(sumoCmd, port=8813)

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
    
    step += 1

# Close TraCI connection
traci.close()

# Save data to a CSV file
df = pd.DataFrame(data, columns=['step', 'lane_id', 'car_count', 'waiting_time'])
df.to_csv('traffic_data2.csv', index=False)
