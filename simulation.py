import traci
import pandas as pd

# Start SUMO simulation
sumoCmd = ["C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo", "-c", "C:\\Coding\\signal syncronization\\configration.sumo.cfg"]
traci.start(sumoCmd)

data = []

step = 0
while step < 50:
    traci.simulationStep()
    
    lane_ids = traci.lane.getIDList()  # Get all lane IDs
    for lane_id in lane_ids:
        car_count = traci.lane.getLastStepVehicleNumber(lane_id)
        waiting_time = traci.lane.getWaitingTime(lane_id)
        data.append([step, lane_id, car_count, waiting_time])
    
    step += 1

traci.close()

# Save data to a CSV file
df = pd.DataFrame(data, columns=['step', 'lane_id', 'car_count', 'waiting_time'])
df.to_csv('traffic_data.csv', index=False)
