import h5py
import pandas as pd

# Open the HDF5 file
with h5py.File('traffic_light_model.h5', 'r') as f:
    data = f['results'][:]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['car_count', 'waiting_time'])

# Save to CSV
df.to_csv('results.csv', index=False)