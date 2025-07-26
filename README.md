# SATS (Synchronized Adaptive Traffic Signaling)

SATS is a machine learning–powered system for **adaptive traffic signal timing** using historical and simulated traffic data.  
Currently, it integrates with **SUMO (Simulation of Urban Mobility)** to simulate intersections and predict **optimized traffic light cycles** using an LSTM-based model.

The long-term goal is to scale SATS to handle **entire regions or cities**, incorporating **real-time data from CCTV feeds and IoT sensors**, with synchronized signal control across multiple intersections.

---

## Current Features
- Runs on a **single intersection** simulation using SUMO.
- Uses **historical traffic data** to train an **LSTM model** for signal timing predictions.
- Automatically updates SUMO configuration files with predicted timings.
- Re-runs the simulation to compare **before vs. after** performance.
- Outputs:
  - **Average waiting time reduction per vehicle**.
  - **Graphical analysis** (saved as an image).
  - **Performance summary in CLI** after each run.

---

## Planned Features
- Scale to **multiple intersections and regions**, with centralized coordination.
- Use **real-time data streams** (CCTV/IoT sensors).
- Deploy as a **cloud-based service** with API endpoints.
- Real-time **adaptive signaling** rather than static configuration.

---

## Project Structure
SATS/
│
├── data/ # Historical traffic datasets
├── models/ # LSTM models and checkpoints
├── configs/ # SUMO configuration files
├── outputs/ # Generated results, graphs, logs
├── main.py # Entry point for SATS (runs the pipeline)
├── preprocessing.py # Data cleaning and feature extraction
├── prediction.py # Model inference logic
├── training.py # Model training (if needed)
└── utils.py # Helper functions

---

## How It Works
1. **Run `main.py`** with the selected SUMO configuration file.
2. Historical data is loaded from `data/` and processed.
3. The **LSTM model** predicts optimal signal timings for each lane.
4. Predicted timings are inserted into the SUMO configuration.
5. SUMO is executed:
   - Once with the default (non-optimized) timings.
   - Once with the predicted (optimized) timings.
6. SATS outputs:
   - CLI summary (average waiting time improvement).
   - A **graph** (saved to `outputs/`).
   - Updated SUMO files.

---

## Requirements
- Python 3.10 or later
- SUMO (Simulation of Urban Mobility)
- Dependencies (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
Usage
Clone the repository:
git clone <your-repo-url>
cd SATS
Ensure SUMO is installed and added to PATH.

Place your SUMO map/configuration files in configs/.

Run SATS:
python main.py --config configs/sample.sumocfg
View results in CLI and generated graph in outputs/.

License
This project is licensed under a Demo Evaluation License (see LICENSE.md):

For evaluation and demonstration purposes only.

No commercial use without explicit permission.

Derivative works allowed only with proper attribution.

Roadmap
 Multi-intersection coordination.

 Real-time data ingestion (CCTV/IoT).

 Cloud-based REST API.

 Web dashboard for live monitoring.
