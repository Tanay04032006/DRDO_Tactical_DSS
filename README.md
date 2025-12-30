This README.md is designed to provide a professional, military-grade technical overview for your repository. It guides users through the installation, data preparation, and execution of the Tactical Decision Support System.

🛡️ DRDO Tactical Decision Support System (TDSS)
Multi-Spectral Neural Analysis for Frontier Security & Logistics
This repository contains a full-stack AI solution for automated terrain intelligence. The system processes 7-channel multi-spectral data (RGB + NIR + Elevation + Slope + Elev-Class) to provide real-time tactical recommendations, risk assessment, and mission-specific logistics.

🚀 Quick Start: How to Run
1. Environment Setup
Ensure you have Python 3.9+ and a CUDA-enabled GPU (recommended) installed.

Bash

# Clone the repository
git clone https://github.com/YourUsername/DRDO-Tactical-DSS.git
cd DRDO-Tactical-DSS

# Install dependencies
pip install -r requirements.txt
2. Data Preparation (Phase I)
The system requires a local Digital Elevation Model (DEM) to calculate slope and elevation classes.

Download ASTER GDEM v3 tiles for your region from NASA Earthdata.

Place the merged TIFF file in the root directory and name it local_gdem_30m.tif.

Run the data engineering pipeline to generate training/inference tiles:

Bash

python phase1_data_engineering.py
3. Model Training (Phase II)
To train the Multi-Task Learning (MTL) CNN from scratch:

Bash

python phase2_training.py
This will generate best_multitask_model.pth in the DRDO_Terrain_Dataset_7Channel/ folder.

4. Launching the Command & Control HUD (Phase III)
Run the interactive Streamlit dashboard:

Bash

streamlit run app.py
🛰️ System Architecture
The TDSS operates across three integrated phases:

Phase I: Data Engineering – Automated STAC API retrieval of Sentinel-2 bands fused with ASTER GDEM data.

Phase II: Neural Engine – A Multi-Task CNN that simultaneously predicts Terrain, Mobility, Access, and Visibility.

Phase III: Tactical Fusion – Merges AI outputs with Live Weather API data to generate Risk Scores and Helicopter Landing Zone (HLZ) safety ratings.

📋 Features
Interactive Targeting: Folium-based map interface for coordinate-specific analysis.

Dynamic Logistics: Automated recommendations for weaponry (Sniper/Carbine), vehicles (Heli/Wheeled), and gear (ECWCS L7).

Safety Engineering: Real-time HLZ slope verification (Green/Amber/Red).

Strategic Export: Downloadable "SECRET // NOFORN" Operational Orders (OPORD).

🛠️ Requirements
streamlit, streamlit-folium, folium

torch, torchvision

numpy, pandas, rasterio, shapely

pystac-client, planetary-computer

requests, streamlit-lottie

🛡️ Operational Note
This system is a proof-of-concept for Decision Support Systems (DSS) in frontier environments. The logistics logic is tailored to high-altitude and arid border regions typical of the Indian subcontinent.
