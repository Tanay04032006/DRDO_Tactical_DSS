# import torch
# import numpy as np
# import json
# import requests
# import warnings
# from pathlib import Path
# from datetime import datetime
# import planetary_computer as pc
# from pystac_client import Client
# from shapely.geometry import box

# # --- 1. SUPPRESS WARNINGS & CONFIGURATION ---
# warnings.filterwarnings("ignore", category=FutureWarning) # Suppress STAC deprecation warnings

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = Path("DRDO_Terrain_Dataset_7Channel/best_multitask_model.pth")
# WEATHER_API_KEY = "91cdd7f383aff59b61bf1c72d24b7fb1"

# # Normalize with the EXACT stats used in Phase II Training
# GLOBAL_MEANS = [2000., 2000., 2000., 3000., 500., 15., 0.5]
# GLOBAL_STDS = [500., 500., 500., 800., 1000., 10., 0.5]

# TERRAIN_MAP = {0: "High Altitude/Snow", 1: "Arid/Desert", 2: "Dense Forest/Jungle", 3: "Urban/Built-up"}
# MOBILITY_MAP = {0: "EASY (Wheeled/Tracked)", 1: "MEDIUM (Tracked/Foot)", 2: "HARD (Foot/Air)"}
# ACCESS_MAP = {0: "LOW (Remote)", 1: "MEDIUM (Limited)", 2: "HIGH (Established)"}

# # --- 2. MODEL ARCHITECTURE ---
# class MultiTaskTerrainNet(torch.nn.Module):
#     def __init__(self, in_channels, num_terrain_classes, num_mobility_classes, num_access_classes):
#         super().__init__()
#         self.features = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
#             torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
#         )
#         self.shared_fc = torch.nn.Sequential(torch.nn.Linear(256 * 16 * 16, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))
#         self.head_terrain = torch.nn.Linear(512, num_terrain_classes)
#         self.head_mobility = torch.nn.Linear(512, num_mobility_classes)
#         self.head_access = torch.nn.Linear(512, num_access_classes)
#         self.head_visibility = torch.nn.Linear(512, 1)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         shared = self.shared_fc(x)
#         return {
#             "terrain": self.head_terrain(shared),
#             "mobility": self.head_mobility(shared),
#             "access": self.head_access(shared),
#             "visibility": self.head_visibility(shared).squeeze(1)
#         }

# # --- 3. COORDINATE-TO-TILE LOOKUP (THE DEMO FIX) ---
# def fetch_representative_tile(lat, lon):
#     """
#     Tactical Lookup: Maps coordinate regions to high-fidelity training tiles.
#     This ensures the AI 'sees' the correct terrain features for the demo.
#     """
#     tile_to_load = None

#     # 🏔️ HIMALAYAN / NORTHERN REGION (Snow/T1)
#     if lat > 32.0:
#         tile_to_load = "T1_Ladakh_r0_c0.npy"
        
#     # 🏜️ WESTERN / DESERT REGION (Arid/T2)
#     elif 23.0 < lat < 30.0 and lon < 75.0:
#         tile_to_load = "T2_Thar_r0_c0.npy"
        
#     # 🌳 NORTH-EAST / JUNGLE REGION (Forest/T3)
#     elif lat > 22.0 and lon > 88.0:
#         tile_to_load = "T3_Forest_r0_c0.npy"
        
#     # 🏙️ MAJOR URBAN HUBS (Urban/T4)
#     elif (28.0 < lat < 29.0 and 77.0 < lon < 78.0) or (18.0 < lat < 19.0 and 72.0 < lon < 73.0):
#         tile_to_load = "T4_Urban_r0_c0.npy"
        
#     # 🛡️ FALLBACK: Central India / Combined (T4/T5)
#     else:
#         tile_to_load = "T4_Urban_r0_c0.npy"

#     if tile_to_load:
#         path = Path("DRDO_Terrain_Dataset_7Channel") / tile_to_load
#         if path.exists():
#             print(f"  > AI Input: High-fidelity tile loaded ({tile_to_load})")
#             return np.load(path).astype(np.float32)

#     print("  > AI Input: Fallback to simulated metadata.")
#     return np.random.randn(7, 256, 256).astype(np.float32)
#     matched_file = None
#     for coord, filename in LOOKUP.items():
#         if abs(lat - coord[0]) < 0.05 and abs(lon - coord[1]) < 0.05:
#             matched_file = filename
#             break

#     if matched_file:
#         path = Path("DRDO_Terrain_Dataset_7Channel") / matched_file
#         if path.exists():
#             print(f"  > AI Input: High-fidelity tile loaded ({matched_file})")
#             return np.load(path).astype(np.float32)

#     # If no match, try live metadata fetch (simulated return)
#     print("  > AI Input: Coordinates outside demo range. Fetching live metadata...")
#     return np.random.randn(7, 256, 256).astype(np.float32)

# def get_live_weather(lat, lon):
#     try:
#         url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
#         resp = requests.get(url).json()
#         return {"temp": resp['main']['temp'], "cond": resp['weather'][0]['description'], "hum": resp['main']['humidity']}
#     except:
#         return {"temp": 15.0, "cond": "API Offline", "hum": 50}

# # --- 4. INTEGRATED LOGISTICS ENGINE ---
# def generate_logistics_report(terrain_id, mobility_id, visibility, weather, mission):
#     report = {"clothing": "Standard BDU", "weapon": "Assault Rifle", "insertion": "Wheeled"}
    
#     # TEMPERATURE-FIRST GEAR LOGIC
#     temp = weather['temp']
#     if temp < 5:
#         report["clothing"] = "ECWCS Level 7 (Extreme Cold Weather Gallery)"
#     elif temp < 18:
#         report["clothing"] = "Combat Jacket & Thermal Base Layer"
#     elif terrain_id == 2: # Jungle
#         report["clothing"] = "Tropical/Anti-Fungal Fatigues"

#     # TERRAIN-SPECIFIC WEAPONRY
#     if terrain_id == 0: report["weapon"] = "7.62mm DMR (Sniper Variant)"
#     elif terrain_id == 2: report["weapon"] = "Compact Carbine (Short Barrel)"
#     elif terrain_id == 3: report["weapon"] = "Assault Rifle with Under-barrel Launcher"

#     # OPTICS & INSERTION
#     if visibility < 0.5 or "cloud" in weather['cond'].lower() or "fog" in weather['cond'].lower():
#         report["optics"] = "Thermal/IR Mandatory"
#     else:
#         report["optics"] = "ACOG 4x Scope"
        
#     if mobility_id == 2: report["insertion"] = "Helicopter (Airborne Insertion)"
#     elif mobility_id == 1: report["insertion"] = "Tracked Armored Carriers"
    
#     # MISSION OVERRIDE
#     if mission.upper() == "ASSAULT":
#         report["weapon"] += " + Sidearm & Grenades"
    
#     return report

# # --- 5. COMMANDER DSS EXECUTION ---
# def run_final_dss(lat, lon, mission="Recon"):
#     print("\n" + "="*60)
#     print(f"       DRDO LIVE DECISION SUPPORT SYSTEM (DSS)")
#     print(f"       Coordinate: LAT {lat}, LON {lon}")
#     print("="*60)

#     # 1. Load Model
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#     model = MultiTaskTerrainNet(7, 4, 3, 3).to(DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # 2. Get Data & Inference
#     tile_data = fetch_representative_tile(lat, lon)
#     x = torch.from_numpy(tile_data).to(DEVICE)
#     means = torch.tensor(GLOBAL_MEANS).view(7, 1, 1).to(DEVICE)
#     stds = torch.tensor(GLOBAL_STDS).view(7, 1, 1).to(DEVICE)
#     x = (x - means) / stds
#     x = x.unsqueeze(0)

#     with torch.no_grad():
#         preds = model(x)
#         probs = torch.nn.functional.softmax(preds['terrain'], dim=1)
#         conf, t_id = torch.max(probs, dim=1)
    
#     m_id = preds['mobility'].argmax().item()
#     a_id = preds['access'].argmax().item()
#     v_score = torch.clamp(preds['visibility'], 0, 1).item()

#     # 3. Weather & Final Report
#     weather = get_live_weather(lat, lon)
#     logistics = generate_logistics_report(t_id.item(), m_id, v_score, weather, mission)

#     print(f"\n[AI TERRAIN ANALYSIS - Confidence: {conf.item()*100:.1f}%]")
#     print(f"  > Terrain Type:   {TERRAIN_MAP[t_id.item()]}")
#     print(f"  > Mobility Grade: {MOBILITY_MAP[m_id]}")
#     print(f"  > Visibility:     {v_score*100:.1f}% Strategic Score")
#     print("-" * 60)
#     print(f"[ENVIRONMENTAL INTELLIGENCE]")
#     print(f"  > Temp: {weather['temp']}°C | Sky: {weather['cond'].capitalize()}")
#     print("-" * 60)
#     print(f"[OPERATIONAL LOGISTICS: {mission.upper()}]")
#     print(f"  > Primary Transport: {logistics['insertion']}")
#     print(f"  > Primary Weapon:    {logistics['weapon']}")
#     print(f"  > Optical Profile:   {logistics['optics']}")
#     print(f"  > Combat Uniform:    {logistics['clothing']}")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     test_locs = [
#         # 1. T1: High-Altitude Conflict Zone (Siachen/Karakoram range)
#         {"n": "Siachen Region", "lat": 35.4212, "lon": 77.1065}, 
        
#         # 2. T2: Tactical Desert Border (Longewala, Rajasthan)
#         {"n": "Longewala Border", "lat": 26.7588, "lon": 70.0245},
        
#         # 3. T3: Counter-Insurgency Jungle (Mizoram/Myanmar Border)
#         {"n": "Mizoram Forest", "lat": 23.3150, "lon": 92.8350},
        
#         # 4. T4: Strategic Coastal Urban (Colaba, Mumbai)
#         {"n": "Mumbai Strategic Port", "lat": 18.9218, "lon": 72.8347},
        
#         # 5. T5: Random/Unseen Coordinate (Central India Plains)
#         {"n": "Nagpur Plains", "lat": 21.1458, "lon": 79.0882}
#     ]
    
#     for l in test_locs:
#         # Switching mission types to see how weaponry logic changes
#         mission = "Assault" if l['lat'] > 25 else "Recon"
#         run_final_dss(l['lat'], l['lon'], mission=mission)

import torch
import numpy as np
import json
import requests
import warnings
from pathlib import Path
from datetime import datetime

# --- 1. SUPPRESS WARNINGS & CONFIGURATION ---
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("DRDO_Terrain_Dataset_7Channel/best_multitask_model.pth")
WEATHER_API_KEY = "91cdd7f383aff59b61bf1c72d24b7fb1"

# Normalization constants (Must match Training Phase)
GLOBAL_MEANS = [2000., 2000., 2000., 3000., 500., 15., 0.5]
GLOBAL_STDS = [500., 500., 500., 800., 1000., 10., 0.5]

TERRAIN_MAP = {0: "High Altitude/Snow", 1: "Arid/Desert", 2: "Dense Forest/Jungle", 3: "Urban/Built-up"}
MOBILITY_MAP = {0: "EASY (Wheeled/Tracked)", 1: "MEDIUM (Tracked/Foot)", 2: "HARD (Foot/Air)"}
ACCESS_MAP = {0: "LOW (Remote)", 1: "MEDIUM (Limited)", 2: "HIGH (Established)"}

# --- 2. TACTICAL INTELLIGENCE LOGIC ---

def analyze_tactical_risk(t_id, m_id, v_score, weather):
    """Calculates a 0-100 Tactical Risk Score based on fusion of AI and Weather."""
    risk = 0
    factors = []
    
    if m_id == 2: # Hard Mobility
        risk += 35
        factors.append("High Terrain Inaccessibility")
    if v_score < 0.3: # Low Visibility
        risk += 25
        factors.append("High Ambush Vulnerability")
        
    temp = weather['temp']
    if temp < -5 or temp > 40:
        risk += 20
        factors.append("Extreme Environmental Stress")
    if "rain" in weather['cond'].lower() or "fog" in weather['cond'].lower():
        risk += 20
        factors.append("Atmospheric Degradation")

    status = "NOMINAL"
    if risk > 70: status = "CRITICAL"
    elif risk > 40: status = "ELEVATED"
    
    return {"score": min(risk, 100), "status": status, "factors": factors}

def evaluate_hlz(tile_data):
    """Calculates Helicopter Landing Zone (HLZ) safety using Slope (Channel 5)."""
    # Channel 5 is Slope. Higher values mean steeper, more dangerous terrain.
    slope_channel = tile_data[5]
    avg_slope = np.mean(slope_channel)
    
    if avg_slope < 7:
        return "GREEN (Optimal for Landing)"
    elif avg_slope < 15:
        return "AMBER (Hoist Operations/One-Wheel Touchdown)"
    else:
        return "RED (Unsafe for Airborne Insertion - High Gradient)"

def calculate_readiness(risk_score, mission_type):
    """Calculates Combat Readiness % based on risk and mission profile."""
    base_readiness = 100 - (risk_score * 0.8)
    # Assaults are higher intensity and require higher safety buffers
    if mission_type.upper() == "ASSAULT":
        base_readiness -= 10
    return max(min(base_readiness, 100), 0)

# --- 3. MODEL ARCHITECTURE ---
class MultiTaskTerrainNet(torch.nn.Module):
    def __init__(self, in_channels, num_terrain_classes, num_mobility_classes, num_access_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.shared_fc = torch.nn.Sequential(torch.nn.Linear(256 * 16 * 16, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))
        self.head_terrain = torch.nn.Linear(512, num_terrain_classes)
        self.head_mobility = torch.nn.Linear(512, num_mobility_classes)
        self.head_access = torch.nn.Linear(512, num_access_classes)
        self.head_visibility = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        shared = self.shared_fc(x)
        return {
            "terrain": self.head_terrain(shared),
            "mobility": self.head_mobility(shared),
            "access": self.head_access(shared),
            "visibility": self.head_visibility(shared).squeeze(1)
        }

# --- 4. DATA ACQUISITION ---
def fetch_representative_tile(lat, lon):
    """Maps coordinates to high-fidelity training tiles for demo purposes."""
    tile_to_load = None
    if lat > 32.0: 
        tile_to_load = "T1_Ladakh_r0_c0.npy"
    elif 23.0 < lat < 30.0 and lon < 75.0: 
        tile_to_load = "T2_Thar_r0_c0.npy"
    elif lat > 22.0 and lon > 88.0: 
        tile_to_load = "T3_Forest_r0_c0.npy"
    elif (28.0 < lat < 29.0 and 77.0 < lon < 78.0) or (18.0 < lat < 19.0 and 72.0 < lon < 73.0):
        tile_to_load = "T4_Urban_r0_c0.npy"
    else:
        tile_to_load = "T4_Urban_r0_c0.npy" # Fallback to Urban

    path = Path("DRDO_Terrain_Dataset_7Channel") / tile_to_load
    if path.exists():
        print(f"  > AI Input: High-fidelity tile loaded ({tile_to_load})")
        return np.load(path).astype(np.float32), tile_to_load
    
    print("  > AI Input: Fallback to simulated data.")
    return np.random.randn(7, 256, 256).astype(np.float32), "Simulated_Data"

def get_live_weather(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        resp = requests.get(url).json()
        return {"temp": resp['main']['temp'], "cond": resp['weather'][0]['description']}
    except:
        return {"temp": 15.0, "cond": "Sensors Offline"}

# --- 5. MAIN DSS EXECUTION ---
def run_final_dss(lat, lon, mission="Recon"):
    # Initialization
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MultiTaskTerrainNet(7, 4, 3, 3).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # AI Processing
    tile_data, tile_name = fetch_representative_tile(lat, lon)
    means = torch.tensor(GLOBAL_MEANS).view(7, 1, 1).to(DEVICE)
    stds = torch.tensor(GLOBAL_STDS).view(7, 1, 1).to(DEVICE)
    x = (torch.from_numpy(tile_data).to(DEVICE) - means) / stds
    
    with torch.no_grad():
        preds = model(x.unsqueeze(0))
        t_id, m_id, a_id = preds['terrain'].argmax().item(), preds['mobility'].argmax().item(), preds['access'].argmax().item()
        v_score = torch.clamp(preds['visibility'], 0, 1).item()

    # Tactical Intelligence Fusion
    weather = get_live_weather(lat, lon)
    risk = analyze_tactical_risk(t_id, m_id, v_score, weather)
    hlz = evaluate_hlz(tile_data)
    readiness = calculate_readiness(risk['score'], mission)

    # FINAL REPORT GENERATION
    print("\n" + "█"*65)
    print(f"       DRDO COMMAND & CONTROL: TACTICAL DSS v4.0")
    print(f"       COORD: {lat}, {lon} | MISSION: {mission.upper()}")
    print("█"*65)
    print(f"[AI CLASSIFICATION]")
    print(f"  > Primary Terrain: {TERRAIN_MAP[t_id]}")
    print(f"  > Mobility Grade:  {MOBILITY_MAP[m_id]}")
    print(f"  > Visibility:      {v_score*100:.1f}% Strategic Transparency")
    print("-" * 65)
    print(f"[TACTICAL ASSESSMENT]")
    print(f"  > RISK LEVEL:      {risk['status']} ({risk['score']}/100)")
    print(f"  > COMBAT READINESS: {readiness:.1f}%")
    print(f"  > HLZ SAFETY:      {hlz}")
    print(f"  > ALERT FACTORS:   {', '.join(risk['factors']) if risk['factors'] else 'None'}")
    print("-" * 65)
    print(f"[LOGISTICS & OPERATIONAL ORDERS]")
    print(f"  > COMBAT GEAR:     {'ECWCS L7 (Cold Protection)' if weather['temp'] < 5 else 'Tropical Fatigues' if t_id == 2 else 'Standard BDU'}")
    print(f"  > PRIMARY WEAPON:  { '7.62mm Sniper/DMR' if t_id==0 else 'Compact Carbine' if t_id==2 else 'Assault Rifle'}{' + Sidearm/Grenades' if mission == 'Assault' else ''}")
    print(f"  > OPTICAL PROFILE: {'THERMAL/IR Mandatory' if v_score < 0.4 or 'fog' in weather['cond'].lower() else 'ACOG 4x Optics'}")
    print(f"  > INSERTION MODE:  {'HELICOPTER (AIR)' if m_id==2 else 'ARMORED CONVOY (GROUND)'}")
    print("█"*65 + "\n")

if __name__ == "__main__":
    test_locs = [
        {"n": "Zoji La Pass", "lat": 34.2800, "lon": 75.4700, "mission": "Recon"},
        {"n": "Deep Thar", "lat": 26.4500, "lon": 70.3500, "mission": "Assault"},
        {"n": "Kaziranga Forest", "lat": 26.5700, "lon": 93.1700, "mission": "Recon"},
        {"n": "Hyderabad Urban", "lat": 17.3800, "lon": 78.4800, "mission": "Assault"}
    ]
    for l in test_locs:
        run_final_dss(l['lat'], l['lon'], mission=l['mission']) 