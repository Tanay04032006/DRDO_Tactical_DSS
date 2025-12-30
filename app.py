import streamlit as st
import torch
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import time
from streamlit_lottie import st_lottie
import requests
from pathlib import Path
from datetime import datetime

# ================================
# 1. BACKEND INTEGRATION
# ================================
try:
    import phase3_final as backend
except ImportError:
    st.error("❌ Cannot find 'phase3_final.py'. Ensure it is in the same folder.")
    st.stop()

# ================================
# 2. PAGE CONFIGURATION & TACTICAL CSS
# ================================
st.set_page_config(page_title="DRDO Tactical DSS v5.2", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0d1117, #05070a); color: #e0e0e0; }
    .hud-panel {
        background: rgba(16, 20, 29, 0.9);
        border: 1px solid #00FF41;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.15);
        padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace;
    }
    .readiness-value {
        font-size: 3.5rem; font-weight: bold; color: #00FF41;
        text-shadow: 0 0 15px rgba(0, 255, 65, 0.5); text-align: center;
    }
    .ticker-wrap {
        position: fixed; bottom: 0; left: 0; width: 100%; 
        background: rgba(0,0,0,0.9); border-top: 1px solid #00FF41; padding: 8px; z-index: 999;
    }
    .ticker-text {
        display: inline-block; white-space: nowrap; animation: ticker 30s linear infinite;
        color: #00FF41; font-family: 'Courier New', monospace; font-size: 0.9rem;
    }
    @keyframes ticker { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
    .op-card {
        background: rgba(22, 27, 34, 0.8); border: 1px solid #30363d;
        border-left: 5px solid #d32f2f; padding: 15px; border-radius: 5px; height: 100%;
    }
    .stButton>button {
        background: linear-gradient(45deg, #8b0000, #d32f2f) !important;
        border: none !important; color: white !important; font-weight: bold;
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# ================================
# 3. UTILITIES
# ================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

radar_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_698wi0tc.json")
drone_anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_7msh8ndu.json")

def live_ticker(lat, lon):
    ts = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""<div class="ticker-wrap"><div class="ticker-text">
    [SYSTEM: ACTIVE] -- [SENTINEL-2 FEED: STABLE] -- [TARGET: {lat}, {lon}] -- [TIME: {ts} Z] -- [ENCRYPTION: RSA-4096] -- [STATUS: SCANNED]
    </div></div>""", unsafe_allow_html=True)

# ================================
# 4. SESSION & MODEL
# ================================
if "lat" not in st.session_state: st.session_state.lat = 35.4212
if "lon" not in st.session_state: st.session_state.lon = 77.1065

@st.cache_resource
def load_dss_engine():
    path = Path("DRDO_Terrain_Dataset_7Channel/best_multitask_model.pth")
    if not path.exists(): return None
    ckpt = torch.load(path, map_location="cpu")
    model = backend.MultiTaskTerrainNet(7, 4, 3, 3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ================================
# 5. UI: SIDEBAR & HEADER
# ================================
with st.sidebar:
    if drone_anim: st_lottie(drone_anim, height=150, key="drone")
    st.header("🎯 STRATEGIC INPUT")
    mission = st.selectbox("Operation Profile", ["Assault", "Recon", "Patrol"])
    lat_in = st.number_input("Lat", value=float(st.session_state.lat), format="%.5f")
    lon_in = st.number_input("Lon", value=float(st.session_state.lon), format="%.5f")
    st.session_state.lat, st.session_state.lon = lat_in, lon_in
    st.divider()
    execute = st.button("🚀 INITIATE SCAN", use_container_width=True)

st.title("🛡️ DRDO COMMAND & CONTROL")
st.markdown("<h5 style='color: #888; margin-top:-15px;'>TACTICAL DECISION SUPPORT SYSTEM v5.2</h5>", unsafe_allow_html=True)

col_map, col_intel = st.columns([2, 1])

with col_map:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=8, tiles="CartoDB dark_matter")
    folium.Marker([st.session_state.lat, st.session_state.lon], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)
    map_out = st_folium(m, height=450, width=800, key="map_v6", returned_objects=["last_clicked"])
    if map_out and map_out.get("last_clicked"):
        nl, nlo = map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"]
        if abs(nl - st.session_state.lat) > 0.0001:
            st.session_state.lat, st.session_state.lon = nl, nlo
            st.rerun()

# ================================
# 6. INFERENCE & LOGISTICS LOGIC
# ================================
if execute:
    model = load_dss_engine()
    with st.status("📡 Processing Multi-Spectral Data...", expanded=False) as status:
        tile, _ = backend.fetch_representative_tile(st.session_state.lat, st.session_state.lon)
        weather = backend.get_live_weather(st.session_state.lat, st.session_state.lon)
        
        # Normalization
        m_t = torch.tensor(backend.GLOBAL_MEANS).view(7, 1, 1)
        s_t = torch.tensor(backend.GLOBAL_STDS).view(7, 1, 1)
        x = (torch.from_numpy(tile).float() - m_t) / s_t
        
        with torch.no_grad():
            preds = model(x.unsqueeze(0))
            t_id, m_id, v_sc = preds["terrain"].argmax().item(), preds["mobility"].argmax().item(), torch.clamp(preds["visibility"], 0, 1).item()
        
        risk = backend.analyze_tactical_risk(t_id, m_id, v_sc, weather)
        hlz = backend.evaluate_hlz(tile)
        readiness = backend.calculate_readiness(risk["score"], mission)
        status.update(label="✅ SCAN COMPLETE", state="complete")

    # Dynamic Logistics logic
    temp = weather.get('temp', 20)
    clothing = "ECWCS L7 (Heavy Cold)" if temp < 5 else "BDU + Combat Jacket" if temp < 18 else "Tropical Fatigues"
    vehicle = "HAL Light Combat Heli" if m_id == 2 else "Tata WhAP (Wheeled Armored)" if m_id == 0 else "Mahindra Marksman (Tracked)"
    weapon = "7.62mm DMR/Sniper" if t_id == 0 else "Compact Carbine" if t_id == 2 else "INSAS/AK-203"

    with col_intel:
        if radar_anim: st_lottie(radar_anim, height=140, key="radar")
        st.markdown(f"""<div class="hud-panel">
            <div style="text-align:center; font-size:0.8rem; color:#888;">MISSION READINESS</div>
            <div class="readiness-value">{readiness:.1f}%</div>
            <hr style="border:0.5px solid #30363d;">
            <p><b>TERRAIN:</b> {backend.TERRAIN_MAP[t_id]}</p>
            <p><b>WEATHER:</b> {temp}°C | {weather.get('cond', 'Clear')}</p>
            <p><b>THREAT:</b> <span style="color:#ff3131;">{risk['status']} ({risk['score']}/100)</span></p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📋 TACTICAL OPERATIONAL ORDERS (OPORD)")
    o1, o2, o3, o4 = st.columns(4)
    o1.markdown(f"<div class='op-card'><b>INSERTION VEHICLE</b><br>{vehicle}</div>", unsafe_allow_html=True)
    o2.markdown(f"<div class='op-card'><b>MISSION GEAR</b><br>{clothing}</div>", unsafe_allow_html=True)
    o3.markdown(f"<div class='op-card'><b>PRIMARY ARMAMENT</b><br>{weapon}</div>", unsafe_allow_html=True)
    o4.markdown(f"<div class='op-card'><b>HLZ STATUS</b><br>{hlz}</div>", unsafe_allow_html=True)

    # Professional Dispatch Report
    report = f"""------------------------------------------------------------
                     DRDO TACTICAL DISPATCH
                     SECRET // NOFORN
------------------------------------------------------------
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
MISSION ID: {mission.upper()}-{int(time.time())}
LOCATION: {st.session_state.lat}, {st.session_state.lon}

[1. ENVIRONMENTAL INTELLIGENCE]
- TERRAIN TYPE: {backend.TERRAIN_MAP[t_id]}
- AMBIENT TEMP: {temp}°C
- VISIBILITY: {v_sc*100:.1f}% Strategic Transparency
- HLZ ADVISORY: {hlz}

[2. RISK ASSESSMENT]
- TACTICAL RISK: {risk['status']} ({risk['score']}/100)
- COMBAT READINESS: {readiness:.1f}%
- ALERT FACTORS: {', '.join(risk['factors']) if risk['factors'] else 'None'}

[3. LOGISTICS & EQUIPMENT]
- INSERTION: {vehicle}
- UNIFORM: {clothing}
- WEAPONRY: {weapon} {'+ Assault Support' if mission == 'Assault' else ''}

[4. COMMANDER'S INTENT]
Proceed with mission if Readiness > 70%. Ensure thermal profile matches {temp}°C ambient.
------------------------------------------------------------
"""
    st.download_button("💾 DOWNLOAD FORMAL OPORD REPORT", data=report, 
                       file_name=f"DRDO_REPORT_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                       use_container_width=True)

else:
    with col_intel: st.info("🛰️ Standby. Initiate scan to populate Sit-Rep.")

live_ticker(st.session_state.lat, st.session_state.lon)