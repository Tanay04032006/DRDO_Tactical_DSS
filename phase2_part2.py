import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any

# Optional progress bar
try:
    from tqdm.auto import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# -----------------------------
# CONFIG & CONSTANTS
# -----------------------------
DATA_DIR = Path("DRDO_Terrain_Dataset_7Channel")
LABELS_CSV = DATA_DIR / "labels_multitask.csv"  # <-- make sure this exists

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1  # epochs

# Loss weights (can tune these)
W_TERRAIN = 1.0
W_MOBILITY = 0.7
W_ACCESS = 0.7
W_VISIBILITY = 0.3

# 🛑 FIXED CLASS COUNTS (BASED ON PROJECT DEFINITIONS)
NUM_TERRAIN_CLASSES = 4     # T1, T2, T3, T4 (0, 1, 2, 3)
NUM_MOBILITY_CLASSES = 3    # easy, medium, hard (0, 1, 2)
NUM_ACCESS_CLASSES = 3      # low, medium, high (0, 1, 2)

# 🛑 FIXED GLOBAL NORMALIZATION STATS (MUST BE CALCULATED FROM FULL DATASET)
# Placeholder values for 7 channels: (B02, B03, B04, B08, DEM, SLOPE, ELEV_CLASS)
GLOBAL_MEANS = [2000., 2000., 2000., 3000., 500., 15., 0.5]
GLOBAL_STDS = [500., 500., 500., 800., 1000., 10., 0.5]

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# -----------------------------
# DSS LOGISTICS MAPPING (PHASE III)
# -----------------------------

# Terrain Mapping (from Phase I/II)
TERRAIN_MAP = {
    0: {"id": "T1", "name": "High Altitude/Snow"},
    1: {"id": "T2", "name": "Arid/Desert"},
    2: {"id": "T3", "name": "Dense Forest/Jungle"},
    3: {"id": "T4", "name": "Urban/Built-up"},
}

# Mobility Mapping (Translating model output to human concept)
MOBILITY_MAP = {0: "EASY (Wheeled/Tracked)", 1: "MEDIUM (Tracked/Foot)", 2: "HARD (Foot/Air)"}
ACCESS_MAP = {0: "LOW (Remote)", 1: "MEDIUM (Limited)", 2: "HIGH (Established)"}


def get_weapons_vehicle_recommendation(terrain_id: int, mobility_id: int, mission_type: str) -> Dict[str, str]:
    """
    Provides vehicle and weapon recommendations based on terrain and predicted mobility.
    """
    
    # 1. Base Weapon and Ancillary Gear based on Terrain (T1-T4)
    if terrain_id == 0: # T1: High Altitude/Snow
        weapon = "Designated Marksman Rifle (7.62mm), Thermal Optics"
        ancillary = "Climbing Gear, Oxygen Tanks, White Camouflage Over-suit"
    elif terrain_id == 1: # T2: Arid/Desert
        weapon = "Standard Assault Rifle (5.56mm), Dust Covers"
        ancillary = "Sand Filters, Large Water Rations"
    elif terrain_id == 2: # T3: Dense Forest/Jungle
        weapon = "Short-Barrel Carbine/SMG, Grenade Launcher (40mm)"
        ancillary = "Machetes, Satellite GPS, NVGs"
    elif terrain_id == 3: # T4: Urban/Built-up
        weapon = "Compact Submachine Gun (SMG), Reflex/Holo Sight"
        ancillary = "Breaching Tools, Riot Shields"
    else: # Default/Catch-all
        weapon = "General Purpose Machine Gun (GPMG), Standard Rifles"
        ancillary = "Standard Comms, Generic Resupply Packs"

    # 2. Vehicle Decision based on Predicted Mobility
    if mobility_id == 0: # Easy
        vehicle = "High-Mobility Wheeled Vehicles (HMVs)"
    elif mobility_id == 1: # Medium
        vehicle = "Tracked Armored Personnel Carriers (APCs)"
    else: # Hard (2)
        vehicle = "Foot Patrol / Helicopter Insertion"

    # 3. Mission-Specific Adjustments
    if mission_type.upper() == "ASSAULT" and mobility_id < 2:
        vehicle = "Heavy Armored Vehicle (Tank/Heavy APC)"
        weapon += ", Dedicated Support Weapon"

    return {
        "Recommended Vehicle": vehicle,
        "Recommended Primary Weapon": weapon,
        "Special Equipment": ancillary
    }


# -----------------------------
# Dataset
# -----------------------------
class MultiTaskTerrainDataset(Dataset):
    """
    x: FloatTensor [C, H, W]
    y_terrain: LongTensor, y_mobility: LongTensor, y_access: LongTensor
    y_visibility: FloatTensor (scalar)
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        required_cols = ["filepath", "terrain_label", "mobility_label", "access_label", "visibility_score"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tile_path = row["filepath"]
        terrain_label = int(row["terrain_label"])
        mobility_label = int(row["mobility_label"])
        access_label = int(row["access_label"])
        visibility_score = float(row["visibility_score"])

        # Load tile: [C, H, W]
        x = np.load(tile_path).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)

        # 🛑 FIX: Apply fixed global normalization
        x_tensor = torch.from_numpy(x)
        
        # Reshape global stats to match tensor shape [C, 1, 1]
        global_means_tensor = torch.tensor(GLOBAL_MEANS, dtype=torch.float32).view(x.shape[0], 1, 1)
        global_stds_tensor = torch.tensor(GLOBAL_STDS, dtype=torch.float32).view(x.shape[0], 1, 1)

        # Perform channel-wise normalization
        x_tensor = (x_tensor - global_means_tensor) / global_stds_tensor 

        y_terrain = torch.tensor(terrain_label, dtype=torch.long)
        y_mobility = torch.tensor(mobility_label, dtype=torch.long)
        y_access = torch.tensor(access_label, dtype=torch.long)
        y_visibility = torch.tensor(visibility_score, dtype=torch.float32)

        return x_tensor, (y_terrain, y_mobility, y_access, y_visibility)


# -----------------------------
# Model
# -----------------------------
class MultiTaskTerrainNet(nn.Module):
    """
    Shared CNN backbone + 4 heads: terrain, mobility, access, visibility (regression)
    """

    def __init__(
        self,
        in_channels,
        num_terrain_classes,
        num_mobility_classes,
        num_access_classes,
    ):
        super().__init__()

        # Shared Feature Extractor (Backbone)
        self.features = nn.Sequential(
            # [C, 256, 256] -> [32, 128, 128]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # [32, 128, 128] -> [64, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # [64, 64, 64] -> [128, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # [128, 32, 32] -> [256, 16, 16]
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

        self.flatten_dim = 256 * 16 * 16

        # Shared fully-connected trunk
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Task-specific heads
        self.head_terrain = nn.Linear(512, num_terrain_classes)
        self.head_mobility = nn.Linear(512, num_mobility_classes)
        self.head_access = nn.Linear(512, num_access_classes)
        self.head_visibility = nn.Linear(512, 1)  # regression

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        shared = self.shared_fc(x)

        terrain_logits = self.head_terrain(shared)
        mobility_logits = self.head_mobility(shared)
        access_logits = self.head_access(shared)
        visibility_pred = self.head_visibility(shared).squeeze(1)  # [B]

        return {
            "terrain": terrain_logits,
            "mobility": mobility_logits,
            "access": access_logits,
            "visibility": visibility_pred,
        }


# -----------------------------
# Training & Evaluation (Unchanged)
# -----------------------------
def train_one_epoch(model, dataloader, criterions, optimizer, device):
    model.train()
    ce_terrain, ce_mobility, ce_access, mse_visibility = criterions
    running_loss, total_samples = 0.0, 0
    correct_terrain, correct_mobility, correct_access = 0, 0, 0
    
    iterable = dataloader
    if USE_TQDM: iterable = tqdm(dataloader, desc="Train", leave=False)

    for inputs, targets in iterable:
        y_terrain, y_mobility, y_access, y_visibility = targets

        inputs = inputs.to(device)
        y_terrain = y_terrain.to(device); y_mobility = y_mobility.to(device)
        y_access = y_access.to(device); y_visibility = y_visibility.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss_terrain = ce_terrain(outputs["terrain"], y_terrain)
        loss_mobility = ce_mobility(outputs["mobility"], y_mobility)
        loss_access = ce_access(outputs["access"], y_access)
        loss_visibility = mse_visibility(outputs["visibility"], y_visibility)

        loss = (W_TERRAIN * loss_terrain + W_MOBILITY * loss_mobility + W_ACCESS * loss_access + W_VISIBILITY * loss_visibility)

        loss.backward(); optimizer.step()

        batch_size = inputs.size(0); running_loss += loss.item() * batch_size
        total_samples += batch_size

        _, pred_terrain = outputs["terrain"].max(1)
        _, pred_mobility = outputs["mobility"].max(1)
        _, pred_access = outputs["access"].max(1)

        correct_terrain += pred_terrain.eq(y_terrain).sum().item()
        correct_mobility += pred_mobility.eq(y_mobility).sum().item()
        correct_access += pred_access.eq(y_access).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc_terrain = correct_terrain / total_samples if total_samples > 0 else 0.0
    acc_mobility = correct_mobility / total_samples if total_samples > 0 else 0.0
    acc_access = correct_access / total_samples if total_samples > 0 else 0.0

    return avg_loss, acc_terrain, acc_mobility, acc_access


def eval_one_epoch(model, dataloader, criterions, device):
    model.eval()
    ce_terrain, ce_mobility, ce_access, mse_visibility = criterions
    running_loss, total_samples = 0.0, 0
    correct_terrain, correct_mobility, correct_access = 0, 0, 0

    with torch.no_grad():
        iterable = dataloader
        if USE_TQDM: iterable = tqdm(dataloader, desc="Val", leave=False)

        for inputs, targets in iterable:
            y_terrain, y_mobility, y_access, y_visibility = targets

            inputs = inputs.to(device); y_terrain = y_terrain.to(device)
            y_mobility = y_mobility.to(device); y_access = y_access.to(device)
            y_visibility = y_visibility.to(device)

            outputs = model(inputs)

            loss_terrain = ce_terrain(outputs["terrain"], y_terrain)
            loss_mobility = ce_mobility(outputs["mobility"], y_mobility)
            loss_access = ce_access(outputs["access"], y_access)
            loss_visibility = mse_visibility(outputs["visibility"], y_visibility)

            loss = (W_TERRAIN * loss_terrain + W_MOBILITY * loss_mobility + W_ACCESS * loss_access + W_VISIBILITY * loss_visibility)

            batch_size = inputs.size(0); running_loss += loss.item() * batch_size
            total_samples += batch_size

            _, pred_terrain = outputs["terrain"].max(1)
            _, pred_mobility = outputs["mobility"].max(1)
            _, pred_access = outputs["access"].max(1)

            correct_terrain += pred_terrain.eq(y_terrain).sum().item()
            correct_mobility += pred_mobility.eq(y_mobility).sum().item()
            correct_access += pred_access.eq(y_access).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc_terrain = correct_terrain / total_samples if total_samples > 0 else 0.0
    acc_mobility = correct_mobility / total_samples if total_samples > 0 else 0.0
    acc_access = correct_access / total_samples if total_samples > 0 else 0.0

    return avg_loss, acc_terrain, acc_mobility, acc_access


# -----------------------------
# Main Execution
# -----------------------------
def main():
    print(f"--- Starting Multi-Task CNN Training ({DEVICE}) ---")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"CSV not found at {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)
    if df.empty: raise RuntimeError("CSV is empty.")

    # Infer input channels from first tile
    try:
        sample_tile = np.load(df["filepath"].iloc[0])
        in_channels = sample_tile.shape[0]
    except Exception as e:
        raise RuntimeError(f"Could not load sample tile or infer channels: {e}")

    # Dataset creation and splitting
    full_dataset = MultiTaskTerrainDataset(df)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print(f"Data: {len(full_dataset)} total samples.")
    print(f"Inferred input channels: {in_channels}. Target classes: T({NUM_TERRAIN_CLASSES}), M({NUM_MOBILITY_CLASSES}), A({NUM_ACCESS_CLASSES})")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / loss / optimizer
    model = MultiTaskTerrainNet(
        in_channels=in_channels,
        num_terrain_classes=NUM_TERRAIN_CLASSES,
        num_mobility_classes=NUM_MOBILITY_CLASSES,
        num_access_classes=NUM_ACCESS_CLASSES,
    ).to(DEVICE)

    ce_terrain = nn.CrossEntropyLoss(); ce_mobility = nn.CrossEntropyLoss()
    ce_access = nn.CrossEntropyLoss(); mse_visibility = nn.MSELoss()
    criterions = (ce_terrain, ce_mobility, ce_access, mse_visibility)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_model_path = DATA_DIR / "best_multitask_model.pth"

    print(f"\nTraining on {DEVICE} for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc_t, train_acc_m, train_acc_a = train_one_epoch(model, train_loader, criterions, optimizer, DEVICE)
        val_loss, val_acc_t, val_acc_m, val_acc_a = eval_one_epoch(model, val_loader, criterions, DEVICE)

        if epoch % PRINT_EVERY == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] "
                f"T Loss: {train_loss:.4f} | V Loss: {val_loss:.4f} | "
                f"T(acc) T/V: {train_acc_t:.3f}/{val_acc_t:.3f} | "
                f"M(acc) T/V: {train_acc_m:.3f}/{val_acc_m:.3f} | "
                f"A(acc) T/V: {train_acc_a:.3f}/{val_acc_a:.3f}"
            )

        # Save best by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "in_channels": in_channels,
                        "num_terrain_classes": NUM_TERRAIN_CLASSES,
                        "num_mobility_classes": NUM_MOBILITY_CLASSES,
                        "num_access_classes": NUM_ACCESS_CLASSES,
                        "epoch": epoch},
                        best_model_path)
            print(f"✅ Saved new best model (val_loss={best_val_loss:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best model stored at: {best_model_path}")
    
    # -----------------------------
    # 🎯 DSS Report Generation Example (Inference Simulation)
    # -----------------------------
    print("\n--- Simulating Final DSS Report ---")
    
    # Simulate inference inputs (using the last batch from the validation loader for a quick test)
    try:
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            
        # Get the first sample's predictions (index 0 of the batch)
        terrain_id = outputs['terrain'][0].argmax().item()
        mobility_id = outputs['mobility'][0].argmax().item()
        access_id = outputs['access'][0].argmax().item()
        visibility_score = outputs['visibility'][0].item()
        
        # Assume a mission type for the report
        MISSION = "Assault"
        
        # Apply DSS Logic
        logistics_report = get_weapons_vehicle_recommendation(terrain_id, mobility_id, MISSION)
        
        # Compile Report
        final_report = {
            "Status": "SIMULATION SUCCESS",
            "Mission": MISSION,
            "--- AI PREDICTIONS ---": "",
            "Terrain Classification": f"T{terrain_id+1}: {TERRAIN_MAP.get(terrain_id, {'name': 'Unknown'})['name']}",
            "Predicted Mobility": MOBILITY_MAP.get(mobility_id, 'N/A'),
            "Predicted Access": ACCESS_MAP.get(access_id, 'N/A'),
            "Predicted Visibility Score": f"{visibility_score:.2f} (0=Poor, 1=Excellent)",
            "--- LOGISTICS RECOMMENDATIONS ---": "",
            "Vehicle": logistics_report['Recommended Vehicle'],
            "Weaponry": logistics_report['Recommended Primary Weapon'],
            "Special Equipment": logistics_report['Special Equipment'],
        }
        
        print(json.dumps(final_report, indent=4))
        
    except StopIteration:
        print("Cannot run simulation: Validation loader is empty.")
    except Exception as e:
        print(f"An error occurred during final report simulation: {e}")


if __name__ == "__main__":
    main()