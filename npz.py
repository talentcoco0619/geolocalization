import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

from cvcities_base.model import TimmModel
from cvcities_base.train import Configuration

cfg = Configuration()

# Preprocessing
def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(cfg.device)

# Feature extraction
def extract_feature(model, img_tensor):
    with torch.no_grad():
        feat = model(img_tensor)
        return feat.cpu().numpy().squeeze(0)

# ⬇️ MAIN FUNCTION: CSV + weights path → feature DB
def build_satellite_feature_db_from_csv(csv_path, output_npz_path, model_weights_path, root_dir="Data"):
    print("🔧 Loading dataset...")

    columns = ["id", "lon", "lat", "city", "sat_path", "pano_path"]
    df = pd.read_csv(csv_path, names=columns)

    # Full image paths
    sat_paths = df['sat_path'].apply(lambda x: os.path.join(root_dir, x)).values
    latitudes = df['lat'].values.astype(np.float32)
    longitudes = df['lon'].values.astype(np.float32)
    gps = np.stack([latitudes, longitudes], axis=1)

    print("🔧 Loading model...")
    model = TimmModel(
        model_name=cfg.model,
        pretrained=cfg.pretrained,
        img_size=cfg.img_size,
        backbone_arch=cfg.backbone_arch,
        agg_arch=cfg.agg_arch,
        agg_config=cfg.agg_config,
        layer1=cfg.layer1
    )
    model.load_state_dict(torch.load(model_weights_path, map_location=cfg.device), strict=False)
    model.to(cfg.device)
    model.eval()

    print("📡 Extracting features...")
    features = []
    valid_img_paths = []  # to store only successfully processed image paths

    for path in tqdm(sat_paths, desc="Extracting satellite features"):
        try:
            tensor = preprocess(path)
            feat = extract_feature(model, tensor)
            features.append(feat)
            valid_img_paths.append(path)
        except Exception as e:
            print(f"❌ Failed for {path}: {e}")
            continue

    features = np.stack(features)
    # Save img_paths as np array of bytes (strings)
    valid_img_paths_np = np.array(valid_img_paths)

    np.savez_compressed(output_npz_path, features=features, gps=gps[:len(valid_img_paths)], img_paths=valid_img_paths_np)
    print(f"✅ Saved satellite feature DB to {output_npz_path}")

# 🧪 Optional: CLI-style entry point
if __name__ == "__main__":
    build_satellite_feature_db_from_csv(
        csv_path="Data/tokyo/img_info.csv",
        output_npz_path="./sat_feature_db.npz",
        model_weights_path="./heavy_files/weights_end.pth",
        root_dir="Data"
    )