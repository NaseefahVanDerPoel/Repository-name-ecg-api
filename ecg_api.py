"""
ecg_api.py — FastAPI inference server for the custom ECG_CNN model

IMPORTANT: this must define the exact same architecture (ConvBlock + ECG_CNN)
used during training, or torch.load's state_dict won't match.
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    r"C:\Users\Admin\OneDrive\Desktop\ecg images\ECG_Project\models\ecg-api-deployment"
)

CHECKPOINT_PATH = "ecg_cnn_best.pth"

IMG_SIZE = 224   # overwritten below by checkpoint's saved img_size if present

# Overwritten below by checkpoint's saved class_names if present.
# Kept here only as a fallback matching training order: index 0 = NORM, index 1 = NON-NORM.
CLASSES = ["NORM", "NON-NORM"]


# ============================================================
# MODEL ARCHITECTURE — must exactly match training
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ECG_CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True, dropout=0.1),
            ConvBlock(32, 64, pool=True, dropout=0.1),
            ConvBlock(64, 128, pool=True, dropout=0.2),
            ConvBlock(128, 256, pool=True, dropout=0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ECG model...")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "class_names" in checkpoint:
        CLASSES = checkpoint["class_names"]

    if "img_size" in checkpoint:
        IMG_SIZE = checkpoint["img_size"]

    # Build the SAME architecture the checkpoint was trained with
    model = ECG_CNN(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("ECG model loaded successfully.")
    print(f"Classes: {CLASSES}")
    print(f"Image size: {IMG_SIZE}")

except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

# ============================================================
# IMAGE PREPROCESSING — must match eval_transforms from training
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FASTAPI APPLICATION
# ============================================================

ecg_api = FastAPI(
    title="ECG Analysis API",
    description="AI-powered ECG anomaly detection API",
    version="1.0.0"
)

ecg_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@ecg_api.get("/")
async def root():
    return {"message": "ECG Analysis API is running"}


@ecg_api.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "classes": CLASSES,
    }


# ============================================================
# ECG PREDICTION
# ============================================================

@ecg_api.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No ECG image was provided.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid ECG image.")

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(dim=1).item()

        prediction = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

        return {
            "success": True,
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"ECG prediction failed: {str(e)}")
