# model.py
from fastapi import FastAPI
from io import BytesIO
from PIL import Image
from typing import Dict
import requests
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.ao.quantization import quantize_dynamic
import time

torch.backends.quantized.engine = "qnnpack"

class ImageModel:
    def __init__(self, device: str | None = None, quantize: bool = False):
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights).eval()
        self.preprocessor = weights.transforms()

        # quantization must live on CPU
        if quantize:
            self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            device = "cpu"

        chosen = (
            device if device is not None
            else ("mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
                  else "cuda" if torch.cuda.is_available()
                  else "cpu")
        )
        self.device = torch.device(chosen)
        self.model.to(self.device)

    def to(self, device: str):
        # NOTE: if model was quantized, device must remain CPU
        if any(p.is_quantized if hasattr(p, "is_quantized") else False for p in self.model.parameters(recurse=True)):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.inference_mode()
    def predict(self, image_url: str) -> Dict:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
        pil_image = Image.open(BytesIO(r.content)).convert("RGB")

        x = self.preprocessor(pil_image).unsqueeze(0).to(self.device)

        # --- Timing start ---
        start = time.perf_counter()
        y = self.model(x)
        end = time.perf_counter()
        # --- Timing end ---

        elapsed_ms = (end - start) * 1000
        return {
            "class_index": int(torch.argmax(y[0])),
            "inference_ms": round(elapsed_ms, 2)
        }

app = FastAPI()

# --- Toggle these for local runs ---
USE_QUANT = False  # set True to test quantized path
DEVICE = ("mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu")
# -----------------------------------

model_instance = ImageModel(device=DEVICE, quantize=USE_QUANT)

@app.get("/health")
async def health():
    return {"ok": True, "device": str(model_instance.device), "quantized": USE_QUANT}

@app.get("/predict")
async def predict(image_url: str) -> Dict:
    return model_instance.predict(image_url)

if __name__ == "__main__":
    import uvicorn
    # run locally on a port that doesn’t collide with your k8s env
    uvicorn.run(app, host="0.0.0.0", port=8001)
