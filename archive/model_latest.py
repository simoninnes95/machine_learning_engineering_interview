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

# Model class with initialization logic.
class ImageModel:
    
    def __init__(self, device: str | None = None, quantize: bool = False):

        # Use the new weights API (replaces deprecated pretrained=True)
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights).eval()

        # Use the exact transforms paired with these weights
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
        response = requests.get(image_url)
        pil_image = Image.open(BytesIO(response.content))
        print("[1/3] Downloaded and parsed image data: {}".format(pil_image))
        
        pil_images = [pil_image]  # Batch size of 1
        input_tensor = torch.cat([self.preprocessor(i).unsqueeze(0) for i in pil_images])
        print("[2/3] Images transformed, tensor shape {}".format(input_tensor.shape))
        
        # --- Timing start ---
        start = time.perf_counter()
        output_tensor = self.model(input_tensor)
        end = time.perf_counter()
        # --- Timing end ---

        elapsed_ms = (end - start) * 1000

        print("[3/3] Inference done in {}".format(round(elapsed_ms, 2)))
        print("DEVICE: {}".format(DEVICE))
        return {
            "class_index": int(torch.argmax(output_tensor[0]))
        }

# Create FastAPI app and model instance.
app = FastAPI()
USE_QUANT = False  # set True to test quantized path
DEVICE = ("mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu")

model_instance = ImageModel(device=DEVICE, quantize=USE_QUANT)

@app.get("/predict")
async def predict(image_url: str) -> Dict:
    # we call the model instance to get the prediction
    prediction = model_instance.predict(image_url)
    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)