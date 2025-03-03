from fastapi import FastAPI, Request
from io import BytesIO
from PIL import Image
from typing import Dict
import requests
import urllib.parse

import torch
from torchvision import transforms
from torchvision.models import vit_b_16

# Model class with initialization logic.
class ImageModel:
    def __init__(self):
        self.model = vit_b_16(pretrained=True).eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image_url: str) -> Dict:
        response = requests.get(image_url)
        pil_image = Image.open(BytesIO(response.content))
        print("[1/3] Downloaded and parsed image data: {}".format(pil_image))
        
        pil_images = [pil_image]  # Batch size of 1
        input_tensor = torch.cat([self.preprocessor(i).unsqueeze(0) for i in pil_images])
        print("[2/3] Images transformed, tensor shape {}".format(input_tensor.shape))
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        print("[3/3] Inference done!")
        return {"class_index": int(torch.argmax(output_tensor[0]))}

# Create FastAPI app and model instance.
app = FastAPI()
model_instance = ImageModel()

@app.get("/predict")
async def predict(image_url: str) -> Dict:
    # we call the model instance to get the prediction
    prediction = model_instance.predict(image_url)
    return prediction


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)