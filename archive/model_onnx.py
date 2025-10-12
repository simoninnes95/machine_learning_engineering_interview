# This file is mostly a copy of model.py but using ONNX Runtime instead of PyTorch
import urllib.parse
import asyncio
import httpx

from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from typing import Dict, List
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import onnxruntime as ort
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.ao.quantization import quantize_dynamic

# Used for testing the quantized model on M2 ARM Macbook
torch.backends.quantized.engine = "qnnpack"


# -----------------------
# Tuning knobs
# -----------------------
QUEUE_MAX = 32          # max requests per batch
FLUSH_MS = 20           # flush interval (milliseconds)
CACHE_SIZE = 512        # max cached outputs (LRU)
HTTP_TIMEOUT = 5.0      # seconds per image fetch

DEVICE = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"


class LRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._od = OrderedDict()

    def get(self, key):
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None

    def set(self, key, value):
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)

class ONNXImageModel:
    def __init__(self, onnx_path: str, prefer_coreml: bool = True):
        self.weights = ViT_B_16_Weights.DEFAULT
        self.preprocessor = self.weights.transforms()

        available = ort.get_available_providers()
        wanted = ["CoreMLExecutionProvider", "CPUExecutionProvider"] if prefer_coreml else ["CPUExecutionProvider"]
        self.providers = [p for p in wanted if p in available]
        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name

        self.device = f"onnxruntime[{','.join(self.providers)}]"


    def preprocess_batch(self, pil_images):
        tensors = []
        valid_idx = []
        for i, pil in enumerate(pil_images):
            if pil is None:
                continue
            try:
                t = self.preprocessor(pil).unsqueeze(0)  # [1,3,224,224]
                tensors.append(t)
                valid_idx.append(i)
            except Exception:
                pass
        if tensors:
            batch = torch.cat(tensors, dim=0)  # [B,3,224,224], CPU float32
            return batch, valid_idx
        return None, []

    def infer(self, batch_tensor: torch.Tensor):
        logits = self.session.run(None, {self.input_name: batch_tensor.numpy()})[0]  # [B,num_classes]
        return torch.from_numpy(logits)


class ImageModel:
    def __init__(self, device: str | None = None, quantize: bool = False):
        # Use the new weights API (replaces deprecated pretrained=True)
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights).eval()

        # Use the exact transforms paired with these weights
        self.preprocessor = weights.transforms()

        # quantization must live on CPU for M2 ARM Macbook
        if quantize:
            self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            device = "cpu"

        chosen = (
            device
            if device is not None
            else (
                "mps"
                if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.device = torch.device(chosen)
        self.model.to(self.device)

    def to(self, device: str):
        if any(
            p.is_quantized if hasattr(p, "is_quantized") else False
            for p in self.model.parameters(recurse=True)
        ):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        return self


    def infer(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # batch_tensor: [B, 3, 224, 224]
        with torch.inference_mode():
            return self.model(batch_tensor)



class RequestItem:
    def __init__(self, image_url: str):
        self.image_url = image_url
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()


class Batcher:
    def __init__(self, model: ImageModel):
        self.model = model
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue()
        self.cache = LRUCache(CACHE_SIZE)
        self.client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True)
        self._task = None
        self._flush_ms = FLUSH_MS
        self._queue_max = QUEUE_MAX

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def shutdown(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()

    async def enqueue(self, image_url: str) -> Dict:
        cached = self.cache.get(image_url)
        if cached is not None:
            return cached

        item = RequestItem(image_url)
        await self.queue.put(item)
        return await item.future  

    async def _run(self):
        # background loop that flushes every FLUSH_MS or when QUEUE_MAX reached
        flush_interval = self._flush_ms / 1000.0
        while True:
            try:
                first = await self.queue.get()
                batch: List[RequestItem] = [first]

                # collect more until timeout or max size
                try:
                    # small timed loop to gather up to QUEUE_MAX
                    end_time = asyncio.get_event_loop().time() + flush_interval
                    while len(batch) < self._queue_max:
                        timeout = max(0, end_time - asyncio.get_event_loop().time())
                        if timeout == 0:
                            break
                        next_item = await asyncio.wait_for(
                            self.queue.get(), timeout=timeout
                        )
                        batch.append(next_item)
                except asyncio.TimeoutError:
                    pass  # time to flush

                # split into cache hits and misses so we only fetch/infer for misses
                misses = [
                    (i, it)
                    for i, it in enumerate(batch)
                    if self.cache.get(it.image_url) is None
                ]

                # if there are misses, fetch + preprocess + infer
                if misses:
                    urls = [it.image_url for _, it in misses]
                    pil_images = await self._fetch_images(
                        urls
                    )  # may include None for failures
                    tensors = []
                    valid_idx: List[int] = []

                    for (miss_idx, it), pil in zip(misses, pil_images):
                        if pil is None:
                            # fail this request
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(
                                        status_code=400, detail="Failed to fetch image"
                                    )
                                )
                            continue
                        try:
                            t = self.model.preprocessor(pil).unsqueeze(
                                0
                            )  # [1,3,224,224]
                            tensors.append(t)
                            valid_idx.append(miss_idx)
                        except Exception:
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(
                                        status_code=400, detail="Preprocessing failed"
                                    )
                                )

                    if tensors:
                        batch_tensor = torch.cat(tensors, dim=0)  # stay on CPU for ORT
                        logits = self.model.infer(batch_tensor)
                        preds = torch.argmax(logits, dim=1).tolist()

                        # write results for valid misses into cache
                        for bi, pred in zip(valid_idx, preds):
                            url = batch[bi].image_url
                            result = {"class_index": int(pred)}
                            self.cache.set(url, result)

                # Now respond to ALL items (hits use cache; misses just populated)
                for it in batch:
                    # if already errored, skip
                    if it.future.done():
                        continue
                    result = self.cache.get(it.image_url)
                    if result is None:
                        # if still missing, treat as failure
                        it.future.set_exception(
                            HTTPException(
                                status_code=500, detail="Unknown inference error"
                            )
                        )
                    else:
                        it.future.set_result(result)

            except Exception as e:
                # Fail-safe: in case of unexpected error, try not to poison the loop
                # Drain one item if present and fail it visibly
                try:
                    it = self.queue.get_nowait()
                    if not it.future.done():
                        it.future.set_exception(
                            HTTPException(status_code=500, detail=str(e))
                        )
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

    async def _fetch_images(self, urls: List[str]) -> List[Image.Image | None]:
        async def fetch_one(u: str):
            try:
                # guard against unescaped URLs
                u = urllib.parse.unquote(u)
                resp = await self.client.get(u)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                return None

        tasks = [fetch_one(u) for u in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)



app = FastAPI()
model_instance = ONNXImageModel("vit_b16.onnx", prefer_coreml=True)
batcher = Batcher(model_instance)


@app.on_event("startup")
async def _startup():
    await batcher.start()


@app.on_event("shutdown")
async def _shutdown():
    await batcher.shutdown()


@app.get("/device-check")
async def health():
    return {"ok": True, "device": str(model_instance.device)}


@app.get("/predict")
async def predict(image_url: str) -> Dict:
    """
    Enqueue request; micro-batcher will group it with neighbors
    and return a per-request result.
    """
    return await batcher.enqueue(image_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)