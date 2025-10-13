import urllib.parse
import asyncio
import httpx
import os

from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from typing import Dict, List
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.ao.quantization import quantize_dynamic


torch.backends.quantized.engine = (
    "qnnpack"  # Used for testing the quantized model on M2 ARM Macbook
)
torch.set_num_threads(
    int(os.getenv("TORCH_NUM_THREADS", "2"))
)  # Used to match the number of CPU cores available from kubernetes
torch.set_num_interop_threads(1)


# -----------------------
# Tuning knobs
# -----------------------
QUEUE_MAX = 20          # max requests per batch
FLUSH_MS = 40           # flush interval (milliseconds)
CACHE_SIZE = 512        # max cached outputs (LRU)
HTTP_TIMEOUT = 5.0      # seconds per image fetch

DEVICE = (
    "mps"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cpu"
)
# DEVICE = "cpu"  # quantized model must run on CPU


class LRUCache:
    """A least-recently-used (LRU) cache implemented using OrderedDict.

    Stores up to a fixed capacity of key-value pairs and evicts the oldest item when full.
    """

    def __init__(self, capacity: int = 256):
        """Initialize the LRU cache.

        Args:
            capacity (int): Maximum number of entries the cache can hold.
        """
        self.capacity = capacity
        self._od = OrderedDict()

    def get(self, key):
        """Retrieve a value from the cache by key.

        Args:
            key: The lookup key.

        Returns:
            The cached value if found, otherwise None.
        """
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None

    def set(self, key, value):
        """Insert or update a key-value pair in the cache.

        Moves the item to the end to mark it as most recently used.
        If the cache exceeds capacity, the oldest item is evicted.

        Args:
            key: The key to insert.
            value: The value to store.
        """
        self._od[key] = value
        self._od.move_to_end(key)
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)


class ImageModel:
    """Wrapper around a pretrained Vision Transformer (ViT-B/16) model with preprocessing,
    device management, and optional quantization support.
    """

    def __init__(self, device: str | None = None, quantize: bool = False):
        """Initialize the model, preprocessing pipeline, and device placement.

        Args:
            device (str | None): Desired compute device ('cpu', 'mps', 'cuda'). If None, auto-detects.
            quantize (bool): Whether to apply dynamic quantization for CPU inference.
        """
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights).eval()
        self.preprocessor = weights.transforms()

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
        """Move the model to the specified device.

        Automatically forces CPU if model contains quantized parameters.

        Args:
            device (str): Target device name ('cpu', 'mps', 'cuda').

        Returns:
            ImageModel: The same instance moved to the new device.
        """
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
        """Run a forward inference pass on a batch of preprocessed images.

        Args:
            batch_tensor (torch.Tensor): A tensor of shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: The raw output logits from the model of shape [B, num_classes].
        """
        with torch.inference_mode():
            return self.model(batch_tensor)


class RequestItem:
    """Represents a single inference request containing an image URL and its associated future."""

    def __init__(self, image_url: str):
        """Initialize a RequestItem with its image URL and response placeholder.

        Args:
            image_url (str): The URL of the image to be processed.
        """
        self.image_url = image_url
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()


class Batcher:
    """Implements micro-batching for efficient model inference.

    Aggregates concurrent requests into batches, handles caching, and performs
    asynchronous image fetching and preprocessing before model inference.
    """

    def __init__(self, model: ImageModel):
        """Initialize the batcher with model, cache, and async client.

        Args:
            model (ImageModel): The image classification model used for inference.
        """
        self.model = model
        self.queue: asyncio.Queue[RequestItem] = asyncio.Queue()
        self.cache = LRUCache(CACHE_SIZE)
        self.client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True)
        self._task = None
        self._flush_ms = FLUSH_MS
        self._queue_max = QUEUE_MAX

    async def start(self):
        """Start the background batching loop."""
        self._task = asyncio.create_task(self._run())

    async def shutdown(self):
        """Cancel the background task and close HTTP connections gracefully."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()

    async def enqueue(self, image_url: str) -> Dict:
        """Add a new image URL to the inference queue.

        Checks for cached results first, otherwise enqueues the request and
        waits for the batch to complete.

        Args:
            image_url (str): URL of the image to be processed.

        Returns:
            Dict: Inference result containing the predicted class index.
        """
        cached = self.cache.get(image_url)
        if cached is not None:
            return cached

        item = RequestItem(image_url)
        await self.queue.put(item)
        return await item.future  # wait for batch result

    async def _run(self):
        """Continuously process queued requests in micro-batches.

        Collects items from the queue until either:
        - the max queue size is reached, or
        - the flush interval elapses.

        Then fetches images, preprocesses them, performs inference,
        caches results, and fulfills futures.
        """
        flush_interval = self._flush_ms / 1000.0
        while True:
            try:
                first = await self.queue.get()
                batch: List[RequestItem] = [first]

                try:
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
                    pass

                costly_processes = [
                    (i, it)
                    for i, it in enumerate(batch)
                    if self.cache.get(it.image_url) is None
                ]

                if costly_processes:
                    urls = [it.image_url for _, it in costly_processes]
                    pil_images = await self._fetch_images(urls)
                    tensors = []
                    valid_idx: List[int] = []

                    for (costly_idx, it), pil in zip(costly_processes, pil_images):
                        if pil is None:
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(
                                        status_code=400, detail="Failed to fetch image"
                                    )
                                )
                            continue
                        try:
                            t = self.model.preprocessor(pil).unsqueeze(0)
                            tensors.append(t)
                            valid_idx.append(costly_idx)
                        except Exception:
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(
                                        status_code=400, detail="Preprocessing failed"
                                    )
                                )

                    if tensors:
                        batch_tensor = torch.cat(tensors, dim=0).to(DEVICE)
                        logits = self.model.infer(batch_tensor)
                        preds = torch.argmax(logits, dim=1).tolist()

                        for bi, pred in zip(valid_idx, preds):
                            url = batch[bi].image_url
                            result = {"class_index": int(pred)}
                            self.cache.set(url, result)

                for it in batch:
                    if it.future.done():
                        continue
                    result = self.cache.get(it.image_url)
                    if result is None:
                        it.future.set_exception(
                            HTTPException(
                                status_code=500, detail="Unknown inference error"
                            )
                        )
                    else:
                        it.future.set_result(result)

            except Exception as e:
                try:
                    it = self.queue.get_nowait()
                    if not it.future.done():
                        it.future.set_exception(
                            HTTPException(status_code=500, detail=str(e))
                        )
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

    async def _fetch_images(self, urls: List[str]) -> List[Image.Image | None]:
        """Download a list of images asynchronously.

        Args:
            urls (List[str]): List of image URLs to download.

        Returns:
            List[Image.Image | None]: A list of PIL Images, or None for failed downloads.
        """

        async def fetch_one(u: str):
            try:
                u = urllib.parse.unquote(u)
                resp = await self.client.get(u)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                return None

        tasks = [fetch_one(u) for u in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


app = FastAPI()
model_instance = ImageModel(device=DEVICE, quantize=False)
batcher = Batcher(model_instance)


@app.on_event("startup")
async def _startup():
    """FastAPI startup hook to start the background batcher."""
    await batcher.start()


@app.on_event("shutdown")
async def _shutdown():
    """FastAPI shutdown hook to gracefully stop the batcher."""
    await batcher.shutdown()


@app.get("/device-check")
async def device_check():
    """Health check endpoint.

    Returns:
        Dict: Contains `"ok": True` and the active device name.
    """
    return {"ok": True, "device": str(model_instance.device)}


@app.get("/predict")
async def predict(image_url: str) -> Dict:
    """
    Enqueue an image URL for batched inference.

    The micro-batcher will group the request with others and return
    a per-request classification result.

    Args:
        image_url (str): The URL of the image to classify.

    Returns:
        Dict: A dictionary containing the predicted class index.
    """
    return await batcher.enqueue(image_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
