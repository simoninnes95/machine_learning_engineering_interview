import urllib.parse
import asyncio
import httpx
import time

from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Response
from typing import Dict, List
from io import BytesIO
from PIL import Image
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.ao.quantization import quantize_dynamic

torch.backends.quantized.engine = "qnnpack"

# -----------------------
# Tuning knobs
# -----------------------
QUEUE_MAX = 32   # max requests per batch
FLUSH_MS = 20    # flush interval (milliseconds)
CACHE_SIZE = 512 # max cached outputs (LRU)
HTTP_TIMEOUT = 5.0  # seconds per image fetch

# DEVICE = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
DEVICE = "cpu"  # quantized model must run on CPU

# -----------------------
# FastAPI app & Prometheus metrics
# -----------------------
app = FastAPI()

# Counters & histograms
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions returned by model")
RESPONSES_TOTAL   = Counter("responses_total", "Total responses returned by service")
CACHE_HITS        = Counter("cache_hits_total", "Cache hits")
CACHE_MISSES      = Counter("cache_misses_total", "Cache misses")
FETCH_ERRORS      = Counter("fetch_errors_total", "Image fetch failures")
PREPROC_ERRORS    = Counter("preprocess_errors_total", "Preprocessing failures")

BATCH_SIZE = Histogram(
    "batch_size",
    "Observed micro-batch sizes",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)
INFER_LATENCY = Histogram(
    "infer_latency_seconds",
    "Model forward latency per batch",
)

# Knob gauges (useful when you sweep configs)
QUEUE_MAX_G = Gauge("queue_max", "Configured micro-batch max size")
FLUSH_MS_G  = Gauge("flush_ms", "Configured flush interval ms")
QUEUE_MAX_G.set(QUEUE_MAX)
FLUSH_MS_G.set(FLUSH_MS)


# -----------------------
# Simple LRU cache (by URL)
# -----------------------
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

# -----------------------
# Model wrapper
# -----------------------
class ImageModel:
    def __init__(self, device: str | None = None, quantize: bool = False):
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights).eval()
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
        # batch_tensor: [B, 3, 224, 224] on self.device
        with torch.inference_mode():
            return self.model(batch_tensor)

# -----------------------
# Batching infrastructure
# -----------------------
class RequestItem:
    def __init__(self, image_url: str):
        self.image_url = image_url
        loop = asyncio.get_running_loop()
        self.future: asyncio.Future = loop.create_future()

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
        # Cache hit? Return immediately
        cached = self.cache.get(image_url)
        if cached is not None:
            CACHE_HITS.inc()
            return cached

        item = RequestItem(image_url)
        await self.queue.put(item)
        return await item.future  # wait for batch result

    async def _run(self):
        # background loop that flushes every FLUSH_MS or when QUEUE_MAX reached
        flush_interval = self._flush_ms / 1000.0
        loop = asyncio.get_running_loop()
        while True:
            try:
                first = await self.queue.get()
                batch: List[RequestItem] = [first]

                # collect more until timeout or max size
                try:
                    end_time = loop.time() + flush_interval
                    while len(batch) < self._queue_max:
                        timeout = max(0, end_time - loop.time())
                        if timeout == 0:
                            break
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(next_item)
                except asyncio.TimeoutError:
                    pass  # time to flush

                # split into cache hits and misses so we only fetch/infer for misses
                misses = [
                    (i, it)
                    for i, it in enumerate(batch)
                    if self.cache.get(it.image_url) is None
                ]
                # count misses at batch time (one increment per request that needs inference)
                if misses:
                    CACHE_MISSES.inc(len(misses))

                # fetch + preprocess + infer for misses
                if misses:
                    urls = [it.image_url for _, it in misses]
                    pil_images = await self._fetch_images(urls)  # may include None for failures
                    tensors = []
                    valid_idx: List[int] = []

                    for (miss_idx, it), pil in zip(misses, pil_images):
                        if pil is None:
                            # fetch failed
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(status_code=400, detail="Failed to fetch image")
                                )
                            continue
                        try:
                            t = self.model.preprocessor(pil).unsqueeze(0)  # [1,3,224,224]
                            tensors.append(t)
                            valid_idx.append(miss_idx)
                        except Exception:
                            PREPROC_ERRORS.inc()
                            if not it.future.done():
                                it.future.set_exception(
                                    HTTPException(status_code=400, detail="Preprocessing failed")
                                )

                    if tensors:
                        # N actually inferred (exclude fetch/preproc failures)
                        N = len(tensors)

                        batch_tensor = torch.cat(tensors, dim=0).to(self.model.device)  # [N,3,224,224]

                        t0 = time.perf_counter()
                        logits = self.model.infer(batch_tensor)  # [N,num_classes]
                        dt = time.perf_counter() - t0

                        # --- Prometheus observations ---
                        INFER_LATENCY.observe(dt)  # seconds for this batch forward
                        BATCH_SIZE.observe(N)      # micro-batch size actually inferred
                        PREDICTIONS_TOTAL.inc(N)   # true QPS comes from rate() of this

                        preds = torch.argmax(logits, dim=1).tolist()

                        # write results for valid misses into cache
                        for bi, pred in zip(valid_idx, preds):
                            url = batch[bi].image_url
                            result = {"class_index": int(pred)}
                            self.cache.set(url, result)

                # respond to ALL items (hits use cache; misses just populated)
                for it in batch:
                    if it.future.done():
                        continue
                    result = self.cache.get(it.image_url)
                    if result is None:
                        it.future.set_exception(
                            HTTPException(status_code=500, detail="Unknown inference error")
                        )
                    else:
                        it.future.set_result(result)

                RESPONSES_TOTAL.inc(len(batch))  # all responses (hits + misses)

            except Exception as e:
                # Fail-safe: try not to poison the loop
                try:
                    it = self.queue.get_nowait()
                    if not it.future.done():
                        it.future.set_exception(HTTPException(status_code=500, detail=str(e)))
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

    async def _fetch_images(self, urls: List[str]) -> List[Image.Image | None]:
        async def fetch_one(u: str):
            try:
                u = urllib.parse.unquote(u)  # guard against unescaped URLs
                resp = await self.client.get(u)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                FETCH_ERRORS.inc()
                return None

        tasks = [fetch_one(u) for u in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

# -----------------------
# App wiring
# -----------------------
model_instance = ImageModel(device=DEVICE, quantize=False)
batcher = Batcher(model_instance)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.on_event("startup")
async def _startup():
    await batcher.start()

@app.on_event("shutdown")
async def _shutdown():
    await batcher.shutdown()

@app.get("/health")
async def health():
    return {"ok": True, "device": str(model_instance.device)}

@app.get("/predict")
async def predict(image_url: str) -> Dict:
    """Enqueue request; micro-batcher will group it with neighbors and return a per-request result."""
    return await batcher.enqueue(image_url)

if __name__ == "__main__":
    import uvicorn
    # Note: use multiple workers/processes only if you also isolate model per-worker (Prometheus multiprocess!)
    uvicorn.run(app, host="0.0.0.0", port=8001)
