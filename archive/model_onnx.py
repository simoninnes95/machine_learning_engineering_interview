# Just the ONNX Model Class and nothing else

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
