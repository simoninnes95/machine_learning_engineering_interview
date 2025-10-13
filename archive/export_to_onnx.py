import torch
import numpy as np 
import onnxruntime as ort
from torchvision.models import vit_b_16, ViT_B_16_Weights

def main():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights).eval()

    dummy = torch.randn(1, 3, 224, 224)  
    torch.onnx.export(
        model,
        dummy,
        "vit_b16.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    weights = ViT_B_16_Weights.DEFAULT
    pt = vit_b_16(weights=weights).eval()
    pre = weights.transforms()

    img = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        y_pt = pt(img).softmax(dim=1).numpy()

    sess = ort.InferenceSession("vit_b16.onnx", providers=[
        "CoreMLExecutionProvider", "CPUExecutionProvider"
    ])
    inp_name = sess.get_inputs()[0].name
    y_ort = sess.run(None, {inp_name: img.numpy()})[0]
    y_ort = torch.tensor(y_ort).softmax(dim=1).numpy()

    print("max abs diff:", float(np.max(np.abs(y_pt - y_ort))))

if __name__ == "__main__":
    main()
