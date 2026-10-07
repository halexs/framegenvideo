import torch
import torch.nn as nn
from train_log.RIFE_HDv3 import Model

class RIFEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.flownet = model.flownet

    def forward(self, img0, img1):
        # Concatenate inputs along channel dimension (6 channels)
        imgs = torch.cat((img0, img1), dim=1)
        
        # Hardcode timestep to 0.5 (mid-point frame interpolation)
        # IFNet expects a scalar float/int or 1D tensor [1]
        timestep = 0.5
        scale_list = [8, 4, 2, 1]
        
        flow, mask, merged = self.flownet(imgs, timestep, scale_list)
        return merged[3]  # Return final frame prediction

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading RIFE model...")
    model = Model()
    model.load_model("./train_log", -1)
    model.eval()

    wrapper = RIFEWrapper(model).to(device)
    wrapper.eval()

    # Dummy inputs matching 704x1280 (divisible by 32)
    img0 = torch.randn(1, 3, 704, 1280, dtype=torch.float32, device=device)
    img1 = torch.randn(1, 3, 704, 1280, dtype=torch.float32, device=device)

    onnx_path = "rife_v3.onnx"
    print(f"Exporting ONNX to {onnx_path}...")

    torch.onnx.export(
        wrapper,
        (img0, img1),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["img0", "img1"],
        output_names=["output"],
        dynamic_axes={
            "img0": {0: "batch", 2: "height", 3: "width"},
            "img1": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}
        },
        dynamo=False
    )

    print("ONNX export completed successfully!")

if __name__ == "__main__":
    main()