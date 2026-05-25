import torch
from torch import nn

ONNX_PATH = "mini_impact_cnn.onnx"


class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def export_onnx():
    model = MiniCNN()
    model.load_state_dict(torch.load("mini_impact_cnn.pt", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )

    print(f"[OK] Modèle exporté en ONNX → {ONNX_PATH}")


if __name__ == "__main__":
    export_onnx()
