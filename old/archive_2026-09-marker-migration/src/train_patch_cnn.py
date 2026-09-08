import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2


class PatchDataset(Dataset):
    def __init__(self, root):
        self.imgs = sorted((root / "images").glob("*.png"))
        self.labels = sorted((root / "labels").glob("*.txt"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]), cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
        with open(self.labels[idx]) as f:
            label = int(f.read().split()[0])
        return img, torch.tensor(label, dtype=torch.long)


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


def train():
    dataset = PatchDataset(Path("data/patches"))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MiniCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        total = 0
        correct = 0
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (pred.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch + 1} – acc: {correct / total:.3f}")

    torch.save(model.state_dict(), "mini_impact_cnn.pt")
    print("[OK] Modèle sauvegardé")


if __name__ == "__main__":
    train()
