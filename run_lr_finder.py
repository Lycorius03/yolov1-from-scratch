import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset.voc_dataset import VOCDataset
from models.yolov1 import YOLOv1
from loss.yolo_loss import YoloLoss
from utils.lr_finder import lr_finder

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    dataset = VOCDataset(
        root_dir="data/VOCdevkit/VOC2007",
        transform=transform,
        split='train'
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    loss_fn = YoloLoss(S=7, B=2, C=20).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=1e-7, momentum=0.9, 
        weight_decay=0.0005
        )

    lr_finder(model, loader, optimizer, loss_fn, DEVICE, start_lr=1e-6, end_lr=1e-1, num_iter=len(loader))