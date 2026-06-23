"""独立评估 runs/20260622_224439 中每个 checkpoint 的 VOC07 mAP"""
import torch
import torchvision.transforms as transforms
from pathlib import Path

from models.yolov1 import YOLOv1
from dataset.voc_dataset import VOCDataset
from utils.map import evaluate_voc07_map
from config import PROJECT_ROOT, VOC2007_DIR

RUN_DIR = PROJECT_ROOT / "runs" / "20260622_224439"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "best_model.pth": "best_model",
    "best_map_model.pth": "best_map_model",
    "best_val_model.pth": "best_val_model",
}


def evaluate_checkpoint(ckpt_path, device=DEVICE):
    print(f"\n{'=' * 60}")
    print(f"评估: {ckpt_path.name}")
    print(f"{'=' * 60}")

    # 加载模型
    model = YOLOv1(S=7, B=2, C=20).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 构建 data loader (VOC2007 test)
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    dataset = VOCDataset(
        root_dirs=[str(VOC2007_DIR)],
        transform=transform,
        split='test',
        use_encoded_target=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch],
        ),
    )

    # 评估 VOC07 mAP
    mAP, aps = evaluate_voc07_map(
        model=model,
        loader=loader,
        device=device,
        conf_threshold=0.1,
        iou_threshold=0.5,
        num_classes=20,
    )

    print(f"VOC07 mAP@0.5: {mAP:.6f}")
    # 打印每个类别的 AP
    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]
    for name, ap in zip(class_names, aps):
        print(f"  {name:>15s}: {ap:.4f}")

    return mAP, aps


if __name__ == "__main__":
    print(f"设备: {DEVICE}")
    print(f"评估运行: {RUN_DIR.name}")

    results = {}
    for filename, label in CHECKPOINTS.items():
        ckpt_path = RUN_DIR / filename
        if not ckpt_path.exists():
            print(f"跳过 {filename} — 文件不存在")
            continue
        mAP, aps = evaluate_checkpoint(ckpt_path, DEVICE)
        results[label] = {"mAP": mAP, "aps": aps}

    # 汇总
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    for label, r in results.items():
        print(f"{label:>20s}: VOC07 mAP@0.5 = {r['mAP']:.6f}")
