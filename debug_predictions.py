import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.yolov1 import YOLOv1
from dataset.voc_dataset import VOCDataset
from detect import predict_batch
from config import VOC2007_DIR, VOC2012_DIR

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    dataset = VOCDataset(
        root_dirs=[str(VOC2007_DIR), str(VOC2012_DIR)],
        transform=transform,
        split='val',
        use_encoded_target=False
    )

    def collate_fn(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        images = torch.stack(images, dim=0)
        return images, targets

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)

    # 改这里
    WEIGHT_PATH = "runs/20260619_213313/best_model.pth"
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    debug_images, _ = next(iter(loader))
    debug_images = debug_images.to(DEVICE)

    batch_results = predict_batch(model, debug_images, conf_threshold=0.01, iou_threshold=0.5)

    total_boxes = 0
    max_score = 0.0
    for i, pred in enumerate(batch_results):
        num = pred.shape[0]
        total_boxes += num
        print(f"图 {i}: {num} 框")
        if num > 0:
            s = pred[:, 4]
            print(f"  score: {s.min().item():.4f} ~ {s.max().item():.4f}")
            if s.max().item() > max_score:
                max_score = s.max().item()

    print(f"\n共 {total_boxes} 框, max score = {max_score:.4f}")
