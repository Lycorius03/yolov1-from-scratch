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
        images, targets = [], []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return torch.stack(images, dim=0), targets

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)

    # 改这里
    WEIGHT_PATH = "runs/20260620_023630/best_model.pth"
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    debug_images, _ = next(iter(loader))
    debug_images = debug_images.to(DEVICE)

    with torch.no_grad():
        raw = model(debug_images)

    pred = raw.reshape(-1, 7, 7, 30)
    bbox = pred[..., :10]
    cls = pred[..., 10:]
    conf1 = pred[..., 4]
    conf2 = pred[..., 9]

    print("\n===== RAW OUTPUT =====")
    print(f"bbox (ch0-9)  mean={bbox.mean():.4f}  max={bbox.max():.4f}  min={bbox.min():.4f}")
    print(f"class (ch10-29) mean={cls.mean():.4f}  max={cls.max():.4f}  min={cls.min():.4f}")

    conf = torch.stack([conf1, conf2], dim=-1)
    print(f"\nconf1 (ch4) mean={conf1.mean():.4f}  max={conf1.max():.4f}  min={conf1.min():.4f}")
    print(f"conf2 (ch9) mean={conf2.mean():.4f}  max={conf2.max():.4f}  min={conf2.min():.4f}")
    print(f"both conf  mean={conf.mean():.4f}  max={conf.max():.4f}  min={conf.min():.4f}")

    cx1 = pred[..., 0]; cy1 = pred[..., 1]; w1 = pred[..., 2]; h1 = pred[..., 3]
    cx2 = pred[..., 5]; cy2 = pred[..., 6]; w2 = pred[..., 7]; h2 = pred[..., 8]
    all_w = torch.cat([w1.flatten(), w2.flatten()])
    all_h = torch.cat([h1.flatten(), h2.flatten()])
    print(f"\nw mean={all_w.mean():.4f}  w max={all_w.max():.4f}  w min={all_w.min():.4f}")
    print(f"h mean={all_h.mean():.4f}  h max={all_h.max():.4f}  h min={all_h.min():.4f}")

    print(f"\nclass max per cell mean={cls.max(dim=-1).values.mean():.4f}  max={cls.max():.4f}")
    print(f"class min per cell mean={cls.min(dim=-1).values.mean():.4f}  min={cls.min():.4f}")

    print("\n===== CONF DISTRIBUTION =====")
    total = conf.numel()
    for t in [0.1, 0.3, 0.5, 0.7]:
        n = (conf > t).sum().item()
        print(f"conf > {t}: {n} / {total} ({100*n/total:.1f}%)")

    print("\n===== DECODED =====")
    batch_results = predict_batch(model, debug_images, conf_threshold=0.01, iou_threshold=0.5)

    total_boxes = 0
    max_score = 0.0
    for i, boxes in enumerate(batch_results):
        num = boxes.shape[0]
        total_boxes += num
        print(f"图 {i}: {num} 框")
        if num > 0:
            s = boxes[:, 4]
            sc = boxes[:, 5]
            print(f"  score: {s.min().item():.4f} ~ {s.max().item():.4f}")
            print(f"  class_ids: {sc.unique().long().tolist()}")
            if s.max().item() > max_score:
                max_score = s.max().item()

    print(f"\n共 {total_boxes} 框, max score = {max_score:.4f}")
