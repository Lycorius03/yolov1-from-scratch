"""Overfit test with per-step logging to CSV, on 3 random images."""
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys, csv, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset.voc_dataset import VOCDataset
from models.yolov1 import YOLOv1
from loss.yolo_loss import YoloLoss
from detect import decode_predictions, CLASS_NAMES
from utils.nms import non_max_suppression
from utils.iou import compute_iou
from config import VOC2007_DIR

S, B, C = 7, 2, 20
device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 500
N_TESTS = 3

transform = transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()])
ds = VOCDataset(root_dirs=[str(VOC2007_DIR)], transform=transform, split='val', use_encoded_target=True)

candidates = []
for idx in range(len(ds)):
    _, target = ds[idx]
    n_obj = (target[..., 4] == 1.0).sum().item()
    if 1 <= n_obj <= 3:
        candidates.append(idx)

random.seed(123)
selected = random.sample(candidates, N_TESTS)

for test_idx, ds_idx in enumerate(selected):
    single_ds = Subset(ds, [ds_idx])
    loader = DataLoader(single_ds, batch_size=1, shuffle=False)
    image, target = next(iter(loader))
    image = image.to(device)
    target = target.to(device)
    img_id = ds.image_ids[ds_idx][1]
    n_obj = (target[..., 4] == 1.0).sum().item()

    gt_info = []
    for rc in (target[0, ..., 4] == 1.0).nonzero(as_tuple=False):
        r, c = rc[0].item(), rc[1].item()
        cls_id = target[0, r, c, 10:].argmax().item()
        gt_info.append((r, c, CLASS_NAMES[cls_id]))

    model = YOLOv1(S=S, B=B, C=C).to(device)
    loss_fn = YoloLoss(S=S, B=B, C=C).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    model.train()

    csv_path = f"overfit_{img_id}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["step","loss","conf_max","n_pred","best_iou","pred_class"])

    print(f"\n{'='*60}")
    print(f"[{test_idx+1}/{N_TESTS}] {img_id}  GT={gt_info}  ({n_obj} obj)")
    print(f"{'='*60}")

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        pred = model(image)
        loss = loss_fn(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()

        # Full eval every 10 steps (also at step 1 and final)
        if step == 1 or step % 10 == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                p = pred.reshape(1, S, S, B * 5 + C)
                conf_max = max(p[..., 4].max().item(), p[..., 9].max().item())
                boxes = decode_predictions(pred, conf_threshold=0.01, device=device)[0]
                boxes = non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.01)

                best_ious = []
                pred_classes = []
                for rc in (target[0, ..., 4] == 1.0).nonzero(as_tuple=False):
                    r, c = rc[0].item(), rc[1].item()
                    gt_box = torch.tensor([
                        (target[0, r, c, 0].item() + c) / S,
                        (target[0, r, c, 1].item() + r) / S,
                        target[0, r, c, 2].item(),
                        target[0, r, c, 3].item(),
                    ], device=device)
                    best = 0.0
                    for pb in boxes:
                        pb_box = torch.tensor([pb[0].item(), pb[1].item(), pb[2].item(), pb[3].item()], device=device)
                        best = max(best, compute_iou(pb_box, gt_box).item())
                    best_ious.append(best)
                for pb in boxes:
                    pred_classes.append(CLASS_NAMES[int(pb[5].item())])

            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([step, f"{loss.item():.4f}", f"{conf_max:.4f}", boxes.shape[0],
                           f"{max(best_ious) if best_ious else 0:.4f}", str(pred_classes)])

            # Print key milestones
            if step == 1:
                print(f"  step {step:4d}: loss={loss.item():.4f}  conf={conf_max:.4f}  preds={boxes.shape[0]}")
            elif step <= 100 and step % 20 == 0:
                iou_str = f"{max(best_ious):.4f}" if best_ious else "0"
                print(f"  step {step:4d}: loss={loss.item():.4f}  conf={conf_max:.4f}  preds={boxes.shape[0]}  iou={iou_str}  {pred_classes}")
            elif step == STEPS:
                iou_str = f"{max(best_ious):.4f}" if best_ious else "0"
                print(f"  step {step:4d}: loss={loss.item():.4f}  conf={conf_max:.4f}  preds={boxes.shape[0]}  iou={iou_str}  {pred_classes}")

            model.train()

    print(f"  CSV saved: {csv_path}")

print(f"\nDone. CSV files saved for all {N_TESTS} tests.")
