import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.yolov1 import YOLOv1
from dataset.voc_dataset import VOCDataset
from detect import decode_predictions, CLASS_NAMES
from utils.nms import non_max_suppression
from utils.iou import compute_iou
from config import VOC2007_DIR, VOC2012_DIR

if __name__ == "__main__":
    DEVICE = "cpu"
    print(f"设备: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    dataset = VOCDataset(
        root_dirs=[str(VOC2007_DIR), str(VOC2012_DIR)],
        transform=transform,
        split='val',
        use_encoded_target=True
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, target = next(iter(loader))
    image = image.to(DEVICE)
    target = target.to(DEVICE)

    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    WEIGHT_PATH = "runs/20260620_141658/best_model.pth"
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        raw = model(image)

    pred = raw.reshape(1, 7, 7, 30)
    bbox1 = pred[..., 0:5]
    bbox2 = pred[..., 5:10]
    cls = pred[..., 10:]

    obj_mask = target[0, ..., 4] == 1.0
    obj_cells = obj_mask.nonzero(as_tuple=False)
    print(f"GT obj cells: {obj_cells.shape[0]}")
    for rc in obj_cells:
        r, c = rc[0].item(), rc[1].item()
        tx, ty, tw, th = target[0, r, c, 0:4].tolist()
        gt_label = target[0, r, c, 10:].argmax().item()
        px1, py1, pw1, ph1, pc1 = bbox1[0, r, c].tolist()
        px2, py2, pw2, ph2, pc2 = bbox2[0, r, c].tolist()
        print(f"  [{r},{c}] GT={CLASS_NAMES[gt_label]} ({tx:.3f},{ty:.3f},{tw:.3f},{th:.3f})")
        print(f"    bbox1: xywh=({px1:.3f},{py1:.3f},{pw1:.3f},{ph1:.3f}) conf={pc1:.4f}")
        print(f"    bbox2: xywh=({px2:.3f},{py2:.3f},{pw2:.3f},{ph2:.3f}) conf={pc2:.4f}")

    iou1 = compute_iou(bbox1[..., :4], target[..., 0:4])
    iou2 = compute_iou(bbox2[..., :4], target[..., 0:4])
    for rc in obj_cells:
        r, c = rc[0].item(), rc[1].item()
        print(f"  [{r},{c}] IoU1={iou1[0,r,c].item():.4f} IoU2={iou2[0,r,c].item():.4f}")

    print(f"\nconf1: mean={bbox1[...,4].mean():.4f} max={bbox1[...,4].max():.4f}")
    print(f"conf2: mean={bbox2[...,4].mean():.4f} max={bbox2[...,4].max():.4f}")
    print(f"class: max={cls.max():.4f}  max_per_cell_mean={cls.max(dim=-1).values.mean():.4f}")
    print(f"w: mean={torch.cat([bbox1[...,2],bbox2[...,2]]).mean():.4f} max={max(bbox1[...,2].max(),bbox2[...,2].max()):.4f}")
    print(f"h: mean={torch.cat([bbox1[...,3],bbox2[...,3]]).mean():.4f} max={max(bbox1[...,3].max(),bbox2[...,3].max()):.4f}")

    for th in [0.4, 0.2, 0.1, 0.05, 0.03, 0.01]:
        boxes = decode_predictions(raw, conf_threshold=th, device=DEVICE)[0]
        boxes = non_max_suppression(boxes, iou_threshold=0.5)
        n = boxes.shape[0]
        if n > 0:
            s = boxes[:, 4]
            print(f"\nconf_th={th}: {n}框, score {s.min():.4f}~{s.max():.4f}")
            for cid in boxes[:, 5].long().unique():
                idx = (boxes[:, 5].long() == cid).nonzero(as_tuple=True)[0]
                sc = s[idx]
                print(f"  {CLASS_NAMES[cid.item()]}: {len(idx)}框, score {sc.min():.4f}~{sc.max():.4f}")
        else:
            print(f"conf_th={th}: 0框")

    # 逐cell看conf×class最高分
    conf = torch.stack([bbox1[..., 4], bbox2[..., 4]], dim=-1)
    best_conf = conf.max(dim=-1).values
    best_class = cls.max(dim=-1).values
    score_matrix = best_conf * best_class
    top5 = score_matrix[0].flatten().topk(min(5, 49))
    print(f"\ntop5 cell scores (conf×class):")
    for val, idx in zip(top5.values, top5.indices):
        r, c = idx.item() // 7, idx.item() % 7
        b1c, b2c = bbox1[0, r, c, 4].item(), bbox2[0, r, c, 4].item()
        cl = cls[0, r, c].argmax().item()
        cp = cls[0, r, c, cl].item()
        print(f"  [{r},{c}] score={val:.4f}  bbox1_conf={b1c:.3f} bbox2_conf={b2c:.3f}  class={CLASS_NAMES[cl]}({cp:.3f})")
