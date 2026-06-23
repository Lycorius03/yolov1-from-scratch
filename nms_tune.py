"""NMS 参数网格搜索 — 找到 conf_threshold × iou_threshold 的最佳组合

评估指标：
  1. 框/图 — 平均每张图检测框数（过多=假阳性泛滥，过少=漏检）
  2. 空图率 — 无检测框的图片占比（太高=阈值过严）
  3. VOC07 mAP@0.5 — 检测精度
  4. 综合评分 — 框数合理 + mAP 高 + 空图率低

输出：
  - 终端表格（各组合的指标对比）
  - 推荐参数
  - 几张采样图片的对比可视化
"""
import torch
import torchvision.transforms as transforms
from pathlib import Path
import itertools

from models.yolov1 import YOLOv1
from dataset.voc_dataset import VOCDataset
from detect import predict_batch, decode_predictions, CLASS_NAMES, CLASS_COLORS, FONT, visualize_predictions
from utils.nms import non_max_suppression
from utils.map import evaluate_voc07_map
from config import PROJECT_ROOT, VOC2007_DIR

RUN_DIR = PROJECT_ROOT / "runs" / "20260622_224439"
WEIGHT_PATH = RUN_DIR / "best_map_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 搜索范围
CONF_THRESHOLDS = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
IOU_THRESHOLDS  = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

# 评估用图片数量（全量 4952 太慢，采样 1024 张快速迭代）
EVAL_LIMIT = 1024


def grid_search(model, dataset, device=DEVICE):
    """遍历所有 conf × iou 组合，返回指标表"""
    results = []

    total_combos = len(CONF_THRESHOLDS) * len(IOU_THRESHOLDS)
    combo_idx = 0

    for conf_th, iou_th in itertools.product(CONF_THRESHOLDS, IOU_THRESHOLDS):
        combo_idx += 1
        print(f"[{combo_idx}/{total_combos}] conf={conf_th:.2f}  iou={iou_th:.2f} ...", end=" ", flush=True)

        # 用 predict_batch 的底层逻辑，但手动控制 conf/iou
        model.eval()
        total_boxes = 0
        empty_images = 0
        total_images = 0

        # 快速统计：框数和空图率
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, range(min(EVAL_LIMIT, len(dataset)))),
            batch_size=16, shuffle=False, num_workers=0,
            collate_fn=lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch]),
        )

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                predictions = model(images)
                boxes_list = decode_predictions(predictions, conf_threshold=conf_th, device=device)

                for boxes in boxes_list:
                    if boxes.shape[0] > 0:
                        boxes = non_max_suppression(boxes, iou_threshold=iou_th, conf_threshold=conf_th)
                    total_images += 1
                    n = boxes.shape[0]
                    total_boxes += n
                    if n == 0:
                        empty_images += 1

        avg_boxes = total_boxes / max(total_images, 1)
        empty_rate = empty_images / max(total_images, 1)

        # 满分制综合评分：
        # 框数在 1~8 之间为佳（太少=漏检，太多=假阳性），空图率 < 30%
        box_score = max(0, 1.0 - abs(avg_boxes - 4.0) / 6.0)  # 4 框/图最佳
        empty_penalty = max(0, 1.0 - empty_rate / 0.3)  # 空图率 0% 满分，30% 零分

        combo_score = box_score * 0.4 + empty_penalty * 0.6

        print(f"框/图={avg_boxes:.1f}  空图率={empty_rate:.1%}  综合={combo_score:.3f}")

        results.append({
            "conf": conf_th,
            "iou": iou_th,
            "avg_boxes": avg_boxes,
            "empty_rate": empty_rate,
            "combo_score": combo_score,
        })

    return results


def evaluate_top_combos(model, dataset, top_results, device=DEVICE, eval_limit=512):
    """对 top-K 组合跑完整 VOC07 mAP"""
    print(f"\n{'='*60}")
    print(f"对 Top 组合跑 VOC07 mAP 精确评估（{eval_limit} 张）...")
    print(f"{'='*60}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(min(eval_limit, len(dataset)))),
        batch_size=16, shuffle=False, num_workers=0,
        collate_fn=lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch]),
    )

    for r in top_results:
        conf_th, iou_th = r["conf"], r["iou"]
        print(f"\n评估 conf={conf_th:.2f} iou={iou_th:.2f} ...", end=" ", flush=True)

        # 注意：evaluate_voc07_map 内部调 predict_batch，后者用固定 conf/iou
        # 所以需要临时 patch
        import detect
        original_predict_batch = detect.predict_batch

        def patched_predict_batch(model, images, conf_threshold=conf_th, iou_threshold=iou_th):
            model.eval()
            device = images.device
            with torch.no_grad():
                predictions = model(images)
            boxes_list = decode_predictions(predictions, conf_threshold=conf_threshold, device=device)
            final_results = []
            for boxes in boxes_list:
                if boxes.shape[0] == 0:
                    final_results.append(torch.zeros((0, 6), device=device))
                    continue
                boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
                final_results.append(boxes)
            return final_results

        detect.predict_batch = patched_predict_batch
        try:
            mAP, aps = evaluate_voc07_map(
                model=model, loader=loader, device=device,
                conf_threshold=conf_th, iou_threshold=iou_th, num_classes=20,
            )
        finally:
            detect.predict_batch = original_predict_batch

        r["mAP"] = mAP
        # 综合评分 v2：mAP 权重
        r["final_score"] = r["combo_score"] * 0.3 + (mAP / 0.5) * 0.7
        print(f"mAP={mAP:.4f}  最终分={r['final_score']:.3f}")


def visualize_top_combos(model, dataset, top_results, device=DEVICE, num_samples=5):
    """对 top-3 组合做可视化对比"""
    from PIL import Image, ImageDraw

    indices = torch.randperm(min(200, len(dataset)))[:num_samples].tolist()

    for rank, r in enumerate(top_results[:3]):
        conf_th, iou_th = r["conf"], r["iou"]
        out_dir = Path(f"outputs/nms_tune/conf{conf_th:.2f}_iou{iou_th:.2f}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n可视化 rank{rank+1}: conf={conf_th:.2f} iou={iou_th:.2f}")

        model.eval()
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        for i, idx in enumerate(indices):
            image_tensor, _ = dataset[idx]
            img_pil = transforms.functional.to_pil_image(image_tensor)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = model(img_tensor)
            boxes_list = decode_predictions(predictions, conf_threshold=conf_th, device=device)
            boxes = non_max_suppression(boxes_list[0], iou_threshold=iou_th, conf_threshold=conf_th)

            draw = ImageDraw.Draw(img_pil)
            w_img, h_img = img_pil.size
            for box in boxes:
                x, y, w, h, score, class_id = box.tolist()
                x1 = (x - w / 2) * w_img
                y1 = (y - h / 2) * h_img
                x2 = (x + w / 2) * w_img
                y2 = (y + h / 2) * h_img
                color = CLASS_COLORS[int(class_id)]
                label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                tb = draw.textbbox((x1, y1 - 14), label, font=FONT)
                draw.rectangle([tb[0]-2, tb[1]-1, tb[2]+2, tb[3]+1], fill=color)
                draw.text((x1, y1 - 14), label, fill="white", font=FONT)

            img_pil.save(out_dir / f"sample_{i+1:02d}.jpg")

        print(f"  保存至 {out_dir}")


if __name__ == "__main__":
    print(f"设备: {DEVICE}")
    print(f"权重: {WEIGHT_PATH}")
    print(f"搜索范围: conf={CONF_THRESHOLDS}, iou={IOU_THRESHOLDS}")
    print(f"评估样本: {EVAL_LIMIT} 张")
    print(f"总组合数: {len(CONF_THRESHOLDS) * len(IOU_THRESHOLDS)}")
    print()

    # 加载模型
    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    model.load_state_dict(torch.load(str(WEIGHT_PATH), map_location=DEVICE))
    model.eval()

    # 构建 dataset (raw format for mAP)
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
    print(f"数据集: {len(dataset)} 张\n")

    # Phase 1: 网格搜索（框数 + 空图率）
    results = grid_search(model, dataset, DEVICE)

    # 排序
    results.sort(key=lambda x: x["combo_score"], reverse=True)

    print(f"\n{'='*60}")
    print("Top 10 组合（按综合评分）")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'conf':<7} {'iou':<7} {'框/图':<8} {'空图率':<9} {'综合分':<8}")
    print("-" * 44)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:<5} {r['conf']:<7.2f} {r['iou']:<7.2f} {r['avg_boxes']:<8.1f} {r['empty_rate']:<9.1%} {r['combo_score']:<8.3f}")

    # Phase 2: 对 Top 5 跑 mAP
    top5 = results[:5]
    evaluate_top_combos(model, dataset, top5, DEVICE, eval_limit=512)

    # 按 final_score 重排
    top5.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    print(f"\n{'='*60}")
    print("最终推荐排序（含 mAP）")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'conf':<7} {'iou':<7} {'框/图':<8} {'空图率':<9} {'mAP':<8} {'最终分':<8}")
    print("-" * 53)
    for i, r in enumerate(top5):
        print(f"{i+1:<5} {r['conf']:<7.2f} {r['iou']:<7.2f} {r['avg_boxes']:<8.1f} {r['empty_rate']:<9.1%} {r.get('mAP',0):<8.4f} {r.get('final_score',0):<8.3f}")

    # Phase 3: 可视化 Top 3
    print(f"\n{'='*60}")
    print("可视化 Top 3 组合")
    print(f"{'='*60}")
    visualize_top_combos(model, dataset, top5[:3], DEVICE, num_samples=5)

    # 最终推荐
    best = top5[0]
    print(f"\n{'='*60}")
    print(f"[BEST] conf_threshold={best['conf']:.2f}  iou_threshold={best['iou']:.2f}")
    print(f"   VOC07 mAP@0.5: {best.get('mAP',0):.4f}")
    print(f"   Avg boxes/img: {best['avg_boxes']:.1f}")
    print(f"   Empty rate: {best['empty_rate']:.1%}")
    print(f"{'='*60}")
