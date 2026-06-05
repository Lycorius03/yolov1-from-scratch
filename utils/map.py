import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate_map(model, loader, device, decode_fn, conf_threshold=0.4, iou_threshold=0.5):

    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            predictions = model(images)

            # 解码预测框
            pred_boxes_list = decode_fn(predictions, conf_threshold=conf_threshold)

            preds = []
            for boxes in pred_boxes_list:
                if len(boxes) == 0:
                    preds.append({
                        "boxes": torch.zeros((0, 4)),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.long)
                    })
                else:
                    # xywh → xyxy
                    xy = boxes[:, :2]
                    wh = boxes[:, 2:4]
                    boxes_xyxy = torch.cat([xy - wh/2, xy + wh/2], dim=1)
                    preds.append({
                        "boxes": boxes_xyxy,
                        "scores": boxes[:, 4],
                        "labels": boxes[:, 5].long()
                    })

            # 解码target
            targets = []
            for i in range(len(images)):
                obj_mask = targets[i, :, :, 4] == 1.0
                cells = obj_mask.nonzero(as_tuple=False)
                gt_boxes = []
                gt_labels = []
                for row, col in cells:
                    cell = targets[i, row, col]
                    cx = (cell[0] + col) / 7
                    cy = (cell[1] + row) / 7
                    w = cell[2]
                    h = cell[3]
                    label = cell[10:].argmax().long()
                    gt_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
                    gt_labels.append(label)

                if gt_boxes:
                    targets.append({
                        "boxes": torch.tensor(gt_boxes),
                        "labels": torch.tensor(gt_labels)
                    })
                else:
                    targets.append({
                        "boxes": torch.zeros((0, 4)),
                        "labels": torch.zeros(0, dtype=torch.long)
                    })

            metric.update(preds, targets)

    result = metric.compute()
    return result["map"].item()