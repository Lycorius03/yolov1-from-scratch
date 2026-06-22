import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from detect import predict_batch
from utils.iou import compute_iou

def _xywh_to_xyxy(boxes):
  x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  return torch.stack([
    (x - w / 2).clamp(0, 1),
    (y - h / 2).clamp(0, 1),
    (x + w / 2).clamp(0, 1),
    (y + h / 2).clamp(0, 1)
  ], dim=1)

def evaluate_map(model, loader, device='cuda', conf_threshold=0.1, iou_threshold=0.5):
  model.eval()
  model.to(device)

  #torchmetrics要求iou_threshold必须是list  
  metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])
  metric.to(device)

  all_preds = []
  all_targets = []
  
  with torch.no_grad():
    for images, targets in loader:
    
      images = images.to(device)
      batch_results = predict_batch(model, images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

      for pred, target in zip(batch_results, targets):
        
        #处理预测标签(pred)的格式 xywh -> xyxy
        if pred.shape[0] == 0:
          pred_dict = {
            'boxes': torch.zeros((0, 4), device=device),
            'scores': torch.zeros((0,), device=device),
            'labels': torch.zeros((0,), dtype=torch.long, device=device)
          }
        else:
          pred_dict = {
            'boxes': _xywh_to_xyxy(pred[:, :4]),
            'scores': pred[:, 4],
            'labels': pred[:, 5].long()
          }
        all_preds.append(pred_dict)
  
        #处理真实标签(pred)的格式 xywh -> xyxy
        if len(target) == 0: 
          target_dict = {
              'boxes': torch.zeros((0, 4), device=device),
              'labels': torch.zeros((0,), dtype=torch.long, device=device)
          }
        else:
          boxes_xywh = target[:, -4:].to(device)
          target_dict = {
              'boxes': _xywh_to_xyxy(boxes_xywh),
              'labels': target[:, :-4].argmax(dim=1).to(device)
          }
        all_targets.append(target_dict)

    metric.update(all_preds, all_targets)
  return metric.compute()['map'].item()


def _voc_ap(rec, prec):
  ap = 0.0
  for t in torch.arange(0.0, 1.1, 0.1, device=rec.device):
    if torch.sum(rec >= t) == 0:
      p = 0
    else:
      p = torch.max(prec[rec >= t])
    ap += p / 11.0
  return ap.item() if hasattr(ap, "item") else float(ap)


def evaluate_voc07_map(model, loader, device='cuda', conf_threshold=0.1, iou_threshold=0.5, num_classes=20):
  model.eval()
  model.to(device)

  preds = [[] for _ in range(num_classes)]
  targets = {}
  image_idx = 0

  with torch.no_grad():
    for images, batch_targets in loader:
      images = images.to(device)
      batch_results = predict_batch(model, images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

      for pred, target in zip(batch_results, batch_targets):
        gt_boxes = _xywh_to_xyxy(target[:, -4:].to(device)) if len(target) else torch.zeros((0, 4), device=device)
        gt_labels = target[:, :-4].argmax(dim=1).to(device) if len(target) else torch.zeros((0,), dtype=torch.long, device=device)
        targets[image_idx] = {
          "boxes": gt_boxes,
          "labels": gt_labels,
          "matched": torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=device)
        }

        if pred.shape[0] > 0:
          pred_boxes = _xywh_to_xyxy(pred[:, :4])
          pred_scores = pred[:, 4]
          pred_labels = pred[:, 5].long()
          for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            preds[int(label.item())].append((image_idx, float(score.item()), box))

        image_idx += 1

  aps = []
  for cls_id in range(num_classes):
    cls_preds = sorted(preds[cls_id], key=lambda x: x[1], reverse=True)
    npos = sum(int((t["labels"] == cls_id).sum().item()) for t in targets.values())
    if npos == 0:
      aps.append(0.0)
      continue

    tp = torch.zeros((len(cls_preds),), device=device)
    fp = torch.zeros((len(cls_preds),), device=device)

    for i, (img_id, _, box) in enumerate(cls_preds):
      gt = targets[img_id]
      cls_mask = gt["labels"] == cls_id
      gt_boxes = gt["boxes"][cls_mask]
      matched_ids = torch.where(cls_mask)[0]

      if gt_boxes.shape[0] == 0:
        fp[i] = 1
        continue

      ious = compute_iou(
        torch.stack([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]).unsqueeze(0),
        torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2, gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]], dim=1)
      )
      best_iou, best_pos = torch.max(ious, dim=0)
      match_id = matched_ids[int(best_pos.item())]

      if best_iou.item() >= iou_threshold and not gt["matched"][match_id]:
        tp[i] = 1
        gt["matched"][match_id] = True
      else:
        fp[i] = 1

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    rec = tp_cum / max(float(npos), 1.0)
    prec = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-8)
    aps.append(_voc_ap(rec, prec))

  return sum(aps) / len(aps), aps


def evaluate_on_loader(model, loader, device="cuda", conf_threshold=0.1, iou_threshold=0.5):
    print("进行 mAP 评估...")
    return evaluate_map(model=model, loader=loader, device=device, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
