import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from detect import predict_batch

def evaluate_map(model, loader, device='cuda', conf_threshold=0.4, iou_threshold=0.5):
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
          px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
          pred_dict = {
            'boxes': torch.stack([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], dim=1),
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
          x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
          target_dict = {
              'boxes': torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=1),
              'labels': target[:, :-4].argmax(dim=1).to(device)
          }
        all_targets.append(target_dict)

    metric.update(all_preds, all_targets)
  return metric.compute()['map'].item()


def evaluate_on_loader(model, loader, device="cuda", conf_threshold=0.4, iou_threshold=0.5):
    print("进行 mAP 评估...")
    return evaluate_map(model=model, loader=loader, device=device, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
