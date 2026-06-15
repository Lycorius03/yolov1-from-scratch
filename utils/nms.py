import torch 
from  torchvision.ops import nms

def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
  if boxes.numel() == 0:
    return []
  
  #筛掉置信度低于conf_threshold的框
  keep_mask = boxes[..., 4] > conf_threshold
  boxes = boxes[keep_mask]

  if boxes.numel() == 0:
    return []
  
  #解包boxes
  x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
  socres = boxes[..., 4]
  class_ids = boxes[..., 5]

  x1 = x - w / 2
  y1 = y - h / 2
  x2 = x + w / 2
  y2 = y + h / 2

  final_boxes = []

  for class_id in class_ids.unique():
    idxs = torch.where(class_ids == class_id)[0]
    if idxs.numel() == 0:
      return []
    boxes_c = torch.stack([x1[idxs], y1[idxs], x2[idxs], y2[idxs]], dim=1)
    scores_c = socres[idxs]
    keep_ids = nms(boxes_c, scores_c, iou_threshold)
    final_boxes.append(boxes[idxs[keep_ids]])

  if len(final_boxes) == 0:
    return []
  
  return [torch.cat(final_boxes, dim=0)]
