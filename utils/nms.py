import torch
from utils.iou import compute_iou

def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
  #过滤置信度低的框
  boxes = boxes[boxes[:, 4] > conf_threshold]

  if len(boxes) == 0:
    return []
  
  result = []

  #对每个类别单独做MNS
  classes = torch.unique(boxes[:, 5])
  
  for cls in classes:
    cls_boxes = boxes[boxes[:, 5] == cls]

    #按照置信度从高到低排序
    sorted_idx = torch.argsort(cls_boxes[:, 4], descending=True)
    cls_boxes = cls_boxes[sorted_idx]

    while len(cls_boxes) > 0:
      best = cls_boxes[0]
      result.append(best)

      if len(cls_boxes) == 1:
        break

      ious = compute_iou(best[:4].unsqueeze(0).expand(len(cls_boxes) -1 , -1), cls_boxes[1:, :4])

      #保留IoU低于阈值的框
      cls_boxes = cls_boxes[1:][ious < iou_threshold]

    return result