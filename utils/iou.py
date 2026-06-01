import torch

def compute_iou(box1, box2):

  #计算box四条边
  box1_x1 = box1[..., 0] - box1[..., 2] / 2
  box1_x2 = box1[..., 0] + box1[..., 2] / 2
  box1_y1 = box1[..., 1] - box1[..., 3] / 2
  box1_y2 = box1[..., 1] + box1[..., 3] / 2

  box2_x1 = box2[..., 0] - box2[..., 2] / 2
  box2_x2 = box2[..., 0] + box2[..., 2] / 2
  box2_y1 = box2[..., 1] - box2[..., 3] / 2
  box2_y2 = box2[..., 1] + box2[..., 3] / 2

  #计算交集四条边
  inter_x1 = torch.max(box1_x1, box2_x1)
  inter_x2 = torch.min(box1_x2, box2_x2)
  inter_y1 = torch.max(box1_y1, box2_y1)
  inter_y2 = torch.min(box1_y2, box2_y2)

  #计算交集面积
  inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
  inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
  inter_area = inter_w * inter_h

  #计算并集面积
  box1_area = box1[..., 2] * box1[..., 3]
  box2_area = box2[..., 2] * box2[..., 3]
  union_area = box1_area + box2_area - inter_area

  #计算IoU
  IoU = inter_area / (union_area + 1e-8)

  return IoU