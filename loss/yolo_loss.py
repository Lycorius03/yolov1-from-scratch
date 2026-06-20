import torch
import torch.nn as nn
from utils.iou import compute_iou

class YoloLoss(nn.Module):
  def __init__(self, S, B, C):
    super().__init__()
    
    self.S = S
    self.B = B
    self.C = C
    self.lambda_coord = 1
    self.lambda_noobj = 0.5
    self.lambda_obj = 3.0
  #前向传播
  def forward(self, predictions, target):
    #重构模型输出的预测值的形状
    predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)

    #提取两个bbox里的预测值
    bbox1_pred = predictions[..., 0:5]
    bbox2_pred = predictions[..., 5:10]
    class_pred = predictions[..., 10:]

    #从target里面提取真实值
    bbox1_target = target[..., 0:5]
    bbox2_target = target[..., 5:10]
    class_target = target[..., 10:] 

    #计算两个bbox和target的IoU
    iou1 = compute_iou(bbox1_pred[..., :4], bbox1_target[..., :4])
    iou2 = compute_iou(bbox2_pred[..., :4], bbox2_target[..., :4])

    #找responsible box
    bbox1_responsible = (iou1 >= iou2)
    bbox2_responsible = ~bbox1_responsible

    #扩展维度，因bbox1_responsible这类的形状是(batch_size, 7, 7),与bbox_pred这类形状为(batch_size, 7, 7, 5)维度不同，相差一个维度
    bbox1_resp_mask = bbox1_responsible.unsqueeze(-1)
    bbox2_resp_mask = bbox2_responsible.unsqueeze(-1)

    #找到obj_mask和noobj_mask
    obj_mask = target[..., 4] == 1.0
    noobj_mask = ~obj_mask

    #扩展维度
    obj_mask_5 = obj_mask.unsqueeze(-1)

    #找到responsible和有物体的预测框和真实框
    resp_pred = (bbox1_resp_mask & obj_mask_5) * bbox1_pred + (bbox2_resp_mask & obj_mask_5) * bbox2_pred
    resp_target = obj_mask_5 * bbox1_target

    #xy Loss
    xy_loss = torch.sum((resp_pred[..., 0:2] - resp_target[..., 0:2]) ** 2)

    #wh Loss
    wh_loss = torch.sum((torch.sqrt(torch.abs(resp_pred[..., 2:4]) + 1e-6) - torch.sqrt(torch.abs(resp_target[..., 2:4]) + 1e-6)) ** 2)

    #coord Loss
    coord_loss = self.lambda_coord * (xy_loss + wh_loss)

    #obj confidence Loss
    obj_conf_target = (bbox1_responsible.float() * torch.clamp(iou1, min=0.3) + bbox2_responsible.float() * torch.clamp(iou2, min=0.3)) * obj_mask.float()
    obj_conf_pred = ((bbox1_responsible * bbox1_pred[..., 4]) + (bbox2_responsible * bbox2_pred[..., 4])) * obj_mask
    obj_conf_loss = torch.sum((obj_conf_pred - obj_conf_target.detach()) ** 2)

    #noobj confidence Loss
    bbox1_noobj_mask = noobj_mask | bbox2_responsible
    bbox2_noobj_mask = noobj_mask | bbox1_responsible

    bbox1_noobj_conf_loss = torch.sum(bbox1_noobj_mask.float() * bbox1_pred[..., 4] ** 2)
    bbox2_noobj_conf_loss = torch.sum(bbox2_noobj_mask.float() * bbox2_pred[..., 4] ** 2)

    noobj_conf_loss = bbox1_noobj_conf_loss + bbox2_noobj_conf_loss
     
    #class Loss
    class_loss = torch.sum(obj_mask.unsqueeze(-1).float() * (class_pred - class_target) ** 2)

    #total Loss
    print(f"obj(x{self.lambda_obj})={self.lambda_obj * obj_conf_loss.item():.2f}  noobj(x{self.lambda_noobj})={self.lambda_noobj * noobj_conf_loss.item():.2f}  coord={coord_loss.item():.2f}  class={class_loss.item():.2f}")
    total_loss = coord_loss + self.lambda_obj * obj_conf_loss + self.lambda_noobj * noobj_conf_loss + class_loss

    return total_loss / predictions.shape[0]