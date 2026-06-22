import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.iou import compute_iou

COMPONENT_LOG = str(Path(__file__).resolve().parent.parent / "loss_components.csv")

class YoloLoss(nn.Module):
  def __init__(self, S, B, C, iou_clamp_epochs=30):
    super().__init__()
    self.S = S
    self.B = B
    self.C = C
    self.lambda_coord = 1
    self.lambda_noobj = 0.05
    self.lambda_obj = 3.0
    self.lambda_noobj_class = 0.001
    self.iou_clamp_epochs = iou_clamp_epochs
    self.current_epoch = 1
    self.phase = "train"

  def set_epoch(self, epoch):
    self.current_epoch = epoch

  def set_phase(self, phase):
    self.phase = phase

  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)

    bbox1_pred = predictions[..., 0:5]
    bbox2_pred = predictions[..., 5:10]
    class_pred = predictions[..., 10:]

    bbox1_target = target[..., 0:5]
    bbox2_target = target[..., 5:10]
    class_target = target[..., 10:] 

    # Convert cell-relative coordinates to image-relative coordinates for IoU calculation
    device = predictions.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(self.S, device=device),
        torch.arange(self.S, device=device),
        indexing="ij"
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(-1).float()
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1).float()

    def to_image_coords(bbox):
        cx_img = (bbox[..., 0:1] + grid_x) / self.S
        cy_img = (bbox[..., 1:2] + grid_y) / self.S
        w_img = bbox[..., 2:3]
        h_img = bbox[..., 3:4]
        return torch.cat([cx_img, cy_img, w_img, h_img], dim=-1)

    iou1 = compute_iou(to_image_coords(bbox1_pred[..., :4]), to_image_coords(bbox1_target[..., :4]))
    iou2 = compute_iou(to_image_coords(bbox2_pred[..., :4]), to_image_coords(bbox2_target[..., :4]))

    bbox1_responsible = (iou1 >= iou2)
    bbox2_responsible = ~bbox1_responsible

    bbox1_resp_mask = bbox1_responsible.unsqueeze(-1)
    bbox2_resp_mask = bbox2_responsible.unsqueeze(-1)

    obj_mask = target[..., 4] == 1.0
    noobj_mask = ~obj_mask
    obj_mask_5 = obj_mask.unsqueeze(-1)

    resp_pred = (bbox1_resp_mask & obj_mask_5) * bbox1_pred + (bbox2_resp_mask & obj_mask_5) * bbox2_pred
    resp_target = obj_mask_5 * bbox1_target

    xy_loss = torch.sum((resp_pred[..., 0:2] - resp_target[..., 0:2]) ** 2)
    wh_loss = torch.sum((torch.sqrt(torch.clamp(resp_pred[..., 2:4], min=1e-6)) - torch.sqrt(torch.clamp(resp_target[..., 2:4], min=1e-6))) ** 2)
    coord_loss = self.lambda_coord * (xy_loss + wh_loss)

    # Object confidence loss
    if self.current_epoch <= self.iou_clamp_epochs:
      iou1_target = torch.clamp(iou1, min=0.3)
      iou2_target = torch.clamp(iou2, min=0.3)
    else:
      iou1_target = iou1
      iou2_target = iou2

    obj_conf_target = (bbox1_responsible.float() * iou1_target + bbox2_responsible.float() * iou2_target) * obj_mask.float()
    obj_conf_pred = ((bbox1_responsible * bbox1_pred[..., 4]) + (bbox2_responsible * bbox2_pred[..., 4])) * obj_mask
    obj_conf_loss = torch.sum((obj_conf_pred - obj_conf_target.detach()) ** 2)

    # No-object confidence loss
    bbox1_noobj_mask = noobj_mask | bbox2_responsible
    bbox2_noobj_mask = noobj_mask | bbox1_responsible

    bbox1_noobj_conf_loss = torch.sum(bbox1_noobj_mask.float() * bbox1_pred[..., 4] ** 2)
    bbox2_noobj_conf_loss = torch.sum(bbox2_noobj_mask.float() * bbox2_pred[..., 4] ** 2)
    noobj_conf_loss = bbox1_noobj_conf_loss + bbox2_noobj_conf_loss
     
    if obj_mask.any():
      class_labels = class_target[obj_mask].argmax(dim=-1)
      class_loss = F.cross_entropy(class_pred[obj_mask], class_labels, reduction='sum')
    else:
      class_loss = class_pred.sum() * 0.0

    # class smooth
    noobj_probs = torch.softmax(class_pred, dim=-1)
    noobj_class_reg = self.lambda_noobj_class * torch.sum(noobj_mask.unsqueeze(-1).float() * (noobj_probs - 1.0/self.C) ** 2)

    # Mean confidence of responsible bboxes in obj cells (diagnostic)
    n_obj = obj_mask.float().sum() + 1e-8
    mean_conf_obj = (obj_conf_pred.abs().sum() / n_obj).item()

    with open(COMPONENT_LOG, 'a') as f:
        f.write(f"{self.current_epoch},{self.phase},{self.lambda_obj * obj_conf_loss.item():.4f},{self.lambda_noobj * noobj_conf_loss.item():.4f},{coord_loss.item():.4f},{class_loss.item():.4f},{mean_conf_obj:.6f}\n")

    total_loss = coord_loss + self.lambda_obj * obj_conf_loss + self.lambda_noobj * noobj_conf_loss + class_loss + noobj_class_reg
    return total_loss / predictions.shape[0]
