import torch
from torchvision import transforms
from PIL import Image, ImageDraw

from models.yolov1 import YOLOv1
from utils.nms import non_max_suppression
from utils.map import evaluate_map

CLASS_NAMES = [
  "aeroplane", "bicycle", "bird", "boat", "bottle",
  "bus", "car", "cat", "chair", "cow",
  "diningtable", "dog", "horse", "motorbike", "person",
  "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def decode_predictions(predictions, S=7, B=2, C=20, conf_threshold=0.4, device="cuda"):
  
  #reshape
  batch_size = predictions.shape[0]

  predictions = predictions.reshape(batch_size, S, S, B * 5 + C).to(device)

  #build grid position map
  grid_y, grid_x = torch.meshgrid(
    torch.arange(S, device=device),
    torch.arange(S, device=device),
    indexing="ij"
  )

  grid_x = grid_x.unsqueeze(-1)
  grid_y = grid_y.unsqueeze(-1)

  all_boxes = []
  #decode bbox-information and calss-information
  for b in range(batch_size):
    cell_preds = predictions[b]

    bbox_preds = cell_preds[..., :B * 5].reshape(S, S, B, 5)
    class_probs = cell_preds[..., B * 5:]

  #transform position
    x = (bbox_preds[..., 0] + grid_x) / S
    y = (bbox_preds[..., 1] + grid_y) / S
    w = bbox_preds[..., 2]
    h  = bbox_preds[..., 3]
    conf = bbox_preds[..., 4] 

  #reshape
    scores = (conf.unsqueeze(-1) * class_probs.unsqueeze(2))
    scores = scores.permute(2, 0, 1, 3).reshape(-1, C)
    boxes = torch.stack([x, y, w, h, conf], dim=-1)
    boxes = boxes.permute(2, 0, 1, 3).reshape(-1, 5)

  #筛掉置信度低于conf_threshold的框
    keep_mask = scores.max(dim=1).values > conf_threshold

    boxes = boxes[keep_mask]
    class_ids = torch.argmax(scores, dim=1)[keep_mask]
    scores = scores.max(dim=1).values[keep_mask]

    if boxes.shape[0] == 0:
      all_boxes.append(torch.zeros((0, 6), device=device))
    else:
      all_boxes.append(
        torch.cat([
          boxes[:, :4],
          scores.unsqueeze(1),
          class_ids.unsqueeze(1).float()
        ], dim=1)
      )

  return all_boxes

#单batch预测
def predict_batch(model, images, conf_threshold=0.4, iou_threshold=0.5):
  #前向传播
  model.eval()
  device = images.device

  with torch.no_grad():
    predictions = model(images)
  #解码 + NMS
  boxes_list = decode_predictions(predictions, conf_threshold=conf_threshold, device=device)
  final_results = []

  for boxes in boxes_list:
    if boxes.shape[0] == 0:
      final_results.append(torch.zeros((0, 6), device=device))
      continue

    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=conf_threshold)

    if len(boxes) == 0:
      final_results.append(torch.zeros((0, 6), device=device))
    else:
      final_results.append(torch.stack(boxes))

  return final_results

#批量预测
def predict_loader(model, loader, device="cuda", conf_threshold=0.4, iou_threshold=0.5):
  all_predictions = []
  model.to(device)

  for images, _ in loader:
    images = images.to(device)

    batch_results = predict_batch(model, images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    all_predictions.extend(batch_results)

  return all_predictions

#可视化
def visualize_predictions(model, loader, image_indices=None, device="cuda", conf_threshold=0.4, iou_threshold=0.5):
  model.eval()
  model.to(device)

  results = []
  global_idx = 0

  with torch.no_grad():
    for images, _ in loader:
      images_device = images.to(device)
      batch_results = predict_batch(model, images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
      batch_size = images.shape[0]
      
      for b in range(batch_size):
        if image_indices is not None and global_idx not in image_indices:
            global_idx += 1
            continue
        image = transforms.functional.to_pil_image(images[b])
        draw = ImageDraw.Draw(image)

        boxes = batch_results[b]
        w_img, h_img = image.size

        for box in boxes:
          x, y, w, h, score, class_id = box.tolist()
          x1 = (x - w / 2) * w_img
          y1 = (y - h / 2) * h_img
          x2 = (x + w / 2) * w_img
          y2 = (y + h / 2) * h_img

          label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
          draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
          draw.text((x1, y1 - 10), label, fill="red")

          results.append(image)
          global_idx += 1
  
  return results

def evaluate_on_loader(model, loader, device="cuda", conf_threshold=0.4, iou_threshold=0.5):
    
    print("进行 mAP 评估...")
    return evaluate_map(model=model, loader=loader, device=device, conf_threshold=conf_threshold, iou_threshold=iou_threshold)