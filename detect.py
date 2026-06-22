import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from models.yolov1 import YOLOv1
from utils.nms import non_max_suppression

CLASS_NAMES = [
  "aeroplane", "bicycle", "bird", "boat", "bottle",
  "bus", "car", "cat", "chair", "cow",
  "diningtable", "dog", "horse", "motorbike", "person",
  "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_COLORS = [
  (0,0,255),   (255,128,0), (255,255,0),   (0,128,255), (128,0,255),
  (0,255,255), (0,255,0),   (255,0,255),   (128,128,0), (255,0,128),
  (0,128,128), (128,0,0),   (255,100,100), (0,128,0),   (0,0,128),
  (128,128,128),(0,100,0),  (100,0,100),   (100,100,0), (200,200,0),
]
def _load_font(size=16):
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try: return ImageFont.truetype(name, size=size)
        except OSError: continue
    return ImageFont.load_default()

FONT = _load_font(16)

def decode_predictions(predictions, S=7, B=2, C=20, conf_threshold=0.1, device="cuda"):
  
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
    class_probs = torch.softmax(cell_preds[..., B * 5:], dim=-1)

  #transform position
    x = (bbox_preds[..., 0] + grid_x) / S
    y = (bbox_preds[..., 1] + grid_y) / S
    w = bbox_preds[..., 2]
    h = bbox_preds[..., 3]
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
      x1 = (boxes[:, 0] - boxes[:, 2] / 2).clamp(0, 1)
      y1 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0, 1)
      x2 = (boxes[:, 0] + boxes[:, 2] / 2).clamp(0, 1)
      y2 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0, 1)
      boxes[:, 0] = (x1 + x2) / 2
      boxes[:, 1] = (y1 + y2) / 2
      boxes[:, 2] = (x2 - x1).clamp(min=1e-6)
      boxes[:, 3] = (y2 - y1).clamp(min=1e-6)
      all_boxes.append(
        torch.cat([
          boxes[:, :4],
          scores.unsqueeze(1),
          class_ids.unsqueeze(1).float()
        ], dim=1)
      )

  return all_boxes

#单batch预测
def predict_batch(model, images, conf_threshold=0.1, iou_threshold=0.5):
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
    final_results.append(boxes)

  return final_results

#批量预测
def predict_loader(model, loader, device="cuda", conf_threshold=0.1, iou_threshold=0.5):
  all_predictions = []
  model.to(device)

  for images, _ in loader:
    images = images.to(device)

    batch_results = predict_batch(model, images, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    all_predictions.extend(batch_results)

  return all_predictions

#可视化
def visualize_predictions(model, loader, image_indices=None, device="cuda", conf_threshold=0.1, iou_threshold=0.5):
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

          color = CLASS_COLORS[int(class_id)]
          label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
          draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
          tb = draw.textbbox((x1, y1 - 14), label, font=FONT)
          draw.rectangle([tb[0]-2, tb[1]-1, tb[2]+2, tb[3]+1], fill=color)
          draw.text((x1, y1 - 14), label, fill="white", font=FONT)

        results.append(image)
        global_idx += 1
  
  return results
