import xml.etree.ElementTree as ET
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from config import VOC2007_DIR, VOC2012_DIR

class VOCDataset(Dataset):
  def __init__(self, root_dirs, transform=None, split='train', use_encoded_target=True):
    if isinstance(root_dirs, str):
      root_dirs = [root_dirs]
    self.root_dirs = root_dirs
    self.transform = transform
    self.split = split
    self.use_encoded_target = use_encoded_target
    self.class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    self.image_ids = self._get_image_ids()

  def _get_image_ids(self):
    all_ids = []
    for root_dir in self.root_dirs:
      if self.split == 'train':
        txt_file = Path(root_dir) / 'ImageSets' / 'Main' / 'trainval.txt'
      elif self.split == 'test':
        txt_file = Path(root_dir) / 'ImageSets' / 'Main' / 'test.txt'
      else:
        txt_file = Path(root_dir) / 'ImageSets' / 'Main' / 'val.txt'

      with open(txt_file, 'r') as f:
        all_ids.extend([(root_dir, line.strip()) for line in f.readlines()])

    return all_ids
     
  def __len__(self):
    return len(self.image_ids)
  
  def __getitem__(self, idx):
    root_dir, image_id = self.image_ids[idx]

    image_path = Path(root_dir) / 'JPEGImages' / f"{image_id}.jpg"
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    annotation_path = Path(root_dir) / 'Annotations' / f"{image_id}.xml"
    boxes, labels = self._parse_annotation(annotation_path)

    # 随机水平翻转（仅训练模式），同时翻转图片和bbox坐标
    if self.split == 'train' and random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      if len(boxes) > 0:
        flipped_boxes = boxes.clone()
        flipped_boxes[:, 0] = orig_w - boxes[:, 2]  # xmin' = W - xmax
        flipped_boxes[:, 2] = orig_w - boxes[:, 0]  # xmax' = W - xmin
        boxes = flipped_boxes

    if self.transform:
      image = self.transform(image)

    if self.use_encoded_target:
      target = self._encode_target(boxes, labels, orig_w, orig_h)
    else:
      target = self._encode_raw_target(boxes, labels, orig_w, orig_h)

    return image, target
  
  def _parse_annotation(self, annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
      class_name = obj.find('name').text
      label = self.class_to_idx.get(class_name, -1)
      
      bbox = obj.find('bndbox')
      xmin = float(bbox.find('xmin').text)
      ymin = float(bbox.find('ymin').text)
      xmax = float(bbox.find('xmax').text)
      ymax = float(bbox.find('ymax').text)

      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(label)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return boxes, labels

  def _encode_target(self, boxes, labels, orig_w, orig_h, S=7):
    target = torch.zeros((S, S, 30))

    for i in range(len(boxes)):
      xmin, ymin, xmax, ymax = boxes[i]
      label = labels[i]

      cx = (xmin + xmax) / 2
      cy = (ymin + ymax) / 2
      w = xmax - xmin
      h = ymax - ymin

      # Normalize relative coordinates to [0, 1] using original image size
      cx /= orig_w
      cy /= orig_h
      w /= orig_w
      h /= orig_h

      col = int(cx * S)
      row = int(cy * S)
      col = min(col, S - 1)
      row = min(row, S - 1)

      cx_cell = cx * S - col
      cy_cell = cy * S - row

      # BBox 1
      target[row, col, 0] = cx_cell
      target[row, col, 1] = cy_cell
      target[row, col, 2] = w
      target[row, col, 3] = h
      target[row, col, 4] = 1.0

      # BBox 2
      target[row, col, 5] = cx_cell
      target[row, col, 6] = cy_cell
      target[row, col, 7] = w
      target[row, col, 8] = h
      target[row, col, 9] = 1.0

      target[row, col, 10 + label] = 1.0

    return target
  
  def _encode_raw_target(self, boxes, labels, orig_w, orig_h):
    if len(boxes) == 0:
      return torch.zeros((0, 24), dtype=torch.float32)

    # xyxy -> xywh, normalized using original image size
    xyxy = boxes.clone()
    xywh = torch.zeros_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2 / orig_w
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2 / orig_h
    xywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / orig_w
    xywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / orig_h       

    num_objs = len(labels)
    onehot = torch.zeros((num_objs, 20), dtype=torch.float32)
    onehot[range(num_objs), labels] = 1.0

    target = torch.cat([onehot, xywh], dim=1)
    return target