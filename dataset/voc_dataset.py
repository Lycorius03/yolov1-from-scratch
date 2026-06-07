import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from config import VOC2007_DIR, VOC2012_DIR

class VOCDataset(Dataset):
  def __init__(self, root_dirs, transform=None, split='train'):
    if isinstance(root_dirs, str):
      root_dirs = [root_dirs]
    self.root_dirs = root_dirs
    self.transform = transform
    self.split = split
    self.class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    # 获取图片数量
    self.image_ids = self._get_image_ids()

    # print(f"数据加载完成！共有{len(self.image_ids)}张图片")

  #_get_image_ids()函数实现
  def _get_image_ids(self):
    all_ids = []
    for root_dir in self.root_dirs:
      if self.split == 'train':
        txt_file = Path(root_dir) / 'ImageSets' / 'Main' / 'trainval.txt'
      else:
        txt_file = Path(root_dir) / 'ImageSets' / 'Main' / 'val.txt'

      with open(txt_file, 'r') as f:
        all_ids.extend([(root_dir, line.strip()) for line in f.readlines()])

    return all_ids
  
  #返回给PyTorch DataLoader数据集大小     
  def __len__(self):
    return len(self.image_ids)
  
  #取用数据的函数实现
  def __getitem__(self, idx):
    """利用索引获取图片以及其标注"""
    root_dir, image_id = self.image_ids[idx]

    #将图片加载到内存
    image_path = Path(root_dir) / 'JPEGImages' / f"{image_id}.jpg"
    image = Image.open(image_path).convert('RGB')

    #将标注文件加载到内存
    annotation_path = Path(root_dir) / 'Annotations' / f"{image_id}.xml"
    boxes, labels = self._parse_annotation(annotation_path)

    #处理图片
    if self.transform:
      image = self.transform(image)

    # #返回简单格式,此种格式只适合通用检测框架，Faster R-CNN, DETR, 数据分析等，YOLO需要特定的格式，也就是哟个_encode_target()转换之后的格式
    # target = {
    #   "boxes" : boxes,
    #   "labels" : labels,
    #   "image_id" : image_id
    # }
    target = self._encode_target(boxes, labels)
    return image, target
  
  #实现XML解析函数
  def _parse_annotation(self,annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
      #读取类别名称
      class_name = obj.find('name').text
      label = self.class_to_idx.get(class_name, -1)
      
      #读取边框坐标
      bbox = obj.find('bndbox')
      xmin = float(bbox.find('xmin').text)
      ymin = float(bbox.find('ymin').text)
      xmax = float(bbox.find('xmax').text)
      ymax = float(bbox.find('ymax').text)

      boxes.append([xmin, ymin, xmax, ymax])
      labels.append(label)

    #转换tensor
    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return boxes, labels

  def _encode_target(self, boxes, labels, S=7, image_size=448):
    target = torch.zeros((S, S, 30))

    for i in range(len(boxes)):
      xmin, ymin, xmax, ymax = boxes[i]
      label = labels[i]
      
      #计算中心点和框的宽高
      cx = (xmin + xmax) / 2
      cy = (ymin + ymax) / 2 
      w = xmax - xmin
      h = ymax - ymin 
      
      #归一化到0-1
      cx /= image_size
      cy /= image_size
      w /= image_size
      h /= image_size

      #判断落在哪个grid cell里
      col = int(cx * S)
      row = int(cy * S)
      col = min(col, S - 1)
      row = min(row, S - 1)

      #计算中心点相对于所在grid cell的偏移
      cx_cell = cx * S - col
      cy_cell = cy * S - row

      #填入bbox
      target[row, col, 0] = cx_cell
      target[row, col, 1] = cy_cell
      target[row, col, 2] = w
      target[row, col, 3] = h
      target[row, col, 4] = 1.0

      target[row, col, 5] = cx_cell
      target[row, col, 6] = cy_cell
      target[row, col, 7] = w
      target[row, col, 8] = h
      target[row, col, 9] = 1.0

      #One_hot编码
      target[row, col, 10 + label] = 1.0

    return target