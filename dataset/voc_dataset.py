import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class VOCDataset(Dataset):
  def __init__(self, root_dir, transform=None, split='train'):
    self.root_dir = root_dir
    self.transform = transform
    self.split = split
    self.class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    # 获取图片数据文件夹和标注文件夹的地址
    self.image_dir = os.path.join(root_dir, 'JPEGImages')
    self.annotation_dir = os.path.join(root_dir, 'Annotations')

    # 获取图片数量
    self.image_ids = self._get_image_ids()

    print(f"数据加载完成！共有{len(self.image_ids)}张图片")

  #_get_image_ids()函数实现
  def _get_image_ids(self):
    if self.split == 'train':
      txt_flie = os.path.join(self.root_dir, 'ImageSets', 'Main', 'trainval.txt')
    else:
      txt_flie = os.path.join(self.root_dir, 'ImageSets', 'Main', 'val.txt')

    with open(txt_flie,'r') as f:
      image_ids = [line.strip() for line in f.readlines()]

    return image_ids
  
  #返回给PyTorch DataLoader数据集大小     
  def __len__(self):
    return len(self.image_ids)
  
  #取用数据的函数实现
  def __getitem__(self, idx):
    """利用索引获取图片以及其标注"""
    image_id = self.image_ids[idx]
    
    #将图片加载到内存
    image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
    image = Image.open(image_path).convert('RGB')

    #将标注文件加载到内存
    annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
    boxes, labels = self._parse_annotation(annotation_path)

    #处理图片
    if self.transform:
      image = self.transform(image)

    #返回简单格式
    target = {
      "boxes" : boxes,
      "labels" : labels,
      "image_id" : image_id
    }

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
