import torch
from torchvision import transforms
from utils.voc_dataset import VOCDataset

if __name__ == "__main__":
  test_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
    ])
  
  dataset = VOCDataset(
    root_dir="data/VOC2007",
    transform=test_transform,
    split='train'
  )

  print(f"数据集加载成功( •̀ ω •́ )✧！总共有{len(dataset)}张图片用于投喂模型ԅ(¯﹃¯ԅ)")

  image, target = dataset[0]
  print(f"图片形状：{image.shape}")
  print(f"这张图片里有{len(target['boxes'])}个物体")
  print(f"类别标签：{target['labels']}")
  print(f"图片ID：{target['image_id']}")
  print(f"\n测试成功了喵ฅ(^៸៸- ⩊-៸៸^)ฅ")