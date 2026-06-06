import torch
from torchvision import transforms
from dataset.voc_dataset import VOCDataset

if __name__ == "__main__":
  test_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
    ])
  
  dataset = VOCDataset(
        root_dirs=[
        "data/VOCdevkit/VOC2007",
        "data/VOCdevkit/VOC2012"
    ],
    transform=test_transform,
    split='train'
  )

  print(f"数据集加载成功( •̀ ω •́ )✧！总共有{len(dataset)}张图片用于投喂模型ԅ(¯﹃¯ԅ)")

  image, target = dataset[0]
  
  assert image.shape == (3, 448, 448), f"老大，标签形状错误了喵！期望(3, 448, 448), 实际为{image.shape}"
  print(f"\n图片形状检查通过了喵：{image.shape}")
  
  assert target.shape == (7, 7, 30)
  print(f"\nYOLOv1标签形状通过了喵：{target.shape}")

  obj_mask = target[:, :, 4] == 1.0
  obj_count = obj_mask.sum().item()

  assert obj_count > 0, f"老大，这张图片的标签里的物体一个都没检测到喵！检查数据或者编码逻辑"
  print(f"\n物体检测检查全部通过了喵：这张图片里有{obj_count}个网络包含了物体中心点")
  print(f"\n有物体的cell位置(row,col):{obj_mask.nonzero(as_tuple=False)}")

  print(f"\n老大，全部都测试成功了喵ฅ(^៸៸- ⩊-៸៸^)ฅ")