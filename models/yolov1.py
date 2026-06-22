import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class YOLOv1(nn.Module):
  def __init__(self, S=7, B=2, C=20, output_mode="sprint"):
    super().__init__()
    #注意虽然现在python2已经不再被广泛使用，但是如果是在python2里，激活父类初始化应写作super(YOLOv1, self).__init__()

    self.S = S
    self.B = B
    self.C = C
    self.output_mode = output_mode

    # print(f"YOLOv1模型初始化完成<(￣︶￣)↗[GO!] → S={S}, B={B}, C={C}")

    # ResNet-50, 去掉最后两层 (GAP + FC), 保留到 layer4
    # 448×448 → 14×14×2048
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    # 14×14×2048 → 7×7×1024
    self.adapter = nn.Sequential(
      nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1)
    )

    # 全连接检测头
    self.fc_layers = nn.Sequential(
      nn.Flatten(),

      #FC1: 50176 → 4096
      nn.Linear(7 * 7 * 1024, 4096),
      nn.LeakyReLU(0.1),
      nn.Dropout(0.7),

      #FC2: 4096 → 1470 (7×7×(20 + 2×5))
      nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
    )

  def forward(self, x):
    # x: (batch_size, 3, 448, 448) → (batch_size, 1470)
    x = self.backbone(x)
    x = self.adapter(x)
    x = self.fc_layers(x)
    x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

    if self.output_mode == "legacy":
      x[..., 2:4] = torch.abs(x[..., 2:4])
      x[..., 7:9] = torch.abs(x[..., 7:9])
      x[..., 4] = torch.sigmoid(x[..., 4])
      x[..., 9] = torch.sigmoid(x[..., 9])
      x[..., 10:] = torch.softmax(x[..., 10:], dim=-1)
    else:
      # bbox gate
      for b in range(self.B):
        start = b * 5
        x[..., start:start + 4] = torch.sigmoid(x[..., start:start + 4])
        x[..., start + 4] = torch.sigmoid(x[..., start + 4])

    return x.reshape(x.shape[0], -1)


ModernYOLOv1 = YOLOv1
