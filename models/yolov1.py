import torch
import torch.nn as nn

class YOLOv1(nn.Module):
  def __init__(self, S=7, B=2, C=20):
    super().__init__()
    #注意虽然现在python2已经不再被广泛使用，但是如果是在python2里，激活父类初始化应写作super(YOLOv1, self).__init__()

    self.S = S
    self.B = B
    self.C = C

    # print(f"YOLOv1模型初始化完成<(￣︶￣)↗[GO!] → S={S}, B={B}, C={C}")

  #========================================================卷积层========================================================
    self.conv_layers = nn.Sequential(
      #Conv1-2负责粗提取特征，降低图片尺寸也为降低硬件负担
      #Conv1 448 × 448 → 224 × 224
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),
  
      #Conv2 224 × 224 → 112 × 112
      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Conv3-6 112 × 112 → 56 × 56 → 28 × 28
      nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Conv7-16 28 × 28 → 14 × 14 
      # (1×1,256 + 3×3,512) ×4
      # 输出 14×14×1024
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Conv17-22 14 × 14 → 7 × 7
      # (1×1,512 + 3×3,1024) ×2
      # 下采样到 7×7
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(0.1),

      #Conv23-24 
      # 最终特征提取
      # 输出 7×7×1024
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1)
    )

    #========================================================全连接层========================================================
    self.fc_layers = nn.Sequential(
      #Feature Flatten
      nn.Flatten(),
      
      #FC1: 50176 → 4096
      nn.Linear(7 * 7 * 1024, 4096),
      nn.LeakyReLU(0.1),
      nn.Dropout(0.5),

      #FC2: 4096 → 1470 (7×7×(20 + 2×5))
      nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
    )
    
  def forward(self, x):
    """
    前向传播
    x: 输入图像的张量，形状为(batch_size, 3, 448, 448)
    return：(bacth_size, 1470)
    """

    #卷积特征提取
    x = self.conv_layers(x)
    
    #全连接层预测
    x = self.fc_layers(x)

    return x
