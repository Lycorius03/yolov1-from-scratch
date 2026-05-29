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
      #Layer1-2负责粗提取特征，降低图片尺寸也为降低硬件负担
      #Layer1
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Layer2
      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Layer3-5
      nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Layer6-8
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
      nn.MaxPool2d(kernel_size=2, stride=2),

      #Layer9-13
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

      #Layer14-20
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.1)
    )

    #========================================================全连接层========================================================
    self.fc_layers = nn.Sequential(
      #展平特征图
      nn.Flatten(),
      
      #Layer1
      nn.Linear(7 * 7 * 1024, 4096),
      nn.LeakyReLU(0.1),
      nn.Dropout(0.5),

      #Layer2
      nn.linear(4096, self.S * self.S * (self.B * 5 + self.C))
    )
    