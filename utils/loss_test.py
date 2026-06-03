import sys

print(__file__)
print(sys.path[0])


import torch
from loss.yolo_loss import YoloLoss

if __name__ == "__main__":

  batch_size = 2
  predictions = torch.randn(batch_size, 1470)
  target = torch.zeros(batch_size, 7, 7, 30)

  #手动给(3, 4)这个grid cell放入一个物体，类别7(cat)
  target[0, 3, 4, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.4, 1.0])
  target[0, 3, 4, 5:10] = torch.tensor([0.5, 0.5, 0.3, 0.4, 1.0])
  target[0, 3, 4, 10 + 7] = 1.0

  loss_fn = YoloLoss(S=7, B=2, C=20)
  loss = loss_fn(predictions, target)

  print(f"Loss值: {loss.item():.4f}")
  print(f"是否为NaN: {torch.isnan(loss).item()}")
  print(f"是否为Inf: {torch.isinf(loss).item()}")