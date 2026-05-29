import torch
from models.yolov1 import YOLOv1

if __name__ == "__main__":
  #创建模型
  model = YOLOv1(S=7, B=2, C=20)

  #测试输入
  batch_size = 4
  test_input = torch.randn(batch_size, 3, 448, 448)

  #前向传播
  output = model(test_input)

  #使用assert断言测试
  excpeted_dim = 7 * 7 * (2 * 5 + 20)
  assert output.shape == (batch_size, excpeted_dim), \
    f"老大，输出形状错误了喵(っ °Д °;)っ！预期：({batch_size}, {excpeted_dim})，实际：{output.shape}"
  print("好耶！老大，我们成功了喵！输出形状完全正确<(￣︶￣)↗[GO!]")
  print(f"模型总参数量：{sum(p.numel() for p in model.parameters()):,}")
