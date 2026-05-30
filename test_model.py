import torch
from models.yolov1 import YOLOv1

if __name__ == "__main__":
    # 创建模型
    model = YOLOv1(S=7, B=2, C=20)
    model.eval()

    # 测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 448, 448)

    with torch.no_grad():
        # 测试 Backbone 输出
        features = model.conv_layers(test_input)

        assert features.shape == (batch_size, 1024, 7, 7), \
            f"老大，Backbone输出尺寸不对喵(っ °Д °;)っ！预期：({batch_size}, 1024, 7, 7)，实际：{features.shape}"

        print(f"Backbone输出尺寸正确喵！→ {features.shape}")

        # 前向传播
        output = model(test_input)

    # 使用assert断言测试
    expected_dim = 7 * 7 * (2 * 5 + 20)

    assert output.shape == (batch_size, expected_dim), \
        f"老大，输出形状错误了喵(っ °Д °;)っ！预期：({batch_size}, {expected_dim})，实际：{output.shape}"

    print("好耶！老大，我们成功了喵！输出形状完全正确<(￣︶￣)↗[GO!]")

    # 输出总参数量
    total_params = sum(p.numel() for p in model.parameters())

    print(f"模型总参数量：{total_params:,}")