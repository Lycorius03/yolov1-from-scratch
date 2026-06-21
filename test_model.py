import torch
from models.yolov1 import YOLOv1

if __name__ == "__main__":
    model = YOLOv1(S=7, B=2, C=20)
    model.eval()

    batch_size = 4
    test_input = torch.randn(batch_size, 3, 448, 448)

    with torch.no_grad():
        backbone_out = model.backbone(test_input)
        assert backbone_out.shape == (batch_size, 2048, 14, 14), \
            f"老大，Backbone输出尺寸不对喵(っ °Д °;)っ！预期：({batch_size}, 2048, 14, 14)，实际：{backbone_out.shape}"
        print(f"Backbone输出尺寸正确喵！→ {backbone_out.shape}")

        adapter_out = model.adapter(backbone_out)
        assert adapter_out.shape == (batch_size, 1024, 7, 7), \
            f"老大，Adapter输出尺寸不对喵(っ °Д °;)っ！预期：({batch_size}, 1024, 7, 7)，实际：{adapter_out.shape}"
        print(f"Adapter输出尺寸正确喵！→ {adapter_out.shape}")

        output = model(test_input)

    expected_dim = 7 * 7 * (2 * 5 + 20)

    assert output.shape == (batch_size, expected_dim), \
        f"老大，输出形状错误了喵(っ °Д °;)っ！预期：({batch_size}, {expected_dim})，实际：{output.shape}"

    print("好耶！老大，我们成功了喵！输出形状完全正确<(￣︶￣)↗[GO!]")

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    head_params = sum(p.numel() for p in model.fc_layers.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backbone (ResNet-50) 参数量：{backbone_params:,}")
    print(f"Adapter (桥接层)    参数量：{adapter_params:,}")
    print(f"Head (全连接检测头)  参数量：{head_params:,}")
    print(f"模型总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")
