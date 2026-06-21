# YOLOv1-Modernized-PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![Backbone](https://img.shields.io/badge/Backbone-ResNet--50-important?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007%20%2B%202012-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## 提取 YOLOv1 核心思想，注入现代 Backbone 的重构实践

---

## 项目介绍

本项目并非对 YOLOv1 (Redmon et al., 2016) 论文的刻板代码复刻，而是一次**提取核心思想、注入现代 Backbone** 的重构实践。

传统的 YOLOv1 复现往往受限于其庞大（271M 参数）的 24 层自定义卷积架构。原论文作者先在 ImageNet（120 万张图）上预训练了整整一周，才在 VOC 上微调——若没有这一关键前提，在 VOC 数据集上从零起步极难收敛。

本项目在保留 YOLOv1 最核心的设计哲学的同时，做出了务实的工程改进：将特征提取主干替换为 ImageNet 预训练的 **ResNet-50**，利用其强大的先验视觉特征，让模型收敛速度和稳定性得到质的飞跃。

> 真正的硬核不在于堆叠毫无意义的卷积层，而在于吃透算法的几何感知逻辑与 Loss 约束机制。通过引入成熟的特征提取器，本仓库将所有算力与代码重心集中在了 YOLO 最惊艳的"回归思想"上。

---

## 核心特性

### 架构换擎 (Modern Backbone)

剥离了原论文难以收敛的随机初始化网络，将特征提取主干替换为 `torchvision.models.resnet50`（ImageNet 预训练权重）。

| 对比项 | 旧架构（24 层自定义） | 新架构（ResNet-50） |
| :--- | :--- | :--- |
| 特征提取器 | 24 层手写卷积，~270M 参数 | ResNet-50，~23M 参数 |
| 预训练 | 无（从零随机初始化） | ImageNet（120 万张图） |
| 收敛难度 | 极难，需大量数据 + 算力 | 大幅降低，可微调 |
| 总参数量 | ~271M | ~254M |

### 原汁原味的 YOLO 哲学 (Core Philosophy Retained)

以下核心代码**一行未改**，完整保留了经历 7 次以上迭代、修复了 8 个隐蔽 Bug 才打磨出来的实现：

- **$7 \times 7$ 网格单元的端到端回归设计** — 将目标检测转化为纯回归问题，每个 grid cell 预测 2 个 bbox + 20 类概率
- **多任务联合损失函数** (`loss/yolo_loss.py`) — 精确计算 IoU、置信度惩罚与坐标权重放缩，含 obj/noobj 信号平衡、IoU 冷启动地板 (clamp 0.3)、λ_obj/λ_noobj 权重等全部细节
- **VOC 联合数据加载器** (`dataset/voc_dataset.py`) — VOC2007 + VOC2012 联合训练、原图尺寸归一化、随机水平翻转增强

### 工程完备性

- **训练**：SGDR 余弦退火重启 (CosineAnnealingWarmRestarts)、线性 Warmup、梯度裁剪、自动 checkpoint 保存与断点续训
- **评估**：每 epoch mAP 评估（VOC2007 test，4952 张独立图片）、Loss 分量逐项记录
- **推理**：单图/目录/验证集采样三种推理模式、NMS 后处理、检测结果可视化
- **追踪**：基于 IoU 匹配的简单多目标追踪，支持摄像头/视频文件入力
- **跨平台**：`pathlib` 统一路径管理，本地 Windows / WSL2 / Linux 云端无缝切换

---

## 项目背景

这是我在 CV 方向学习路径中"**目标级视觉 2D 感知**"阶段的核心实践项目。

选择 YOLOv1 而非直接上手更现代的检测器的原因：

1. **YOLOv1 是 anchor-free 单阶段检测器的思想起点** — "把目标检测转化为端到端回归问题"这一核心决策在 v1 中就已经完成。理解 v1，是理解后续一切改进的前提。
2. **从零复现能完整经历每一个设计决策** — 为什么要分 7×7 的网格？为什么每个网格要预测 2 个 box？Loss 里坐标权重和 noobj 权重的权衡从何而来？这些问题在调用高层 API 时是看不见的。
3. **全链路实践** — 覆盖数据加载、模型搭建、损失设计、训练循环、推理后处理、视频追踪的完整开发链路。

---

## 项目目标与学习收获

通过本项目，我系统掌握了以下能力：

- **理解 YOLOv1 的核心思想**：网格单元划分、边界框回归、多任务联合损失函数的设计哲学
- **掌握 Pascal VOC 数据集的完整处理流程**：XML 标注解析、类别映射、原图尺寸归一化
- **深入理解目标检测 Loss 的各个分项**：坐标损失 (xy + wh)、置信度损失 (obj + noobj)、分类损失的数学推导与代码实现
- **系统化 Bug 排查能力**：从标注编码到 IoU 计算再到 Loss 信号平衡，累计定位并修复了 8 个隐蔽 Bug（详见 [DEVLOG.md](DEVLOG.md)）
- **从"造轮子"到"用轮子"的认知升级**：理解了预训练 Backbone 不是偷懒，而是深度学习工程中不可回避的先验基础

---

## 当前进度 (Phase)

| Phase | 模块 | 状态 | 说明 |
| :---: | :--- | :---: | :--- |
| 1 | 📁 项目结构搭建 | ✅ 完成 | 目录规范、模块划分、代码风格统一 |
| 1 | 📦 Pascal VOC 数据准备 | ✅ 完成 | VOC2007 + VOC2012 联合训练，含独立 test 集 |
| 2 | 🗂️ `VOCDataset` 数据加载器 | ✅ 完成 | XML 解析、原图尺寸归一化、水平翻转增强 |
| 3 | 🏗️ YOLOv1 模型结构 | ✅ 完成 | ResNet-50 Backbone + 桥接层 + 全连接检测头 |
| 4 | 📉 Loss 函数 | ✅ 完成 | 多任务联合损失（坐标/置信度/分类，含 IoU 坐标转换） |
| 5 | 🔁 训练循环 + 可视化 | ✅ 完成 | SGDR + Warmup + 自动 checkpoint + 训练曲线绘制 |
| 6 | 🔍 推理与 NMS + mAP 评估 | ✅ 完成 | NMS 后处理 + mAP@0.5 每 epoch 评估 |
| 6 | 🖼️ 推理入口脚本 | ✅ 完成 | 单图/目录/验证集采样三种模式 |
| 7 | 📊 画图可视化模块 | ✅ 完成 | 双 y 轴训练曲线 + 单指标曲线 |
| 8 | 🎥 视频目标追踪 | ✅ 完成 | IoU 匹配的简单多目标追踪 |
| 9 | 🚀 架构升级 | ✅ 完成 | 24 层自定义卷积 → ResNet-50 (ImageNet 预训练) |

---

## 技术要点

### 架构设计

**ResNet-50 Backbone** (`self.backbone`)：去掉最后的 AvgPool 和 FC，保留到 layer4。输入 $448 \times 448 \times 3$，输出 $14 \times 14 \times 2048$。ImageNet 预训练权重提供了强大的先验视觉特征。

**桥接层** (`self.adapter`)：单层 `Conv2d(2048→1024, k=3, stride=2) + BN + LeakyReLU(0.1)`，将 ResNet 的特征图压回 YOLOv1 原汁原味的 $7 \times 7 \times 1024$ 网格空间。

**全连接检测头** (`self.fc_layers`)：完全沿用 YOLOv1 原始设计 — `Flatten → FC(50176→4096) → LeakyReLU → Dropout(0.5) → FC(4096→1470)`，输出严格对齐 $S \times S \times (B \times 5 + C) = 1470$。

### YOLOv1 损失函数的多任务结构

详见 [DEVLOG.md](DEVLOG.md) 中 Part 2 的完整 Bug 追踪和设计分析。

---

## 项目结构

```text
yolov1-from-scratch/
├── data/                          # 数据集（gitignore 忽略）
│   └── VOCdevkit/
│       ├── VOC2007/
│       │   ├── Annotations/      # XML 标注（含 test 集）
│       │   ├── ImageSets/Main/   # train/val/trainval/test 划分
│       │   └── JPEGImages/       # 原始图像（含 test 集）
│       └── VOC2012/
│           ├── Annotations/
│           ├── ImageSets/
│           └── JPEGImages/
│
├── models/
│   ├── __init__.py
│   └── yolov1.py                 # YOLOv1 模型定义（ResNet-50 + 桥接 + 检测头）
│
├── dataset/
│   ├── __init__.py
│   └── voc_dataset.py            # VOCDataset 数据加载器（多数据集联合）
│
├── loss/
│   ├── __init__.py
│   └── yolo_loss.py              # YOLOv1 多任务联合损失函数（未改）
│
├── utils/
│   ├── __init__.py
│   ├── voc_dataset_test.py       # 数据加载单元测试
│   ├── loss_test.py              # Loss Sanity Test
│   ├── iou.py                    # IoU 计算工具
│   ├── lr_finder.py              # LR Finder 学习率范围测试
│   ├── nms.py                    # NMS 后处理
│   ├── map.py                    # mAP 评估
│   └── plot_utils.py             # 训练曲线可视化
│
├── train.py                      # 训练入口脚本
├── run_lr_finder.py              # LR Finder 运行脚本
├── run_detect.py                 # 推理入口（单图/目录/验证集采样）
├── detect.py                     # 图像推理、批量检测、可视化（纯函数库）
├── gui_detect.py                 # GUI 检测入口（摄像头/图片两种模式）
├── track.py                      # 视频目标追踪（IoU 匹配 + track ID）
├── test_model.py                 # 模型前向传播测试
├── overfit_test.py               # 过拟合回归测试
├── config.py                     # 统一路径配置（本地/云端一键切换）
├── runs/                         # 训练输出（gitignore 忽略）
├── requirements.txt              # 项目依赖
├── README.md
├── DEVLOG.md                     # 开发者日志（完整调试记录）
└── LICENSE
```

---

## 安装与运行

### 1. 克隆仓库

```bash
git clone https://github.com/Lycorius03/yolov1-from-scratch.git
cd yolov1-from-scratch
```

### 2. 安装依赖

建议使用 conda 或 venv 创建独立环境：

```bash
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 中**不包含 PyTorch 及其相关依赖**（torch、torchvision 等）。这是为了防止 PyTorch 与云算力实例（AutoDL、Colab 等）预装环境发生 CUDA 版本冲突。请根据自身环境自行安装：
>
> ```bash
> # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. 准备数据集

从 [Pascal VOC 官网](http://host.robots.ox.ac.uk/pascal/VOC/) 下载以下三个压缩包：

- `VOCtrainval_06-Nov-2007.tar` — VOC2007 训练+验证集（5011 张）
- `VOCtrainval_11-May-2012.tar` — VOC2012 训练+验证集（11540 张）
- `VOCtest_06-Nov-2007.tar` — VOC2007 测试集（4952 张，**独立于训练集，用于 mAP 评估**）

解压到 `data/VOCdevkit/` 目录，最终结构如下：

```text
data/VOCdevkit/
├── VOC2007/
│   ├── Annotations/       # 9963 个 XML（5011 trainval + 4952 test）
│   ├── ImageSets/Main/    # train.txt / val.txt / trainval.txt / test.txt
│   └── JPEGImages/        # 9963 张图片
└── VOC2012/
    ├── Annotations/       # 17125 个 XML
    ├── ImageSets/Main/    # train.txt / val.txt / trainval.txt
    └── JPEGImages/        # 17125 张图片
```

**数据划分**（与论文一致）：

| 用途 | 数据 | 图片数 |
| :--- | :--- | :--- |
| 训练 | VOC2007 trainval + VOC2012 trainval | **16,551** |
| 评估 | VOC2007 test（完全独立，无数据泄露） | **4,952** |

### 4. 测试数据加载器

```bash
python utils/voc_dataset_test.py
```

### 5. 测试模型前向传播

```bash
python test_model.py
```

验证输出形状严格等于 `(batch_size, 1470)`。

### 6. 运行推理（需要先完成训练生成 best_model.pth）

在 `run_detect.py` 顶部修改以下变量后直接运行：

```python
RUN_NAME = "20260607_165143"   # runs/ 下训练记录的时间戳文件夹名
MODE = "single"                # single | dir | val_sample
INPUT = "test.jpg"             # single/dir 模式下的输入路径
NUM_SAMPLES = 5                # val_sample 模式下采样张数
```

```bash
python run_detect.py
```

---

## 后续计划

- **冻结 Backbone 实验**：冻结 ResNet-50 前几层，仅微调高层和检测头，进一步加速收敛
- **更强数据增强**：RandomAffine / MixUp / Mosaic，缓解过拟合
- **锚框 (Anchor Box)**：引入 YOLOv2 的 anchor 机制，改善密集小目标检测

---

## 致谢

- **原始论文**：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，Joseph Redmon et al., CVPR 2016
- **Backbone**：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，He et al., CVPR 2016
- **数据集**：[The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

---

持续更新中 · Last updated: 2026-06-21 (架构重构：ResNet-50 Backbone 替换 24 层自定义卷积)
