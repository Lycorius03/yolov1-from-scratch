# YOLOv1 From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007%20%26%202012-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## 从零手写 YOLOv1 —— 不依赖任何检测框架，逐行理解目标检测的底层逻辑

---

## 项目介绍

本项目是对经典目标检测算法 **YOLOv1**（You Only Look Once v1, Redmon et al., 2016）的完整从零复现，使用 **PyTorch** 实现，不借助任何目标检测框架（如 Detectron2、MMDetection 等）。

与直接调用高层 API 的学习方式不同，本项目的每一行代码均独立编写，涵盖数据加载、模型搭建、损失函数设计、训练流程、评估指标、推理可视化，直至视频目标追踪的完整开发链路。这是我深入理解计算机视觉**目标检测全栈开发流程**的核心实践项目，也是我 CV 方向研究学习的重要组成部分。

> 「手写」不是目的，理解是目的。手写是理解最有效的路径。

---

## 项目目标与学习收获

通过本项目，我系统掌握了以下能力：

- **理解 YOLOv1 的核心思想**：网格单元（Grid Cell）划分、边界框（Bounding Box）回归、多任务联合损失函数的设计哲学；
- **掌握 Pascal VOC 数据集的完整处理流程**：XML 标注解析、类别映射、数据增强策略；
- **从头构建深度神经网络**：理解 YOLOv1 的 24 层卷积 + 全连接层架构，以及每层的感受野与特征语义；
- **深入理解目标检测 Loss 的各个分项**：坐标损失、置信度损失、分类损失的数学推导与代码实现；
- **构建完整的深度学习工程训练链路**：从数据准备到模型保存、从推理到可视化的工程闭环；
- **初探多目标追踪（MOT）**：在帧级检测结果的基础上实现目标 ID 关联与轨迹跟踪。

---

## 当前进度 (Phase)

| Phase | 模块 | 状态 | 说明 |
| :---: | :--- | :---: | :--- |
| 1 | 📁 项目结构搭建 | ✅ 完成 | 目录规范、模块划分、代码风格统一 |
| 1 | 📦 Pascal VOC2007 数据准备 | ✅ 完成 | 数据集下载、目录组织、路径配置 |
| 2 | 🗂️ `VOCDataset` 数据加载器 | ✅ 完成 | XML 解析、类别映射、transform 接口 |
| 2 | 🧪 数据加载单元测试 | ✅ 完成 | 独立测试文件，验证 Bounding Box 与标签正确性 |
| 3 | 🏗️ YOLOv1 模型结构 | ✅ 完成 | 24 层卷积主干 + 全连接头架构完整搭建（手搓过程中补齐了官方 Backbone 漏掉的两层卷积，并清晰化了全部层注释） |
| 4 | 📉 前向传播与 Loss 函数 | ✅ 前向传播已完成 | `forward` 函数已实现并通过维度断言测试（Smoke Test），输出严格对齐 `(batch_size, 1470)`；Loss 函数待实现 |
| 5 | 🔁 训练循环 | ⏳ 计划中 | `train.py` + `config.py` + `tools/train_utils.py` |
| 6 | 🔍 推理与可视化 | ⏳ 计划中 | `detect.py` + `utils/nms.py` + `utils/visualize.py` |
| 7 | 🎥 视频目标追踪 | ⏳ 计划中 | `track.py`，基于 SORT 或简单 IoU 匹配 |

---

## 核心技术理解

### 网络架构设计哲学

YOLOv1 的 24 层网络由 **卷积特征提取主干**（20 层）和 **全连接检测头**（4 层）组成，参照 GoogLeNet 的设计思路，通过 1×1 卷积降维再 3×3 卷积扩展通道数的交替策略，在控制参数量的同时保证特征表达能力。

### 核心超参数配置

本项目严格遵循 YOLOv1 原始论文的配置：网格数 **S=7**，每个网格预测边界框数 **B=2**，Pascal VOC 类别数 **C=20**。输出特征图形状严格对齐为 `(batch_size, 1470)`，符合数学契约 `7 × 7 × (2 × 5 + 20) = 1470`。模型总参数量为 **271,703,550**，其中约 75% 的参数集中在第一层全连接层（50176 → 4096）。

### 逐层空间几何与数学本质

**14×14 → 7×7 降采样（Layer 13）**
该层为 `Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)`，通过步幅为 2 的卷积直接完成空间降采样，将特征图从 14×14 缩小到 7×7。与"先池化再卷积"的传统做法不同，步幅卷积在降采样的同时进行了一次 3×3 的局部特征融合，使每个输出位置综合了输入 3×3 邻域的信息，减少了直接池化造成的细节丢失。

**7×7 特征图上的两次 3×3 卷积（Layer 14–15）**
经过前面 13 层的卷积和池化，每个位置的感受野已经覆盖原始图像的大部分区域。这两层 3×3 卷积的主要目的不是继续扩大感受野，而是**在高分维特征空间（1024 通道）内做进一步的空间混合与特征整合**，让最终输出给全连接层的每个 7×7 位置特征都经过充分的上下文聚合，为后续的网格单元（Grid Cell）级预测提供更稳定的特征基础。

**全连接层 50176 → 4096（FC Layer 1）**
`7×7×1024 = 50176` 维的展平向量经过全连接层映射到 4096 维，构成一个高维空间中的线性超平面投影。该层结构为 `Linear → LeakyReLU(0.1) → Dropout(0.5)`：卷积提取的分布式空间特征被压缩为紧凑的全局语义向量，完成从局部感知到全局表征的转换。50176 维中大量冗余的空间信息（相邻位置的强相关性）被线性变换消解，保留的是跨网格单元（Grid Cell）的全局关联特征。LeakyReLU 在投影后注入非线性，使网络能够建模网格单元坐标、置信度与类别之间的联合分布。

**全连接层 4096 → 1470（FC Layer 2，检测头）**
将 4096 维全局特征映射到 `S×S×(B×5+C) = 7×7×30 = 1470` 维输出张量。每个网格单元（Grid Cell）的 30 维包含 2 个边界框（各 5 维：中心坐标 x, y、宽高 w, h、置信度）和 20 个类别概率。这一层完成了从全局语义表征到局部检测预测的映射——网络在前向传播中先通过卷积和瓶颈层提取全局上下文，最终在这一层将语义信息重新分配到每个网格单元的检测参数上。

**LeakyReLU 维持训练早期梯度流通**
全网络使用 LeakyReLU(α=0.1) 而非 ReLU。对于一个 24 层的网络，训练初期权重处于随机初始化状态，若使用 ReLU，约一半的神经元输出被截断为零，导致对应位置梯度完全消失——在多层叠加的情况下，梯度能够回传到浅层的概率急剧下降，网络陷入**训练停滞**。LeakyReLU 在负区保留 0.1 的斜率，确保无论激活值正负，梯度始终存在且能逐层回传，避免了训练早期的梯度枯竭问题。

**最后一层不加激活函数的原因**
全连接输出层直接使用 `nn.Linear`，不施加任何激活函数。这是因为 YOLOv1 需要输出 Bounding Box 的中心坐标 (x, y) 和宽高 (w, h)，这些值在归一化后仍需允许**负值输出**（例如目标中心位于网格单元（Grid Cell）边界之外时）。如果施加 ReLU 或 Sigmoid，会强制将输出限制在非负区间，导致网络无法表达正确的坐标回归值。

**Dropout 的解耦协同适应机制（位于两个全连接层之间, p=0.5）**
在 FC Layer 1 的 4096 维输出之后、FC Layer 2 检测头之前施加 Dropout(p=0.5)。其作用不是简单的"防止过拟合"——更深层的机制是**解耦协同适应**（Decoupled Co-adaptation）。在全连接层这样的高密度参数区域，神经元倾向于联合记忆训练样本的特定模式而非学习泛化特征。Dropout 通过随机屏蔽 50% 的神经元，强制每个神经元独立学习有意义的特征表示，而不是依赖其他神经元的"兜底"，确保最终送入检测头的 4096 维特征具备足够的泛化能力。

---

## 项目结构

```text
yolov1-from-scratch/
├── data/                          # 数据集（.gitignore 忽略）
│   └── VOCdevkit/
│       ├── VOC2007/
│       │   ├── Annotations/      # XML 标注文件
│       │   ├── ImageSets/        # 训练/验证/测试集划分
│       │   └── JPEGImages/       # 原始图像
│       └── VOC2012/              # 后续扩展添加
│
├── models/
│   ├── __init__.py
│   └── yolov1.py                 # YOLOv1 模型定义
│
├── dataset/
│   ├── __init__.py
│   └── voc_dataset.py            # 数据加载器
│
├── loss/
│   ├── __init__.py
│   └── yolo_loss.py              # YOLOv1 专用 Loss
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py             # 数据增强
│   ├── nms.py                    # 非极大值抑制
│   ├── visualize.py              # 画框可视化
│   └── metrics.py                # mAP 计算等
│
├── tools/
│   ├── convert_voc_to_yolo.py    # 标注格式转换（可选）
│   └── train_utils.py            # 训练辅助函数
│
├── config.py                     # 超参数配置文件
├── train.py                      # 训练主脚本
├── detect.py                     # 单张图片/摄像头检测
├── track.py                      # 视频目标追踪
├── test_model.py                 # 模型测试
├── utils/
│   └── voc_dataset_test.py       # 数据加载单元测试
├── requirements.txt
├── README.md
└── .gitignore
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

主要依赖包括：

```text
torch >= 2.0
torchvision
numpy
Pillow
opencv-python
matplotlib
```

### 3. 准备数据集

从 [Pascal VOC 官网](http://host.robots.ox.ac.uk/pascal/VOC/) 下载 VOC2007 数据集，并按照以下结构放置：

```text
data/VOCdevkit/VOC2007/
```

### 4. 测试数据加载器

```bash
python utils/voc_dataset_test.py
```

运行后将输出图像尺寸、Bounding Box 坐标及类别标签，用于验证数据管道的正确性。

### 5. 测试模型前向传播

```bash
python test_model.py
```

运行后执行前向传播维度断言测试（Smoke Test）：以随机生成的 `(4, 3, 448, 448)` 张量作为输入，验证输出形状严格等于 `(batch_size, 1470)`，并打印模型总参数量。当前已通过全部测试。

### 6. 训练（即将支持）

```bash
python train.py --epochs 135 --batch_size 64 --lr 0.001
```

### 7. 推理（即将支持）

```bash
python detect.py --image path/to/image.jpg --weights checkpoints/yolov1.pth
```

---

## 后续计划 (Phase 4 及以后)

**Phase 4 — 前向传播与 Loss 函数**
`forward` 函数已实现并通过维度断言测试：输入 `(batch_size, 3, 448, 448)`，输出严格对齐 `(batch_size, 1470)`。在手搓过程中补齐了最初版本 Backbone 漏掉的两层卷积（最终特征提取阶段的两个 3×3 卷积），并清晰化了原本模糊的层注释。下一步实现 YOLO 的多任务联合损失，包括：坐标回归损失（仅对负责预测的 box）、置信度损失（有物体 vs 无物体分开加权）、分类交叉熵损失。深入理解 `λ_coord = 5`、`λ_noobj = 0.5` 的设计动机。

**Phase 5 — 训练流程**
搭建完整训练循环，含学习率分段衰减策略、模型权重 checkpoint 保存、训练过程 loss 曲线记录与可视化（TensorBoard 或 matplotlib）。

**Phase 6 — 推理与可视化**
实现 NMS（Non-Maximum Suppression）后处理，在原图上绘制检测框与置信度，支持单张图像与批量图像推理。

**Phase 7 — 视频目标追踪**
在帧级检测结果的基础上，实现基于 IoU 匹配的简单多目标追踪（Simple SORT），为每个目标分配稳定的轨迹 ID，输出追踪视频。

---

## 致谢

- **原始论文**：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，Joseph Redmon et al., CVPR 2016
- **数据集**：[The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

本项目是我计算机视觉学习路径中的实践核心，记录了从数据处理到完整系统落地的每一个思考过程。如果你也在从零学习目标检测，欢迎交流。

---

持续更新中 · Last updated: 2026-05-30
