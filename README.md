# YOLOv1 From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## 从零手写 YOLOv1 —— 不依赖任何检测框架，逐行理解目标检测的底层逻辑

---

## 项目介绍

本项目是对经典目标检测算法 **YOLOv1**（You Only Look Once v1, Redmon et al., 2016）的完整从零复现，使用 **PyTorch** 实现，不借助任何目标检测框架（如 Detectron2、MMDetection 等）。

与直接调用高层 API 的学习方式不同，本项目的每一行代码均独立编写，涵盖数据加载、模型搭建、损失函数设计、训练流程、推理可视化，直至视频目标追踪的完整开发链路。这是我深入理解计算机视觉**目标检测全栈开发流程**的核心实践项目，也是我 CV 方向研究学习的重要组成部分。

> 「手写」不是目的，理解是目的。手写是理解最有效的路径。

---

## 学习目标

通过本项目，我希望系统掌握以下能力：

- **理解 YOLOv1 的核心思想**：Grid Cell 划分、Bounding Box 回归、多任务联合损失函数的设计哲学；
- **掌握 Pascal VOC 数据集的完整处理流程**：XML 标注解析、类别映射、数据增强策略；
- **从头构建深度神经网络**：理解 YOLOv1 的 24 层卷积 + 全连接层架构，以及每层的感受野与特征语义；
- **深入理解目标检测 Loss 的各个分项**：坐标损失、置信度损失、分类损失的数学推导与代码实现；
- **构建完整的深度学习工程训练链路**：从数据准备到模型保存、从推理到可视化的工程闭环；
- **初探多目标追踪（MOT）**：在帧级检测结果的基础上实现目标 ID 关联与轨迹跟踪。

---

## 当前进度

| 模块 | 状态 | 说明 |
| :--- | :---: | :--- |
| 📁 项目结构搭建 | ✅ 完成 | 目录规范、模块划分、代码风格统一 |
| 📦 Pascal VOC2007 数据准备 | ✅ 完成 | 数据集下载、目录组织、路径配置 |
| 🗂️ `VOCDataset` 数据加载器 | ✅ 完成 | XML 解析、类别映射、transform 接口 |
| 🧪 数据加载单元测试 | ✅ 完成 | 独立测试文件，验证 Bounding Box 与标签正确性 |
| 🏗️ YOLOv1 模型结构 | 🔄 进行中 | 24 层卷积 + FC 层，参照原论文架构手写 |
| 📉 YOLO Loss 函数 | ⏳ 计划中 | 坐标 / 置信度 / 分类三项损失联合实现 |
| 🔁 训练循环 | ⏳ 计划中 | 含学习率调度、模型 checkpoint 保存 |
| 🔍 推理与可视化 | ⏳ 计划中 | NMS 后处理、Bounding Box 绘制 |
| 🎥 视频目标追踪 | ⏳ 计划中 | 基于 SORT 或简单 IoU 匹配的多目标追踪 |

---

## 项目结构

```text
yolov1-from-scratch/
├── data/
│   └── VOCdevkit/
│       └── VOC2007/              # Pascal VOC 2007 数据集
│           ├── Annotations/      # XML 标注文件
│           ├── ImageSets/        # 训练/验证/测试集划分
│           └── JPEGImages/       # 原始图像
│
├── dataset/
│   ├── __init__.py
│   └── voc_dataset.py            # VOCDataset 数据加载器（XML 解析 + transform）
│
├── model/
│   ├── __init__.py
│   └── yolov1.py                 # YOLOv1 网络结构（24 Conv + FC）[进行中]
│
├── loss/
│   ├── __init__.py
│   └── yolo_loss.py              # YOLO 多任务 Loss 函数 [计划中]
│
├── utils/
│   ├── __init__.py
│   ├── transforms.py             # 数据增强变换
│   ├── nms.py                    # Non-Maximum Suppression [计划中]
│   └── visualize.py              # 检测结果可视化 [计划中]
│
├── train.py                      # 训练入口脚本 [计划中]
├── detect.py                     # 推理入口脚本 [计划中]
├── track.py                      # 视频追踪入口脚本 [计划中]
├── test_dataset.py               # VOCDataset 单元测试
├── requirements.txt
└── README.md
```

---

## 安装与使用

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
python test_dataset.py
```

运行后将输出图像尺寸、Bounding Box 坐标及类别标签，用于验证数据管道的正确性。

### 5. 训练（即将支持）

```bash
python train.py --epochs 135 --batch_size 64 --lr 0.001
```

### 6. 推理（即将支持）

```bash
python detect.py --image path/to/image.jpg --weights checkpoints/yolov1.pth
```

---

## 后续计划

**Day 3 — 模型结构**
按照原论文手写 YOLOv1 网络，包含 20 层预训练卷积骨干网络（参照 GoogLeNet 思路）+ 4 层检测卷积 + 2 层全连接层，输出 `7 × 7 × 30` 的预测张量。

**Day 4 — Loss 函数**
实现 YOLO 的多任务联合损失，包括：坐标回归损失（仅对负责预测的 box）、置信度损失（有物体 vs 无物体分开加权）、分类交叉熵损失。深入理解 `λ_coord = 5`、`λ_noobj = 0.5` 的设计动机。

**Day 5 — 训练流程**
搭建完整训练循环，含学习率分段衰减策略、模型权重 checkpoint 保存、训练过程 loss 曲线记录与可视化（TensorBoard 或 matplotlib）。

**Day 6 — 推理与可视化**
实现 NMS（Non-Maximum Suppression）后处理，在原图上绘制检测框与置信度，支持单张图像与批量图像推理。

**Day 7 — 视频目标追踪**
在帧级检测结果的基础上，实现基于 IoU 匹配的简单多目标追踪（Simple SORT），为每个目标分配稳定的轨迹 ID，输出追踪视频。

---

## 致谢

- **原始论文**：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，Joseph Redmon et al., CVPR 2016
- **数据集**：[The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

本项目是我计算机视觉学习路径中的实践核心，记录了从数据处理到完整系统落地的每一个思考过程。如果你也在从零学习目标检测，欢迎交流。

---

持续更新中 · Last updated: 2026
