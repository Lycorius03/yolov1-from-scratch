# YOLOv1 From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![Dataset](https://img.shields.io/badge/Dataset-Pascal%20VOC%202007%20%2B%202012-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Phase%207%20完成%20%7C%20调试中-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## 从零手写 YOLOv1 —— 不依赖任何检测框架，逐行理解目标检测的底层逻辑

---

## 项目介绍

本项目是对经典目标检测算法 **YOLOv1**（You Only Look Once v1, Redmon et al., 2016）的完整从零复现，使用 **PyTorch** 实现，不借助任何目标检测框架（如 Detectron2、MMDetection 等）。

与直接调用高层 API 的学习方式不同，本项目的每一行代码均独立编写，涵盖数据加载、模型搭建、损失函数设计、训练流程、评估指标、推理可视化，直至视频目标追踪的完整开发链路。这是我深入理解计算机视觉**目标检测全栈开发流程**的核心实践项目，也是我 CV 方向研究学习的重要组成部分。

> 「手写」不是目的，理解是目的。手写是理解最有效的路径。

---

## 项目背景

本项目是我 CV 方向学习路径的第一个核心实践项目，位于"**目标级视觉 2D 感知**"阶段。

选择 YOLOv1 而非直接上手 YOLOv5 或 YOLO 系列最新版，原因有三：

1. **YOLOv1 是 anchor-free 单阶段检测器的思想起点**——后续的 YOLO 版本虽然引入了 anchor box、特征金字塔、解耦头等大量工程优化，但核心的"把目标检测转化为端到端回归问题"这一设计决策在 v1 中就已完成。理解 v1，是理解后续一切改进的前提。
2. **从零复现能完整经历每一个设计决策**——为什么要分 7×7 的网格？为什么每个网格要预测 2 个 box？Loss 里 `λ_coord = 5` 和 `λ_noobj = 0.5` 的权衡从何而来？这些问题在调用高层 API 时是看不见的，但手搓每一行代码时会自然浮现。
3. **24 层网络是理解深度学习工程化的最小完整体**——足够小（单卡可训），足够大（涵盖数据加载、模型搭建、损失设计、训练循环、推理后处理的全链路），是一个极佳的学习载体。

---

## 项目目标与学习收获

通过本项目，我系统掌握了以下能力：

- **理解 YOLOv1 的核心思想**：网格单元（Grid Cell）划分、边界框（Bounding Box）回归、多任务联合损失函数的设计哲学；
- **掌握 Pascal VOC 数据集的完整处理流程**：XML 标注解析、类别映射、数据增强策略；
- **从头构建深度神经网络**：理解 YOLOv1 的 24 层卷积 + 2 层全连接层架构，以及每层的感受野与特征语义；
- **深入理解目标检测 Loss 的各个分项**：坐标损失、置信度损失、分类损失的数学推导与代码实现；
- **构建完整的深度学习工程训练链路**：从数据准备到模型保存、从推理到可视化的工程闭环；
- **初探多目标追踪（MOT）**：在帧级检测结果的基础上实现目标 ID 关联与轨迹跟踪。

---

## 当前进度 (Phase)

| Phase | 模块 | 状态 | 说明 |
| :---: | :--- | :---: | :--- |
| 1 | 📁 项目结构搭建 | ✅ 完成 | 目录规范、模块划分、代码风格统一 |
| 1 | 📦 Pascal VOC2007 数据准备 | ✅ 完成 | 数据集下载（含 test 集）、目录组织、路径配置 |
| 1 | 📦 Pascal VOC2012 数据准备 | ✅ 完成 | 引入 VOC2012 与 VOC2007 联合训练，数据量提升 3.3 倍 |
| 1 | 📦 VOC2007 test 集 | ✅ 完成 | 下载 VOCtest_06-Nov-2007.tar，4952 张独立评估集 |
| 2 | 🗂️ `VOCDataset` 数据加载器 | ✅ 完成 | XML 解析、类别映射、transform 接口 |
| 2 | 🧪 数据加载单元测试 | ✅ 完成 | 独立测试文件，验证 Bounding Box 与标签正确性 |
| 3 | 🏗️ YOLOv1 模型结构 | ✅ 完成 | 24 层卷积 + 2 层全连接检测头架构完整搭建 |
| 4 | 📉 前向传播与 Loss 函数 | ✅ 完成 | `forward` 输出严格对齐 `(batch_size, 1470)`；Loss 完整实现（坐标/置信度/分类三分项 + IoU 工具）并已通过 Sanity Test |
| 5 | 🔁 训练循环 + 可视化 | ✅ 完成 | `train.py` + `utils/lr_finder.py` + LR Finder 自动调优 |
| 6 | 🔍 推理与 NMS + mAP 评估 | ✅ 完成 | ✅ `utils/nms.py` + ✅ `detect.py`（纯函数库） + ✅ `utils/map.py` |
| 6 | 🖼️ 推理入口脚本 | ✅ 完成 | `run_detect.py`：单图/目录/验证集采样三种模式，加载 `best_model.pth` 直接出检测结果图 |
| 7 | 📊 画图可视化模块 | ✅ 完成 | `utils/plot_utils.py`：双 y 轴训练曲线 + 单指标曲线绘制 |
| 8 | 🎥 视频目标追踪 (SORT) | ⏳ 待开始 | `track.py`，基于 SORT 或简单 IoU 匹配 |

---

## 核心技术理解

### 网络架构设计哲学

YOLOv1 的检测网络由 **24 层卷积特征提取主干** 和 **2 层全连接检测头** 组成。每层卷积后均配有 **BatchNorm** 和 **LeakyReLU(0.1)** 激活——BatchNorm 对从零训练至关重要，它归一化各层激活分布，防止梯度在 24 层深度网络中消失或爆炸，同时抑制模型陷入固定模板输出（mode collapse）。Conv1-2（7×7 conv → maxpool → 3×3 conv → maxpool）做初始特征提取与快速降采样；Conv3-16 借鉴 GoogLeNet 的 bottleneck 思想，通过 **1×1 卷积降维 + 3×3 卷积扩展通道数** 的交替堆叠，在 112×112 到 7×7 的各级特征尺度上逐步加深通道（128→1024）；Conv23-24 在 7×7 分辨率下做最终的特征整合。全连接部分先将 7×7×1024 展平后投影到 4096 维，再直接回归到 1470 维的检测输出。

### 核心超参数配置

本项目严格遵循 YOLOv1 原始论文的配置：网格数 **S=7**，每个网格预测边界框数 **B=2**，Pascal VOC 类别数 **C=20**。输出特征图形状严格对齐为 `(batch_size, 1470)`，符合数学契约 `7 × 7 × (2 × 5 + 20) = 1470`。模型总参数量为 **271,703,550**，其中约 75% 的参数集中在 FC1（50176 → 4096）。

### 逐层空间几何与数学本质

**14×14 → 7×7 降采样（Conv17-22）**
该层为 `Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)`，通过步幅为 2 的卷积直接完成空间降采样，将特征图从 14×14 缩小到 7×7。与"先池化再卷积"的传统做法不同，步幅卷积在降采样的同时进行了一次 3×3 的局部特征融合，使每个输出位置综合了输入 3×3 邻域的信息，减少了直接池化造成的细节丢失。

**7×7 特征图上的两次 3×3 卷积（Conv23-24）**
经过前面 13 层的卷积和池化，每个位置的感受野已经覆盖原始图像的大部分区域。这两层 3×3 卷积的主要目的不是继续扩大感受野，而是**在高分维特征空间（1024 通道）内做进一步的空间混合与特征整合**，让最终输出给全连接层的每个 7×7 位置特征都经过充分的上下文聚合，为后续的网格单元（Grid Cell）级预测提供更稳定的特征基础。

**全连接层 50176 → 4096（FC1）**
`7×7×1024 = 50176` 维的展平向量经过全连接层映射到 4096 维，构成一个高维空间中的线性超平面投影。该层结构为 `Linear → LeakyReLU(0.1) → Dropout(0.5)`：卷积提取的分布式空间特征被压缩为紧凑的全局语义向量，完成从局部感知到全局表征的转换。50176 维中大量冗余的空间信息（相邻位置的强相关性）被线性变换消解，保留的是跨网格单元（Grid Cell）的全局关联特征。LeakyReLU 在投影后注入非线性，使网络能够建模网格单元坐标、置信度与类别之间的联合分布。

**全连接层 4096 → 1470（FC2，检测头）**
将 4096 维全局特征映射到 `S×S×(B×5+C) = 7×7×30 = 1470` 维输出张量。每个网格单元（Grid Cell）的 30 维包含 2 个边界框（各 5 维：中心坐标 x, y、宽高 w, h、置信度）和 20 个类别概率。这一层完成了从全局语义表征到局部检测预测的映射——网络在前向传播中先通过卷积和瓶颈层提取全局上下文，最终在这一层将语义信息重新分配到每个网格单元的检测参数上。

**LeakyReLU 维持训练早期梯度流通**
全网络使用 LeakyReLU(α=0.1) 而非 ReLU。对于一个 24 层的网络，训练初期权重处于随机初始化状态，若使用 ReLU，约一半的神经元输出被截断为零，导致对应位置梯度完全消失——在多层叠加的情况下，梯度能够回传到浅层的概率急剧下降，网络陷入**训练停滞**。LeakyReLU 在负区保留 0.1 的斜率，确保无论激活值正负，梯度始终存在且能逐层回传，避免了训练早期的梯度枯竭问题。

**最后一层不加激活函数的原因**
全连接输出层直接使用 `nn.Linear`，不施加任何激活函数。这是因为 YOLOv1 需要输出 Bounding Box 的中心坐标 (x, y) 和宽高 (w, h)，这些值在归一化后仍需允许**负值输出**（例如目标中心位于网格单元（Grid Cell）边界之外时）。如果施加 ReLU 或 Sigmoid，会强制将输出限制在非负区间，导致网络无法表达正确的坐标回归值。

**Dropout 的解耦协同适应机制（位于两个全连接层之间, p=0.5）**
在 FC1 的 4096 维输出之后、FC2 检测头之前施加 Dropout(p=0.5)。其作用不是简单的"防止过拟合"——更深层的机制是**解耦协同适应**（Decoupled Co-adaptation）。在全连接层这样的高密度参数区域，神经元倾向于联合记忆训练样本的特定模式而非学习泛化特征。Dropout 通过随机屏蔽 50% 的神经元，强制每个神经元独立学习有意义的特征表示，而不是依赖其他神经元的"兜底"，确保最终送入检测头的 4096 维特征具备足够的泛化能力。

### YOLOv1 损失函数的多任务结构

YOLOv1 的损失不是一个统一误差，而是一个由四个加权子损失组成的多任务系统。

#### 四项损失分解

- **坐标损失（coord loss）**：仅对负责预测的 bbox 计算。中心坐标 (x, y) 使用线性误差，宽高 (w, h) 先取平方根再计算误差。√wh 的作用是压缩大框的梯度尺度，防止大尺寸框主导整体 loss。权重 `λ_coord = 1`（loss 已按 batch 归一化）。
- **置信度损失（obj / noobj loss）**：分为有物体和无物体两部分。负责框学习真实置信度（IoU ≥ 0.3 时用实际 IoU，< 0.3 时用 0.3 冷启动地板），非负责框被压制到接近 0。两者通过权重 `λ_obj = 3.0` 和 `λ_noobj = 0.05` 平衡——后者显著低于原论文的 0.5，因为从零训练时每图 obj:noobj 信号比约 3:95，过高的 noobj 权重会压死置信度。
- **分类损失（class loss）**：在有物体的 grid cell 上计算 MSE；在无物体的 grid cell 上施加极小的均匀化正则（权重 0.001），防止模型坍缩到"永远预测 person"。

#### 责任分配机制（Hard Assignment）

每个 grid cell 预测 B=2 个 bbox，但只有一个 bbox 负责学习真实目标。判定标准是与 ground truth 的 IoU——IoU 更高的 bbox 成为 responsible predictor，另一个 bbox 只学习 confidence ≈ 0。这种硬分配（hard assignment）机制简洁高效，但也是 YOLOv1 训练不稳定的来源之一。

#### 框的本质：参数化输出而非几何区域

检测框不是从图像中"裁"出来的区域，而是神经网络直接输出的参数化向量 (x, y, w, h)。所有后续操作（IoU 计算、NMS、可视化绘制）都基于这些预测值在张量层面的运算完成——理解这一点是将数学公式与 tensor 计算图对应起来的关键。

---

## 训练与调试历程

从零复现 YOLOv1 的过程并非一帆风顺，而且仍在进行中。从学习率策略、Loss 权重平衡、到架构层面的 BatchNorm 缺失导致 mode collapse、再到数据划分泄露，累计发现并修复了 9 个问题。每一步的思考都在日志中留下了痕迹。

当前训练配置：warmup 5 epoch（1e-4→5e-4），主干 80 epoch（5e-4），收敛 55 epoch（2e-4），微调 30 epoch（7e-5）。λ_coord=1, λ_obj=3.0, λ_noobj=0.05。

> 📖 **[开发者日志（DEVLOG.md）](DEVLOG.md)** —— 实时记录的学习笔记，伴随项目推进持续更新

---

## 项目结构

```text
yolov1-from-scratch/
├── data/                          # 数据集（gitignore 忽略）
│   └── VOCdevkit/
│       ├── VOC2007/
│       │   ├── Annotations/      # XML 标注文件（含 test 集）
│       │   ├── ImageSets/Main/   # train/val/trainval/test 划分
│       │   └── JPEGImages/       # 原始图像（含 test 集）
│       └── VOC2012/
│           ├── Annotations/
│           ├── ImageSets/
│           └── JPEGImages/
│
├── models/
│   ├── __init__.py
│   └── yolov1.py                 # YOLOv1 模型定义
│
├── dataset/
│   ├── __init__.py
│   └── voc_dataset.py            # VOCDataset 数据加载器（支持多数据集）
│
├── loss/
│   ├── __init__.py
│   └── yolo_loss.py              # YOLOv1 多任务联合损失函数
│
├── utils/
│   ├── __init__.py
│   ├── voc_dataset_test.py       # 数据加载单元测试
│   ├── loss_test.py              # Loss Sanity Test
│   ├── iou.py                    # IoU 计算工具
│   ├── lr_finder.py              # LR Finder 学习率范围测试
│   ├── nms.py                    # NMS 后处理（已完成）
│   ├── map.py                    # mAP 评估
│   └── plot_utils.py             # 训练曲线可视化（双 y 轴 + 单指标）
│
├── train.py                      # 训练入口脚本（跨系统兼容，DEVICE 自适应）
├── run_lr_finder.py              # LR Finder 运行脚本
├── run_detect.py                 # 推理入口脚本（单图/目录/验证集采样，智能检测框绘制）
├── detect.py                     # 图像推理、批量检测、预测可视化（纯函数库）
├── test_model.py                 # 模型前向传播测试（Smoke Test）
├── overfit_test.py               # 过拟合回归测试（1 图 500 步，验证流水线正确性）
├── config.py                     # 统一路径配置（pathlib，本地/云端一键切换）
├── lr_finder.png                 # LR Finder 结果图
├── lr_finder_defect.png          # 第二版 LR Finder 曲线
├── lr_finder_defect1.png         # 第三版 LR Finder 曲线
├── runs/                         # 训练输出（gitignore 忽略，含 checkpoint 和训练日志）
├── requirements.txt              # 项目依赖（不含 PyTorch，避免云端 CUDA 版本冲突）
├── README.md
├── LICENSE
├── .gitattributes
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

> **注意**：`requirements.txt` 中**不包含 PyTorch 及其相关依赖**（torch、torchvision 等）。这是为了防止 PyTorch 与云算力实例（如 AutoDL、Colab 等）预装环境发生 CUDA 版本冲突。请根据自身环境自行安装，推荐使用 CUDA 11.8 ~ 12.8 对应的 PyTorch 版本。安装示例：
>
> ```bash
> # CUDA 11.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

主要依赖包括：

```text
torch >= 2.0
torchvision
numpy
Pillow
opencv-python
matplotlib
torchmetrics
```

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

运行后将输出图像尺寸、Bounding Box 坐标及类别标签，用于验证数据管道的正确性。

### 5. 测试模型前向传播

```bash
python test_model.py
```

运行后执行前向传播维度断言测试（Smoke Test）：以随机生成的 `(4, 3, 448, 448)` 张量作为输入，验证输出形状严格等于 `(batch_size, 1470)`，并打印模型总参数量。当前已通过全部测试。

### 6. 运行推理（需要先完成训练生成 best_model.pth）

训练完成后，在 `run_detect.py` 顶部修改以下变量后直接运行：

```python
RUN_NAME = "20260607_165143"   # runs/ 下训练记录的时间戳文件夹名
MODE = "single"                # single | dir | val_sample       
INPUT = "test.jpg"             # single/dir 模式下的输入路径
NUM_SAMPLES = 5                # val_sample 模式下采样张数
```

```bash
python run_detect.py
```

- `single` 模式：对单张图片推理并保存检测结果图
- `dir` 模式：对一个目录下的所有图片批量推理
- `val_sample` 模式：从验证集随机采样指定张数，可视化检测效果，无需额外准备图片

---

## 后续计划

**Phase 8 — 视频目标追踪 (SORT)**
在帧级检测结果的基础上，实现基于 IoU 匹配的简单多目标追踪（Simple SORT），为每个目标分配稳定的轨迹 ID，输出追踪视频。`detect.py` 作为纯函数库提供逐帧检测接口，`run_detect.py` 作为独立推理入口验证单帧效果，二者共同为 Phase 8 的视频帧级推理做好准备。

---

## 致谢

- **原始论文**：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，Joseph Redmon et al., CVPR 2016
- **数据集**：[The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

本项目是我计算机视觉学习路径中的实践核心，记录了从数据处理到完整系统落地的每一个思考过程。如果你也在从零学习目标检测，欢迎交流。

---

持续更新中 · Last updated: 2026-06-21 (数据泄漏修复 + 每 Epoch mAP 评估)
