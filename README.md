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
| 1 | 📦 Pascal VOC2007 数据准备 | ✅ 完成 | 数据集下载、目录组织、路径配置 |
| 2 | 🗂️ `VOCDataset` 数据加载器 | ✅ 完成 | XML 解析、类别映射、transform 接口 |
| 2 | 🧪 数据加载单元测试 | ✅ 完成 | 独立测试文件，验证 Bounding Box 与标签正确性 |
| 3 | 🏗️ YOLOv1 模型结构 | ✅ 完成 | 24 层卷积 + 2 层全连接检测头架构完整搭建 |
| 4 | 📉 前向传播与 Loss 函数 | ✅ 完成 | `forward` 输出严格对齐 `(batch_size, 1470)`；Loss 完整实现（坐标/置信度/分类三分项 + IoU 工具）并已通过 Sanity Test |
| 5 | 🔁 训练循环 + 可视化 | ✅ 完成 | `train.py` + `utils/lr_finder.py` + LR Finder 自动调优 |
| 6 | 🔍 推理与 NMS + mAP 评估 | 🔄 进行中 | `utils/nms.py` + `utils/map.py` |
| 7 | 🎥 视频目标追踪 (SORT) | ⏳ 待开始 | `track.py`，基于 SORT 或简单 IoU 匹配 |

---

## 核心技术理解

### 网络架构设计哲学

YOLOv1 的检测网络由 **24 层卷积特征提取主干** 和 **2 层全连接检测头** 组成。前 2 层（7×7 conv → maxpool → 3×3 conv → maxpool）做初始特征提取与快速降采样；第 3-20 层借鉴 GoogLeNet 的 bottleneck 思想，通过 **1×1 卷积降维 + 3×3 卷积扩展通道数** 的交替堆叠，在 112×112 到 7×7 的各级特征尺度上逐步加深通道（128→1024）；最后两层 3×3 卷积（Layer 23-24）在 7×7 分辨率下做最终的特征整合。全连接部分先将 7×7×1024 展平后投影到 4096 维，再直接回归到 1470 维的检测输出。

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

### YOLOv1 损失函数的多任务结构

YOLOv1 的损失不是一个统一误差，而是一个由四个加权子损失组成的多任务系统。

#### 四项损失分解

- **坐标损失（coord loss）**：仅对负责预测的 bbox 计算。中心坐标 (x, y) 使用线性误差，宽高 (w, h) 先取平方根再计算误差。√wh 的作用是压缩大框的梯度尺度，防止大尺寸框主导整体 loss。
- **置信度损失（obj / noobj loss）**：分为有物体和无物体两部分。负责框学习真实置信度（IoU 相关），非负责框被压制到接近 0。两者通过不同权重（λ_noobj = 0.5）平衡。
- **分类损失（class loss）**：仅在有物体的 grid cell 上计算，使用类别交叉熵。分类和定位共享骨干特征，但在输出层被解耦为独立的回归头和分类头。

#### 责任分配机制（Hard Assignment）

每个 grid cell 预测 B=2 个 bbox，但只有一个 bbox 负责学习真实目标。判定标准是与 ground truth 的 IoU——IoU 更高的 bbox 成为 responsible predictor，另一个 bbox 只学习 confidence ≈ 0。这种硬分配（hard assignment）机制简洁高效，但也是 YOLOv1 训练不稳定的来源之一。

#### 框的本质：参数化输出而非几何区域

检测框不是从图像中"裁"出来的区域，而是神经网络直接输出的参数化向量 (x, y, w, h)。所有后续操作（IoU 计算、NMS、可视化绘制）都基于这些预测值在张量层面的运算完成——理解这一点是将数学公式与 tensor 计算图对应起来的关键。

---

### 学习率调优：从梯度爆炸到 LR Finder

#### 问题发现

原论文的训练基于 ImageNet 预训练权重，分段阶梯学习率为 `1e-3 → 1e-2 → 1e-3 → 1e-4`，共 135 个 epoch。本项目从零随机初始化，直接照搬这套 schedule 后出现训练失稳：loss 在 125 附近陷入平台期，每个 epoch 内 batch loss 剧烈震荡，出现梯度爆炸。

#### 第一次调整（经验调参）

将 warmup 阶段延长至 10 轮（保持 `1e-3`），第二阶梯学习率折半至 `5e-3`。结果：前 10 轮正常，第 11 轮 lr 跳升后 train_loss 从 124 反弹至 128，之后卡死在 126 平台，val_loss 同步卡在 124，两者同时停滞——排除过拟合，确认是学习率过大导致的震荡平台。等第二阶梯结束、lr 回落到 `1e-3` 后，loss 重新开始正常下降，震荡幅度也明显收窄。

这说明 `1e-3` 很可能就是这个模型的最优主干训练学习率，但凭直觉判断没有说服力，需要实验依据。

#### 引入 LR Finder

了解到 Leslie Smith 提出的 Learning Rate Range Test 后，实现了 `utils/lr_finder.py`，通过指数级扫描 lr 区间，系统化定位最优学习率，而不是靠猜。

初版参数 `start_lr=1e-7, end_lr=1, num_iter=100`，曲线在 loss 轴上剧烈震荡，趋势不明显，且在 lr=1 处出现断崖式爆炸。调整为 `start_lr=1e-6, end_lr=1e-1, num_iter=len(loader)`（即 314 次迭代）后好很多，但单 batch loss 的随机噪声仍然很强，频繁出现尖峰毛刺，难以精确定位拐点。

#### 引入 EMA + 偏差修正

在 LR Finder 中加入指数移动平均（EMA）对 loss 曲线做平滑处理，同时引入偏差修正：

$$\hat{v}_t = \frac{v_t}{1 - \beta^t}$$

偏差修正的必要性：初始 $v_0 = 0$，前几轮迭代的估计值天然偏低，修正后曲线前段才真实可信。

EMA 本身存在一个固有的系统性局限：过往所有时刻的 loss 都参与当前值的计算，旧观测持续残留权重，新 loss 的突变需要多轮迭代才能逐渐冲淡历史影响，曲线永远滞后于真实信号。调整衰减系数 β 可以缓解滞后，但平滑度和响应速度是数学上互斥的，无法同时最优，只能取一个合适的折中值。读图时需配合最陡处法（Steepest）或谷底倒退法（Valley）人工判断最优区间。

#### LR Finder 结果

![LR Finder](lr_finder.png)

- `1e-6` 至 `1e-4`：loss 平缓，lr 过小，学习几乎停滞
- `1e-4` 至 `1e-2`：loss 持续下降，**最优区间**
- `1e-2` 以上：曲线趋平

#### 最终学习率策略

| 阶段 | Epoch | lr | 说明 |
| :--- | :---- | :- | :--- |
| Warmup | 1-10 | 1e-4 | 从甜区下限开始，稳定初始化 |
| 主干训练 | 11-125 | 1e-3 | 甜区中心，持续收敛 |
| 精细收敛 | 126-155 | 3e-4 | 缩小步长，逼近极值 |
| 微调 | 156+ | 1e-4 | 最终精调 |

---

## 实验结果

> 训练进行中（基于修正后的 lr schedule 重新训练）

### 训练配置

- 数据集：VOC2007（train+val 共 5011 张）
- Batch size：16，优化器：SGD（momentum=0.9，decay=0.0005）
- lr schedule：warmup(1e-4) → 1e-3 → 3e-4 → 1e-4，共 175 epochs
- 梯度裁剪：max_norm=10.0

计划记录指标：

- Train/Val Loss 曲线（每 epoch）
- Batch 级别 loss 曲线（每 20 个 batch）
- mAP@0.5 曲线
- 各类别 AP 对比

---

## 项目结构

```text
yolov1-from-scratch/
├── data/                          # 数据集（gitignore 忽略）
│   └── VOCdevkit/
│       └── VOC2007/
│           ├── Annotations/      # XML 标注文件
│           ├── ImageSets/        # 训练/验证/测试集划分
│           └── JPEGImages/       # 原始图像
│
├── models/
│   ├── __init__.py
│   └── yolov1.py                 # YOLOv1 模型定义
│
├── dataset/
│   ├── __init__.py
│   └── voc_dataset.py            # VOCDataset 数据加载器
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
│   ├── nms.py                    # NMS 后处理
│   └── map.py                    # mAP 评估
│
├── train.py                      # 训练入口脚本
├── run_lr_finder.py              # LR Finder 运行脚本
├── test_model.py                 # 模型前向传播测试（Smoke Test）
├── lr_finder.png                 # LR Finder 结果图
├── runs/                         # 训练输出（gitignore 忽略，含 checkpoint 和训练日志）
├── requirements.txt
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

---

## 后续计划 (Phase 6 及以后)

**Phase 6 — 推理与 NMS + mAP 评估**
实现 NMS（Non-Maximum Suppression）后处理，在原图上绘制检测框与置信度，支持单张图像与批量图像推理；基于 mAP 指标客观评估检测精度。

**Phase 7 — 视频目标追踪 (SORT)**
在帧级检测结果的基础上，实现基于 IoU 匹配的简单多目标追踪（Simple SORT），为每个目标分配稳定的轨迹 ID，输出追踪视频。

---

## 致谢

- **原始论文**：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)，Joseph Redmon et al., CVPR 2016
- **数据集**：[The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

本项目是我计算机视觉学习路径中的实践核心，记录了从数据处理到完整系统落地的每一个思考过程。如果你也在从零学习目标检测，欢迎交流。

---

持续更新中 · Last updated: 2026-06-05
