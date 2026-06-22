# 开发者日志

这份日志伴随项目推进持续更新，记录从零复现 YOLOv1 过程中的每一个决策、每一次调试、每一次"为什么不行"的追问。

---

## 目录

- [Part 1: 学习率调优全记录](#part-1-学习率调优全记录)
- [Part 2: Bug 追踪](#part-2-bug-追踪)
- [Part 3: 训练历史数据（2026-06-01 ~ 2026-06-19）](#part-3-训练历史数据2026-06-01--2026-06-19)
- [Part 4: 阶段反思（2026-06-17）](#part-4-阶段反思2026-06-17)
- [Part 5: 架构折戟与思想重构（2026-06-21）](#part-5-架构折戟与思想重构2026-06-21)
- [Part 6: 输出激活函数 + 防过拟合 + 数据增强（2026-06-22）](#part-6-输出激活函数--防过拟合--数据增强2026-06-22)

---

## Part 1: 学习率调优全记录

### 2026-06-01 — 问题发现

原论文的训练基于 ImageNet 预训练权重，分段阶梯学习率为 `1e-3 → 1e-2 → 1e-3 → 1e-4`，共 135 个 epoch。本项目从零随机初始化，直接照搬这套 schedule 后出现训练失稳：loss 降至约 125 后陷入平台期，每个 epoch 内 batch loss 剧烈震荡，出现梯度爆炸。

### 2026-06-02 — 第一次调整（经验调参）

将 warmup 阶段延长至 10 轮（保持 `1e-3`），第二阶梯学习率折半至 `5e-3`。结果：前 10 轮正常，第 11 轮 lr 跳升后 train_loss 从 124 反弹至 128，之后卡死在 126 平台，val_loss 同步卡在 124，两者同时停滞——排除过拟合，确认是学习率过大导致的震荡平台。等第二阶梯结束、lr 回落到 `1e-3` 后，loss 重新开始正常下降，震荡幅度也明显收窄。

这说明 `1e-3` 很可能就是这个模型的最优主干训练学习率，但凭直觉判断没有说服力，需要实验依据。

### 2026-06-04 — 第二次调整（引入 LR Finder）

考虑到这是一个从零初始化的模型而非预训练模型，学习率策略需要仔细考量。当时评估了两种方案——分段阶梯衰减（原论文做法）和余弦退火（Cosine Annealing）。初期选择分段阶梯衰减的原因：

1. **降低变量数**——当时 Loss 实现尚未验证正确性、数据划分未修复、BN 也未加入，同时改变学习率策略会增加调试难度。
2. **LR Finder 可独立指导阶梯参数**——LR Finder 能直接给出有效 lr 区间和最优值，阶梯衰减的参数（每级 lr 值和停留轮数）可以据此设定。

了解到 Leslie Smith 提出的 Learning Rate Range Test 后，实现了 `utils/lr_finder.py`，通过指数级扫描 lr 区间，系统化定位最优学习率，而不是靠猜。

初版参数 `start_lr=1e-7, end_lr=1, num_iter=100`，曲线在 loss 轴上剧烈震荡，趋势不明显，且在 lr=1 处出现断崖式爆炸。调整为 `start_lr=1e-6, end_lr=1e-1, num_iter=len(loader)`（即 314 次迭代）后好很多，但单 batch loss 的随机噪声仍然很强，频繁出现尖峰毛刺，难以精确定位拐点。

#### 引入 EMA + 偏差修正

在 LR Finder 中加入指数移动平均（EMA）对 loss 曲线做平滑处理，同时引入偏差修正：

$$
\hat{v}_{t} = \frac{v_{t}}{1 - \beta^{t}}
$$

偏差修正的必要性：初始 $v_{0} = 0$，前几轮迭代的估计值天然偏低，修正后曲线前段才真实可信。

EMA 本身有一个无法回避的短板：过往所有时刻的 loss 都参与当前值的计算，旧观测持续残留权重，新 loss 的突变需要多轮迭代才能逐渐冲淡历史影响，曲线永远滞后于真实信号。调整衰减系数 β 可以缓解滞后，但平滑度和响应速度是数学上互斥的，无法同时最优，只能取一个合适的折中值。读图时需配合最陡处法（Steepest）或谷底倒退法（Valley）人工判断最优区间。

#### 第二次 LR Finder 结果

首次实现 EMA + 偏差修正后的 LR Finder 曲线：

![LR Finder (Second)](lr_finder_defect.png)

- `1e-6` 至 `1e-4`：loss 平缓，lr 过小，学习几乎停滞
- `1e-4` 至 `1e-2`：loss 持续下降，有效收敛区间
- `1e-2` 以上：曲线趋平

> **读图提示**：EMA 滞后性使谷底位置偏右，Steepest 法指向 `1e-3` 附近。

第二次 LR Finder 已经证明当前学习率策略（`1e-3`）位于有效收敛区间内。但训练仍然失败——模型长期停留在约 123 的 Loss 平台附近，**说明问题已经不是学习率**。

### 2026-06-05 — 第三次调整（从学习率转向损失函数排查）

LR Finder 给出了收敛区间，但正式训练依然失败。进一步分析训练日志与 Loss 实现后，定位到三个问题。

#### 1. 阶跃式 Warmup 的优化冲击

最初的 Warmup 并非真正意义上的 Warmup。训练前 10 个 Epoch 使用 `1e-4`，随后直接跳变到 `1e-3`。这种非连续的学习率变化会将参数更新步长瞬间放大 10 倍，破坏刚刚建立的梯度方向。

最终改为线性 Warmup：

```python
def get_lr(epoch):
    if epoch <= 10:
        start_lr = 1e-4
        end_lr = 1e-3
        return start_lr + (end_lr - start_lr) * (epoch - 1) / 10
    elif epoch <= 125:
        return 1e-3
    elif epoch <= 155:
        return 3e-4
    else:
        return 1e-4
```

学习率从离散跳变变为连续增长，更符合 Warmup 的设计初衷。

#### 2. No-Object Mask 的责任分配漏洞

重新审查 YOLOv1 Loss 实现时，发现了一个极其隐蔽的逻辑漏洞。

YOLOv1 的设计要求：Responsible Box 学习目标，Non-Responsible Box 学习置信度趋近于 0。但原实现中，部分非负责框没有正确参与 `noobj_conf_loss` 的计算。

修复后采用显式布尔逻辑：

```python
bbox1_noobj_mask = noobj_mask | bbox2_responsible
bbox2_noobj_mask = noobj_mask | bbox1_responsible

bbox1_noobj_conf_loss = torch.sum(
    bbox1_noobj_mask.float() * bbox1_pred[..., 4] ** 2
)
bbox2_noobj_conf_loss = torch.sum(
    bbox2_noobj_mask.float() * bbox2_pred[..., 4] ** 2
)
noobj_conf_loss = bbox1_noobj_conf_loss + bbox2_noobj_conf_loss
```

这样无论是背景网格还是未被选中的预测框，都能获得正确的监督信号，符合 YOLOv1 原论文的责任分配机制。

#### 3. Loss 量级与梯度裁剪的相互影响

所有误差项采用 `torch.sum()` 聚合，Loss 量级会随 Batch Size 线性增长。与此同时训练启用了 `clip_grad_norm_(max_norm=10.0)`，导致大量梯度在反向传播阶段被强制截断——模型表面上使用 `1e-3` 学习率，但实际有效更新步长远低于设定值。

最终修改：

```python
# Loss 按 Batch Size 归一化
return total_loss / predictions.shape[0]
```

同时将梯度裁剪阈值调整为 `max_norm=50.0`，恢复合理的梯度动态范围。

#### 第三次 LR Finder 结果

修复 Loss 实现、Warmup 连续化与梯度尺度问题后，重新运行 LR Finder：

![LR Finder (Third)](lr_finder_defect1.png)

- `1e-6` 至 `1e-4`：学习率过小，Loss 几乎不下降
- `1e-4` 至 `1e-2`：Loss 持续下降，有效收敛区间
- `1e-3` 附近：曲线斜率最大，**最优学习率**（Steepest 法）
- `1e-2` 以上：接近谷底，Loss 趋于平缓

> **读图提示**：EMA 平滑曲线存在时序滞后性，真实信号的最优点比平滑曲线显示的略早。若仅看谷底位置（`1e-2` 附近），实际上已经过了最优区间——谷底处 Loss 已经趋于平缓，学习率已经偏大。正确的做法是取曲线最陡下降处（Steepest 法）或谷底向回退一个数量级（Valley 法），二者都指向 `1e-3` 附近。
>
> 需要注意的是：LR Finder 仅能判断"哪个学习率区间有效"，无法回答"有效区间内的学习率是否足以让模型充分收敛"。第三次 LR Finder 确认了 `1e-3` 位于有效区间内，但正式训练仍然卡在 ~7.9 的平台期——**LR Finder 结论正确，但仅凭它并不能保证收敛**。

#### 两条曲线的对比分析

| 对比项 | 第二次 LR Finder | 第三次 LR Finder |
| :--- | :--- | :--- |
| Loss 实现 | 存在 noobj mask 漏洞、Loss 未归一化 | 已修复 |
| Warmup | 阶跃式跳变 | 线性连续 |
| 梯度裁剪 | max_norm=10.0（过度截断） | max_norm=50.0 |
| 有效收敛区间 | `1e-4` ~ `1e-2` | `1e-4` ~ `1e-2` |
| 最优学习率（Steepest 法） | `1e-3` 附近 | `1e-3` 附近 |

两次 LR Finder 曲线的收敛区间高度一致，有效学习率范围几乎完全重合。这从实验上确认了一个关键结论：**在第二次与第三次调整之间，模型无法收敛的根本原因不是学习率策略，而是 Loss 实现有漏洞、梯度裁剪截断过度。**。修复底层实现后，`1e-3` 学习率依然位于有效收敛区间内，因此保留当前学习率策略不变。

> 但需要明确的是：第三次调整修复了 Loss 实现中的问题，使 Loss 量级回归合理范围，**并不等于收敛问题已被解决**。修复后的正式训练仍然长期停留在 ~7.9 的平台期，无法继续深入收敛。这说明还有更深层的因素在限制模型的学习能力。

#### 第三次学习率策略

| 阶段 | Epoch | lr | 说明 |
| :--- | :---- | :- | :--- |
| Warmup（线性） | 1-10 | 1e-4 → 1e-3 | 连续增长，避免阶跃冲击 |
| 主干训练 | 11-125 | 1e-3 | 甜区中心，持续收敛 |
| 精细收敛 | 126-155 | 3e-4 | 缩小步长，逼近极值 |
| 微调 | 156+ | 1e-4 | 最终精调 |

### 2026-06-07 — 第四次调整（引入 VOC2012，突破数据瓶颈）

#### 1. 为什么怀疑是数据量不足？

三次训练的调整轨迹呈现出一条清晰的线索——**模型有能力学习，但始终无法继续深入收敛**。

第一次与第二次训练（仅使用 VOC2007）中，Loss 的下降轨迹呈现出高度相似的模式：前几十个 epoch 快速下降后，训练 Loss 稳定在约 124 左右、验证 Loss 稳定在约 122 左右，之后便不再有明显进展。第三次修复了 Loss 归一化问题后，Loss 量级下降到 7.9 / 7.7 水平，但**平台期现象依然存在**。

从量纲角度统一对比三次结果：前两次使用 `torch.sum()` 聚合 Loss（未除以 Batch Size），第三次改为 `sum() / batch_size` 归一化。若将前两次的 Loss 也按 Batch Size = 16 进行归一化折算，其稳定平台约为 **7.6–7.8**——与第三次的 7.9 平台**惊人地接近**。这说明三次训练在同一个 Loss 量级上遇到了几乎相同的收敛天花板。

同时观察 batch 级别的 Loss 波动：第三次训练中，单个 batch 的 Loss 在 5–15 之间剧烈跳动，epoch 级别的平均 Loss 则被这种高方差平滑到一条缓慢下降后停滞的曲线。这种"宏观停滞 + 微观高噪"的模式是**数据不足时的经典信号**——模型在有限样本上反复记忆后，无法通过更多样化的样本来获得进一步的泛化梯度。

VOC2007 的 trainval 集仅有约 5011 张图像，对于一个 24 层卷积 + 2 层全连接、参数量超过 2.7 亿的检测网络而言，每个参数能"看到"的有效样本非常有限。数据量的天花板一旦触达，再精细的学习率调优也无法突破——这就是第四次调整从"调参"转向"扩数据"的根本原因。

#### 2. 数据扩展方案

引入 Pascal VOC2012 数据集，与 VOC2007 合并使用：

- **VOC2007 trainval**：约 5011 张
- **VOC2012 trainval**：约 11540 张
- **合并后训练集**：约 **16551 张**，数据量提升 **3.3 倍**

修改涉及三个核心文件：

- `dataset/voc_dataset.py`：`VOCDataset` 的 `__init__` 接收 `root_dirs` 列表参数，自动聚合多目录数据
- `utils/voc_dataset_test.py`：测试脚本同步更新多目录配置
- `train.py`：训练集与验证集均指向 VOC2007 + VOC2012 联合路径

#### 3. 第四次 LR Finder 结果

加入新数据集后重新运行 LR Finder，得到以下曲线：

![LR Finder (Fourth)](lr_finder.png)

与第三次 LR Finder 的曲线形态高度相似，但在细节上有所不同：

- `1e-6` 至 `1e-4`：学习率过小，Loss 几乎不下降
- `1e-4` 至 `5e-4`：Loss 快速下降，最陡斜率集中区域（**Steepest 法 → 最优学习率**）
- `1e-3` 附近：曲线已明显趋平，Loss 接近谷底（Valley 法可参考此位置）
- `1e-2` 以上：Loss 趋于平缓

> **读图提示**：与第三次 LR Finder 相比，第四次的 Loss 谷值更低（约 8.0 vs 约 8.5），说明在更大的数据集上、相同的归一化 Loss 实现下，模型有更大的收敛潜力。但需特别注意：EMA 平滑曲线存在时序滞后性，Steepest 法指向的最陡下降段（`5e-4` 附近）比谷底位置更靠左——若按 Valley 法从谷底（`1e-2` 附近）向回退，反而容易落在曲线已趋平的区域，因此此处应以最陡处为主。

#### 4. 更新后的学习率策略

基于第四次 LR Finder 的结果，对分段阶梯学习率做微调：

```python
def get_lr(epoch):
  if epoch <= 10:
    start_lr = 1e-4
    end_lr = 5e-4
    return start_lr + (end_lr - start_lr) * (epoch - 1) / 10
  elif epoch <= 125:
      return 5e-4
  elif epoch <= 155:
      return 1.5e-4
  else:
      return 5e-5
```

主干训练学习率从 `1e-3` 调整为 `5e-4`——更靠近 LR Finder 曲线最陡处（约 `8e-4`）但略保守，在更大的数据集上追求更稳健的收敛节奏。

### 2026-06-07 — 第五次调整（跨系统运行，迁移 AutoDL）

总 epoch 数调整为 150，最后的微调阶段（lr=5e-5）从第 131 轮持续到第 150 轮（共 20 轮）。

#### 1. 为什么需要跨系统运行？

本地硬件条件有限，GPU 算力不足以支撑 VOC2007+VOC2012 联合训练的高效迭代。为解决这一问题，项目迁移至 **AutoDL** 云平台进行训练——通过租用云端 GPU 实例，训练速度得到数量级提升，完整训练流程已在云端成功跑通。

#### 2. 跨系统兼容性改造

为确保代码在本地（Windows）和云端（Linux）之间无缝切换，做了以下适配：

- **路径系统独立**：所有路径配置统一收口到 `config.py`（基于 `pathlib`），不硬编码绝对路径。本地与云端只需修改 `config.py` 中 `DATA_ROOT`、`VOC2007_DIR`、`VOC2012_DIR`、`RUNS_DIR` 的指向即可，无需改动训练逻辑。
- **设备自适应**：`train.py` 中 `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` 保证代码在无 GPU 环境下不会报错。
- **PyTorch 依赖排除**：`requirements.txt` 中不包含 PyTorch 及其相关包，避免本地与云端 CUDA 版本冲突。云端实例通常已预装对应 CUDA 版本的 PyTorch，只需 `pip install -r requirements.txt` 安装其余依赖即可直接运行。
- **运行脚本标准化**：训练入口统一通过 `python train.py` 启动，LR Finder 通过 `python run_lr_finder.py` 启动，推理通过 `python run_detect.py` 启动，不依赖任何平台特定的启动方式。

#### 3. 更新后的学习率策略

| 阶段 | Epoch | lr | 说明 |
| :--- | :---- | :- | :--- |
| Warmup（线性） | 1-10 | 1e-4 → 5e-4 | 连续增长，避免阶跃冲击 |
| 主干训练 | 11-80 | 5e-4 | 甜区中心，80 轮后 loss 平台 |
| 精细收敛 | 81-130 | 1.5e-4 | 缩小步长，逼近极值 |
| 微调 | 131-150 | 5e-5 | 最终精调 |

总训练周期：150 epochs。

### 2026-06-19 — 第六次调整（mAP 集成后的正式训练策略）

mAP 评估模块集成完毕、NMS 与推理链路修复后，正式训练启动。前五次调试训练的经验指向两个核心改进方向：**延长预热以匹配从零初始化的参数量**、**缩小学习率衰减幅度以平滑收敛轨迹**。

#### 1. 调整依据

第五次训练虽然成功收敛，但其学习率策略仍有优化空间。10 轮预热对于一个从零随机初始化、参数量 2.7 亿的模型而言偏短——参数还处在初始混乱状态时学习率已触及峰值，梯度方向尚未稳定就被迫全速前进。同时每次衰减幅度约 3.3×，属于断崖式降速，模型需要多个 epoch 重新适应新步长。

此外主干阶段 5e-4 停留了 70 轮，但实际观察显示 loss 在第 50-60 轮附近已进入平台——后 20 轮几乎没有有效进展。

#### 2. 更新后的学习率策略

```python
def get_lr(epoch):
  if epoch <= 20:
      start_lr = 1e-4
      end_lr = 5e-4
      return start_lr + (end_lr - start_lr) * (epoch - 1) / 20
  elif epoch <= 80:
      return 5e-4
  elif epoch <= 135:
      return 2e-4
  else:
      return 7e-5
```

| 阶段 | Epoch | lr | 说明 |
| :--- | :---- | :- | :--- |
| Warmup（线性） | 1-20 | 1e-4 → 5e-4 | 从 10 轮延长至 20 轮，匹配从零初始化的 271M 参数体量 |
| 主干训练 | 21-80 | 5e-4 | 60 轮，比此前缩短以规避平台期尾部 |
| 粗粒度收敛 | 81-135 | 2e-4 | 降幅从 3.3× 收窄至 2.5×，55 轮充分消化 |
| 精细收敛 | 136-170 | 7e-5 | 再降 2.9×，避免步长过小卡死在浅谷 |
| **总计** | **170** | | 比此前多 20 轮，补偿延长预热带来的训练总量 |

#### 3. 核心变化

| 对比项 | 第五次调整 | 第六次调整 |
| :--- | :--- | :--- |
| Warmup 轮数 | 10 | 20 |
| 主干轮数 | 70（11-80） | 60（21-80） |
| 第一次衰减倍率 | 3.3×（5e-4 → 1.5e-4） | 2.5×（5e-4 → 2e-4） |
| 第二次衰减倍率 | 3.0×（1.5e-4 → 5e-5） | 2.9×（2e-4 → 7e-5） |
| 总轮数 | 150 | 170 |

#### 4. 为什么衰减幅度不是均等的

第一次衰减（5e-4 → 2e-4）倍率 2.5×，第二次（2e-4 → 7e-5）倍率 2.9×。两次衰减幅度不相等，是基于一个简单的考量：**学习率越小，越需要留足余地**。5e-4 降到 2e-4 时，模型仍然有可观的更新动量，后续还有 90 轮可以用来调整。而 2e-4 降到 7e-5 后只剩 35 轮，如果降得太狠（如 5e-5），更新步长可能小到推不动参数，模型在最后阶段反而停滞。7e-5 是一个保守的上限——不会太小导致卡死，也不会太大导致末段剧烈震荡。

---

## Part 2: Bug 追踪

### 2026-06-19 — Bug 记录：`lambda_noobj` 未参与总损失计算

**发现日期**：2026-06-19

**症状**：第六次调整后启动正式训练，170 epochs 完成，loss 从 9.2 下降至 7.3 后进入平台。诊断脚本显示模型最高预测置信度仅 0.049，连 0.05 阈值都无法通过，mAP 全程为 0。

**根因**：`loss/yolo_loss.py` 中 `self.lambda_noobj = 0.5` 在 `__init__` 中定义，但 `forward()` 计算总损失时未乘入：

```python
# 错误（修复前）
total_loss = coord_loss + obj_conf_loss + noobj_conf_loss + class_loss

# 正确（修复后）
total_loss = coord_loss + obj_conf_loss + self.lambda_noobj * noobj_conf_loss + class_loss
```

**为什么这一行导致 mAP = 0**：

每张图片中，YOLOv1 的 7×7 网格里大约只有 3 个 cell 包含目标中心点，其余 ~46 个 cell 都是背景。每个 cell 预测 B=2 个 bbox，因此每张图的损失信号分布为：

| 信号方向 | 数量（每张图） | 意图 |
| :--- | :---: | :--- |
| confidence → 1 | ~3 个 responsible bbox | 有目标，需要高置信度 |
| confidence → 0 | ~95 个（46×2 背景 + 3 非负责） | 无目标，需要低置信度 |

论文设计 `λ_noobj = 0.5` 正是为了平衡这一不对称——将背景信号等效权重砍半，让"正向信号"和"负向信号"的力量对比从 3 vs 95（≈ 1:32）改善到 3 vs 47.5（≈ 1:16），给模型留出学习"这里有东西"的梯度空间。

`lambda_noobj` 漏乘后，背景压制力翻倍，网络的最优策略变成**所有 cell 全部输出 confidence ≈ 0**——这正是 170 轮训练后 observed 的现象：max score < 0.05，任何 conf_threshold 下都生成不了有效检测框。

**影响评估**：此 bug 可能不是 mAP=0 的唯一原因。模型坐标回归能力、类别判别能力是否同样受损，需要本次修复后的训练数据来验证。不排除 Loss 实现中还有其他隐藏问题。

**后续验证**（2026-06-19）：修复 lambda_noobj 后重新训练 170 epochs。train loss 从 9.24 → 7.33，val loss 从 7.52 → 7.25，与修复前几乎一致。mAP 仍为 0。lambda_noobj 修复虽然正确，但**不是导致 loss 卡在 ~7.3 平台的唯一原因**。

### 2026-06-19 — Bug 记录：Confidence Target 偏离原论文设计

**发现过程**：lambda_noobj 修复后重新训练，loss 曲线和最终数值与修复前几乎没有差异，说明还有一个更根本的问题限制了模型的学习。

回到原论文和 Loss 代码逐行对照，发现 `obj_conf_loss` 的计算中，confidence target 被设为固定值 1.0，而非论文要求的 IoU(pred, GT)。

**原论文设计**：

> "Formally we define confidence as Pr(Object) × IoU^{truth}_{pred}. If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the IoU between the predicted box and the ground truth."

论文的意图很清楚：置信度 = 预测框与真实框的 IoU。训练时 confidence target 应该是 box 当前的 IoU，而不是固定的 1。

**代码问题**（`loss/yolo_loss.py` 修复前）：

```python
obj_conf_loss = torch.sum((obj_conf_pred - obj_mask.float()) ** 2)
#                                              ^^^^^^^^^^^^^^
#                                              固定 1.0，不管框好不好
```

所有有目标的 cell 的 responsible bbox，都被要求输出 confidence=1.0。模型刚初始化、框的位置完全随机时，也被告知"你的置信度应为 1"。这在逻辑上是矛盾的——一个随机乱放的框不应该有高置信度，原论文用 IoU 作为 target 正是为了解决这个矛盾。

**修复**：

```python
obj_conf_target = (bbox1_responsible.float() * iou1 + bbox2_responsible.float() * iou2) * obj_mask.float()
obj_conf_loss = torch.sum((obj_conf_pred - obj_conf_target.detach()) ** 2)
```

`obj_conf_target` 现在等于 responsible bbox 与 GT 的 IoU：

- 框烂时 IoU≈0.05 → target≈0.05 → 模型学会降低这类框的置信度
- 框好时 IoU≈0.7 → target≈0.7 → 模型学会提高这类框的置信度
- 训练越久，框越准，IoU 越高，置信度自然跟着上升

`.detach()` 的作用：IoU 是预测框的函数，依赖 (x, y, w, h) 四个输出。如果不 detach，obj_conf_loss 的梯度会通过 IoU 回流到坐标预测上，与 coord_loss 形成竞争。coord_loss 管定位、obj_conf_loss 管控置信度校准——两条梯度应该各走各的路。

**为什么 lambda_noobj 修复后 loss 仍然停在 7.3**：

现在可以拼出完整的因果链了：

1. `lambda_noobj` 漏乘 → noobj 信号权重翻倍，背景将 confidence 压向 0
2. Confidence target = 固定 1.0 → 模型同时被要求对所有有目标的框输出 confidence=1
3. 这两股力相互矛盾，模型的一个自然妥协是输出非常小的框（w, h → 0），使 IoU 容易计算且 coord loss 小，同时维持一个被压制的中等 confidence 水平

loss=7.3 就是这个妥协策略的平衡点。修复 lambda_noobj 只减半了背景压力，但没有解决"框不好也必须 conf=1"的矛盾——所以 loss 纹丝不动。

**影响评估**：这个修复与 lambda_noobj 修复共同作用，应该能解锁 confidence 的学习。预期下一轮训练中 max score 会从 0.049 明显上升，mAP 开始有数值。

---

**后续验证**（2026-06-19）：conf target=IoU 修复后重新训练。loss 降得更快（epoch 1 即 6.95 vs 上一次 9.04），但 epoch 20 后 train loss 卡在 5.09，val loss 卡在 4.99，mAP 仍为 0。

诊断脚本输出：max score = 0.013，比修复前（0.049）更低。每张图刚好 2 框，score 完全一致——说明 conf target=IoU 在从零训练时遇到冷启动问题。

### 2026-06-19 — Bug 记录：IoU 冷启动导致 conf target 过低

**根因**：原论文使用 ImageNet 预训练权重。预训练后的特征提取层已有一定表达能力，初始预测框不至于完全随机——IoU 起点远高于从零初始化。所以直接用 IoU 做 conf target 是可行的。

但从零训练时 IoU ≈ 0.01，conf target ≈ 0.01。模型被告知"你置信度应该是 0.01"，准确但没有任何推力。死循环：框不好 → IoU 低 → target 低 → 没有改善 conf 的梯度 → 框继续不好。

epoch 23 的诊断结果完全吻合这个推断：max score 从 0.049（conf target=1 那次）降到 0.013，说明模型正在正确汇报自己很差——但它需要一个最低推力来打破死循环。

**修复**（`loss/yolo_loss.py`）：

```python
obj_conf_target = (bbox1_responsible.float() * torch.clamp(iou1, min=0.3) +
                   bbox2_responsible.float() * torch.clamp(iou2, min=0.3)) * obj_mask.float()
```

`torch.clamp(iou, min=0.3)` 设了一个 0.3 的地板。框烂时 target = 0.3 给梯度推力，框好（IoU > 0.3）后 clamp 自动失效，target 切回纯 IoU。

0.3 的选择：不能太高（会退化成 conf target=1 的旧 bug），不能太低（冷启动推力不够）。0.3 是一个合理的冷启动档位——告诉模型"先试试输出 0.3，然后根据框的质量调整"。

**后续验证**（2026-06-19）：clamp(0.3) 修复后继续训练。框数从 2 升到 14，max score 从 0.013 升到 0.029——进展有但随后回落到 0.025，42 轮后 score 和 loss 都停滞。关键线索：每张图输出完全一致的 12-14 框、完全一致的 score 范围——说明 class 那侧没学到区分能力，模型对每个输入给出相同预测。

更根本的问题是：conf 停在 0.03-0.04 附近不再动。从训练日志的 loss 组成来看，obj_loss 和 noobj_loss 的梯度在这个值附近达到了信号平衡，模型没有动机输出更高的置信度。

### 2026-06-19 — 调整记录：λ_noobj 从 0.5 降至 0.1

**根因**：原论文 λ_noobj=0.5 是为 ImageNet 预训练设计的——预训练后初始 conf 偏高，需要较强压制。从零训练初始 conf≈0，noobj 信号太多了。前三次修复每次都有改善，但没改变"背景信号远多于目标信号"这一基本事实。

**修复**（`loss/yolo_loss.py`）：

```python
self.lambda_noobj = 0.1  # 原 0.5
```

选 0.1 的逻辑：原论文 lr=1.0 搭配 λ_noobj=0.5，本项目 lr=5e-4，按有效更新步长比例缩放，λ_noobj 大致该在 0.1 附近。太小（0.01）会让 noobj 失效导致大量假正例，太大（0.3+）还是压不住。

**后续验证**（2026-06-19）：λ_noobj=0.1 训练 133 轮。loss 从 7.0 降到 5.13，但 max score 卡在 0.066，mAP 仍为 0。诊断暴露了关键线索——输出并非完全不学习，而是 class 和 conf 高度不对称：class max=0.446 已开始激活，conf max=0.237、conf > 0.3 的数量为 0。所有图输出全为 person、框数固定 44——模型找到了一种能稳定降低 loss 的平衡态，而非真正学会了检测。

### 2026-06-20 — 调整记录：新增 λ_obj 权重

**根因**：obj_loss 和 noobj_loss 的信号不对称，仅靠 λ_noobj=0.1 未能完全解决。训练日志中 conf 被锁在 ~0.07——obj 信号和 noobj 信号在此处达到梯度均衡，模型没有进一步调整置信度的动力。原论文依赖 ImageNet 预训练提供初始特征表达能力，obj 信号一开始就占优。从零训练则相反，obj 信号从零起步，天然处于弱势。

**修复**（`loss/yolo_loss.py`）：

```python
self.lambda_obj = 3.0  # 新增，原论文隐式 = 1

total_loss = coord_loss + self.lambda_obj * obj_conf_loss
           + self.lambda_noobj * noobj_conf_loss + class_loss
```

λ_obj=3.0 将 obj 信号放大到与 noobj 信号基本对等的水平，恢复了合理的梯度竞争关系。这是对原论文设计的补全：预训练模型下 obj/noobj 天然平衡，从零训练则需要显式加权。

### 2026-06-20 — Bug 记录：训练 IoU 计算中的坐标尺度不匹配

**发现过程**：λ_obj 调整后重新训练 170 epochs。train loss 降至 ~4.8 后停滞，mAP 仍为 0。诊断脚本发现：**无论阈值设为多少，模型对所有输入输出完全相同的预测，每个框 conf 精确锁死在 0.28**。

0.28 太整齐了，说明有一个固定的 target 在约束 conf。检查了 obj_conf_target 的 clamp 逻辑和当前权重配置，怀疑 IoU 计算本身有问题——如果 IoU 恒为 0，clamp(min=0.3) 就会把所有 target 锁死到 0.3，而这恰好能让 conf 收敛到 0.28 附近。

回头逐行审查 `compute_iou()` 的输入数据格式。

**根因**（`loss/yolo_loss.py` 第 34-35 行）：

```python
# 修复前
iou1 = compute_iou(bbox1_pred[..., :4], bbox1_target[..., :4])
iou2 = compute_iou(bbox2_pred[..., :4], bbox2_target[..., :4])
```

`compute_iou()` 期望所有坐标在同一尺度下（`cx, cy, w, h` 均为 image-relative，范围 0~1）。但传入的 bbox 格式是混合坐标系：

| 坐标分量 | 实际尺度 | 范围 |
| :--- | :--- | :--- |
| `cx_cell, cy_cell` | cell-relative（单个 grid cell 的偏移） | 0~1 ≡ 图宽的 1/7 |
| `w, h` | image-relative（整张图的比例） | 0~1 ≡ 全图 |

`compute_iou()` 计算 `cx - w/2` 时，从 cell 尺度的中心坐标减去 image 尺度的半宽——两种不同单位的量直接做减法，等同于把物体的宽高在计算中被等比缩小了 7 倍。

**后果**：在混合坐标系中，`w/2` 只有真实值的 1/7。预测框稍微偏离 GT，两个框的 x 范围就完全不重叠 → `IoU = 0`。诊断脚本验证了一个典型场景：小物体（w=0.08, h=0.10）+ 微小中心偏移 → 混合坐标 IoU = **0.000000**，正确坐标 IoU = **0.5949**。

IoU 恒为 0 触发冷启动保护的 clamp(min=0.3)，`obj_conf_target` 被永久锁死在 0.3，模型无论怎么优化都被告知"你的置信度应该是 0.3"。
→ 网络学到的最优策略就是输出 conf ≈ 0.28（obj/noobj 均衡点）
→ **confidence 永远无法突破 0.4 的检测阈值**
→ **mAP 全程为 0**

这解释了此前所有训练中观察到的现象：loss 在 ~7 附近进入平台、max score 卡在 0.03-0.28 区间、每张图输出固定的框数和 score——模型并非没有学习，而是在一个数学上不可能赢的游戏中找到了唯一的纳什均衡。

**修复**（`loss/yolo_loss.py`）：

```python
# 创建 grid 偏移映射
device = predictions.device
grid_y, grid_x = torch.meshgrid(
    torch.arange(self.S, device=device),
    torch.arange(self.S, device=device),
    indexing="ij"
)
grid_x = grid_x.unsqueeze(0).unsqueeze(-1).float()  # (1, S, S, 1)
grid_y = grid_y.unsqueeze(0).unsqueeze(-1).float()

def to_image_coords(bbox):
    """将 bbox 从 cell-relative 转换到 image-relative 坐标。
    Input:  [cx_cell, cy_cell, w_img, h_img]
    Output: [cx_img,  cy_img,  w_img,  h_img]
    """
    cx_img = (bbox[..., 0:1] + grid_x) / self.S   # cell偏移 + grid位置 → 全图坐标
    cy_img = (bbox[..., 1:2] + grid_y) / self.S
    w_img = torch.abs(bbox[..., 2:3])
    h_img = torch.abs(bbox[..., 3:4])
    return torch.cat([cx_img, cy_img, w_img, h_img], dim=-1)

# 转换后再计算 IoU
iou1 = compute_iou(to_image_coords(bbox1_pred[..., :4]),
                   to_image_coords(bbox1_target[..., :4]))
iou2 = compute_iou(to_image_coords(bbox2_pred[..., :4]),
                   to_image_coords(bbox2_target[..., :4]))
```

关键点说明：

- `cx_img = (cx_cell + grid_col) / S`：cell 偏移（0~1）加上 grid 列号（0~6），除以 S=7，得到 0~1 的全图相对坐标
- `torch.abs(w)`：预测的 w/h 可能为负（网络输出无约束），取绝对值防止 `w/2` 符号翻转
- `to_image_coords` 同时对 prediction 和 target 做转换，保证两者在统一坐标系内比较

**验证**：

单元测试 `utils/test_iou_scale_fix.py` 覆盖三种场景：

| 场景 | Buggy IoU（混合坐标） | Fixed IoU（图像坐标） |
| :--- | :--- | :--- |
| 小物体 + 微小中心偏移 | **0.0000** | 0.5949 |
| 近完美预测（offset=0.02） | **0.0000** | 0.8257 |
| 完美预测（pred == target） | 1.0000 | 1.0000 |

**与前序 Bug 的关系**：这个 Bug 解释了为什么 conf target=IoU 的修复（2026-06-19）和 λ_obj 调整（2026-06-20）之后，模型依然无法突破。此前每一步修复都正确且必要，但都建立在一个有缺陷的 IoU 计算之上——IoU 永远是 0，所有围绕 confidence target 的设计都无法发挥作用。修复 IoU 坐标转换后，前序所有修复（IoU target、clamp 地板、λ_obj 权重）才能第一次真正影响训练。

---

### 2026-06-20 — Bug 记录：标注坐标归一化未使用原图尺寸

**发现过程**：修复 IoU 坐标转换后，loss 函数侧的坐标系已经一致。但随即想到另一个问题——训练数据的标签坐标本身是否准确？如果标签一开始就编码到了错误的 grid cell，那 IoU 算得再准也是对着错误的 target 在算。

**根因**（`dataset/voc_dataset.py` 第 108-125 行）：

```python
# 修复前
def _encode_target(self, boxes, labels, S=7, image_size=448):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    cx /= image_size   # ← 始终除以 448
    cy /= image_size
    w /= image_size
    h /= image_size
```

Pascal VOC 的 XML 标注坐标是相对于**原始图片分辨率**的像素值。常见 VOC 图片尺寸有 $500 \times 375$、$500 \times 333$、$375 \times 500$ 等，只有少数恰好是 $448 \times 448$。但代码在 DataLoader 的 `transform` 中将图片 Resize 到了 $448 \times 448$，而标签归一化却始终除以固定值 448。

对于一张 $500 \times 375$ 的图片：

- `cx_new = 200 / 500 = 0.40`（正确）
- `cx_old = 200 / 448 = 0.446`（错误，偏大 11.5%）
- `cy_new = 187.5 / 375 = 0.50`（正确）
- `cy_old = 187.5 / 448 = 0.419`（错误，偏小 16.3%）

中心坐标误差高达 `dx=0.046, dy=0.081`。对于 7×7 的网格，每个 cell 跨度为 1/7 ≈ 0.143。dy=0.081 的误差相当于**超过半个 cell**，足以将物体编码到错误的 grid cell。

单元测试验证了一个典型场景：

| 对比项 | 旧代码（÷448） | 新代码（÷500/375） |
| :--- | :--- | :--- |
| cx | 0.4464 | 0.4000 |
| cy | 0.4185 | 0.5000 |
| **grid cell** | **(row=2, col=3)** | **(row=3, col=2)** |

物体被编码到了**完全不同的 grid cell**——行和列都错了。

**后果**：

- Resize 后的图片中，物体视觉特征出现在某个位置，但标签告诉模型物体在另一个位置
- 模型被要求学会"看到 A 处的猫，在 B 处输出检测框"——一个不可能的任务
- 即使用最准确的 IoU 计算，也是在对着**错位的标签**做监督

**修复**（`dataset/voc_dataset.py`）：

`__getitem__` 中在 transform 之前获取原图尺寸：

```python
image = Image.open(image_path).convert('RGB')
orig_w, orig_h = image.size   # ← 在 Resize 之前获取

# ... transform ...

target = self._encode_target(boxes, labels, orig_w, orig_h)
```

`_encode_target` 签名变更：

```python
def _encode_target(self, boxes, labels, orig_w, orig_h, S=7):
    cx /= orig_w    # 用原图宽度归一化 x 坐标
    cy /= orig_h    # 用原图高度归一化 y 坐标
    w /= orig_w     # 宽度只和原图宽度有关
    h /= orig_h     # 高度只和原图高度有关
```

`_encode_raw_target` 同步修改，接收 `orig_w, orig_h` 参数。

**为什么 w 用 orig_w、h 用 orig_h**：YOLOv1 的 bbox 回归目标中，w 和 h 分别是框宽和框高占全图宽和全图高的比例。图片被 Resize 到 448×448 后，宽高比被拉伸，但标签的归一化应该在 Resize 之前完成——因为网络学习的是"框占全图百分比"这个几何属性，而非像素宽高。w 只与原始宽度有关，h 只与原始高度有关。

**验证**：单元测试覆盖了非正方形图像（500×375）、正方形非 448 图像、以及 448×448 图像（确保向后兼容）。448×448 图像的新旧结果完全一致。

---

**两个 Bug 的联合效应**：

至此可以拼出 mAP=0 的完整因果链：

1. **标注坐标错误**（Bug 2）→ 网格标签错位，模型对着错误的位置学
2. **IoU 计算尺度错误**（Bug 1）→ obj_conf_target 被 clamp 锁死在 0.3
3. **conf target=IoU + clamp(0.3)**（之前的设计）→ 理论上正确，但因 Bug 1 导致 IoU 恒为 0，clamp 从未真正"失效"
4. **λ_obj=3.0**（之前的调整）→ 正确的信号平衡，但增加的只是"推 conf 往 0.3"的力度

每一步修复都是正确且必要的，但 Bug 1 和 Bug 2 位于整个训练流水线的最底层——标注编码和 IoU 计算是所有后续 Loss 设计的前提。这两个根基修正后，前序所有 Loss 层面的优化才第一次有机会真正发挥作用。当然，具体效果如何，mAP 能不能突破 0、confidence 能不能跨过 0.3 门槛，还得等下一轮训练跑完才知道。

### 2026-06-20 — Bug 记录：`λ_noobj` 过高导致置信度均衡点锁死在阈值以下

**发现过程**：修复 IoU 坐标转换和标注归一化后，重新训练 170 epochs。mAP 全程仍为 0，只有极少数 batch 偶尔出现非零值。这一现象无法用之前已修复的 Bug 解释——IoU 计算正确、标注坐标正确，为什么仍然检测不到物体？

编写端到端诊断脚本 `diagnose_pipeline.py`，逐链路排查数据编码→Loss 计算→解码推理。所有前置链路通过验证（坐标编码正确、IoU 计算正确、Loss 计算无 NaN）。但在"置信度均衡点分析"模块找到了根因。

**根因**：`λ_noobj = 0.5` 对从零训练的模型过高。

原论文用 ImageNet 预训练，初始 conf 偏高。从零训练初始 conf≈0，每图约 3 个 obj 信号 vs 约 95 个 noobj 信号。`λ_obj=3.0, λ_noobj=0.5` 配置下，noobj 压制力约是 obj 推力的 5 倍。诊断脚本确认：在 clamp(0.3) 生效时 conf 均衡点约 0.048，IoU 涨到 0.5 后也只有 0.080——都低于 0.1 检测阈值。**模型在数学上就不可能输出通过阈值的置信度。**

**修复**（`loss/yolo_loss.py`）：

```python
self.lambda_noobj = 0.1  # 从 0.5 降低
```

调整后 obj/noobj 信号接近 1:1，conf 均衡点升至 clamp 时 ~0.15、真实 IoU 时 ~0.24，均超过 0.1 阈值。

**为什么之前 λ_noobj=0.1 也失败了**：DEVLOG 中曾记录 λ_noobj 降至 0.1 后训练 133 轮 max score 仍卡在 0.066。但那时 IoU 坐标转换 Bug 尚未修复——IoU 恒为 0，obj_conf_target 被 clamp 锁死在 0.3，而 λ_noobj=0.1 虽然降低了背景压制力，但 coord_loss 和 obj_conf_loss 都建立在一个有缺陷的 IoU 计算之上。现在 IoU 修复后，λ_noobj=0.1 的效果完全不同。

**验证**（2026-06-20）：

| 指标 | 修复前 (λ_noobj=0.5) | 修复后 (λ_noobj=0.1, 仅2 epoch) |
| :--- | :--- | :--- |
| mean_conf_obj | ~0.005（被压制） | 0.008 → 0.288（2 epoch内飙升） |
| coord_loss | 下降缓慢 | 54 → 8（快速改善） |
| mAP (epoch 2) | 0.0000 | **0.0004**（首次出现非零值！） |

- 2 epoch 训练：train_loss 6.46 → 3.41，val_loss 4.87 → 3.23
- mean_conf_obj 在 2 epoch 内从 0.008 升至 0.288，稳定在 clamp 地板 0.3 附近
- mAP 从 0 突破到 0.0004，虽是微小数值但证明了"模型开始产生有效检测"这一质变
- 模型需要更多 epoch 来学习类别区分——class_loss 目前仍然较高

**与历史修复的关系**：这 6 个修复构成了一条完整的因果链（时间倒序）：

1. **Bug 1 (IoU 尺度)** 和 **Bug 2 (标注归一化)** → 修复了数据流水线的最底层
2. **Bug 3 (λ_noobj 漏乘)** → 修复了 loss 公式隐患
3. **Bug 4 (conf_target=1)** → 修复了 target 设计
4. **Bug 5 (IoU 冷启动)** → 添加了 clamp 地板保护
5. **Bug 6 (λ_obj 新增)** → 增强了 obj 信号推力
6. **Bug 7 (λ_noobj 过高, 本次)** → 降低了 noobj 信号压制力

每一步都正确且必要。但 Bug 7 的特别之处在于：前面的修复虽然让流水线变得"正确"，但信号不对称问题一直存在。只有把 λ_noobj 降到合适的水平，obj 信号才能真正"胜出"，模型才有数学上的可能性去学习"输出高于阈值的置信度"。

---

## Part 3: 训练历史数据（2026-06-01 ~ 2026-06-19）

以下五次训练均发生在 mAP 评估模块集成之前。当时的训练只能通过 loss 判断收敛状态，缺乏检测精度的量化指标，因此这五轮本质上属于调试阶段——用于排查 loss 实现漏洞、验证学习率策略、测试数据量扩展效果。

### 共享训练配置

- 数据集：VOC2007 + VOC2012（train+val 共约 16551 张）
- Batch size：16，优化器：SGD（momentum=0.9，weight_decay=5e-4）
- 梯度裁剪：max_norm=50.0

### 第五次训练结果

| 指标 | 数值 | 所在 Epoch |
| :--- | :--- | :---: |
| Best Train Loss | 7.302 | 146 |
| Best Val Loss | 7.219 | 111 |
| Best Batch Loss | 3.346 | 61 |

### 历史训练记录

| 轮次 | 数据集 | Loss 归一化 | 平台期 train / val Loss | 根因分析 |
| :--- | :--- | :--- | :--- | :--- |
| 第一次 | VOC2007 | `sum()`（未除 batch_size） | ~124 / ~122 （÷16 ≈ 7.8 / 7.6） | 阶跃式 warmup + lr 过大 |
| 第二次 | VOC2007 | `sum()`（未除 batch_size） | ~124 / ~122 （÷16 ≈ 7.8 / 7.6） | lr 策略已正确，但 Loss 实现有漏洞 |
| 第三次 | VOC2007 | `sum()/batch_size` | ~7.9 / ~7.7 | Loss 归一化修复，但数据量成为新瓶颈 |
| 第四次 | VOC2007 + VOC2012 | `sum()/batch_size` | ~7.30 / ~7.20（lr=5e-4 约 80 轮后平台） | 数据量提升 3.3 倍，lr 主干降至 5e-4 |
| 第五次 | VOC2007 + VOC2012 | `sum()/batch_size` | ✅ 收敛，~7.302 / ~7.219 | lr 主干缩短至 80 轮，迁移 AutoDL 云算力 |

> 注：表中的 Loss 量级差异仅来自聚合方式不同。若将第一次、第二次的 `sum()` 结果除以 batch_size（16），或将第三次的归一化结果乘回 batch_size，三者的平台期 Loss 会落在同一个量级（约 120–126 / ~7.6–7.8）。这一数值上的高度一致，排除了"学习率不同导致不同平台"的可能性，进一步指向数据量不足的假设。

---

## Part 4: 阶段反思（2026-06-17）

由于是第一次独立做完整的深度学习项目，经验不足，导致前期过度关注 Loss 值的变化。`train.py` 训练脚本完成后，迫不及待地就开始训练，并长时间陷入"调整学习率 → 重新训练"的循环，苛求 Loss 要降到足够低，却忽略了一个基本事实：YOLOv1 毕竟是十年前的模型了，在当时的技术水平下，Loss 的收敛天花板本身就有限。

后续了解到 mAP 评估指标后，思路发生了变化——不再过多纠结于 Loss 值是否好看，而是着手加入 mAP 评测模块，并同步推进 NMS 以及推理模块（`detect.py`）的完成，同时对训练脚本进行了多轮改进。目前这一阶段已初步告一段落，但 lambda_noobj 的发现说明 Loss 实现中可能还藏着其他问题——调试远未结束。既然加入了 mAP 评估，前面五次调整学习率的训练记录依然会保留——毕竟是自己一步步走过来的路，每一次实验也都是货真价实做了的。但从现在开始，后续的训练才算得上是真正正式、有效的训练。

此外，新增了 `utils/plot_utils.py` 模块用于训练数据的可视化，专门绘制训练曲线。其中主要包含两个函数：

- **`plot_training_curve`**：绘制训练综合曲线，采用双 y 轴设计——左轴为 Loss 曲线（Train Loss + Val Loss），右轴为 mAP 曲线，一张图即可纵览训练全貌。
- **`plot_single_metric`**：用于专门绘制单指标曲线，支持 mAP、Train Loss、Val Loss 三种指标的独立可视化。

坦白来说，现在的程序里面仍然还有很多需要优化的地方——不只是功能模块需要更规范的组织，还有一些过于保守的冗余防御性编程写法，其实是可以简化的。不过我觉得当前阶段的重心应该暂时放在把完整流程跑完，而非过早陷入细节打磨。

---

### 2026-06-20 — 交叉验证：逐项排查 + Overfit 测试

基于训练曲线和 DEVLOG 中的历史记录，整理了按概率排序的怀疑列表。核心直觉：**"模型不像完全没学会，更像评估告诉你它没学会"**——train loss 和 val loss 几乎重合且已进入平台，这通常不是学习率问题，而是监督信号或评估链路的问题。

逐一编写测试验证每个怀疑：

#### 验证 1: 训练 vs 推理坐标一致性

直接比对 `yolo_loss.py` 的 `to_image_coords()` 和 `detect.py` 的 `decode_predictions()` 中坐标转换公式：

| 组件 | 公式 |
| :--- | :--- |
| 训练 (loss) | `cx_img = (cx_cell + grid_x) / S` |
| 推理 (detect) | `x = (bbox_preds[..., 0] + grid_x) / S` |

对相同输入的手动计算结果：训练 `cx=0.485714`，推理 `x=0.485714` → **完全一致**。

→ **排除**：推理解码 Bug。

#### 验证 2: mAP 评估链路

逐行审查 `map.py`：torchmetrics API 使用正确（`boxes=[x1,y1,x2,y2]`，归一化 [0,1]），xywh→xyxy 转换正确，pred 与 target 坐标空间一致。手动构造完美匹配的预测和目标框，验证转换公式无误。

→ **排除**：mAP 计算代码 Bug。

#### 验证 3: Label 编码抽样

随机抽取 10 个 VOC 验证集样本，检查 `_encode_target` 的编码结果。9/10 通过验证（cx_cell/cy_cell 在 [0,1)、w/h > 0、conf==1.0、单个类别激活）。1/10 出现一个 cell 内有 2 个 class 被激活的情况——这是因为两个不同类别的物体落在了同一个 7×7 grid cell 中，这是 YOLOv1 的已知架构限制，而非实现 Bug。

→ **部分确认**：Label 编码在 YOLOv1 架构限制内正确。

#### 验证 4: Responsible Box 逻辑

构造受控场景（bbox1 near-perfect, bbox2 far-off），验证 loss 函数中 responsible bbox 的选择逻辑。结果：`iou1=0.892, iou2=0.071`，bbox1 正确选为 responsible，`obj_conf_target=0.892`（clamp 未激活，因为 IoU > 0.3）。

→ **排除**：Responsible Box 逻辑错误。

#### 验证 5: Class 分支

2 epoch 模型在验证集上的 class 分布：person 86.4%，chair 9.1%，diningtable 4.5%，其余 17 类为 0。这与 DEVLOG 中记录的"所有图输出 person"一致——class 分支确实存在严重的类别偏向。但这在 2 epoch 阶段是预期行为（class_loss 仍高达 ~40），需要更多训练才能收敛。

→ **确认**：Class 分支存在偏向，需要持续监控。

#### 验证 6: Overfit 测试（最权威）

**这是行业中排查检测器最有效的方法**——用 1 张图训练，检验模型是否能过拟合。如果 1 张图训 1000 步 mAP 仍然是 0，那就 100% 是代码 Bug。

选取 VOC2007 的 `000007.jpg`（含 1 个 GT: class=car, cell(3,4)），训练 500 steps（lr=0.001, SGD+momentum）：

| Step | Loss | Conf Max | N Pred | Best IoU |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 3.772 | 0.016 | 0 | 0.000 |
| 50 | 0.148 | 0.738 | 1 | **0.953** |
| 100 | 0.048 | 1.022 | 1 | 0.922 |
| 300 | 0.006 | 0.960 | 1 | 0.974 |
| 500 | **0.012** | **0.979** | 1 | **0.971** |

最终预测：`[0.641, 0.571, 0.716, 0.838] score=0.961 class=car` ——与 GT 几乎完美匹配。

**结论：模型能在 50 步内学会一张图上的目标检测。训练流水线（数据编码→前向→Loss→反向→解码）完好。**

#### 排查结论

| 怀疑项 | 验证结果 |
| :--- | :--- |
| mAP 计算代码 Bug | ❌ 排除 — map.py 正确 |
| 推理解码 Bug | ❌ 排除 — 与训练完全一致 |
| Label 编码问题 | ⚠️ 架构限制，非 Bug |
| Responsible Box 错误 | ❌ 排除 — 逻辑正确 |
| Conf 分支被压死 | ✅ **确认 — λ_noobj=0.5 太高 (已修复)** |
| Class 分支异常 | ⚠️ 存在偏向，需持续监控 |
| 学习率问题 | ❌ 排除 — Overfit 测试证明 lr=0.001 有效 |

最初的直觉（"模型不是在乱学，而是评估不让你看到它在学"）对了一半：模型确实在学，但瓶颈不在评估链路——而在 **训练链路的信号平衡**（λ_noobj），obj/noobj 信号比 1:5.3 把置信度天花板压在了 0.048。

Overfit 测试是最终裁决：模型**能**学会检测，mAP=0 不是代码逻辑错误，而是 obj/noobj 信号不对称导致的置信度天花板。修复 λ_noobj 后，2 epoch 训练已显示 mAP 破零（0.0004），mean_conf 从 0.008 升至 0.288。

### 2026-06-21 — Bug 记录：Mode Collapse（模式坍塌）

**症状**：λ_noobj 从 0.5 降至 0.1 后训练 11 epoch，mAP 在 epoch 5 出现 0.0006 后随即归零。验证集上每张图输出完全相同的预测——20 张随机图片只有 1 种 class 序列（chair×9 + diningtable×2 + person×49），所有图 conf max 精确一致到 3 位小数。模型根本没在看图。

**诊断过程**：

编写诊断脚本直接加载 best_model.pth，对 20 张随机验证集图片做逐图推理。发现：

- 所有图输出完全相同的 class 分布
- conf_max 在所有图上精确一致
- 每张图 60 个预测框（conf_th=0.01），无一例外

这不是过拟合（train loss ≈ val loss），而是 mode collapse——模型找到了一个不依赖输入的捷径：输出固定模板就能在平均意义上最小化 loss。

逐一尝试：

- λ_noobj 0.1→0.05：mAP 仍 0.0002，坍塌持续
- warmup 20→5 epoch：无效
- noobj cell class 均匀化正则（权重 0.001）：无效
- lr 拉到 0.001：无效

**根因**：`models/yolov1.py` 的 24 层卷积 **没有任何归一化层**。原 YOLOv1 论文中大量使用了 BatchNorm，但当前实现只有 `Conv2d → LeakyReLU`。对于堆叠了 24 层的深度网络，从零训练时如果没有 BN：

- 各层激活值尺度不可控，深层的梯度呈指数衰减
- 模型很快收敛到常数输出——常数输出在激活分布不稳定的网络中是一种"安全策略"
- 常数输出在损失函数上能找到局部最优（预测 person 在多数情况下碰对，低 conf 满足 noobj 约束）
- 一旦陷入这个局部最优，SGD 在此处的梯度极小，即使提高 lr 也难以逃逸

这解释了为什么 Overfit 测试能完美工作（单张图不存在"平均化"退路，IoU 0.97, conf 0.98），但全量训练却坍缩为模板。

**修复**（`models/yolov1.py`）：

为全部 24 层 Conv2d 添加 BatchNorm：

```python
# 修复前
nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
nn.LeakyReLU(0.1),

# 修复后
nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, bias=False),
nn.BatchNorm2d(128),
nn.LeakyReLU(0.1),
```

同时将所有 Conv2d 的 `bias` 设为 `False`——BN 自带可学习的 `γ`（scale）和 `β`（shift），卷积层的 bias 完全冗余。

**验证**（2026-06-21）：

加 BN 后 2 epoch 训练：

| 指标 | 加 BN 前 | 加 BN 后 |
| :--- | :--- | :--- |
| mAP (epoch 2) | 0.0006 | **0.0017** |
| 输出模式数/40 图 | 1（坍塌） | **28** |
| 激活类别数 | 3 | **10** |
| 每图平均类别 | 1.0 | **3.5** |

模式坍塌完全解除。person 仍占 77%（VOC 中 person 本身就是最大类），随训练会自然分散。

**为什么之前没发现**：

λ_noobj=0.5 时，noobj_loss 异常强大，强制 conf≈0 遍布所有 cell——这反而"意外地"阻止了模板形成（模型必须看输入才能决定哪里放 conf）。当 λ_noobj 降到 0.1/0.05 后，noobj 约束减轻，模型发现了输出中等置信度 + 固定模板的捷径。BN 缺失这个架构层面的问题在强 noobj 约束下被掩盖了。

**当前训练配置总结**（经历 8 个 Bug 后）：

| 配置项 | 值 | 说明 |
| :--- | :--- | :--- |
| λ_coord | 1 | loss 按 batch 归一化后的等效权重 |
| λ_obj | 3.0 | 从零训练需强化 obj 推力 |
| λ_noobj | **0.05** | 从 0.5→0.1→0.05，对抗信号不对称 |
| noobj class reg | 0.001 | 防止 class 坍塌 |
| warmup | 5 epoch | 1e-4→5e-4，从 20 大幅缩短 |
| 主干 lr | 5e-4 | 80 epoch |
| BN | ✅ **全部 24 层** | 架构修复 |
| Overfit 测试 | ✅ 5/5 通过 | 回归验证 |

---

### 2026-06-21 — 修复记录：数据划分泄露 + 每 Epoch mAP 评估

**发现过程**：Mode Collapse 修复后复盘训练配置，重新审视数据划分方式。发现了一个早该发现的问题——**验证集被训练集完全包含**。

**数据泄露的诊断**：

VOC2007 和 VOC2012 的 `ImageSets/Main/` 下有三个文件：

- `train.txt`：纯训练图片
- `val.txt`：纯验证图片
- `trainval.txt`：train + val 合并

之前的代码：

```python
# voc_dataset.py（修复前）
if self.split == 'train':
    txt_file = ... / 'trainval.txt'   # ← 读了 train+val 的全部数据
else:
    txt_file = ... / 'val.txt'        # ← val 是 trainval 的子集
```

数学上 `trainval = train ∪ val`，因此 `val ⊂ trainval`——验证集的 **8333 张图片全部被训练过**。任何 mAP 数值都是"考原题"得出的，对泛化能力的评估毫无意义。

**修复方案——还原论文做法**：

论文原文：**训练用 VOC2007 trainval + VOC2012 trainval，评估用 VOC2007 test**。

之前无法这么做是因为只下载了 `VOCtrainval_06-Nov-2007.tar` 和 `VOCtrainval_11-May-2012.tar`，缺少 `VOCtest_06-Nov-2007.tar`（VOC2007 测试集）。补齐后：

1. 解压 `VOCtest_06-Nov-2007.tar` 到 `VOCdevkit/` 目录
2. 测试集自动合并进 `VOC2007/`：
   - `JPEGImages/`：5011 + 4952 = **9963** 张
   - `Annotations/`：5011 + 4952 = **9963** 个 XML
   - `ImageSets/Main/test.txt`：**4952 行**（与 trainval 完全隔离）

**代码修改**：

`dataset/voc_dataset.py`：

```python
if self.split == 'train':
    txt_file = ... / 'trainval.txt'     # 训练：全量 train+val
elif self.split == 'test':              # 新增
    txt_file = ... / 'test.txt'         # 评估：完全独立
else:
    txt_file = ... / 'val.txt'          # 保留，向后兼容
```

`train.py` 验证集改为：

```python
val_dataset = VOCDataset(
    root_dirs=[str(VOC2007_DIR)],       # test.txt 仅在 VOC2007 中存在
    split='test'                        # 独立于训练数据
)
```

**最终数据划分**：

| 集合 | 图片数 | 来源 |
| :--- | :--- | :--- |
| 训练集 | **16,551** | VOC2007 trainval + VOC2012 trainval |
| 评估集 | **4,952** | VOC2007 test（完全独立） |
| 训练 ∩ 评估 | **0** | ✓ 无数据泄露 |

### 额外改进：每 Epoch 评估 mAP

此前 mAP 仅在 `epoch % 5 == 0` 时评估，其余 epoch 的 CSV 中 mAP 列填 0.0。现改为每 epoch 都运行 `evaluate_map()` 并在终端输出 `mAP@0.5: {value}`。训练曲线图上 mAP 曲线从稀疏变为连续，无需修改绘图代码。

**影响评估**：

这是修了 8 个 Bug 后第一次能在**完全正确的数据划分**下评估模型。此前所有训练的 mAP 数值（包括非零的 epoch）都是在验证集泄漏的前提下的结果——模型的真实泛化能力可能比记录中更弱，也可能更强（训练数据不再被验证集"污染"，模型需要在完全未见过的 test 集上证明自己）。

---

### 2026-06-21 — 调整记录：学习率切换为余弦退火

**背景**：修复数据划分后首次在 VOC2007 test 上完成 170 epoch 训练。mAP 达到 0.124，但训练日志暴露出阶梯衰减的严重问题：

- `lr=5e-4` 阶段（epoch 6-80）：**75% 的 epoch val_loss 爆炸**（最高 1355），参数在 loss landscape 上剧烈震荡
- `lr=2e-4` 阶段（epoch 81-135）：爆炸几乎消失（仅 2/55），mAP 从 0.065 稳步升至 0.124
- `lr=7e-5` 阶段（epoch 136-170）：完全稳定，但 mAP 停滞在 0.123

数据很清楚：lr=5e-4 让 75 个 epoch 的算力浪费在震荡中，lr=2e-4 才是稳定工作区间，而阶梯衰减的每级幅度需要人工猜。LR Finder 能测短期梯度响应但测不出长期稳定性。

之前在 Part 1 中选择阶梯衰减是为了减少调试变量——当时 Loss 实现未验证、数据划分有泄漏、BN 未加入。现在这些前提都已修复，是时候换用更先进的调度策略。

**修改**（`train.py`）：

```python
# 原 get_lr / set_lr 注释保留，新增调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,   # 170
    eta_min=1e-5
)
```

`eta_min=1e-5` 设一个很小的地板，让模型在末段有足够空间精细收敛。主循环中 `scheduler.step()` 在 val_loss 计算之后执行，lr 通过 `scheduler.get_last_lr()[0]` 记录。

**对比**：

| 对比项 | 阶梯衰减（旧） | 余弦退火（新） |
| :--- | :--- | :--- |
| 衰减方式 | 离散跳变 | 连续平滑 |
| 参数数量 | 4 级 lr + 3 个衰减点，共 7 个超参 | T_max + eta_min，2 个 |
| 最高 lr 持续时间 | 75 epoch 全速，后 70 epoch 震荡 | 从峰值持续衰减，无震荡风险 |
| 断点续训 | 需手动跟踪 epoch 确定当前 lr | 加载 scheduler_state_dict 即可 |

**其他修复**（同步完成）：checkpoint 中新增 `scheduler_state_dict`，resume 时自动恢复调度器状态。

---

### 2026-06-21 — 调整记录：训练数据增强加入随机水平翻转

mAP 0.124 与论文 0.634 之间最根本的差距来自过拟合——train_loss 0.33 vs val_loss 2.70，差了 8 倍。数据增强只有 ColorJitter（亮度+饱和度），缺少 VOC 标准增强中影响最大的**随机水平翻转**。

**修改**（`dataset/voc_dataset.py`）：在 `__getitem__` 中，仅训练模式下 50% 概率水平翻转 PIL 图像，同时翻转 bbox 坐标（`xmin' = W - xmax, xmax' = W - xmin`），标注不会错位。验证/测试模式不受影响。

---

持续更新中 · Last updated: 2026-06-22 (输出激活函数 + 防过拟合 + 数据增强 + SGDR 周期缩短)

---

### 2026-06-21 — 调整记录：余弦退火重启 + 数据驱动参数分析

**背景**：前两次正式训练（step decay: 5e-4→2e-4→7e-5, 同一配置）产出了大量可对比数据。Run 083804（数据泄露，val⊂train）mAP=0.83 但无意义；Run 152120（独立 test）mAP=0.124 才是真实泛化水平。两轮数据对比揭示了三个精确问题。

#### 1. 学习率各阶段效率量化

| 阶段 | lr | Epoch | 爆炸率 | 平均 mAP | mAP 变化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 主干 | 5e-4 | 6-80 (75轮) | **32/75 (43%)** | 0.049 | 震荡无方向 |
| 收敛 | 2e-4 | 81-135 (55轮) | 1/55 (2%) | 0.100 | 0.065→0.099 ↑ |
| 微调 | 7e-5 | 136-170 (35轮) | 0/35 | 0.118 | **完全停滞** |

**关键发现**：

- 5e-4 时 val_loss 爆炸至最高 1355，75 个 epoch 算力浪费在震荡中
- 2e-4 是唯一稳定有效的学习区间
- 7e-5 太低，mAP 纹丝不动
- 阶梯衰减的每级幅度靠猜，且 75+35=110 个 epoch 在做无效功

#### 2. 最终状态 Loss 组成

| 组件 | 占比 |
| :--- | :--- |
| class | **59.5%** |
| obj | 18.1% |
| coord | 16.4% |
| noobj | 5.9% |

分类占 loss 的 60%——模型主要卡在类别区分，不是定位或置信度。DEVLOG 中记录的"person 占 77% 输出"与此一致。

#### 3. 过拟合程度

train_loss 0.33 vs val_loss 2.70，8 倍差距。但 Run 083804（数据泄露）mAP=0.83 证明架构本身有能力学好——问题在于泛化。

#### 修改（SGDR + 增强）

**`train.py` — 学习率调度重构**：

```python
# 旧: CosineAnnealingLR, lr=1e-3→1e-5, 单次下降
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=170, eta_min=1e-5)

# 新: SGDR, η_max=3e-4→η_min=1e-5, 多周期重启
cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-5)
# 配合 SequentialLR: 5 epoch linear warmup → SGDR
```

参数选择逻辑：

- η_max=3e-4：5e-4 爆炸 / 2e-4 稳定 → 取中间，短暂触及不会引发爆炸
- T_0=40, T_mult=2：首周期 40 轮充分收敛，之后 80 轮、160 轮逐步延长
- weight_decay 5e-4→1e-3：加重正则对抗 8x 过拟合

**`loss/yolo_loss.py` — 分类正则加强**：

```python
# 旧
noobj_class_reg = 0.001 * ...

# 新
noobj_class_reg = 0.005 * ...   # class 占 60% loss，需更强均匀化推力
```

**为什么 SGDR 而不是普通余弦退火**：SGDR 论文 (Loshchilov & Hutter, 2017) 的核心优势是周期性重启——每次从 η_max 重新开始，模型有机会跳出当前局部最优。普通余弦退火只有一个下降通道，一旦卡在次优极小值无法自救。两次重启等于两次"重新选择路径"的机会。

**预期**：lr 稳定在 1e-4~2e-4 有效区间的比例大幅提升，爆炸风险降至近零，两个重启点可能带来 mAP 跳跃。

---

## Part 5: 架构折戟与思想重构（2026-06-21）

### 2026-06-21 — 架构折戟与思想重构：放弃"从零造轮子"，拥抱现代 Backbone

#### 问题复盘

累计修复 8 个 Bug、反复调优学习率策略和数据增强后，24 层自定义卷积的模型确实能够收敛——train loss 降到了 0.33，mAP 也突破了 0。但收敛后的泛化效果惨不忍睹：train_loss 0.33 vs val_loss 2.70，8 倍差距，mAP 卡在 0.124 再也上不去。模型在训练集上学到了东西，却完全无法推广到未见过的图片上。

重新精读原论文 2.2 节后，查明了一个被严重忽略的客观事实：

> 原作者的 24 层网络并非在 VOC 上从零训练，而是在 ImageNet 上预训练了整整一周。

论文原文清清楚楚写了，但第一次读的时候根本没意识到这意味着什么。ImageNet 有 120 万张图，VOC 训练集只有 1.6 万张——差了 75 倍。作者用 120 万张图教会网络"什么是边缘、什么是纹理、什么是形状"，然后才在 VOC 上微调检测能力。我让一个随机初始化的 271M 参数网络在 1.6 万张图上同时学几何纹理和高阶空间坐标映射——网络确实学到了东西，但泛化能力被数据量天花板牢牢锁死。

#### 第一性原理反思

目标检测本质上是 **"特征提取"** + **"位置/类别回归"**。这两个子任务对数据量的需求截然不同：

| 子任务 | 需要学习的知识 | 所需数据量 |
| :--- | :--- | :--- |
| 特征提取 | 边缘、纹理、形状、语义 | **极大**（ImageNet 级别的百万级） |
| 位置/类别回归 | 网格映射、坐标归一化、类别区分 | 中等（VOC 级别即可） |

让模型在仅有几万张图的 VOC 数据集上同时从零学习两者，违背了深度学习的数据规模规律。**这不是代码写得好不好的问题，是数学上采样不足的问题。**

#### 架构修正

放弃毫无意义的"原教旨主义复刻"。今天将 Backbone 替换为 `torchvision.models.resnet50`（ImageNet 预训练权重）。

具体改动（`models/yolov1.py`）：

- **删除**：手写的 24 层自定义卷积 + padding/stride 对齐代码（约 100 行）
- **新增**：`self.backbone` — ResNet-50，去掉最后的 GAP 和 FC，保留到 layer4（输出 14×14×2048）
- **新增**：`self.adapter` — 单层 Conv2d(2048→1024, stride=2) + BN + LeakyReLU，将 14×14 压到 7×7
- **保留**：`self.fc_layers` — 原来的全连接检测头，一行未改

**一行代码都没改的模块**：

- `loss/yolo_loss.py` — 连同那 8 个 Bug 的修复历史、obj/noobj 信号平衡、IoU 坐标转换逻辑，全部原样保留
- `dataset/voc_dataset.py` — VOC2007+VOC2012 联合训练、水平翻转增强、原图尺寸归一化，全部原样保留

损失函数才是 YOLOv1 的灵魂，我并没有丢失它。

#### 核心领悟

这 8 个 Bug 的修复历程教会了我如何系统地排查深度学习流水线——从标注编码到 IoU 计算再到 Loss 信号平衡，每一步都经得起推敲。这段经历不会因为换了一个 Backbone 就贬值——恰恰相反，正是因为经历了这些，我才真正理解了 YOLOv1 论文中每一个设计决策背后的工程考量。

但同样重要的是认识到自己的能力边界。作为一个个人开发者，时间、算力、金钱都是有限的。**真正的硬核不在于堆叠毫无意义的卷积层，而在于吃透算法的几何感知逻辑与 Loss 约束机制。** 通过引入成熟的特征提取器，本仓库将所有算力与代码重心集中在了 YOLO 最惊艳的"回归思想"上。

这也算是对自己有了一个清晰的认知：从零训练一个 271M 参数的模型需要的精力和资源，对个人开发者来说是一种难以估量的投入。现在踩坑，以后少踩坑。

**参数量变化**：

| 对比项 | 旧架构（24 层自定义） | 新架构（ResNet-50） |
| :--- | :--- | :--- |
| Backbone | ~270M（24 层手写卷积，从零训练） | **23.5M**（ResNet-50，预训练） |
| Adapter | — | 18.9M（2048→1024，stride=2） |
| Head | ~0.6M | 211.5M（未变，FC 50176→4096→1470） |
| 总参数 | ~271M | **~254M** |
| 可训参数 | 全部 | 全部（254M，backbone 可冻结为 0） |

**架构替换前后的训练效果对比：**

旧架构（24层手写卷积，数据泄露修复后首次独立 test 评估）最终 mAP = 0.124，且 train_loss 0.33 vs val_loss 2.70 差距达 8 倍——严重过拟合：

![旧架构训练曲线：mAP 天花板 0.12](runs/20260621_152120/train_curve.png)

换 ResNet-50 后首次训练（`20260622_000500`）mAP 立刻冲到 0.359，翻了近 3 倍。预训练 backbone 的本质不是"换了层卷积"，而是把特征提取的学习成本从「VOC 的 1.6 万张图」转移到了「ImageNet 的 120 万张图」上：

![ResNet-50 首次训练曲线](runs/20260622_000500/train_curve.png)

> 两张图放一起对比非常直观：旧架构 mAP 整条曲线在 0.12 以下蠕动，ResNet 从 epoch 20 起就稳定在 0.3 以上。这不是参数量的差异，是预训练特征 vs 从零学特征的差异。

---

## Part 6: 输出激活函数 + 防过拟合 + 数据增强（2026-06-22）

### 2026-06-22 — 修复记录：输出层激活函数 + 防过拟合 + 数据增强

#### 背景

ResNet-50 首次训练完成（`20260622_000500`）。mAP 峰值 0.359（vs 旧架构 0.124），翻近 3 倍，但 val_loss 卡在 1.5 下不去，推理时每张图仍只输出一种类别。

#### 诊断过程

对随机噪声跑前向传播，发现致命问题：

```text
conf  bbox1: mean=-2.55  max=0.47  min=-11.66
conf  bbox2: mean=-2.72  max=0.67  min=-9.45
class prob max per cell: mean=1.44
```

**置信度可以为负数，class "概率"可以远超 1.0。** 模型末层 `nn.Linear(4096, 1470)` 之后没有任何激活函数，输出完全无界。MSE loss 虽然也能推着走，但模型要浪费大量容量去学一个本该由激活函数免费提供的输出范围约束。

#### 修复（`models/yolov1.py`）

forward 末尾加 sigmoid 和 softmax：

```python
x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
x[..., 4]  = torch.sigmoid(x[..., 4])      # conf bbox1 ∈ [0,1]
x[..., 9]  = torch.sigmoid(x[..., 9])      # conf bbox2 ∈ [0,1]
x[..., 10:] = torch.softmax(x[..., 10:], dim=-1)  # class sum=1
return x.reshape(x.shape[0], -1)
```

**注意**：sigmoid/softmax 改变了输出空间，旧权重不兼容，必须重新训练。

#### λ_noobj 重分析

sigmoid 让 conf 初始值从 ~0 变成 ~0.5，之前的 λ_noobj 分析前提变了。重新用加 sigmoid 的模型跑前向，统计 obj/noobj 梯度量级对比：

| λ_noobj | obj : noobj 梯度比 | 判断 |
| :--- | :---: | :--- |
| 0.05 | ≈ 1 : 1.6 | 略偏压制，clamp(0.3) 地板托住 obj 方向 |
| 0.1 | ≈ 1 : 3.2 | 压制过强 |

**结论：λ_noobj=0.05 不需要改。** 历史上从 0.5→0.1→0.05 的调试路径，本质上是在补偿"没有 sigmoid 时 conf 可以为负"的问题。sigmoid 补上了，0.05 刚好。

#### Head 过拟合

Head FC 50176→4096 单层 205M 参数，占总参数 83%。Dropout(0.5) 在 1.6 万张图上挡不住。

**修复**：Dropout 0.5 → 0.7。

#### 数据增强

仅 ColorJitter(brightness, saturation) + HorizontalFlip。

**修复**（`train.py`）：ColorJitter 加 contrast=0.5、hue=0.1，新增 GaussianBlur(k=3, σ 0.1~1.5)。

#### SGDR 重启周期缩短

旧配置 T_0=40, T_mult=2，170 epoch 只有 2 次重启。训练数据（`20260622_000500`）显示：每次重启 lr 跳回 η_max 后 mAP 反而涨一波，说明模型需要高 lr 震荡来跳出记忆化。

周期太长时，大量时间耗在低 lr 区域（如 cycle 2 末尾 40+ epoch 在 1e-4 以下蠕动），512×512 维的 FC 参数空间在极低 lr 下几乎只做局部插值——这为过拟合提供了完美温床。

**修复**（`train.py`）：T_0 40 → 20，170 epoch 内从 2 次重启变为 4 次。更频繁的高 lr 震荡充当隐式正则化。

#### 本次修复清单

| 优先级 | 问题 | 修复 | 文件 |
| :---: | :--- | :--- | :--- |
| P0 | 输出无界 | sigmoid(conf) + softmax(class) | models/yolov1.py |
| P1 | class loss 占比高 | softmax 天然约束输出范围 | models/yolov1.py |
| P1 | Head 过拟合 | Dropout 0.5→0.7 | models/yolov1.py |
| P2 | 数据增强弱 | GaussianBlur + ColorJitter 扩展 | train.py |
| P2 | 重启周期过长 | SGDR T_0 40→20，2次→4次重启 | train.py |
| P3 | λ_noobj | 不调，sigmoid 下数学上刚好 | — |

---

## 2026-06-22 — 全量冲刺：输出约束 + CE分类 + VOC07评估

### 诊断

`runs/20260622_122515` 给出的结论很明确：

- best mAP = 0.3633，epoch 143
- best val_loss = 1.3055，epoch 144
- epoch 146 重启回 3e-4 后，mAP 从 0.36 附近掉到 0.33
- 最后 epoch 170 mAP = 0.3104，不代表最好模型
- 512 张 test，conf=0.1 时平均 14.1 框/图，假阳性偏多
- 预测类别偏 person / chair / car
- bbox 最大 w/h 超过 1.0，展示和评估都需要 clamp

**SGDR 重启破坏的完整 epoch-by-epoch 数据：**

| epoch | lr | mAP | 变化 |
| :--- | :--- | :--- | :--- |
| 143 | 1.1e-5 | **0.3633** | ← 全训练最佳 |
| 144 | 1.0e-5 | 0.3599 | |
| 145 | 1.0e-5 | 0.3614 | |
| **146** | **3.0e-4** | **0.3328** | ← SGDR 重启，lr 翻 30 倍，mAP 暴跌 3 个点 |
| 147 | 3.0e-4 | 0.3225 | |
| 148 | 3.0e-4 | 0.3248 | |
| 149 | 3.0e-4 | 0.3093 | |
| 170 | 2.8e-4 | 0.3104 | ← 最终，比 best 低 5 个点 |

![训练曲线：SGDR 重启破坏](runs/20260622_122515/train_curve.png)

> 图中 epoch 146 处 mAP 曲线（蓝线）的断崖下跌清晰可见。lr 从 ~1e-5 跳回 3e-4 后，模型再也没回到 0.36 以上。**best model 在 epoch 143，不在 epoch 170。** 这就是为什么新策略去掉 SGDR 重启——宁可慢，不自杀。

判断：模型能学，但上限被三件事压住：class 梯度弱、后期 lr 重启太猛、bbox 输出空间不干净。

### 修改

- `models/yolov1.py`
  - x/y/w/h 全部 sigmoid
  - conf sigmoid
  - class 保持 logits，不在 forward softmax

- `loss/yolo_loss.py`
  - class MSE 改 CrossEntropy
  - wh 去掉 abs
  - IoU target 前 30 epoch clamp，之后真实 IoU
  - noobj class reg 降为辅助项

- `dataset/voc_dataset.py`
  - 加 bbox-aware scale / translate
  - 裁剪越界 bbox
  - 过滤无效 bbox

- `train.py`
  - backbone / adapter / head 分组 lr
  - warmup + cosine，不再 SGDR 大重启
  - 保存 best_val_model / best_map_model / best_model
  - training_log 增加 VOC07 mAP、best、保存原因、框数统计

- `detect.py`
  - class logits decode 时 softmax
  - bbox decode 不再 abs
  - xyxy clamp 到 [0,1]
  - visualize_predictions 修复无框/多框返回问题

- `utils/map.py`
  - 保留 torchmetrics mAP
  - 新增 VOC2007 11-point mAP

### 风险

- 新输出空间和旧权重不兼容，需要重新训练。
- sigmoid(w/h) 可能限制超大物体，但 VOC 中大多数目标仍在可表达范围内。
- CE 分类可能让 early loss 变大，这是正常现象。

### 预期

- 更少越界框
- class collapse 缓解
- best 权重管理更准确
- mAP 曲线不再被后期高 lr 重启打穿

### Smoke Test 验证结果（2026-06-22）

以下所有测试在 `F:\.anaconda\envs\torchenv\python.exe` 环境下执行，使用 `cuda` 设备。

#### 1. 语法检查 — 全部通过

8 个已修改文件 `py_compile` 全部 OK：

```text
models/yolov1.py      OK
loss/yolo_loss.py      OK
dataset/voc_dataset.py OK
train.py               OK
detect.py              OK
utils/map.py           OK
utils/loss_test.py     OK
run_detect.py          OK
```

#### 2. 前向 shape + 梯度流通 — 通过

- Sprint 模式：输出 `[1, 1470]`，bbox x/y/w/h/conf ∈ [0,1]（sigmoid 约束正确），class logits 无界（设计目标）
- Legacy 模式：bbox w/h ≥ 0（abs 约束），class softmax sum = 1.0（向后兼容）
- 梯度流通：166 个参数全部有有效梯度，无死节点
- 最大梯度范数 5352，最小 4.8（梯度动态范围正常）

#### 3. Loss sanity test — 通过

`utils/loss_test.py` 运行结果：Loss = 11.86，无 NaN，无 Inf。新 CE 分类 + sigmoid bbox 参数化下 loss 计算稳定。

#### 4. Mini overfit（单图 40 步）— 通过

选取 VOC2007 train 中含目标的图片（idx=0），1 张图训练 40 step（lr=1e-4, SGD+momentum=0.9）：

| 指标 | 数值 |
| :--- | :--- |
| First loss | 18.92 |
| Last loss | 2.28 |
| 预测框数 | 8 |
| Top box class | 8 (chair), score=0.41 |

**结论**：新 CE 分类 + sigmoid bbox + IoU clamp 阶段开关的 loss/输出链路可以训练，单图能收敛、能出框、类别正确。

#### 5. VOC07 评估链路 — 通过

128 张 VOC2007 test 图片上运行 `evaluate_voc07_map()`：

- 无崩溃、无 NaN、无 shape mismatch
- Fresh model mAP = 0.0（预期，未训练模型无检测能力）
- 11-point AP 计算逻辑验证通过

#### 5.5 512 张阈值扫描 — 通过

| conf | 总框数 | 框/图 | 空图数 |
| :--- | :--- | :--- | :--- |
| 0.01 | 22928 | 44.8 | 0/512 |
| 0.05 | 0 | 0.0 | 512/512 |
| 0.10 | 0 | 0.0 | 512/512 |
| 0.20 | 0 | 0.0 | 512/512 |
| 0.30 | 0 | 0.0 | 512/512 |
| 0.50 | 0 | 0.0 | 512/512 |

未训练模型 score = conf × softmax(class) ≈ 0.5 × 0.05 = 0.025，conf≥0.05 全部为空符合预期。conf=0.01 时 44.8 框/图来自 98 个 cell×bbox 中 score>0.01 的部分，NMS 前原始输出。

#### 6. 推理出图 — 通过

5 张随机 VOC2007 test 图片推理，`decode_predictions` + NMS + PIL 绑框全链路无崩溃。未训练模型 0 框输出（score = conf × softmax(class) ≈ 0.5 × 0.05 = 0.025 < 0.1），符合预期。

#### 7. 1 epoch smoke train — 通过

配置：`YOLO_EPOCHS=1 YOLO_TRAIN_LIMIT=32 YOLO_EVAL_LIMIT=64`。

| 检查项 | 结果 |
| :--- | :--- |
| train_loss | 9.90（正常，CE 分类初始高） |
| val_loss | 9.15 |
| mAP / VOC07 mAP | 0.0（32 张 1 epoch 预期） |
| best_val_model.pth | ✅ 已保存 |
| best_map_model.pth | ✅ 已保存（best_map=-1.0 初始值保证首轮落盘） |
| best_model.pth | ✅ 已保存（兼容入口） |
| checkpoint.pth | ✅ 含 model/optimizer/scheduler state |
| training_log.csv | ✅ 13 字段齐全 |
| loss_components.csv | ✅ epoch/phase 可区分 |
| train_curve.png | ✅ 已生成 |
| saved_by | val+map（双条件触发正确） |

**loss 组成分析**（smoke train epoch 1）：

| 组件 | 均值 | 说明 |
| :--- | :--- | :--- |
| class (CE) | ~120 | CE 分类初始高，正常现象 |
| noobj | ~19 | sigmoid 初始 conf≈0.5，noobj 压制力合理 |
| coord | ~12 | 坐标尚未学习 |
| obj (×3.0) | ~3 | obj 信号弱，与未学习状态一致 |

随着训练进行，class loss 会从 ~120 逐步下降，coord 和 obj 也会改善。noobj 应随 conf 下降而同步降低。

### 最终文件改动清单

| 文件 | +行 | -行 | 核心改动 |
| :--- | :---: | :---: | :--- |
| `models/yolov1.py` | +16 | -4 | x/y/w/h/conf sigmoid, class logits, legacy 模式 |
| `loss/yolo_loss.py` | +34 | -11 | class MSE→CE, wh 去 abs, IoU clamp 阶段开关 |
| `dataset/voc_dataset.py` | +43 | -1 | bbox-aware scale/translate/crop/filter |
| `train.py` | +179 | -115 | 分组 lr, warmup+cosine, 双 best 权重, CSV 扩展 |
| `detect.py` | +34 | -26 | class logits→softmax, bbox 去 abs, xyxy clamp |
| `utils/map.py` | +116 | -23 | VOC07 11-point AP/mAP |
| `run_detect.py` | +7 | -5 | LEGACY_OUTPUT 开关 |
| `utils/loss_test.py` | +10 | -8 | 修复导入路径 |
| `DEVLOG.md` | +64 | -0 | 诊断、改动、风险、测试记录 |

### 已知风险与后续建议

1. **旧权重不兼容**：`runs/20260622_122515` 的最佳模型（mAP=0.3633）需通过 `LEGACY_OUTPUT=True` 加载，否则 sigmoid 双重作用导致输出异常。`run_detect.py` 已设此开关。
2. **CE 分类早期 loss 大**：class_loss ~120 是正常现象（20 类均匀分布 CE = -ln(0.05) ≈ 3.0，× S×S 个 cell 累计），随训练快速下降。
3. **sigmoid(w/h) 约束**：w/h ∈ [0,1] 要求目标宽高不超过全图。VOC 中绝大多数目标满足此条件，但极端长宽比物体（如 train 类）的回归精度需监控。
4. **正式训练建议**：使用全量数据（16551 张），`YOLO_EPOCHS=170` 或更高，不设 `YOLO_TRAIN_LIMIT`/`YOLO_EVAL_LIMIT` 限制。
