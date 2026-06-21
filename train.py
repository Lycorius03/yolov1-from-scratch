import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import csv
from pathlib import Path
from datetime import datetime

from dataset.voc_dataset import VOCDataset
from models.yolov1 import YOLOv1
from loss.yolo_loss import YoloLoss
import loss.yolo_loss as yolo_loss_module
from config import VOC2007_DIR, VOC2012_DIR, RUNS_DIR
from utils.map import evaluate_map
from utils.plot_utils import plot_training_curve

# torch.backends.cudnn.enabled = False

def collate_fn(batch):
  images = []
  targets = []

  for img, target in batch:
    images.append(img)
    targets.append(target)

  images = torch.stack(images, dim=0)
  return images, targets

# SGDR: CosineAnnealingWarmRestarts，T_0=40, T_mult=2, η_max=3e-4, η_min=1e-5

#Train
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch ,batch_log_file):
  model.train()
  total_loss = 0.0

  for batch_idx, (images, targets) in enumerate(loader):
    images = images.to(device)
    targets = targets.to(device)

    #Forward + Backward
    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_fn(predictions, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
    optimizer.step()
    
    total_loss += loss.item()

    #Log loss
    if batch_idx % 20 == 0:
      print(f"batch {batch_idx}/{len(loader)}, loss: {loss.item():.4f}")
      with open(batch_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, batch_idx, loss.item()])

  return total_loss / len(loader) 

#Validation
def val_one_epoch(model, loader, loss_fn, device):
  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, targets in loader:
      images = images.to(device)
      targets = targets.to(device)
      predictions = model(images)
      loss = loss_fn(predictions, targets)
      total_loss += loss.item()

  return total_loss / len(loader)


if __name__ == "__main__":
  # 超参数配置
  S = 7
  B = 2
  C = 20
  BATCH_SIZE = 16
  NUM_EPOCHS = 170
  # η_max: 5e-4爆炸, 2e-4稳定, 取中间
  LEARNING_RATE = 3e-4
  WARMUP_EPOCHS = 5
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"使用设备: {DEVICE}")

  #数据预处理
  transform = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ColorJitter(brightness=0.5, saturation=0.5),
    transforms.ToTensor(),
  ])

  #训练集
  train_dataset = VOCDataset(
    root_dirs=[
        str(VOC2007_DIR),
        str(VOC2012_DIR)
    ],
    transform=transform,
    split='train'
  )

  #验证集——VOC2007 test（独立于训练，无数据泄露）
  val_encoded_dataset = VOCDataset(
        root_dirs=[
        str(VOC2007_DIR),
    ],
    transform=transforms.Compose([
      transforms.Resize((448, 448)),
      transforms.ToTensor(),
    ]),
    split='test',
    use_encoded_target=True
  )

  #mAP评估集——VOC2007 test
  val_raw_dataset = VOCDataset(
        root_dirs=[
        str(VOC2007_DIR),
    ],
    transform=transforms.Compose([
      transforms.Resize((448, 448)),
      transforms.ToTensor(),
    ]),
    split='test',
    use_encoded_target=False
  )

  #DataLoader
  train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
  )

  val_encoded_loader = DataLoader(
    val_encoded_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
  )

  val_raw_loader = DataLoader(
    val_raw_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
  )

  #定义模型, Loss, 优化器
  model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
  loss_fn = YoloLoss(S=S, B=B, C=C).to(DEVICE)

  optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    # 5e-4→1e-3，对抗8x过拟合
    weight_decay=1e-3
  )

  # Warmup 5轮: 1e-4 → 3e-4, 然后切入SGDR
  def warmup_lambda(epoch):
      if epoch < WARMUP_EPOCHS:
          warmup_lr = 1e-4 + (LEARNING_RATE - 1e-4) * (epoch + 1) / WARMUP_EPOCHS
          return warmup_lr / LEARNING_RATE
      return 1.0

  warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
  cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer,
      T_0=40,
      T_mult=2,
      eta_min=1e-5
  )
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup, cosine],
      milestones=[WARMUP_EPOCHS]
  )

  #数据记录系统
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = RUNS_DIR / timestamp
  log_dir.mkdir(parents=True, exist_ok=True)

  log_file = log_dir / "training_log.csv"
  with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch","train_loss","val_loss","mAP","lr"])

  batch_log_file = log_dir / "batch_log.csv"
  with open(batch_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "batch", "loss"])

  comp_log_file = log_dir / "loss_components.csv"
  with open(comp_log_file, 'w', newline='') as f:
    f.write("obj_w,noobj_w,coord,class,mean_conf_obj\n")
  yolo_loss_module.COMPONENT_LOG = str(comp_log_file)

  print(f"训练日志保存至:{log_file}")
  # NUM_EPOCHS = 10

  best_val_loss = float('inf')

  RESUME_PATH = None
  start_epoch = 1

  if RESUME_PATH and Path(RESUME_PATH).exists():
    checkpoint = torch.load(RESUME_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"从epoch {start_epoch}继续训练")

  for epoch in range(start_epoch, NUM_EPOCHS + 1):
    current_lr = scheduler.get_last_lr()[0]
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} lr={current_lr:.6f}")
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch ,batch_log_file)
    val_loss = val_one_epoch(model, val_encoded_loader, loss_fn, DEVICE)

    print(f"train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    #mAP —— 每个epoch评估
    mAP = evaluate_map(model=model, loader=val_raw_loader, device=DEVICE, conf_threshold=0.1, iou_threshold=0.5)
    print(f"mAP@0.5: {mAP:.4f}")

    scheduler.step()

    #写入CSV
    with open(log_file, 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_loss, val_loss, mAP, current_lr])

    #保存最优模型
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), log_dir / "best_model.pth")
      torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'train_loss': train_loss,
      'val_loss': val_loss,
      'mAP': mAP
    }, log_dir / "checkpoint.pth")
      print(f"模型已保存 (val_loss:{val_loss:.4f}, mAP:{mAP:.4f})")

  #绘制曲线
  print("\n=== 老大，训练完成，正在绘制训练曲线喵 ===")
  plot_path = log_dir / "train_curve.png"
  plot_training_curve(log_file=str(log_file), save_path=str(plot_path))
  print("老大，训练曲线绘制完了喵！")