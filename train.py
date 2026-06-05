import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import csv
from datetime import datetime

from dataset.voc_dataset import VOCDataset
from models.yolov1 import YOLOv1
from loss.yolo_loss import YoloLoss

# torch.backends.cudnn.enabled = False


#学习率
def get_lr(epoch):
  if epoch <= 10:
    return 1e-4      
  elif epoch <= 125:
    return 1e-3      
  elif epoch <= 155:
    return 3e-4     
  else:
    return 1e-4 
  
def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

#Train
def train_one_epoch(model, loader, optimizer, loss_fn, device, batch_log_file):
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
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
  NUM_EPOCHS = 175
  LEARNING_RATE = 1e-3
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
    root_dir="data/VOCdevkit/VOC2007",
    transform=transform,
    split='train'
  )

  #验证集
  val_dataset = VOCDataset(
    root_dir="data/VOCdevkit/VOC2007",
    transform=transforms.Compose([
      transforms.Resize((448, 448)),
      transforms.ToTensor(),
    ]),
    split='val'
  )

  #DataLoader
  train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
  )

  val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
  )

  #定义模型, Loss, 优化器
  model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
  loss_fn = YoloLoss(S=S, B=B, C=C).to(DEVICE)

  optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4
  )

  #数据记录系统
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = f"runs/{timestamp}"
  os.makedirs(log_dir, exist_ok=True)

  log_file = os.path.join(log_dir, "training_log.csv")
  with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch","train_loss","val_loss","lr"])

  batch_log_file = os.path.join(log_dir, "batch_log.csv")
  with open(batch_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "batch", "loss"])

  print(f"训练日志保存至:{log_file}")
  # NUM_EPOCHS = 10

  best_val_loss = float('inf')

  RESUME_PATH = None  
  start_epoch = 1

  if RESUME_PATH and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"从epoch {start_epoch}继续训练")

  for epoch in range(start_epoch, NUM_EPOCHS + 1):
    #更新学习率
    lr = get_lr(epoch)
    set_lr(optimizer, lr)
    
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} lr={lr}")

    #训练和验证
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, batch_log_file)
    val_loss = val_one_epoch(model, val_loader, loss_fn, DEVICE)

    print(f"train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    #写入CSV
    with open(log_file, 'a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_loss, val_loss, lr])

    #保存最优模型
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
      torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'train_loss': train_loss,
      'val_loss': val_loss,
    }, os.path.join(log_dir, "checkpoint.pth"))
      
    print(f"模型已保存 (val_loss:{val_loss:.4f})")