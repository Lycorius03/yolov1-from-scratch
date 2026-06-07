import torch
import matplotlib.pyplot as plt
import copy
from pathlib import Path

from config import LOG_DIR

def lr_finder(model, loader, optimizer, loss_fn, DEVICE, start_lr=1e-6, end_lr=1e-1, num_iter=300):

  #保存原始状态
  model_state = copy.deepcopy(model.state_dict())
  optimizer_state = copy.deepcopy(optimizer.state_dict())

  lrs = []
  losses = []
  best_loss = float('inf')
  
  avg_loss = 0.0
  beta = 0.98

  #计算指数衰减学习率倍率
  mult = (end_lr / start_lr) ** (1 / num_iter)
  lr = start_lr



  #设置初始lr
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  model.train()
  iterator = iter(loader)

  for i in range(num_iter):
    try:
      images, targets = next(iterator)
    except StopIteration:
      iterator = iter(loader)
      images, targets = next(iterator)

    images = images.to(DEVICE)
    targets = targets.to(DEVICE)

    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_fn(predictions, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    
    #EMA
    current_loss = loss.item()
    avg_loss = beta * avg_loss + (1 - beta) * current_loss
    smoothed_loss = avg_loss / (1 - beta ** (i + 1))

    lrs.append(lr)
    losses.append(smoothed_loss)

    lr *= mult
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    #记录best_loss，loss爆炸前提前停止
    if smoothed_loss < best_loss:
      best_loss = smoothed_loss
    if smoothed_loss > 4 * best_loss:
      print(f"老大，Loss值炸了，快停下喵(っ °Д °;)っ！提前停止于iter{i}")
      break

  #恢复原始状态
  model.load_state_dict(model_state)
  optimizer.load_state_dict(optimizer_state)

  #画图
  min_loss = min(losses)
  plt.figure(figsize=(10, 6))
  plt.plot(lrs, losses)
  plt.xscale('log')
  plt.ylim(min_loss * 0.9, min_loss * 3.0)
  plt.xlabel('Learning Rate (log scale)')
  plt.ylabel('Loss')
  plt.title('Learning Rate Range Test')
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  save_path = LOG_DIR / "lr_finder.png"
  plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.show()
  print(f"老大，图已保存为 {save_path} 了喵")

  return lrs, losses