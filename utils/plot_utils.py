import matplotlib.pyplot as plt
import csv
from pathlib import Path

def plot_training_curve(log_file: str, save_path:str=None):
  epochs = []
  train_losses = []
  val_losses = []
  mAPs = []

  #读取数据
  with open(log_file, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
      if len(row) < 5:
        continue
      epochs.append(int(row[0]))
      train_losses.append(float(row[1]))
      val_losses.append(float(row[2]))
      mAPs.append(float(row[3]))

  #创建图像
  fig, ax1 = plt.subplots(figsize=(10, 6))

  #(左轴)画Loss曲线
  ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
  ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss', color='blue')
  ax1.tick_params(axis='y', labelcolor='blue')
  ax1.legend(loc='upper left')

  #(右轴)画mAP曲线
  ax2 = ax1.twinx()
  ax2.plot(epochs, mAPs, 'g-', label='mAP@0.5', linewidth=2)
  ax2.set_ylabel('mAP', color='green')
  ax2.set_ylim(0, 1)
  ax2.tick_params(axis='y', labelcolor='green')
  ax2.legend(loc='upper right')

  plt.title('Training Curve (Loss & mAP)')

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"老大，训练曲线已经保存到：{save_path}了喵")
  else:
    plt.show()

def plot_single_metric(log_file: str, metric_name: str = "mAP", save_path: str = None):
  epochs = []
  train_losses = []
  val_losses = []
  mAPs = []

  #读取数据
  with open(log_file, 'r', newline='') as f:
      reader = csv.reader(f)
      next(reader)  
      for row in reader:
          if len(row) < 5:
              continue
          epochs.append(int(row[0]))
          train_losses.append(float(row[1]))
          val_losses.append(float(row[2]))
          mAPs.append(float(row[3]))

  #创建图像
  fig, ax = plt.subplots(figsize=(10, 6))

  #根据输入的指标名称，分流画图
  if metric_name == "mAP":
      ax.plot(epochs, mAPs, 'g-', label='mAP@0.5', linewidth=2)
      ax.set_ylabel('mAP', color='green')
      ax.tick_params(axis='y', labelcolor='green')
      
  elif metric_name == "Train Loss":
      ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
      ax.set_ylabel('Loss', color='blue')
      ax.tick_params(axis='y', labelcolor='blue')
      
  elif metric_name == "Val Loss":
      ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
      ax.set_ylabel('Loss', color='red')
      ax.tick_params(axis='y', labelcolor='red')
      
  else:
      return f"老大，你的指标名称输入错误！ '{metric_name}'是未知名称 ，应该输入'mAP', 'Train Loss' 或 'Val Loss'喵"

  ax.set_xlabel('Epoch')
  plt.title(f'Training Metric: {metric_name}')
  ax.legend(loc='upper left')

  if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      print(f"{metric_name} 曲线已保存至: {save_path}")
  else:
      plt.show()
