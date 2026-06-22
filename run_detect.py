import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from models.yolov1 import YOLOv1
from dataset.voc_dataset import VOCDataset
from detect import decode_predictions, CLASS_NAMES, CLASS_COLORS
from utils.nms import non_max_suppression
from config import PROJECT_ROOT, VOC2007_DIR, VOC2012_DIR


RUN_NAME = "20260607_165143"
MODE = "single"
INPUT = "test.jpg"
NUM_SAMPLES = 5
OUTPUT_DIR = "./outputs/detect"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
FONT = ImageFont.truetype("arial.ttf", size=16)

#加载 best_model.pth 权重，返回 eval 模式的模型
def load_best_model(weight_path, device="cuda"):
  model = YOLOv1(S=7, B=2, C=20).to(device)
  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()
  return model

#对单张 PIL Image 做前向推理 + NMS，返回画好检测框的 PIL Image
def detect_single_image(model, image, device="cuda", conf_threshold=0.4, iou_threshold=0.5):
  transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
  ])

  img_tensor = transform(image).unsqueeze(0).to(device)

  with torch.no_grad():
    predictions = model(img_tensor)

  boxes_list = decode_predictions(predictions, conf_threshold=conf_threshold, device=device)
  boxes = non_max_suppression(boxes_list[0], iou_threshold=iou_threshold)

  draw = ImageDraw.Draw(image)
  w_img, h_img = image.size

  for box in boxes:
    x, y, w, h, score, class_id = box.tolist()
    x1 = (x - w / 2) * w_img
    y1 = (y - h / 2) * h_img
    x2 = (x + w / 2) * w_img
    y2 = (y + h / 2) * h_img

    color = CLASS_COLORS[int(class_id)]
    label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    tb = draw.textbbox((x1, y1 - 14), label, font=FONT)
    draw.rectangle([tb[0]-2, tb[1]-1, tb[2]+2, tb[3]+1], fill=color)
    draw.text((x1, y1 - 14), label, fill="white", font=FONT)

  return image


if __name__ == "__main__":
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"使用设备: {DEVICE}")

  # 拼出 best_model.pth 路径
  weight_path = PROJECT_ROOT / "runs" / RUN_NAME / "best_model.pth"
  if not weight_path.exists():
    raise FileNotFoundError(f"老大，找不到 {weight_path} 喵，检查一下 RUN_NAME")
  print(f"加载模型权重: {weight_path}")
  model = load_best_model(str(weight_path), device=DEVICE)

  output_dir = Path(OUTPUT_DIR)
  output_dir.mkdir(parents=True, exist_ok=True)

  if MODE == "single":
    image = Image.open(INPUT).convert("RGB")
    result = detect_single_image(
      model, image, device=DEVICE,
      conf_threshold=CONF_THRESHOLD,
      iou_threshold=IOU_THRESHOLD
    )
    save_path = output_dir / f"detect_{Path(INPUT).stem}.jpg"
    result.save(save_path)
    print(f"推理完成，结果保存至: {save_path}")

  elif MODE == "dir":
    input_dir = Path(INPUT)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]
    print(f"找到 {len(image_paths)} 张图片，开始推理...")

    for img_path in image_paths:
      image = Image.open(img_path).convert("RGB")
      result = detect_single_image(
        model, image, device=DEVICE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
      )
      save_path = output_dir / f"detect_{img_path.stem}.jpg"
      result.save(save_path)

    print(f"批量推理完成，共 {len(image_paths)} 张，结果保存至: {output_dir}")

  elif MODE == "val_sample":
    print(f"从验证集随机采样 {NUM_SAMPLES} 张...")

    transform = transforms.Compose([
      transforms.Resize((448, 448)),
      transforms.ToTensor(),
    ])

    val_dataset = VOCDataset(
      root_dirs=[str(VOC2007_DIR), str(VOC2012_DIR)],
      transform=transform,
      split='val',
      use_encoded_target=False
    )

    indices = torch.randperm(len(val_dataset))[:NUM_SAMPLES].tolist()

    for idx in indices:
      image_tensor, _ = val_dataset[idx]
      image = transforms.functional.to_pil_image(image_tensor)
      image_id = val_dataset.image_ids[idx][1]
      result = detect_single_image(
        model, image, device=DEVICE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
      )
      save_path = output_dir / f"detect_{image_id}.jpg"
      result.save(save_path)

    print(f"验证集采样推理完成，共 {len(indices)} 张，结果保存至: {output_dir}")

  else:
    raise ValueError(f"老大，MODE 只能是 single / dir / val_sample，你给的是 '{MODE}' 喵")
