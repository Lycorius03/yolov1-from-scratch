import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from models.yolov1 import YOLOv1
from detect import decode_predictions, CLASS_NAMES, CLASS_COLORS
from utils.nms import non_max_suppression
from config import PROJECT_ROOT


CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PIL 字体（画中文/英文标签用）
FONT = ImageFont.truetype("arial.ttf", size=20)

def load_model(run_name="20260622_122515"):
    weight_path = PROJECT_ROOT / "runs" / run_name / "best_model.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")
    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    model.load_state_dict(torch.load(str(weight_path), map_location=DEVICE))
    model.eval()
    print(f"权重: {weight_path}")
    return model


# 预处理
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


# PIL Image → 检测框列表 [(cx,cy,w,h,score,class_id), ...] 归一化
def detect_frame(model, image_pil):
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = model(img_tensor)
    boxes_list = decode_predictions(predictions, conf_threshold=CONF_THRESHOLD, device=DEVICE)
    boxes = non_max_suppression(boxes_list[0], iou_threshold=IOU_THRESHOLD, conf_threshold=CONF_THRESHOLD)
    return boxes


def draw_boxes_pil(image_pil, boxes):
    """用 PIL 画框和标签（解决 cv2 中文乱码、字体过小的问题）"""
    draw = ImageDraw.Draw(image_pil)
    w, h = image_pil.size
    for box in boxes:
        cx, cy, bw, bh, score, class_id = box.tolist()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        color = CLASS_COLORS[int(class_id)]
        label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tb = draw.textbbox((x1, y1 - 18), label, font=FONT)
        draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=color)
        draw.text((x1, y1 - 18), label, fill="white", font=FONT)
    return image_pil


# ── 摄像头实时检测 ──
def run_camera(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    win_name = "YOLOv1 - Camera (q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("Camera mode - press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → PIL
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = detect_frame(model, image_pil)
        image_pil = draw_boxes_pil(image_pil, boxes)
        # PIL → BGR
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        cv2.putText(frame, f"Objects: {boxes.shape[0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


# ── 图片检测 ──
def run_image(model):
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    image_pil = Image.open(path).convert("RGB")
    ow, oh = image_pil.size
    boxes = detect_frame(model, image_pil)
    image_pil = draw_boxes_pil(image_pil, boxes)
    print(f"Detected {boxes.shape[0]} objects")

    # 保存
    save_path = Path(path).parent / f"detect_{Path(path).name}"
    image_pil.save(str(save_path))
    print(f"Saved: {save_path}")

    # 显示（自适应窗口大小）
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    scale = min(1200 / max(w, h), 1.0)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    cv2.putText(disp, f"Objects: {boxes.shape[0]}  |  Press any key to close",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    win_name = "YOLOv1 - Image"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── 主菜单 ──
def main():
    model = load_model()
    print(f"Device: {DEVICE}")

    root = tk.Tk()
    root.title("YOLOv1")
    root.geometry("300x200")
    root.resizable(False, False)

    tk.Label(root, text="YOLOv1 Detection", font=("Arial", 14, "bold")).pack(pady=15)
    tk.Label(root, text=f"Device: {DEVICE}", font=("Arial", 9)).pack()

    tk.Button(root, text="Camera", width=20, height=2,
              command=lambda: [root.withdraw(), run_camera(model), root.deiconify()]
              ).pack(pady=10)

    tk.Button(root, text="Image", width=20, height=2,
              command=lambda: [root.withdraw(), run_image(model), root.deiconify()]
              ).pack(pady=5)

    tk.Label(root, text="20260622_122515 | conf=0.5 iou=0.35", font=("Arial", 8), fg="gray").pack(side="bottom", pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
