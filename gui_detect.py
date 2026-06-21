import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from models.yolov1 import YOLOv1
from detect import decode_predictions, CLASS_NAMES
from utils.nms import non_max_suppression
from config import PROJECT_ROOT


CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 颜色池
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (128, 255, 0), (0, 128, 255),
    (255, 0, 128), (128, 0, 255), (0, 255, 128),
    (192, 192, 192), (128, 128, 128),
]


def load_model(run_name="20260621_152120"):
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


def draw_boxes(frame_bgr, boxes):
    """在 BGR 帧上画框"""
    h, w = frame_bgr.shape[:2]
    for box in boxes:
        cx, cy, bw, bh, score, class_id = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = COLORS[int(class_id) % len(COLORS)]
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
        cv2.putText(frame_bgr, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame_bgr


# ── 模式1: 摄像头实时检测 ──
def run_camera(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("错误", "打不开摄像头")
        return

    print("摄像头模式 — 按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = detect_frame(model, image_pil)
        frame = draw_boxes(frame, boxes)
        cv2.putText(frame, f"Objects: {boxes.shape[0]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLOv1 — Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── 模式2: 单张图片检测 ──
def run_image(model):
    path = filedialog.askopenfilename(
        title="选择图片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    image_pil = Image.open(path).convert("RGB")
    boxes = detect_frame(model, image_pil)

    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    frame = draw_boxes(frame, boxes)
    print(f"检测到 {boxes.shape[0]} 个物体")

    # 保存
    save_path = Path(path).parent / f"detect_{Path(path).name}"
    cv2.imwrite(str(save_path), frame)
    print(f"结果: {save_path}")

    # 显示
    h, w = frame.shape[:2]
    scale = min(800 / max(w, h), 1.0)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    cv2.putText(disp, f"Objects: {boxes.shape[0]}  |  Press any key to close",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("YOLOv1 — Image", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── 主菜单 ──
def main():
    model = load_model()
    print(f"设备: {DEVICE}")

    root = tk.Tk()
    root.title("YOLOv1 — 选择模式")
    root.geometry("300x200")
    root.resizable(False, False)

    tk.Label(root, text="YOLOv1 Detection", font=("Arial", 14, "bold")).pack(pady=15)
    tk.Label(root, text=f"设备: {DEVICE}", font=("Arial", 9)).pack()

    tk.Button(root, text="摄像头实时检测", width=20, height=2,
              command=lambda: [root.withdraw(), run_camera(model), root.deiconify()]
              ).pack(pady=10)

    tk.Button(root, text="图片检测", width=20, height=2,
              command=lambda: [root.withdraw(), run_image(model), root.deiconify()]
              ).pack(pady=5)

    tk.Label(root, text="Run: 20260621_152120 | conf=0.1", font=("Arial", 8), fg="gray").pack(side="bottom", pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
