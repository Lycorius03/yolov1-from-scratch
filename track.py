import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from models.yolov1 import YOLOv1
from detect import decode_predictions, CLASS_NAMES
from utils.nms import non_max_suppression
from config import PROJECT_ROOT


# ── 追踪器 ──
class Track:
    def __init__(self, track_id, bbox, class_id, score):
        self.id = track_id
        self.bbox = bbox           # (cx, cy, w, h) 归一化 0~1
        self.class_id = class_id
        self.score = score
        self.age = 0
        self.hits = 1


# tracks N×4, detections M×4 归一化 cxcywh → N×M IoU
def iou_matrix(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    t = np.array([t.bbox for t in tracks])   # (N, 4)
    d = np.array(detections)                   # (M, 4)

    t_x1 = t[:, 0] - t[:, 2] / 2
    t_y1 = t[:, 1] - t[:, 3] / 2
    t_x2 = t[:, 0] + t[:, 2] / 2
    t_y2 = t[:, 1] + t[:, 3] / 2

    d_x1 = d[:, 0] - d[:, 2] / 2
    d_y1 = d[:, 1] - d[:, 3] / 2
    d_x2 = d[:, 0] + d[:, 2] / 2
    d_y2 = d[:, 1] + d[:, 3] / 2

    inter_x1 = np.maximum(t_x1[:, None], d_x1[None, :])
    inter_y1 = np.maximum(t_y1[:, None], d_y1[None, :])
    inter_x2 = np.minimum(t_x2[:, None], d_x2[None, :])
    inter_y2 = np.minimum(t_y2[:, None], d_y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    t_area = t[:, 2] * t[:, 3]
    d_area = d[:, 2] * d[:, 3]
    union = t_area[:, None] + d_area[None, :] - inter + 1e-8

    return inter / union


# greedy matching，返回 (matched, unmatched_tracks, unmatched_dets)
def associate(tracks, detections, iou_threshold=0.3, max_age=30):
    if len(tracks) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
    if len(detections) == 0:
        return [], list(range(len(tracks))), []

    ious = iou_matrix(tracks, detections)
    matched = []
    unmatched_t = set(range(len(tracks)))
    unmatched_d = set(range(len(detections)))

    # 按 IoU 降序贪婪匹配
    pairs = [(i, j, ious[i, j]) for i in range(len(tracks)) for j in range(len(detections))]
    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, iou in pairs:
        if iou < iou_threshold:
            break
        if i in unmatched_t and j in unmatched_d:
            matched.append((i, j))
            unmatched_t.remove(i)
            unmatched_d.remove(j)

    return matched, list(unmatched_t), list(unmatched_d)


# ── 主逻辑 ──
if __name__ == "__main__":
    # 配置
    RUN_NAME = "20260622_122515"
    VIDEO_PATH = None            # None = 摄像头
    OUTPUT = "outputs/track/tracked.mp4"
    CONF_THRESHOLD = 0.1         # 检测阈值，放低让人框尽量出来
    IOU_THRESHOLD = 0.5          # NMS 阈值
    TRACK_IOU = 0.3              # 追踪匹配 IoU 阈值
    MAX_AGE = 30                  # 消失多久后丢弃
    PERSON_ONLY = True           # 只追踪 person
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"设备: {DEVICE}")

    # 加载模型
    weight_path = PROJECT_ROOT / "runs" / RUN_NAME / "best_model.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"权重不存在: {weight_path}")
    model = YOLOv1(S=7, B=2, C=20).to(DEVICE)
    model.load_state_dict(torch.load(str(weight_path), map_location=DEVICE))
    model.eval()
    print(f"权重: {weight_path}")

    # 视频源
    if VIDEO_PATH:
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)     # 摄像头
    if not cap.isOpened():
        raise RuntimeError("打不开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"输入: {w_in}x{h_in} @ {fps:.1f}fps")

    # 输出
    out_dir = Path(OUTPUT).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps if fps > 0 else 15, (w_in, h_in))

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    # 追踪状态
    tracks = []
    next_id = 0
    frame_idx = 0
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    print("追踪中... 按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 检测
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            predictions = model(img_tensor)

        boxes_list = decode_predictions(predictions, conf_threshold=CONF_THRESHOLD, device=DEVICE)
        boxes = non_max_suppression(boxes_list[0], iou_threshold=IOU_THRESHOLD, conf_threshold=CONF_THRESHOLD)

        # 筛 person
        if PERSON_ONLY and boxes.shape[0] > 0:
            mask = boxes[:, 5] == CLASS_NAMES.index("person")
            boxes = boxes[mask]

        detections = []
        if boxes.shape[0] > 0:
            detections = boxes[:, :4].cpu().numpy()   # (M, 4) cxcywh

        # 匹配
        matched, unmatched_t, unmatched_d = associate(tracks, detections, TRACK_IOU, MAX_AGE)

        # 更新匹配到的 track
        for ti, di in matched:
            tracks[ti].bbox = detections[di]
            tracks[ti].age = 0
            tracks[ti].hits += 1
            tracks[ti].score = boxes[di, 4].item()

        # 删除过期 track
        tracks = [t for i, t in enumerate(tracks) if i not in unmatched_t or t.age < MAX_AGE]
        for t in tracks:
            t.age += 1

        # 新建未匹配 detections
        for di in unmatched_d:
            tracks.append(Track(next_id, detections[di],
                                int(boxes[di, 5].item()),
                                boxes[di, 4].item()))
            next_id += 1

        # 画框
        for t in tracks:
            if t.age > 0:
                continue  # 不画首次未确认的
            color = colors[t.id % len(colors)]
            cx, cy, w, h = t.bbox
            x1 = int((cx - w / 2) * w_in)
            y1 = int((cy - h / 2) * h_in)
            x2 = int((cx + w / 2) * w_in)
            y2 = int((cy + h / 2) * h_in)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.id} {CLASS_NAMES[t.class_id]} {t.score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(frame, f"Frame:{frame_idx} Tracks:{sum(1 for t in tracks if t.age==0)}",
                    (10, h_in - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        cv2.imshow("Track", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"完成，保存至: {OUTPUT}")
