from pathlib import Path

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent

# ── 数据目录 ──
DATA_ROOT = PROJECT_ROOT / "data" / "VOCdevkit"
VOC2007_DIR = DATA_ROOT / "VOC2007"
VOC2012_DIR = DATA_ROOT / "VOC2012"

# ── 输出目录 ──
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
