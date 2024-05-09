from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'hand.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'hand_labeled.jpg'

# ML Model config
MODEL_DIR = ROOT / 'weights'
HAND_FRACTURE_DETECTION_YOLOV8N = MODEL_DIR / 'YOLOv8n-hands.pt'
HAND_FRACTURE_DETECTION_YOLOV8X = MODEL_DIR / 'YOLOv8x-hands.pt'
HAND_FRACTURE_DETECTION_YOLOV9E = MODEL_DIR / 'YOLOv9e-hands.pt'