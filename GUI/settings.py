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
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'hand.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'hand_labeled.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DEFAULT_DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
DEFAULT_SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'
HAND_FRACTURE_DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Webcam
WEBCAM_PATH = 0
