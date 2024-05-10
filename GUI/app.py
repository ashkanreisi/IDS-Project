# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Hand Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}

        </style>""".format(
        padding_top=1, padding_bottom=1
    ),
    unsafe_allow_html=True,
)

# Main page heading
st.title("Hand Fracture Detection", anchor=False)

# Sidebar
st.sidebar.header("Model Configuration")

# Model Options
model_type = st.sidebar.radio(
    "Select Model", ['YOLOv8n 3M - Faster, less accurate', 'YOLOv8x 68M - Slower, more accurate', 'YOLOv9e 58M - Slower, most accurate'])

confidence = float(st.sidebar.slider(
    "Select Detection Confidence Threshold", 10, 100, 55, format="%d%%")) / 100

# Selecting Detection Or Segmentation
if model_type == 'YOLOv8n 3M - Faster, less accurate':
    model_path = Path(settings.HAND_FRACTURE_DETECTION_YOLOV8N)
elif model_type == 'YOLOv8x 68M - Slower, more accurate':
    model_path = Path(settings.HAND_FRACTURE_DETECTION_YOLOV8X)
elif model_type == 'YOLOv9e 58M - Slower, most accurate':
    model_path = Path(settings.HAND_FRACTURE_DETECTION_YOLOV9E)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

source_img = None

source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png"))

col1, col2 = st.columns(2)
with col1:
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                     use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                     use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(
            default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        if st.sidebar.button('Detect Fractures'):
            res = model.predict(uploaded_image,
                                conf=confidence,
                                imgsz=640,
                                device=0)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)
st.sidebar.markdown('''<small>[Zach Estreito](https://github.com/zestreito/) & [Ashkan Reisi](https://github.com/ashkanreisi) 2024</small>''', unsafe_allow_html=True)
