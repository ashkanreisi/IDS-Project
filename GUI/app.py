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
    page_title="Bone Fracture Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Bone Fracture Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Model", ['Hand Fracture Detection - YOLOv8x', 'Hand Fracture Detection - YOLOv8n'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence Threshold", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Hand Fracture Detection - YOLOv8x':
    model_path = Path(settings.HAND_FRACTURE_DETECTION_YOLOV8X)
elif model_type == 'Hand Fracture Detection - YOLOv8n':
    model_path = Path(settings.HAND_FRACTURE_DETECTION_YOLOV8N)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

source_img = None

source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
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
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                # st.write(ex)
                st.write("No image is uploaded yet!")
