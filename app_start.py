# https://github.com/CodingMantras/yolov8-streamlit-detection-tracking

from pathlib import Path
import PIL
import streamlit.components.v1 as components

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
import requests

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("i3LUniversity.png", 
             use_container_width=True)


# image = Image.open('i3LUniversity.png')

st.markdown(
    "<h1 style='text-align: center;'>AI-based Gram Staining Detection</h1>",
    unsafe_allow_html=True
)
classes = ["Bacilli_N", "Bacilli_P", "Cocci_N", "Cocci_P", "Fungus"]

model_path = "best_gramstain.pt"

if not os.path.exists(model_path):
    url = "https://huggingface.co/Sadrawi/gram_stain/resolve/main/best_gramstain.pt"
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)

model = YOLO(model_path)

# ...existing code...

# ...existing code...

col1, col2 = st.columns(2)

with col1:
    source_img = st.file_uploader(
        "Choose an image...", 
        type=("jpg", "jpeg", "png"))
    detect_clicked = st.button('Detect')

if source_img is not None and detect_clicked:
    raw_img = PIL.Image.open(source_img)
    res = model.predict(raw_img)
    res_plotted = res[0].plot()[:, :, ::-1]

    col_raw, col_detected = st.columns(2)
    with col_raw:
        st.image(raw_img, caption="Raw Image", use_container_width=True)
    with col_detected:
        st.image(res_plotted, caption="Classification Result", use_container_width=True)

# ...existing code...

# ...existing code...



