# https://github.com/CodingMantras/yolov8-streamlit-detection-tracking

from pathlib import Path
import PIL
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

image = Image.open('i3l_logo.png')

col1, col2 = st.columns([1,3])
with col1:
    st.image(image)
with col2:
    st.title("i3L AI-based Gram Staining Detection")


classes = ["Bacilli_N", "Bacilli_P", "Cocci_N", "Cocci_P", "Fungus"]

covid_model_path = Path('best_gramstain.pt')
covid_model = YOLO(covid_model_path)

col1, col2 = st.columns(2)

with col1:
    source_img = st.file_uploader(
        "Choose an image...", 
        type=("jpg", "jpeg", "png"))

with col2:
    if st.button('Detect'):
        source_img = PIL.Image.open(source_img)
        res = covid_model.predict(source_img)
        # boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, 
            caption='Detected Image',
            use_container_width =True  )
