import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🔥 Fire Supervision")
st.write("YOLOv8 火灾目标检测系统")

model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "上传图片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="原始图片")

    results = model.predict(image, conf=0.25)

    result_image = results[0].plot()

    st.image(result_image, caption="检测结果")
