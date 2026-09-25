import streamlit as st
from ultralytics import YOLO
from PIL import Image

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="Fire Supervision",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Fire Supervision")
st.write("基于 YOLOv8 的火灾目标检测系统")


# =========================
# 加载 YOLO 模型
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================
# 参数设置
# =========================
confidence = st.sidebar.slider(
    "置信度阈值",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)


# =========================
# 上传图片
# =========================
uploaded_file = st.file_uploader(
    "请选择需要检测的图片",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # 原始图片
    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)

    # YOLO 推理
    results = model.predict(
        source=image,
        conf=confidence
    )

    # YOLO plot() 返回 BGR
    result_image = results[0].plot()
    result_image = result_image[:, :, ::-1]

    # 检测结果
    with col2:
        st.subheader("检测结果")
        st.image(result_image, use_container_width=True)

    # 显示检测数量
    number_of_objects = len(results[0].boxes)

    st.success(
        f"检测完成，共检测到 {number_of_objects} 个目标。"
    )
