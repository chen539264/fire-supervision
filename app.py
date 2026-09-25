import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="Fire Supervision",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Fire Supervision")
st.caption("基于 YOLOv8 的火灾智能检测系统")


# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================
# 侧边栏
# =========================
st.sidebar.title("⚙️ 检测设置")

confidence = st.sidebar.slider(
    "置信度阈值",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

mode = st.sidebar.radio(
    "检测模式",
    ["📷 图片检测", "🎬 视频检测"]
)


# ==========================================================
# 图片检测
# ==========================================================
if mode == "📷 图片检测":

    st.header("📷 图片火灾检测")

    uploaded_file = st.file_uploader(
        "上传待检测图片",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原始图片")
            st.image(image, use_container_width=True)

        # YOLO推理
        results = model.predict(
            source=image,
            conf=confidence
        )

        result = results[0]

        # 获取检测后的图片
        result_image = result.plot()
        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        with col2:
            st.subheader("检测结果")
            st.image(result_image, use_container_width=True)

        # =========================
        # 检测信息
        # =========================
        boxes = result.boxes
        number_of_objects = len(boxes)

        st.divider()
        st.subheader("📊 检测信息")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "检测目标数量",
                number_of_objects
            )

        if number_of_objects > 0:

            confidences = boxes.conf.cpu().numpy()

            max_conf = float(confidences.max())
            avg_conf = float(confidences.mean())

            with col2:
                st.metric(
                    "最高置信度",
                    f"{max_conf:.2%}"
                )

            with col3:
                st.metric(
                    "平均置信度",
                    f"{avg_conf:.2%}"
                )

            if max_conf >= 0.70:
                st.error("🔥 检测到高置信度火灾目标！")

            elif max_conf >= 0.40:
                st.warning("⚠️ 检测到疑似火灾目标")

            else:
                st.info("检测到低置信度目标")

        else:

            with col2:
                st.metric(
                    "最高置信度",
                    "—"
                )

            with col3:
                st.metric(
                    "平均置信度",
                    "—"
                )

            st.success("✅ 当前图片未检测到火灾目标")


# ==========================================================
# 视频检测
# ==========================================================
elif mode == "🎬 视频检测":

    st.header("🎬 视频火灾检测")

    uploaded_video = st.file_uploader(
        "上传待检测视频",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        # 保存上传的视频到临时文件
        temp_input = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_input.write(uploaded_video.read())
        temp_input.close()

        st.subheader("原始视频")
        st.video(temp_input.name)

        if st.button(
            "🔥 开始检测",
            type="primary",
            use_container_width=True
        ):

            cap = cv2.VideoCapture(temp_input.name)

            fps = cap.get(cv2.CAP_PROP_FPS)

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_number = 0
            fire_frames = 0
            max_confidence = 0.0

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                # YOLO检测
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                # 是否检测到目标
                if len(result.boxes) > 0:

                    fire_frames += 1

                    current_max = float(
                        result.boxes.conf.max()
                    )

                    max_confidence = max(
                        max_confidence,
                        current_max
                    )

                # 绘制检测框
                annotated_frame = result.plot()

                writer.write(annotated_frame)

                frame_number += 1

                # 更新进度
                if total_frames > 0:

                    progress = min(
                        frame_number / total_frames,
                        1.0
                    )

                    progress_bar.progress(progress)

                status_text.write(
                    f"正在检测：{frame_number} / {total_frames} 帧"
                )

            cap.release()
            writer.release()

            progress_bar.progress(1.0)

            st.success("✅ 视频检测完成")

            # =========================
            # 视频统计
            # =========================
            st.subheader("📊 视频检测统计")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "总帧数",
                    frame_number
                )

            with col2:
                st.metric(
                    "检测到火灾的帧数",
                    fire_frames
                )

            with col3:
                st.metric(
                    "最高置信度",
                    f"{max_confidence:.2%}"
                )

            st.subheader("检测后视频")

            st.video(output_path)

            # 下载结果
            with open(output_path, "rb") as f:

                video_bytes = f.read()

            st.download_button(
                label="⬇️ 下载检测结果",
                data=video_bytes,
                file_name="fire_detection_result.mp4",
                mime="video/mp4",
                use_container_width=True
            )

            # 删除输入临时文件
            try:
                os.unlink(temp_input.name)
            except:
                pass
