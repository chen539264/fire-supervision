import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time


# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="Fire Supervision",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

/* 主页面 */
[data-testid="stAppViewContainer"] {
    background: #f4f6f9;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* 隐藏 Streamlit 默认元素 */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* 卡片圆角 */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
    background: white;
}

/* 上传框 */
[data-testid="stFileUploaderDropzone"] {
    background: #fafbfc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 14px;
}

/* Metric */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e7eaf0;
    padding: 16px;
    border-radius: 14px;
}

/* 图片 */
[data-testid="stImage"] img {
    border-radius: 14px;
}

/* 主按钮 */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(
        90deg,
        #e84032,
        #ff6840
    );
    color: white;
    font-size: 16px;
    font-weight: 700;
}

div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: none;
    opacity: 0.94;
}

/* 下载按钮 */
.stDownloadButton > button {
    width: 100%;
    min-height: 45px;
    border-radius: 11px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 顶部标题
# 注意：HTML 不缩进，避免再次显示成代码
# =========================================================
st.markdown("""
<div style="background:linear-gradient(115deg,#111827,#202936 65%,#44211c);
padding:28px 34px;
border-radius:22px;
margin-bottom:22px;
box-shadow:0 12px 30px rgba(15,23,42,0.08);">

<div style="display:flex;
justify-content:space-between;
align-items:center;
gap:20px;
flex-wrap:wrap;">

<div>

<div style="font-size:12px;
font-weight:700;
letter-spacing:2px;
color:#ff8a6c;
margin-bottom:7px;">
AI FIRE DETECTION PLATFORM
</div>

<div style="font-size:36px;
font-weight:800;
color:white;
line-height:1.2;">
🔥 Fire Supervision
</div>

<div style="font-size:15px;
color:#cbd5e1;
margin-top:9px;">
基于 YOLOv8 的智能火灾目标检测与分析系统
</div>

</div>

<div style="padding:9px 15px;
border-radius:999px;
border:1px solid rgba(255,255,255,0.16);
background:rgba(255,255,255,0.07);
color:#e2e8f0;
font-size:12px;">
● YOLOv8 MODEL ONLINE
</div>

</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# 主布局
#
# 左侧：设置
# 右侧：原图 + 检测结果
# =========================================================
setting_col, workspace_col = st.columns(
    [1, 3.2],
    gap="large"
)


# =========================================================
# 左侧：检测控制面板
# =========================================================
with setting_col:

    with st.container(border=True):

        st.markdown("### ⚙️ 检测设置")

        st.caption(
            "上传图片并设置检测灵敏度。"
        )

        st.divider()

        with st.form("detection_form"):

            st.markdown("#### ① 上传图片")

            uploaded_file = st.file_uploader(
                "JPG / JPEG / PNG",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )

            st.markdown("")

            st.markdown("#### ② 置信度等级")

            confidence_level = st.select_slider(
                "检测灵敏度",
                options=[
                    "10% 高灵敏度",
                    "30% 标准模式"
                ],
                value="30% 标准模式",
                label_visibility="collapsed"
            )


            # 选择实际阈值
            if confidence_level == "10% 高灵敏度":

                confidence = 0.10

                st.info(
                    "高灵敏度模式\n\n"
                    "能够发现更多疑似火焰目标，"
                    "但低置信度误检也可能增加。"
                )

            else:

                confidence = 0.30

                st.info(
                    "标准检测模式\n\n"
                    "过滤置信度较低的候选目标，"
                    "结果通常更加稳健。"
                )


            st.markdown("")

            detect_button = st.form_submit_button(
                "🔥 开始检测",
                use_container_width=True
            )


        st.divider()

        st.markdown("#### 阈值说明")

        st.caption(
            "10% 表示模型认为目标属于火焰的概率达到 "
            "0.10 后即可显示检测框。"
        )

        st.caption(
            "30% 表示只有置信度达到 0.30 及以上的目标"
            "才会保留。"
        )


# =========================================================
# 检测状态变量
# =========================================================
if "result_data" not in st.session_state:
    st.session_state.result_data = None


# =========================================================
# 执行检测
# =========================================================
if detect_button:

    if uploaded_file is None:

        st.warning(
            "请先在左侧上传需要检测的图片。"
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        start_time = time.time()


        with st.spinner(
            "AI 正在分析图像，请稍候..."
        ):

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )


        inference_time = time.time() - start_time

        result = results[0]


        # YOLO绘制结果
        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )


        boxes = result.boxes

        object_count = len(boxes)


        # 统计置信度
        if object_count > 0:

            conf_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            max_conf = float(
                np.max(conf_values)
            )

            avg_conf = float(
                np.mean(conf_values)
            )

        else:

            max_conf = 0.0
            avg_conf = 0.0


        st.session_state.result_data = {
            "image": image,
            "annotated_rgb": annotated_rgb,
            "annotated_bgr": annotated_bgr,
            "object_count": object_count,
            "max_conf": max_conf,
            "avg_conf": avg_conf,
            "confidence": confidence,
            "inference_time": inference_time
        }


# =========================================================
# 右侧：检测工作区
# =========================================================
with workspace_col:

    st.markdown("### 🔍 检测工作区")

    st.caption(
        "原始图像与 YOLOv8 分析结果将同时显示在这里。"
    )


    # -----------------------------------------------------
    # 尚未检测
    # -----------------------------------------------------
    if st.session_state.result_data is None:

        with st.container(border=True):

            st.markdown("")

            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding:95px 20px;
                ">

                    <div style="
                        font-size:54px;
                        margin-bottom:16px;
                    ">
                        🔥
                    </div>

                    <div style="
                        font-size:21px;
                        font-weight:700;
                        color:#273142;
                    ">
                        等待检测
                    </div>

                    <div style="
                        color:#8a94a6;
                        font-size:14px;
                        margin-top:8px;
                    ">
                        请在左侧上传图片并点击「开始检测」
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # 已经完成检测
    # -----------------------------------------------------
    else:

        data = st.session_state.result_data


        # =================================================
        # 图片左右双视图
        # =================================================
        original_col, detection_col = st.columns(
            2,
            gap="large"
        )


        # -------------------------------------------------
        # 原始图像
        # -------------------------------------------------
        with original_col:

            with st.container(border=True):

                st.markdown("#### 🖼️ 原始图像")

                st.image(
                    data["image"],
                    use_container_width=True
                )

                st.markdown(
                    """
                    **图像含义**

                    此区域显示用户上传的原始图像，
                    未经过目标检测算法处理。

                    可用于与右侧 AI 检测结果进行直接对比。
                    """
                )


        # -------------------------------------------------
        # 检测结果
        # -------------------------------------------------
        with detection_col:

            with st.container(border=True):

                st.markdown("#### 🔥 AI 检测结果")

                st.image(
                    data["annotated_rgb"],
                    use_container_width=True
                )

                st.markdown(
                    """
                    **结果含义**

                    蓝色检测框表示 YOLOv8 认为可能存在
                    **火焰目标的位置**。

                    检测框上方的数字表示该目标的
                    **模型置信度**。
                    """
                )


        # =================================================
        # 图例说明
        # =================================================
        with st.container(border=True):

            st.markdown("### 📖 检测结果说明")

            meaning1, meaning2, meaning3 = st.columns(3)

            with meaning1:

                st.markdown("#### ▣ 检测框")

                st.write(
                    "方框表示模型定位到的疑似火焰区域。"
                )

            with meaning2:

                st.markdown("#### % 置信度")

                st.write(
                    "表示模型对该目标属于火焰类别的判断程度。"
                )

            with meaning3:

                st.markdown("#### 🎚 检测阈值")

                st.write(
                    "低于当前设置阈值的候选目标不会显示。"
                )


        # =================================================
        # 检测统计
        # =================================================
        st.markdown("### 📊 检测数据")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "检测目标",
            f'{data["object_count"]} 个'
        )

        m2.metric(
            "最高置信度",
            (
                f'{data["max_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )

        m3.metric(
            "平均置信度",
            (
                f'{data["avg_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )

        m4.metric(
            "检测阈值",
            f'{data["confidence"] * 100:.0f}%'
        )


        # =================================================
        # 检测判断
        # =================================================
        with st.container(border=True):

            st.markdown("### 🚨 AI 分析状态")

            if data["object_count"] == 0:

                st.success(
                    "当前图像中未检测到达到设定阈值的火焰目标。"
                )

            else:

                if data["max_conf"] >= 0.60:

                    st.error(
                        "检测到较高置信度的火焰目标，"
                        "建议进一步核查现场情况。"
                    )

                elif data["max_conf"] >= 0.30:

                    st.warning(
                        "检测到疑似火焰目标，"
                        "建议结合实际场景进行判断。"
                    )

                else:

                    st.info(
                        "检测到低置信度疑似火焰目标。"
                    )


            st.caption(
                f'本次模型处理耗时约 '
                f'{data["inference_time"]:.2f} 秒。'
            )


        # =================================================
        # 下载
        # =================================================
        success, encoded_image = cv2.imencode(
            ".png",
            data["annotated_bgr"]
        )

        if success:

            st.download_button(
                "⬇️ 下载检测结果图片",
                data=encoded_image.tobytes(),
                file_name="fire_detection_result.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# footer
# =========================================================
st.markdown("---")

st.caption(
    "Fire Supervision · YOLOv8 Intelligent Fire Detection System"
)
