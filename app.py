import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np


# =========================================================
# 页面设置
# =========================================================
st.set_page_config(
    page_title="Fire Supervision",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 页面美化 CSS
# =========================================================
st.markdown(
    """
    <style>

    /* 页面背景 */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 10% 0%, rgba(255, 91, 59, 0.08), transparent 28%),
            #f5f7fa;
    }

    /* 主体宽度 */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* 上传框 */
    [data-testid="stFileUploaderDropzone"] {
        background: #fafbfc;
        border: 1.5px dashed #cbd5e1;
        border-radius: 16px;
        padding: 18px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #f05a47;
        background: #fff8f6;
    }

    /* 卡片 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8edf3;
        border-radius: 16px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    /* 普通按钮 */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    /* form submit 主按钮 */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 54px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(
            90deg,
            #e93f33 0%,
            #ff6b35 100%
        );
        color: white;
        font-size: 17px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(233, 63, 51, 0.18);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        color: white;
        border: none;
        transform: translateY(-1px);
    }

    /* 图片圆角 */
    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Hero 顶部区域
# =========================================================
st.markdown(
    """
    <div style="
        background:
            linear-gradient(
                120deg,
                #111827 0%,
                #1f2937 55%,
                #3a1f1b 100%
            );
        padding: 32px 38px;
        border-radius: 24px;
        margin-bottom: 26px;
        box-shadow: 0 18px 40px rgba(17,24,39,0.10);
    ">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            flex-wrap:wrap;
            gap:16px;
        ">

            <div>

                <div style="
                    color:#ff8a65;
                    font-size:13px;
                    font-weight:700;
                    letter-spacing:2px;
                    margin-bottom:8px;
                ">
                    AI FIRE DETECTION PLATFORM
                </div>

                <div style="
                    color:white;
                    font-size:38px;
                    line-height:1.2;
                    font-weight:800;
                ">
                    🔥 Fire Supervision
                </div>

                <div style="
                    color:#cbd5e1;
                    font-size:16px;
                    margin-top:10px;
                ">
                    基于 YOLOv8 的智能火灾目标检测系统
                </div>

            </div>

            <div style="
                background:rgba(255,255,255,0.08);
                border:1px solid rgba(255,255,255,0.12);
                padding:10px 16px;
                border-radius:999px;
                color:#f1f5f9;
                font-size:13px;
                font-weight:600;
            ">
                ● MODEL ONLINE
            </div>

        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 加载 YOLO
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# 简介
# =========================================================
st.markdown("### 图像检测")

st.caption(
    "上传待检测图像，选择检测灵敏度，然后点击「开始检测」。"
)


# =========================================================
# 表单区域
# =========================================================
with st.container(border=True):

    with st.form("fire_detection_form"):

        col_upload, col_setting = st.columns(
            [1.7, 1],
            gap="large"
        )

        # -------------------------
        # 上传图片
        # -------------------------
        with col_upload:

            st.markdown("#### ① 上传图像")

            uploaded_file = st.file_uploader(
                "支持 JPG、JPEG、PNG",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )

        # -------------------------
        # 置信度
        # -------------------------
        with col_setting:

            st.markdown("#### ② 检测灵敏度")

            confidence_mode = st.select_slider(
                "选择置信度阈值",
                options=[
                    "高灵敏度 · 10%",
                    "标准模式 · 30%"
                ],
                value="标准模式 · 30%"
            )

            if confidence_mode == "高灵敏度 · 10%":

                confidence = 0.10

                st.info(
                    "10%：更容易发现疑似火焰，但误报可能更多。"
                )

            else:

                confidence = 0.30

                st.info(
                    "30%：过滤低置信度结果，检测结果更加稳健。"
                )

        st.markdown("")

        # -------------------------
        # 开始按钮
        # -------------------------
        detect_button = st.form_submit_button(
            "🔥 开始检测",
            use_container_width=True
        )


# =========================================================
# 开始检测
# =========================================================
if detect_button:

    if uploaded_file is None:

        st.warning("请先上传一张需要检测的图片。")

    else:

        # 读取图像
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        # -------------------------------------------------
        # 检测中的交互提示
        # -------------------------------------------------
        with st.status(
            "正在执行 AI 火灾检测...",
            expanded=True
        ) as status:

            st.write("正在读取图像...")
            st.write(
                f"当前置信度阈值：{int(confidence * 100)}%"
            )

            # YOLO 推理
            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            st.write("YOLOv8 推理完成")
            st.write("正在生成检测结果...")

            status.update(
                label="检测完成",
                state="complete",
                expanded=False
            )

        # =================================================
        # 检测结果
        # =================================================
        st.markdown("## 检测结果")

        original_col, result_col = st.columns(
            2,
            gap="large"
        )

        # 原图
        with original_col:

            st.markdown("#### 原始图像")

            st.image(
                image,
                use_container_width=True
            )

        # YOLO 输出
        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )

        with result_col:

            st.markdown("#### AI 检测结果")

            st.image(
                annotated_rgb,
                use_container_width=True
            )


        # =================================================
        # 数据统计
        # =================================================
        boxes = result.boxes

        object_count = len(boxes)

        if object_count > 0:

            conf_array = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            max_conf = float(
                np.max(conf_array)
            )

            avg_conf = float(
                np.mean(conf_array)
            )

        else:

            max_conf = 0.0
            avg_conf = 0.0


        st.markdown("### 检测数据")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "检测目标",
            f"{object_count} 个"
        )

        metric2.metric(
            "最高置信度",
            (
                f"{max_conf * 100:.1f}%"
                if object_count > 0
                else "—"
            )
        )

        metric3.metric(
            "平均置信度",
            (
                f"{avg_conf * 100:.1f}%"
                if object_count > 0
                else "—"
            )
        )

        metric4.metric(
            "检测阈值",
            f"{confidence * 100:.0f}%"
        )


        # =================================================
        # 状态判断
        # =================================================
        st.markdown("### 检测状态")

        if object_count == 0:

            st.success(
                "✅ 当前图像中未检测到符合阈值的火灾目标。"
            )

        else:

            if max_conf >= 0.60:

                st.error(
                    "🔥 检测到较高置信度的火灾目标。"
                )

            elif max_conf >= 0.30:

                st.warning(
                    "⚠️ 检测到疑似火灾目标，请结合实际场景进一步确认。"
                )

            else:

                st.info(
                    "ℹ️ 检测到低置信度疑似目标。"
                )


        # =================================================
        # 下载结果
        # =================================================
        success, encoded_image = cv2.imencode(
            ".png",
            annotated_bgr
        )

        if success:

            st.download_button(
                "⬇️ 下载检测结果",
                data=encoded_image.tobytes(),
                file_name="fire_detection_result.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# 页脚
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:13px;
        padding:6px 0 14px 0;
    ">
        Fire Supervision · YOLOv8 Intelligent Detection
    </div>
    """,
    unsafe_allow_html=True
)
