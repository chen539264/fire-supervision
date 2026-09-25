import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time


# =========================================================
# 页面设置
# =========================================================
st.set_page_config(
    page_title="Fire Supervision",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 简约 CSS
# 中文：宋体
# 西文、数字：Times New Roman
# =========================================================
st.markdown(
    """
    <style>

    /* 整体页面 */
    html, body, [class*="css"] {
        font-family:
            "Times New Roman",
            "SimSun",
            "Songti SC",
            serif;
    }

    [data-testid="stAppViewContainer"] {
        background: white;
        color: #111111;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* 标题 */
    .main-title {
        text-align: center;
        font-family:
            "Times New Roman",
            "SimSun",
            serif;
        font-size: 34px;
        font-weight: bold;
        color: #111111;
        margin-bottom: 4px;
    }

    .sub-title {
        text-align: center;
        font-family:
            "SimSun",
            "Songti SC",
            serif;
        font-size: 17px;
        color: #444444;
        margin-bottom: 18px;
    }

    /* 横线 */
    .line {
        border: none;
        border-top: 1px solid #888888;
        margin: 12px 0 22px 0;
    }

    .thin-line {
        border: none;
        border-top: 1px solid #d0d0d0;
        margin: 18px 0;
    }

    /* 小标题 */
    h1, h2, h3, h4 {
        font-family:
            "Times New Roman",
            "SimSun",
            serif !important;
        color: #111111;
    }

    /* 上传组件 */
    [data-testid="stFileUploaderDropzone"] {
        background: white;
        border: 1px solid #bdbdbd;
        border-radius: 2px;
    }

    /* 下拉菜单 */
    [data-baseweb="select"] > div {
        border-radius: 2px !important;
        border-color: #999999 !important;
    }

    /* 按钮 */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 46px;
        border-radius: 2px;
        border: 1px solid #333333;
        background: #222222;
        color: white;
        font-family:
            "Times New Roman",
            "SimSun",
            serif;
        font-size: 16px;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #444444;
        color: white;
        border-color: #222222;
    }

    /* 图片 */
    [data-testid="stImage"] img {
        border-radius: 0px;
        border: 1px solid #dddddd;
    }

    /* Metric 去掉卡片感 */
    [data-testid="stMetric"] {
        background: white;
        border: none;
        padding: 4px 0;
    }

    [data-testid="stMetricLabel"] {
        font-family:
            "SimSun",
            "Songti SC",
            serif;
        font-size: 15px;
    }

    [data-testid="stMetricValue"] {
        font-family:
            "Times New Roman",
            "SimSun",
            serif;
    }

    /* 下载按钮 */
    .stDownloadButton > button {
        border-radius: 2px;
        border: 1px solid #555555;
        background: white;
        color: #111111;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 顶部标题区
# =========================================================
st.markdown(
    '<div class="main-title">Fire Supervision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">基于 YOLOv8 的火灾目标检测系统</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<hr class="line">',
    unsafe_allow_html=True
)


# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# 保存检测结果
# =========================================================
if "result_data" not in st.session_state:
    st.session_state.result_data = None


# =========================================================
# 主布局
#
# 左：设置
# 右：检测结果
# =========================================================
left_col, right_col = st.columns(
    [1, 3.2],
    gap="large"
)


# =========================================================
# 左侧：检测设置
# =========================================================
with left_col:

    st.markdown("### 检测设置")

    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )

    with st.form("detection_form"):

        # -------------------------------------------------
        # 上传图片
        # -------------------------------------------------
        st.markdown("#### 1. 上传图片")

        uploaded_file = st.file_uploader(
            "请选择待检测图像",
            type=["jpg", "jpeg", "png"]
        )


        st.markdown("")


        # -------------------------------------------------
        # 检测强度
        # -------------------------------------------------
        st.markdown("#### 2. 检测强度")

        detection_level = st.selectbox(
            "请选择检测强度",
            [
                "高强度",
                "标准",
                "稳健"
            ],
            index=0,
            label_visibility="collapsed"
        )


        # =================================================
        # 内部置信度
        #
        # 页面中不显示
        # =================================================
        confidence_map = {
            "高强度": 0.10,
            "标准": 0.20,
            "稳健": 0.30
        }

        confidence = confidence_map[
            detection_level
        ]


        # -------------------------------------------------
        # 模式说明
        # -------------------------------------------------
        if detection_level == "高强度":

            st.caption(
                "检测灵敏度较高，可保留更多疑似火焰目标。"
            )

        elif detection_level == "标准":

            st.caption(
                "兼顾检测灵敏度与结果稳定性。"
            )

        else:

            st.caption(
                "过滤较弱候选目标，检测结果更加保守。"
            )


        st.markdown("")


        # -------------------------------------------------
        # 开始检测
        # -------------------------------------------------
        detect_button = st.form_submit_button(
            "开始检测",
            use_container_width=True
        )


    # =====================================================
    # 左侧说明
    # =====================================================
    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )

    st.markdown("#### 模式说明")

    st.caption(
        "高强度：优先发现更多潜在目标。"
    )

    st.caption(
        "标准：兼顾目标数量与检测稳定性。"
    )

    st.caption(
        "稳健：主要保留较明显的火焰目标。"
    )


# =========================================================
# 点击开始检测
# =========================================================
if detect_button:

    if uploaded_file is None:

        st.warning(
            "请先上传待检测图片。"
        )

    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        start_time = time.time()


        with st.spinner(
            "正在进行目标检测..."
        ):

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )


        inference_time = (
            time.time() - start_time
        )


        result = results[0]


        # =================================================
        # YOLO 检测图像
        # =================================================
        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # 检测框
        # =================================================
        boxes = result.boxes

        object_count = len(boxes)


        # =================================================
        # 置信度
        # =================================================
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


        # =================================================
        # 保存结果
        # =================================================
        st.session_state.result_data = {

            "image":
                image,

            "annotated_rgb":
                annotated_rgb,

            "annotated_bgr":
                annotated_bgr,

            "object_count":
                object_count,

            "max_conf":
                max_conf,

            "avg_conf":
                avg_conf,

            "detection_level":
                detection_level,

            "inference_time":
                inference_time
        }


# =========================================================
# 右侧：检测区域
# =========================================================
with right_col:

    st.markdown("### 检测结果")

    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )


    # =====================================================
    # 右侧始终分为两栏
    # =====================================================
    original_col, detection_col = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # 尚未检测
    # =====================================================
    if st.session_state.result_data is None:

        # -------------------------------------------------
        # 原始图片
        # -------------------------------------------------
        with original_col:

            st.markdown("#### 原始图像")

            st.markdown("")

            st.markdown(
                "<div style='height:320px;"
                "border:1px solid #d5d5d5;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#999999;"
                "font-family:SimSun,serif;'>"
                "等待上传图像"
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("")

            st.caption(
                "原始图像：显示用户上传、尚未经过目标检测处理的图像。"
            )


        # -------------------------------------------------
        # 检测图片
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 检测图像")

            st.markdown("")

            st.markdown(
                "<div style='height:320px;"
                "border:1px solid #d5d5d5;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#999999;"
                "font-family:SimSun,serif;'>"
                "等待检测结果"
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("")

            st.caption(
                "检测图像：显示模型识别出的疑似火焰位置及相应检测框。"
            )


    # =====================================================
    # 已检测
    # =====================================================
    else:

        data = st.session_state.result_data


        # -------------------------------------------------
        # 原始图像
        # -------------------------------------------------
        with original_col:

            st.markdown("#### 原始图像")

            st.image(
                data["image"],
                use_container_width=True
            )

            st.caption(
                "原始图像：用户上传的原始输入图像，"
                "用于与模型检测结果进行对照。"
            )


        # -------------------------------------------------
        # 检测图像
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 检测图像")

            st.image(
                data["annotated_rgb"],
                use_container_width=True
            )

            st.caption(
                "检测图像：检测框表示模型识别出的疑似火焰区域，"
                "框上数值表示对应目标的置信度。"
            )


        # =================================================
        # 横线
        # =================================================
        st.markdown(
            '<hr class="thin-line">',
            unsafe_allow_html=True
        )


        # =================================================
        # 检测数据
        # =================================================
        st.markdown("#### 检测数据")

        metric1, metric2, metric3, metric4 = st.columns(
            4
        )


        metric1.metric(
            "检测目标数量",
            f'{data["object_count"]}'
        )


        metric2.metric(
            "最高置信度",
            (
                f'{data["max_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        metric3.metric(
            "平均置信度",
            (
                f'{data["avg_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        metric4.metric(
            "检测模式",
            data["detection_level"]
        )


        # =================================================
        # 横线
        # =================================================
        st.markdown(
            '<hr class="thin-line">',
            unsafe_allow_html=True
        )


        # =================================================
        # 结果说明
        # =================================================
        st.markdown("#### 结果说明")


        if data["object_count"] == 0:

            st.write(
                "当前图像中未检测到符合当前检测模式要求的火焰目标。"
            )

        else:

            st.write(
                f'本次共检测到 '
                f'{data["object_count"]} 个疑似火焰目标。'
            )

            st.write(
                "检测框用于表示模型定位到的疑似火焰区域；"
                "置信度反映模型对该目标属于火焰类别的判断程度。"
            )


        st.caption(
            f'本次检测耗时：'
            f'{data["inference_time"]:.2f} s'
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
                "下载检测结果",
                data=encoded_image.tobytes(),
                file_name="fire_detection_result.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# 页面底部
# =========================================================
st.markdown(
    '<hr class="line">',
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align:center;"
    "font-family:Times New Roman,SimSun,serif;"
    "font-size:12px;"
    "color:#777777;'>"
    "Fire Supervision · YOLOv8 Fire Detection System"
    "</div>",
    unsafe_allow_html=True
)
