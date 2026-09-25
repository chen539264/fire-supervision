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
# 页面样式
# 中文：宋体
# 西文 / 数字：Times New Roman
# =========================================================
st.markdown(
    """
    <style>

    /* =====================================================
       整体页面
    ===================================================== */

    html,
    body,
    [class*="css"],
    .stApp {
        font-family:
            "Times New Roman",
            "SimSun",
            "Songti SC",
            serif;
    }

    [data-testid="stAppViewContainer"] {
        background: #ffffff;
        color: #111111;
    }

    /*
       尽量利用整个屏幕
       左右只留很小的边距
    */
    .block-container {
        max-width: 97%;
        padding-top: 0.8rem;
        padding-bottom: 1.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }


    /* =====================================================
       隐藏 Streamlit 默认元素
    ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =====================================================
       标题
    ===================================================== */

    .main-title {
        text-align: center;

        font-family:
            "Times New Roman",
            "SimSun",
            serif;

        font-size: 31px;
        font-weight: bold;

        color: #111111;

        margin-top: 0;
        margin-bottom: 3px;
    }


    .sub-title {
        text-align: center;

        font-family:
            "SimSun",
            "Songti SC",
            serif;

        font-size: 16px;

        color: #444444;

        margin-bottom: 12px;
    }


    /* =====================================================
       横线
    ===================================================== */

    .main-line {
        border: none;
        border-top: 1px solid #777777;
        margin: 8px 0 16px 0;
    }

    .thin-line {
        border: none;
        border-top: 1px solid #d3d3d3;
        margin: 13px 0;
    }


    /* =====================================================
       各级标题
    ===================================================== */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        font-family:
            "Times New Roman",
            "SimSun",
            "Songti SC",
            serif !important;

        color: #111111;

        letter-spacing: 0;
    }


    /* =====================================================
       上传区域
    ===================================================== */

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;

        border: 1px solid #b7b7b7;

        border-radius: 0;

        padding: 10px;
    }


    /* =====================================================
       下拉菜单
    ===================================================== */

    [data-baseweb="select"] > div {
        border-radius: 0 !important;
        border-color: #999999 !important;
        min-height: 43px;
    }


    /* =====================================================
       开始检测按钮
    ===================================================== */

    div[data-testid="stFormSubmitButton"] button {

        width: 100%;

        min-height: 45px;

        border-radius: 0;

        border: 1px solid #222222;

        background: #222222;

        color: #ffffff;

        font-family:
            "Times New Roman",
            "SimSun",
            serif;

        font-size: 16px;

        font-weight: normal;
    }


    div[data-testid="stFormSubmitButton"] button:hover {

        background: #444444;

        color: #ffffff;

        border-color: #222222;
    }


    /* =====================================================
       图片
    ===================================================== */

    [data-testid="stImage"] img {

        border-radius: 0;

        border: 1px solid #dddddd;
    }


    /* =====================================================
       指标
    ===================================================== */

    [data-testid="stMetric"] {

        background: #ffffff;

        border: none;

        padding: 2px 0;
    }


    [data-testid="stMetricLabel"] {

        font-family:
            "SimSun",
            "Songti SC",
            serif;

        font-size: 14px;
    }


    [data-testid="stMetricValue"] {

        font-family:
            "Times New Roman",
            "SimSun",
            serif;

        font-size: 26px;
    }


    /* =====================================================
       下载按钮
    ===================================================== */

    .stDownloadButton > button {

        width: 100%;

        border-radius: 0;

        border: 1px solid #555555;

        background: #ffffff;

        color: #111111;

        min-height: 43px;

        font-family:
            "Times New Roman",
            "SimSun",
            serif;
    }


    /* =====================================================
       caption
    ===================================================== */

    [data-testid="stCaptionContainer"] {

        font-family:
            "Times New Roman",
            "SimSun",
            "Songti SC",
            serif;

        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 顶部标题区域
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
    '<hr class="main-line">',
    unsafe_allow_html=True
)


# =========================================================
# 加载 YOLO 模型
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


# =========================================================
# Session State
# =========================================================
if "result_data" not in st.session_state:
    st.session_state.result_data = None


# =========================================================
# 主体布局
#
# 左侧尽量窄
# 右侧尽量宽
# =========================================================
left_col, right_col = st.columns(
    [0.65, 4.35],
    gap="small"
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
        # 1. 上传图片
        # -------------------------------------------------
        st.markdown("#### 1. 上传图片")

        uploaded_file = st.file_uploader(
            "请选择待检测图像",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        st.markdown("")


        # -------------------------------------------------
        # 2. 检测强度
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
        # 内部置信度阈值
        #
        # 页面不显示具体数字
        #
        # 高强度：
        # 阈值最低，因此通常检测目标更多
        # =================================================
        confidence_map = {

            "高强度":
                0.10,

            "标准":
                0.20,

            "稳健":
                0.30
        }


        confidence = confidence_map[
            detection_level
        ]


        # -------------------------------------------------
        # 模式简短说明
        # -------------------------------------------------
        if detection_level == "高强度":

            st.caption(
                "灵敏度最高，可保留更多疑似火焰目标。"
            )


        elif detection_level == "标准":

            st.caption(
                "兼顾检测数量与结果稳定性。"
            )


        else:

            st.caption(
                "过滤较弱目标，检测结果更加保守。"
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
    # 左下方模式说明
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
        "标准：兼顾检测灵敏度和稳定性。"
    )


    st.caption(
        "稳健：主要保留更加明显的目标。"
    )


# =========================================================
# 点击开始检测
# =========================================================
if detect_button:


    # -----------------------------------------------------
    # 未上传图片
    # -----------------------------------------------------
    if uploaded_file is None:

        st.warning(
            "请先上传待检测图片。"
        )


    # -----------------------------------------------------
    # 已上传
    # -----------------------------------------------------
    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        start_time = time.time()


        # -------------------------------------------------
        # YOLO 推理
        # -------------------------------------------------
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
        # 绘制检测结果
        # =========================================================
        annotated_bgr = result.plot()


        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # 获取检测框
        # =========================================================
        boxes = result.boxes


        object_count = len(
            boxes
        )


        # =================================================
        # 置信度统计
        # =========================================================
        if object_count > 0:

            conf_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )


            max_conf = float(
                np.max(
                    conf_values
                )
            )


            avg_conf = float(
                np.mean(
                    conf_values
                )
            )


        else:

            max_conf = 0.0

            avg_conf = 0.0


        # =================================================
        # 保存本次结果
        # =========================================================
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
# 右侧区域
# =========================================================
with right_col:

    st.markdown("### 检测结果")

    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )


    # =====================================================
    # 原图 / 检测图
    #
    # 使用 small gap，
    # 尽可能让两张图片大
    # =====================================================
    original_col, detection_col = st.columns(
        [1, 1],
        gap="small"
    )


    # =====================================================
    # 尚未进行检测
    # =====================================================
    if st.session_state.result_data is None:


        # -------------------------------------------------
        # 原始图像等待区
        # -------------------------------------------------
        with original_col:

            st.markdown("#### 原始图像")


            st.markdown(
                "<div style='"
                "height:390px;"
                "border:1px solid #d0d0d0;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#888888;"
                "font-family:SimSun,Songti SC,serif;"
                "font-size:15px;"
                "'>"
                "等待上传图像"
                "</div>",
                unsafe_allow_html=True
            )


            st.markdown("")


            st.caption(
                "原始图像：显示用户上传的原始输入图像，"
                "未经目标检测算法处理。"
            )


        # -------------------------------------------------
        # 检测图像等待区
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 检测图像")


            st.markdown(
                "<div style='"
                "height:390px;"
                "border:1px solid #d0d0d0;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#888888;"
                "font-family:SimSun,Songti SC,serif;"
                "font-size:15px;"
                "'>"
                "等待检测结果"
                "</div>",
                unsafe_allow_html=True
            )


            st.markdown("")


            st.caption(
                "检测图像：显示 YOLOv8 识别得到的"
                "疑似火焰位置及对应检测框。"
            )


    # =====================================================
    # 已完成检测
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
                "作为模型检测结果的对照。"
            )


        # -------------------------------------------------
        # 检测结果
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 检测图像")


            st.image(
                data["annotated_rgb"],
                use_container_width=True
            )


            st.caption(
                "检测图像：检测框表示模型识别出的疑似火焰区域；"
                "框上的数值表示对应目标的置信度。"
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
            [1, 1, 1, 1]
        )


        # -------------------------------------------------
        # 目标数量
        # -------------------------------------------------
        metric1.metric(
            "检测目标数量",
            f'{data["object_count"]}'
        )


        # -------------------------------------------------
        # 最高置信度
        # -------------------------------------------------
        metric2.metric(
            "最高置信度",

            (
                f'{data["max_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        # -------------------------------------------------
        # 平均置信度
        # -------------------------------------------------
        metric3.metric(
            "平均置信度",

            (
                f'{data["avg_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        # -------------------------------------------------
        # 检测模式
        # -------------------------------------------------
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
        # 检测结果说明
        # =================================================
        st.markdown("#### 结果说明")


        if data["object_count"] == 0:

            st.write(
                "当前图像中未检测到符合当前检测模式要求的"
                "疑似火焰目标。"
            )


        else:

            st.write(
                f'本次共检测到 '
                f'{data["object_count"]} '
                f'个疑似火焰目标。'
            )


            st.write(
                "检测框表示模型定位到的疑似火焰区域；"
                "置信度用于反映模型对该目标属于火焰类别"
                "的判断程度。"
            )


        st.caption(
            f'模型本次检测耗时：'
            f'{data["inference_time"]:.2f} s'
        )


        # =================================================
        # 下载检测结果
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
    '<hr class="main-line">',
    unsafe_allow_html=True
)


st.markdown(
    "<div style='"
    "text-align:center;"
    "font-family:Times New Roman,SimSun,serif;"
    "font-size:12px;"
    "color:#777777;"
    "'>"
    "Fire Supervision · YOLOv8 Fire Detection System"
    "</div>",
    unsafe_allow_html=True
)
