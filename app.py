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
    page_title="智能火灾识别与辅助研判系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 页面样式
# 中文：宋体
# 英文、数字：Times New Roman
# =========================================================
st.markdown(
    """
    <style>

    /* 整体字体 */
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

    /* 页面 */
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
        color: #111111;
    }

    /* 恢复正常页面宽度 */
    .block-container {
        max-width: 1450px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* 隐藏 Streamlit 默认菜单 */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* 主标题 */
    .main-title {
        text-align: center;

        font-family:
            "SimSun",
            "Songti SC",
            serif;

        font-size: 32px;
        font-weight: bold;

        color: #111111;

        margin-top: 0;
        margin-bottom: 5px;
    }

    /* 英文副标题 */
    .sub-title {
        text-align: center;

        font-family:
            "Times New Roman",
            serif;

        font-size: 16px;

        color: #555555;

        margin-bottom: 16px;
    }

    /* 主横线 */
    .main-line {
        border: none;
        border-top: 1px solid #777777;

        margin: 10px 0 20px 0;
    }

    /* 普通横线 */
    .thin-line {
        border: none;
        border-top: 1px solid #d0d0d0;

        margin: 16px 0;
    }

    /* 标题字体 */
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

    /* 上传区域 */
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff;

        border: 1px solid #b8b8b8;

        border-radius: 2px;
    }

    /* 下拉框 */
    [data-baseweb="select"] > div {
        border-radius: 2px !important;

        border-color: #999999 !important;

        min-height: 44px;
    }

    /* 开始检测按钮 */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;

        min-height: 46px;

        border-radius: 2px;

        border: 1px solid #222222;

        background: #222222;

        color: white;

        font-family:
            "Times New Roman",
            "SimSun",
            serif;

        font-size: 16px;

        font-weight: normal;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #444444;

        color: white;

        border-color: #222222;
    }

    /* 图片 */
    [data-testid="stImage"] img {
        border-radius: 0;

        border: 1px solid #dddddd;
    }

    /* 数据指标 */
    [data-testid="stMetric"] {
        background: #ffffff;

        border: none;

        padding: 4px 0;
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
    }

    /* 下载按钮 */
    .stDownloadButton > button {
        width: 100%;

        min-height: 43px;

        border-radius: 2px;

        border: 1px solid #555555;

        background: #ffffff;

        color: #111111;

        font-family:
            "Times New Roman",
            "SimSun",
            serif;
    }

    /* 小字 */
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
# 顶部标题
# =========================================================
st.markdown(
    '<div class="main-title">'
    '智能火灾识别与辅助研判系统'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Intelligent Fire Detection and Assessment System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<hr class="main-line">',
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
# 左边：检测设置
# 右边：检测结果
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
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        st.markdown("")


        # -------------------------------------------------
        # 检测灵敏度
        # -------------------------------------------------
        st.markdown("#### 2. 识别灵敏度")

        detection_level = st.selectbox(
            "请选择识别灵敏度",
            [
                "高灵敏度",
                "标准",
                "稳健"
            ],
            index=0,
            label_visibility="collapsed"
        )


        # =================================================
        # 模型内部阈值
        #
        # 不在页面显示具体数字
        # =========================================================
        confidence_map = {
            "高灵敏度": 0.10,
            "标准": 0.20,
            "稳健": 0.30
        }

        confidence = confidence_map[
            detection_level
        ]


        # -------------------------------------------------
        # 当前模式说明
        # -------------------------------------------------
        if detection_level == "高灵敏度":

            st.caption(
                "对疑似火焰区域更加敏感，"
                "通常能够保留更多潜在目标。"
            )

        elif detection_level == "标准":

            st.caption(
                "兼顾识别灵敏度与结果稳定性，"
                "适用于一般检测场景。"
            )

        else:

            st.caption(
                "主要保留较明显的疑似目标，"
                "检测结果更加保守。"
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
    # 模式说明
    # =====================================================
    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )

    st.markdown("#### 模式说明")

    st.caption(
        "高灵敏度：优先发现更多潜在目标。"
    )

    st.caption(
        "标准：兼顾检测数量与识别稳定性。"
    )

    st.caption(
        "稳健：主要保留较明显的疑似目标。"
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

        # -------------------------------------------------
        # 读取图片
        # -------------------------------------------------
        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # 记录检测时间
        # -------------------------------------------------
        start_time = time.time()


        # -------------------------------------------------
        # 模型推理
        # -------------------------------------------------
        with st.spinner(
            "正在进行图像识别，请稍候..."
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

        object_count = len(boxes)


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
# 右侧：检测结果
# =========================================================
with right_col:

    st.markdown("### 检测结果")

    st.markdown(
        '<hr class="thin-line">',
        unsafe_allow_html=True
    )


    # =====================================================
    # 原始图像 / 识别结果
    # =====================================================
    original_col, detection_col = st.columns(
        [1, 1],
        gap="large"
    )


    # =====================================================
    # 尚未进行检测
    # =====================================================
    if st.session_state.result_data is None:


        # -------------------------------------------------
        # 原始图像
        # -------------------------------------------------
        with original_col:

            st.markdown("#### 原始图像")

            st.markdown("")

            st.markdown(
                "<div style='"
                "height:320px;"
                "border:1px solid #d5d5d5;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#999999;"
                "font-family:SimSun,Songti SC,serif;"
                "'>"
                "等待上传图像"
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("")

            st.caption(
                "原始图像：显示用户上传的图像，"
                "作为识别结果的原始参照。"
            )


        # -------------------------------------------------
        # 检测结果
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 识别结果")

            st.markdown("")

            st.markdown(
                "<div style='"
                "height:320px;"
                "border:1px solid #d5d5d5;"
                "display:flex;"
                "align-items:center;"
                "justify-content:center;"
                "color:#999999;"
                "font-family:SimSun,Songti SC,serif;"
                "'>"
                "等待识别结果"
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("")

            st.caption(
                "识别结果：显示系统识别得到的"
                "疑似火焰区域及其位置。"
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
                "未经识别结果标注处理。"
            )


        # -------------------------------------------------
        # 检测图像
        # -------------------------------------------------
        with detection_col:

            st.markdown("#### 识别结果")

            st.image(
                data["annotated_rgb"],
                use_container_width=True
            )

            st.caption(
                "识别结果：方框表示系统识别出的疑似火焰区域；"
                "框上的数值表示对应目标的识别置信度。"
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
        st.markdown("#### 识别数据")


        metric1, metric2, metric3, metric4 = st.columns(
            4
        )


        # 检测目标数量
        metric1.metric(
            "识别目标数量",
            f'{data["object_count"]}'
        )


        # 最高置信度
        metric2.metric(
            "最高置信度",
            (
                f'{data["max_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        # 平均置信度
        metric3.metric(
            "平均置信度",
            (
                f'{data["avg_conf"] * 100:.1f}%'
                if data["object_count"] > 0
                else "—"
            )
        )


        # 当前模式
        metric4.metric(
            "识别模式",
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
                "当前图像中未识别到符合当前设置要求的"
                "疑似火焰目标。"
            )

        else:

            st.write(
                f'本次共识别到 '
                f'{data["object_count"]} '
                f'个疑似火焰目标。'
            )

            st.write(
                "识别框用于表示系统定位到的疑似火焰区域；"
                "置信度用于反映系统对相应目标判断结果的可信程度。"
            )


        st.caption(
            f'本次识别耗时：'
            f'{data["inference_time"]:.2f} s'
        )


        # =================================================
        # 下载结果
        # =================================================
        success, encoded_image = cv2.imencode(
            ".png",
            data["annotated_bgr"]
        )


        if success:

            st.download_button(
                "下载识别结果",
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
    "Intelligent Fire Detection and Assessment System"
    "</div>",
    unsafe_allow_html=True
)
