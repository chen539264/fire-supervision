import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time


# =========================================================
# 页面基础设置
# =========================================================
st.set_page_config(
    page_title="Fire Supervision",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 页面 CSS 美化
# =========================================================
st.markdown(
    """
    <style>

    /* ---------------- 页面背景 ---------------- */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(239, 68, 68, 0.05),
                transparent 28%
            ),
            #f5f7fa;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
    }


    /* ---------------- 隐藏默认元素 ---------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ---------------- 卡片 ---------------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        background: rgba(255,255,255,0.86);
        border-color: #e4e8ee;
    }


    /* ---------------- 上传框 ---------------- */

    [data-testid="stFileUploaderDropzone"] {
        background: #fafbfd;
        border: 1.5px dashed #ccd3dc;
        border-radius: 14px;
        padding: 16px;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ef5b49;
        background: #fff9f7;
    }


    /* ---------------- 下拉选择 ---------------- */

    [data-baseweb="select"] > div {
        border-radius: 12px !important;
        min-height: 48px;
    }


    /* ---------------- 图片 ---------------- */

    [data-testid="stImage"] img {
        border-radius: 13px;
    }


    /* ---------------- 指标 ---------------- */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e9ef;
        border-radius: 15px;
        padding: 17px;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }


    /* ---------------- 开始检测按钮 ---------------- */

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 54px;

        border: none;
        border-radius: 13px;

        background:
            linear-gradient(
                90deg,
                #e83e32 0%,
                #ff6741 100%
            );

        color: white;

        font-size: 16px;
        font-weight: 700;

        box-shadow:
            0 8px 18px
            rgba(239, 68, 68, 0.16);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        color: white;
        border: none;
        opacity: 0.94;
        transform: translateY(-1px);
    }


    /* ---------------- 下载按钮 ---------------- */

    .stDownloadButton > button {
        width: 100%;
        min-height: 47px;

        border-radius: 12px;

        font-weight: 650;
    }


    /* ---------------- 标题间距 ---------------- */

    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 顶部标题区域
# 单行 HTML，防止被 Streamlit 当代码显示
# =========================================================
st.markdown(
    '<div style="background:linear-gradient(115deg,#111827 0%,#1f2937 62%,#4a211b 100%);padding:27px 34px;border-radius:22px;margin-bottom:24px;box-shadow:0 15px 36px rgba(15,23,42,0.10);"><div style="display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;"><div><div style="color:#ff8b6d;font-size:12px;font-weight:750;letter-spacing:2px;margin-bottom:7px;">AI FIRE DETECTION PLATFORM</div><div style="color:white;font-size:36px;font-weight:800;line-height:1.2;">🔥 Fire Supervision</div><div style="color:#cbd5e1;font-size:15px;margin-top:9px;">基于 YOLOv8 的智能火灾目标检测与分析系统</div></div><div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);padding:9px 15px;border-radius:999px;color:#e8edf4;font-size:12px;font-weight:650;">● MODEL ONLINE</div></div></div>',
    unsafe_allow_html=True
)


# =========================================================
# 加载 YOLOv8 模型
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
# 主体布局
#
# 左边：设置
# 右边：工作区
# =========================================================
setting_col, workspace_col = st.columns(
    [1.05, 3.4],
    gap="large"
)


# =========================================================
# 左侧参数设置
# =========================================================
with setting_col:

    with st.container(border=True):

        st.markdown("## ⚙️ 检测设置")

        st.caption(
            "上传待检测图像，并选择 AI 检测强度。"
        )

        st.divider()


        # =================================================
        # 使用 form：
        # 用户必须点击开始检测才执行模型
        # =================================================
        with st.form("fire_detection_form"):

            # ---------------------------------------------
            # 上传图片
            # ---------------------------------------------
            st.markdown("### ① 上传图片")

            uploaded_file = st.file_uploader(
                "支持 JPG / JPEG / PNG",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ]
            )


            st.markdown("")


            # ---------------------------------------------
            # 检测强度
            # ---------------------------------------------
            st.markdown("### ② 检测强度")

            detection_level = st.selectbox(
                "请选择检测模式",
                [
                    "🔥 高强度检测",
                    "⚡ 标准检测",
                    "🛡️ 稳健检测"
                ],
                index=0,
                label_visibility="collapsed"
            )


            # ---------------------------------------------
            # 内部阈值
            #
            # 用户界面不会显示这些数字
            #
            # 阈值越低：
            # 保留候选目标越多
            # ---------------------------------------------
            confidence_map = {
                "🔥 高强度检测": 0.10,
                "⚡ 标准检测": 0.20,
                "🛡️ 稳健检测": 0.30
            }

            confidence = confidence_map[
                detection_level
            ]


            # ---------------------------------------------
            # 模式介绍
            # ---------------------------------------------
            if detection_level == "🔥 高强度检测":

                st.info(
                    "🔥 **高强度模式**\n\n"
                    "优先发现更多疑似火焰区域，"
                    "检测灵敏度最高，适合排查潜在风险。"
                )

            elif detection_level == "⚡ 标准检测":

                st.info(
                    "⚡ **标准模式**\n\n"
                    "兼顾检测灵敏度与结果稳定性，"
                    "适用于常规火灾目标检测。"
                )

            else:

                st.info(
                    "🛡️ **稳健模式**\n\n"
                    "更加关注明显的火焰目标，"
                    "减少弱目标及疑似误检。"
                )


            st.markdown("")


            # ---------------------------------------------
            # 开始检测
            # ---------------------------------------------
            detect_button = st.form_submit_button(
                "🔥 开始检测",
                use_container_width=True
            )


        # =================================================
        # 模式说明
        # =================================================
        st.divider()

        st.markdown("### 模式说明")

        st.markdown(
            "**🔥 高强度**  \n"
            "优先发现更多潜在火焰目标。"
        )

        st.markdown(
            "**⚡ 标准**  \n"
            "兼顾检测数量和结果稳定性。"
        )

        st.markdown(
            "**🛡️ 稳健**  \n"
            "过滤较弱候选目标，结果更加保守。"
        )

        st.caption(
            "检测强度越高，模型通常会保留更多"
            "疑似目标，因此检测框数量可能增加。"
        )


# =========================================================
# 点击开始检测之后执行
# =========================================================
if detect_button:

    # -----------------------------------------------------
    # 未上传图片
    # -----------------------------------------------------
    if uploaded_file is None:

        st.warning(
            "请先在左侧上传一张需要检测的图片。"
        )


    # -----------------------------------------------------
    # 开始检测
    # -----------------------------------------------------
    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        start_time = time.time()


        with st.spinner(
            "YOLOv8 正在分析图像，请稍候..."
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
        # YOLO 绘制检测框
        # =========================================================

        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB
        )


        # =================================================
        # 获取 Boxes
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
        # 保存到 Session
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
# 右侧工作区域
# =========================================================
with workspace_col:

    st.markdown("## 🔍 检测工作区")

    st.caption(
        "原始图像与 AI 检测结果将在下方进行同步对比。"
    )


    # =====================================================
    # 工作区左右分栏
    # =====================================================
    original_col, detection_col = st.columns(
        2,
        gap="large"
    )


    # =====================================================
    # 没有检测结果时
    # =====================================================
    if st.session_state.result_data is None:


        # -------------------------------------------------
        # 原始图片等待框
        # -------------------------------------------------
        with original_col:

            with st.container(border=True):

                st.markdown("### 🖼️ 原始图像")

                st.markdown("")

                st.markdown(
                    "<div style='text-align:center;font-size:50px;padding-top:68px;'>🖼️</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<div style='text-align:center;font-size:19px;font-weight:700;color:#303846;margin-top:10px;'>等待图像</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<div style='text-align:center;color:#8c96a5;font-size:13px;margin-top:8px;margin-bottom:76px;'>上传的原始图片将在这里显示</div>",
                    unsafe_allow_html=True
                )

                st.divider()

                st.markdown(
                    "**原始图像含义**"
                )

                st.caption(
                    "展示未经模型处理的原始输入图像，"
                    "用于和右侧 AI 检测结果进行对比。"
                )


        # -------------------------------------------------
        # AI 结果等待框
        # -------------------------------------------------
        with detection_col:

            with st.container(border=True):

                st.markdown("### 🔥 AI 检测结果")

                st.markdown("")

                st.markdown(
                    "<div style='text-align:center;font-size:50px;padding-top:68px;'>🔥</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<div style='text-align:center;font-size:19px;font-weight:700;color:#303846;margin-top:10px;'>等待检测</div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    "<div style='text-align:center;color:#8c96a5;font-size:13px;margin-top:8px;margin-bottom:76px;'>点击左侧「开始检测」后生成结果</div>",
                    unsafe_allow_html=True
                )

                st.divider()

                st.markdown(
                    "**检测结果含义**"
                )

                st.caption(
                    "YOLOv8 将在疑似火焰区域绘制检测框，"
                    "用于展示模型定位到的目标区域。"
                )


    # =====================================================
    # 已经有检测结果
    # =====================================================
    else:

        data = st.session_state.result_data


        # -------------------------------------------------
        # 左：原始图
        # -------------------------------------------------
        with original_col:

            with st.container(border=True):

                st.markdown("### 🖼️ 原始图像")

                st.image(
                    data["image"],
                    use_container_width=True
                )

                st.divider()

                st.markdown(
                    "**图像含义**"
                )

                st.caption(
                    "这是用户上传的原始图像，"
                    "未经过任何检测框绘制或目标标注。"
                )

                st.caption(
                    "它作为模型输入，也是判断检测效果的"
                    "原始参照。"
                )


        # -------------------------------------------------
        # 右：检测结果
        # -------------------------------------------------
        with detection_col:

            with st.container(border=True):

                st.markdown("### 🔥 AI 检测结果")

                st.image(
                    data["annotated_rgb"],
                    use_container_width=True
                )

                st.divider()

                st.markdown(
                    "**结果含义**"
                )

                st.caption(
                    "检测框表示 YOLOv8 识别到的"
                    "疑似火焰区域。"
                )

                st.caption(
                    "检测框上显示的数值表示模型对该目标"
                    "属于火焰类别的判断置信程度。"
                )


        # =================================================
        # 检测解释
        # =================================================
        st.markdown("### 📖 检测结果说明")

        with st.container(border=True):

            legend1, legend2, legend3 = st.columns(
                3,
                gap="large"
            )


            with legend1:

                st.markdown("#### ▣ 检测框")

                st.write(
                    "框选模型认为可能存在火焰"
                    "目标的位置与范围。"
                )


            with legend2:

                st.markdown("#### % 置信度")

                st.write(
                    "反映模型对当前目标判断结果的"
                    "可信程度。"
                )


            with legend3:

                st.markdown("#### 🔥 检测强度")

                st.write(
                    "强度越高，系统越倾向于保留"
                    "更多疑似火焰目标。"
                )


        # =================================================
        # 数据统计
        # =================================================
        st.markdown("### 📊 检测数据")


        mode_name_map = {

            "🔥 高强度检测":
                "高强度",

            "⚡ 标准检测":
                "标准",

            "🛡️ 稳健检测":
                "稳健"
        }


        metric1, metric2, metric3, metric4 = st.columns(
            4,
            gap="medium"
        )


        # -------------------------------------------------
        # 检测数量
        # -------------------------------------------------
        metric1.metric(
            "检测目标",
            f'{data["object_count"]} 个'
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
            mode_name_map[
                data["detection_level"]
            ]
        )


        # =================================================
        # AI 分析状态
        # =================================================
        st.markdown("### 🚨 AI 分析状态")

        with st.container(border=True):

            # ---------------------------------------------
            # 无目标
            # ---------------------------------------------
            if data["object_count"] == 0:

                st.success(
                    "✅ 当前图像中未检测到符合"
                    "当前模式要求的火焰目标。"
                )


            # ---------------------------------------------
            # 有目标
            # ---------------------------------------------
            else:

                if data["max_conf"] >= 0.60:

                    st.error(
                        "🔥 检测到较高置信度的火焰目标，"
                        "建议进一步核查现场情况。"
                    )

                elif data["max_conf"] >= 0.30:

                    st.warning(
                        "⚠️ 检测到疑似火焰目标，"
                        "建议结合现场环境进一步确认。"
                    )

                else:

                    st.info(
                        "ℹ️ 检测到低置信度疑似火焰目标，"
                        "建议结合其他信息综合判断。"
                    )


            st.caption(
                f'模型本次处理耗时约 '
                f'{data["inference_time"]:.2f} 秒。'
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
                "⬇️ 下载检测结果图片",
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
    "<div style='text-align:center;color:#9aa3b1;font-size:12px;padding:4px 0 10px 0;'>Fire Supervision · YOLOv8 Intelligent Fire Detection System</div>",
    unsafe_allow_html=True
)
