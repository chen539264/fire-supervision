from ultralytics import YOLO
import gradio as gr
import numpy as np

# 加载你训练好的 YOLOv8 模型
model = YOLO("best.pt")


def predict(image):
    # YOLO 推理
    results = model.predict(
        source=image,
        conf=0.25
    )

    # 获取 YOLO 绘制好检测框的图片
    result_image = results[0].plot()

    # BGR -> RGB
    result_image = result_image[:, :, ::-1]

    return result_image


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="上传图片"),
    outputs=gr.Image(type="numpy", label="检测结果"),
    title="YOLOv8 目标检测系统",
    description="上传一张图片，系统将使用训练好的 YOLOv8 模型进行目标检测。"
)


if __name__ == "__main__":
    demo.launch()
