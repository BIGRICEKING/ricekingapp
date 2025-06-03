import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import os
import gdown
import psutil

# 用量計算
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # 轉成 MB
    print(f"[Memory] {note}: {mem:.2f} MB")

# ==== 自定義ResizeLayer放在前面 ==== 
@tf.keras.utils.register_keras_serializable("Custom")
class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config

# ==== 註冊自定義層給Keras ==== 
custom_objects = tf.keras.utils.get_custom_objects()
custom_objects["Custom>ResizeLayer"] = ResizeLayer

# 獲取目前執行目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 自動下載 segmentation.h5（如果不存在）
seg_path = os.path.join(BASE_DIR, "segmentation.h5")
if not os.path.exists(seg_path):
    gdown.download("https://drive.google.com/uc?id=1d44Rt7ihKTdkdhn2grWh6nlEJ4k4OUKh", seg_path, quiet=False)

print("segmentation.h5 是否存在：", os.path.exists(seg_path))
print("segmentation.h5 路徑：", seg_path)

# 檢查 analysis.h5 是否存在，否則從 Google Drive 下載
cls_path = os.path.join(BASE_DIR, "analysis.h5")
if not os.path.exists(cls_path):
    print("找不到 analysis.h5，開始從 Google Drive 下載")
    gdown.download("https://drive.google.com/uc?id=1G6q_AKZi7MyNJAX9b9pIOkvT8DRwpwMU", cls_path, quiet=False)

print("檢查 analysis.h5 是否存在：", os.path.exists(cls_path))
print("目前 BASE_DIR：", BASE_DIR)

# ==== 載入模型 ==== 
print_memory_usage("載入 segmentation.h5 前")
model_seg = tf.keras.models.load_model(seg_path, custom_objects=custom_objects)
print_memory_usage("載入 segmentation.h5 後")

print_memory_usage("載入 analysis.h5 前")
model_cls = tf.keras.models.load_model(cls_path, custom_objects=custom_objects)
print_memory_usage("載入 analysis.h5 後")

# # 模型 warm-up（初始化做一次假推論）
# dummy_input = np.zeros((1, 300, 400, 3), dtype=np.float32)  # 模擬一張圖片的大小
# print("模型 warm-up 開始")
# print_memory_usage("warm-up 前")
# _ = model_seg.predict(dummy_input)
# _ = model_cls.predict(dummy_input)
# print_memory_usage("warm-up 後")
# print("模型 warm-up 完成")

# ==== 建立 Flask 應用 ==== 
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Server is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:

        print_memory_usage("收到請求")

        if 'image' not in request.files:
            print("沒有收到 'image' 欄位")
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        print(f"收到圖片：{file.filename}")

        # 圖片處理
        img = Image.open(file).convert('RGB')
        img_np = np.array(img)
        print_memory_usage("圖片轉 numpy 後")

        # ==== 預處理 ==== 
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_image = tf.expand_dims(input_image, axis=0)
        print_memory_usage("預處理完成")

        # ==== 模型 1：分割 ==== 
        segmentation_result = model_seg.predict(input_image)[0]
        print_memory_usage("分割模型預測後")
        mask = tf.squeeze(segmentation_result, axis=-1)
        binary_mask = tf.where(mask > 0.5, 1.0, 0.0)
        binary_mask = tf.expand_dims(binary_mask, axis=-1)
        masked_image = input_image[0] * binary_mask

        # ==== 模型 2：分類 ==== 
        input_image_cls = tf.image.resize(masked_image, (300, 400))
        input_image_cls = tf.expand_dims(input_image_cls, axis=0)

        prediction = model_cls.predict(input_image_cls)[0]
        predicted_class = int(np.argmax(prediction))

        predicted_class = int(np.argmax(prediction))
        return jsonify({'result': predicted_class})

    except Exception as e:
        import traceback
        print("predict 發生錯誤：", str(e))
        traceback.print_exc()  # 印出詳細錯誤
        return jsonify({'error': str(e)}), 500


# ==== 啟動伺服器 ==== 
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


print("Current working directory:", os.getcwd())

