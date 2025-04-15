import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS  # 支援跨域

# ==== 自定義 ResizeLayer ====
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

# ==== 註冊自定義層到 TensorFlow ====
custom_objects = tf.keras.utils.get_custom_objects()
custom_objects["Custom>ResizeLayer"] = ResizeLayer

# ==== 載入模型 ====
model_seg = tf.keras.models.load_model(r"C:\Users\JIMMY\Desktop\rice_disease_api\segmentation.h5", custom_objects=custom_objects)
model_cls = tf.keras.models.load_model(r"C:\Users\JIMMY\Desktop\rice_disease_api\analysis.h5", custom_objects=custom_objects)

# ==== Flask 應用 ====
app = Flask(__name__)
CORS(app)  # 啟用跨域支持

# ==== 預測 API ====
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img_np = np.array(img)

    # ==== 預處理：轉為模型格式 ==== 
    # 對於第一個模型 (假設需要 300x400)
    input_image = tf.image.resize(img_np, (300, 400))  # 將圖像調整為 300x400
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_image = tf.expand_dims(input_image, axis=0)  # (1, 300, 400, 3)

    # ==== 模型 1：圖像分割 (保持 300x400) ====
    segmentation_result = model_seg.predict(input_image)[0]  # (H, W, 1)

    # ==== 轉為遮罩 ====
    mask = tf.squeeze(segmentation_result, axis=-1)  # (H, W)
    binary_mask = tf.where(mask > 0.5, 1.0, 0.0)  # 二值化 (H, W)
    binary_mask = tf.expand_dims(binary_mask, axis=-1)  # (H, W, 1)

    # ==== 疊圖遮罩到原圖 ====
    masked_image = input_image[0] * binary_mask  # (H, W, 3)

    # ==== 對於第二個模型，將圖像大小調整為 300x400 (這是 model_cls 所需的尺寸) ====
    input_image_cls = tf.image.resize(masked_image, (300, 400))  # 調整為 300x400
    input_image_cls = tf.cast(input_image_cls, tf.float32) / 255.0
    input_image_cls = tf.expand_dims(input_image_cls, axis=0)  # (1, 300, 400, 3)

    # ==== 模型 2：圖像辨識 ====
    prediction = model_cls.predict(input_image_cls)[0]  # e.g., [0.1, 0.7, 0.2]
    predicted_class = int(np.argmax(prediction))  # 0 or 1 or 2

    return jsonify({'result': predicted_class})


# ==== 啟動伺服器 ====
if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return 'Server is running!'

# 在 api.py 的最下方
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

