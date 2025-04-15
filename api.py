import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import os
import gdown

# ==== 自定義 ResizeLayer：必須放在前面 ==== 
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

# ==== 註冊自定義層給 Keras ==== 
custom_objects = tf.keras.utils.get_custom_objects()
custom_objects["Custom>ResizeLayer"] = ResizeLayer

# ==== 確保模型檔案存在，否則從 Google Drive 下載 ====
if not os.path.exists("segmentation.h5"):
    gdown.download("https://drive.google.com/uc?id=1d44Rt7ihKTdkdhn2grWh6nlEJ4k4OUKh", "segmentation.h5", quiet=False)

# 如果你的 classification 模型也在 Google Drive，也加這段：
if not os.path.exists("analysis.h5"):
    gdown.download("https://drive.google.com/uc?id=你的分析模型ID", "analysis.h5", quiet=False)

# ==== 載入模型 ==== 
model_seg = tf.keras.models.load_model("segmentation.h5", custom_objects=custom_objects)
model_cls = tf.keras.models.load_model("analysis.h5", custom_objects=custom_objects)

# ==== 建立 Flask 應用 ==== 
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Server is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img_np = np.array(img)

    # ==== 預處理 ==== 
    input_image = tf.image.resize(img_np, (300, 400))
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_image = tf.expand_dims(input_image, axis=0)

    # ==== 模型 1：分割 ==== 
    segmentation_result = model_seg.predict(input_image)[0]
    mask = tf.squeeze(segmentation_result, axis=-1)
    binary_mask = tf.where(mask > 0.5, 1.0, 0.0)
    binary_mask = tf.expand_dims(binary_mask, axis=-1)
    masked_image = input_image[0] * binary_mask

    # ==== 模型 2：分類 ==== 
    input_image_cls = tf.image.resize(masked_image, (300, 400))
    input_image_cls = tf.cast(input_image_cls, tf.float32) / 255.0
    input_image_cls = tf.expand_dims(input_image_cls, axis=0)

    prediction = model_cls.predict(input_image_cls)[0]
    predicted_class = int(np.argmax(prediction))

    return jsonify({'result': predicted_class})

# ==== 啟動伺服器 ==== 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
