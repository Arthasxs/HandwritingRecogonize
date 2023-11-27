import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载训练好的模型
model = load_model('t1.h5')

# 读取手写数字图像
image_path = 'hand.jpg'  # 替换图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (28, 28))
normalized_image = resized_image.reshape(1, 28, 28, 1) / 255.0

# 使用模型进行预测
prediction = model.predict(normalized_image)
predicted_label = np.argmax(prediction)

print("Predicted label:", predicted_label)

