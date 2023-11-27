import tensorflow as tf

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 打印数据集的形状
print(f"Train Images Shape: {train_images.shape}") #训练集的三维张量(数量,高度,宽度)
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}")#测试集的三维张量(数量,高度,宽度)
print(f"Test Labels Shape: {test_labels.shape}")

