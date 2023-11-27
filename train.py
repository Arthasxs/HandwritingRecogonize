import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# 加载数据集
# x为特征数据，y为标签数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 这里通过'reshape'将其形状调整为(样本数量, 28, 28, 1)，其中1表示图像是灰度图像
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = x_train.astype('float32') / 255  # 像素值转换,归一化
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train)  # onehot编码
y_test = to_categorical(y_test)

# 创建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 卷积层relu激活函数,单侧抑制
model.add(MaxPooling2D((2, 2)))  # 最大池化层减小特征图的空间尺寸，同时保留最重要的信息
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())  # 展平层,展平为一维数组
model.add(Dense(64, activation='relu'))  # 全连接层
model.add(Dense(10, activation='softmax'))  # 激活函数softmax对应概率分布

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])  # 随机梯度下降优化器,损失函数,性能指标

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=64)#迭代轮数,批量大小

# 保存模型
model.save('t1.h5')

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)#评估函数,返回损失与评估指标
print('Test accuracy:', test_acc)

# 预测
predictions = model.predict(x_test)#包含模型对测试集的每个图像的预测结果的数组

# 显示一些预测结果
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')#显示第i个测试集图像的像素值28*28,以灰度图表示
    plt.show()
    print("Prediction:", np.argmax(predictions[i]))#返回数组中对应概率最大的索引

