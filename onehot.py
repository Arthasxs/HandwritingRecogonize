from tensorflow.keras.utils import to_categorical

# 假设有5个类别
labels = [0, 2, 4, 1, 3]

# 进行独热编码
#标签之间的关系被编码成向量的位置关系
one_hot_labels = to_categorical(labels)

print("原始标签:", labels)
print("独热编码后:\n", one_hot_labels)

