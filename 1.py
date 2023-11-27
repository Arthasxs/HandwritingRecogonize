import tkinter as tk
from tkinter import Canvas, Button, Label, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")

        # 创建画板
        self.initial_canvas_width = 200
        self.initial_canvas_height = 200
        self.canvas = Canvas(root, width=self.initial_canvas_width, height=self.initial_canvas_height, bg="black")
        self.canvas.pack(pady=10)

        # 创建按钮
        self.predict_button = Button(root, text="预测", command=self.predict_digit)
        self.predict_button.pack(pady=5)

        self.clear_button = Button(root, text="清空", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

        self.open_button = Button(root, text="打开文件", command=self.open_file)
        self.open_button.pack(pady=5)

        # 预测结果标签
        self.result_label = Label(root, text="预测结果：", font=("Helvetica", 16), fg="white")
        self.result_label.pack(pady=10)

        # 初始化画板
        self.image = Image.new("L", (self.initial_canvas_width, self.initial_canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.image_on_canvas = None

        # 加载训练好的模型
        self.model = load_model('t1.h5')

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_line(x1, y1, x2, y2, fill="white", width=15)
        self.draw.line([x1, y1, x2, y2], fill="white", width=15)

    def predict_digit(self):
        # 保存画板内容为图像文件
        self.image.save("canvas.png")

        # 读取图像，预处理，并进行预测
        img = image.load_img("canvas.png", color_mode="grayscale", target_size=(28, 28))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # 进行预测
        prediction = self.model.predict(img_array)

        # 获取概率最大的数字
        predicted_label = np.argmax(prediction)

        # 在标签中显示预测结果
        self.result_label.config(text=f"预测结果：{predicted_label}")

        # 打印概率分布到命令行
        formatted_probs = [f"{prob:.3g}" for prob in prediction[0]]
        print("概率分布：", formatted_probs)

    def clear_canvas(self):
        # 保存当前画板尺寸
        current_canvas_width = self.canvas.winfo_width()
        current_canvas_height = self.canvas.winfo_height()

        # 清空画板
        self.canvas.delete("all")
        self.image = Image.new("L", (self.initial_canvas_width, self.initial_canvas_height), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="预测结果：")
        # 清空画板上的图像
        if self.image_on_canvas:
            self.canvas.delete(self.image_on_canvas)

        # 还原画板尺寸
        self.canvas.config(width=self.initial_canvas_width, height=self.initial_canvas_height)

    def open_file(self):
        # 打开文件对话框，不限制文件类型
        file_path = filedialog.askopenfilename(initialdir='/home/hqyj/Documents/number', filetypes=[])
        
        if file_path:
            # 读取图像
            img = Image.open(file_path)

            # 更新画板上的图像
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=ImageTk.PhotoImage(img))
            # 使画板大小适应图像大小
            self.canvas.config(width=img.width, height=img.height)

            # 保存图像对象，以便在清空画板时能够删除
            self.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.bind("<B1-Motion>", app.paint)
    root.mainloop()

