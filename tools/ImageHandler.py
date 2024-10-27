"""
  filename      : ImageHandler
  author        : quanzhou.li
  date          : 2024/10/21
  Description   :
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tools import T_linear


class ImageHandler:

    def __init__(self, image: Image):
        self.im = image
        # 将图片转化为矩阵处理
        self.image_matrix = np.array(self.im)
        self.output_matrix = self.image_matrix

    def draw_histogram(self):
        # 使用Unique函数统计每个值的出现次数
        values, counts = np.unique(self.image_matrix, return_counts=True)
        # 绘制直方图
        plt.bar(values, counts)
        plt.title('Pixel Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()

    def linear_transform(self):
        # 初始化线性变换参数
        a, c = 100.0, 1.0
        b, d = 180.0, 255.0

        # 绘制初始图像和曲线
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        ax[0].imshow(self.image_matrix, cmap='gray')
        ax[0].set_aspect('equal')
        ax[0].set_title('Image')
        ax[0].axis('off')

        # 绘制编辑后的图像和曲线
        ax[1].imshow(T_linear(self.image_matrix, a, b, c, d), cmap='gray')
        ax[1].set_aspect('equal')
        ax[1].set_title('Image')
        ax[1].axis('off')

        # 绘制变换曲线图
        x = np.arange(256)
        y = T_linear(x, a, b, c, d)
        ax[2].plot(x, y)
        ax[2].set_title('Linear Transform')
        ax[2].set_xlim(0, 255)
        ax[2].set_ylim(0, 255)

        plt.show()
        self.output_matrix = T_linear(self.image_matrix, a, b, c, d)

