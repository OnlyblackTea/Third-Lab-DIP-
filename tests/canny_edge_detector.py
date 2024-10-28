"""
  filename      : canny_edge_detector
  author        : 13105
  date          : 2024/10/28
  Description   : canny边缘检测
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tools import *


def canny_edge_detector(image, sigma=1.4, low_threshold=50, high_threshold=150):
    # 1. 高斯模糊(降噪处理)
    blurred = gaussian_filter(image, sigma=sigma)

    # 2. 计算梯度(使用sobel算子，3x3的卷积核)
    # Sobel算子
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    Gx = conv_with_core(blurred, sobel_x)
    Gy = conv_with_core(blurred, sobel_y)

    # 计算梯度幅值和方向
    gradient_magnitude = np.hypot(Gx, Gy)
    gradient_direction = np.arctan2(Gy, Gx) * (180.0 / np.pi) % 180  # 梯度方向

    # print("Gradient Magnitude Min:", gradient_magnitude.min())
    # print("Gradient Magnitude Max:", gradient_magnitude.max())

    # 3. 非极大值抑制
    height, width = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_direction[i, j]
            # 确定相邻像素的坐标
            q = r = 255  # 先假设相邻值为255
            # 0度方向
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # 45度方向
            elif (22.5 <= angle < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # 90度方向
            elif (67.5 <= angle < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # 135度方向
            elif (112.5 <= angle < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            # 非极大值抑制
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    # print("Suppressed Image:", suppressed)

    # 4. 双阈值处理
    strong_edges = (suppressed > high_threshold)
    thresholded_edges = np.zeros_like(suppressed)
    weak_edges = ((suppressed >= low_threshold) & (suppressed <= high_threshold))
    # print("Strong Edges Count:", np.sum(strong_edges))
    # print("Weak Edges Count:", np.sum(weak_edges))
    # 强边缘
    thresholded_edges[strong_edges] = 1

    # 弱边缘：连接强边缘
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if weak_edges[i, j] and (
                    strong_edges[i + 1, j] or strong_edges[i - 1, j] or
                    strong_edges[i, j + 1] or strong_edges[i, j - 1] or
                    strong_edges[i + 1, j + 1] or strong_edges[i - 1, j - 1] or
                    strong_edges[i + 1, j - 1] or strong_edges[i - 1, j + 1]):
                thresholded_edges[i, j] = 1
    # 5. 返回边缘图像
    return thresholded_edges


# 示例使用
if __name__ == "__main__":
    # 生成一个简单的测试图像
    image = Image.open('../resources/test3_3.jpg')

    r, g, b = image.split()
    r_array = np.array(r)
    g_array = np.array(g)
    b_array = np.array(b)
    l_array = np.array(image.convert('L'))

    # Canny边缘检测
    edges = canny_edge_detector(r_array)
    # 高斯滤波
    # blurred_edges = gaussian_filter(edges, sigma=1, size=5)

    # 显示结果
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(r_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Canny Edges')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.show()