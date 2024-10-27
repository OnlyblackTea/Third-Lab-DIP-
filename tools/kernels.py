"""
  filename      : kernels
  author        : quanzhou.li
  date          : 2024/10/27
  Description   : 存放算子以及产生算子的函数
"""

import numpy as np

laplace_core = np.array([[ 1,  1,  1],
                          [ 1, -8,  1],
                          [ 1,  1,  1]])

sobel_x = np.array([[ -1,  0,  1],
                     [ -2,  0,  2],
                     [ -1,  0,  1]])

sobel_y = np.array([[ -1, -2, -1],
                     [  0,  0,  0],
                     [  1,  2,  1]])

sobel_diagonal_1 = np.array([[ 0,  1,  2],
                              [-1,  0,  1],
                              [-2, -1,  0]])

sobel_diagonal_2 = np.array([[ -2, -1,  0],
                              [ -1,  0,  1],
                              [  0,  1,  2]])

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

def gaussian_kernel(size, sigma):
    """
    生成高斯核
    :param size:
    :param sigma:
    :return:
    """
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) *
                     np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)),
                    # 在生成高斯核时，我们希望核的中心与图像的当前像素对齐。
                    # 使用(x-(size-1)/2)可以确保高斯核的中心位于核的中心像素
        (size, size)
    )
    return kernel / np.sum(kernel)  # 归一化

def laplace_of_gaussian_kernel(size, sigma):
    """
    生成高斯拉普拉斯核（deprecated）
    :param size:
    :param sigma:
    :return:
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    center = size // 2
    x, y = np.indices((size, size)) - center
    r_squared = x ** 2 + y ** 2

    # 计算高斯拉普拉斯值
    kernel = (1 / (2 * np.pi * sigma ** 4)) * (sigma ** 2 - r_squared) * np.exp(-r_squared / (2 * sigma ** 2))

    # 归一化
    # kernel /= np.sum(np.abs(kernel))

    return kernel