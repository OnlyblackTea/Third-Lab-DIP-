"""
  filename      : functions
  author        : quanzhou.li
  date          : 2024/10/27
  Description   : 基本函数
"""
import math
import numpy as np

# 卷积
def conv_with_core(img, knl, pad = False):
    """
    卷积运算
    :param img: array格式的图片
    :param knl: 卷积核
    :param pad: 是否填充
    :return: 卷积后的输出
    """
    knl_height, knl_width = knl.shape
    if pad:
        pad_height = knl_height // 2
        pad_width = knl_width // 2
        img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    img_height, img_width = img.shape
    ret_height = img_height - knl_height + 1
    ret_width = img_width - knl_width + 1
    ret = np.zeros((ret_height, ret_width))
    for i in range(ret_height):
        for j in range(ret_width):
            rgn = img[i:i+knl_height, j:j+knl_width]
            ret[i, j] = np.sum(rgn * knl)
    return ret

def median_filter(img, knl_size):
    """
    对图像应用中值滤波
    :param img:
    :param knl_size:
    :return:
    """
    if knl_size % 2 == 0:
        raise ValueError("knl_size must be odd number")
    pad_size = knl_size // 2
    pad_img = np.pad(img, pad_size, mode='edge')
    ret = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 提取当前像素及其邻域
            rgn = pad_img[i:i+knl_size, j:j+knl_size]
            ret[i, j] = np.median(rgn)
    return ret

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)

# 阈值函数
def apply_threshold(img, threshold):
    ret = np.zeros_like(img)
    ret[img > threshold] = img[img > threshold]
    return ret

def clip_and_convert(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        return np.clip(ret, 0, 255).astype(np.uint8)
    return wrapper

# log变换
@clip_and_convert
def T_log(r, base=2):
    norm_r = r/255 + 1
    return math.log(r, base) * 255

# 灰度反转
@clip_and_convert
def T_reverse(r):
    return 255 - r

# 线性变换
@clip_and_convert
def T_linear(r, a, b, c, d):
    """
    线性变换
    :param r: 输入像素值
    :param a: 暗处拐点输入值
    :param b: 亮处拐点输入值
    :param c: 暗处拐点输出值
    :param d: 亮处拐点输出值
    :return s: 变换后的像素值
    """
    return ((d - c) / (b - a)) * (r - a) + c