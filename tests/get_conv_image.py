"""
  filename      : get_conv_image
  author        : quanzhou.li
  date          : 2024/10/27
  Description   : 获得卷积后的图像
"""
from PIL import Image
from matplotlib import pyplot as plt

from tools import *

image = Image.open('../resources/test3_3.jpg')

r, g, b = image.split()
r_array = np.array(r)

# 直接卷积
conv_array = conv_with_core(r_array, laplace_core)

# 中值去噪
conv_array = median_filter(conv_array, 3)

# 激活函数
conv_array = relu(conv_array)

# 阈值去噪
conv_array = conv_with_core(conv_array, gaussian_kernel(3, 3), pad=True)
conv_array = apply_threshold(conv_array, 100)

# conv_array = sigmoid(conv_array)

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

# 显示红色通道
ax[0].imshow(r_array, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# 显示卷积后的图像
ax[1].imshow(conv_array, cmap='gray')
ax[1].set_title('Conv Layer')
ax[1].axis('off')

plt.tight_layout()
plt.show()