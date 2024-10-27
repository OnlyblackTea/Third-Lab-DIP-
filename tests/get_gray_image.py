"""
  filename      : get_gray_image
  author        : quanzhou.li
  date          : 2024/10/26
  Description   : 获取图像灰度图
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image = Image.open('../resources/test3_3.jpg')

r, g, b = image.split()
r_array = np.array(r)
g_array = np.array(g)
b_array = np.array(b)
l_array = np.array(image.convert('L'))

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# 显示红色通道
ax[0, 0].imshow(r_array)
ax[0, 0].set_title('Red Channel')
ax[0, 0].axis('off')

# 显示绿色通道
ax[0, 1].imshow(g_array)
ax[0, 1].set_title('Green Channel')
ax[0, 1].axis('off')

# 显示蓝色通道
ax[1, 0].imshow(b_array)
ax[1, 0].set_title('Blue Channel')
ax[1, 0].axis('off')

# 显示灰度通道
ax[1, 1].imshow(l_array)
ax[1, 1].set_title('Light Channel')
ax[1, 1].axis('off')

plt.tight_layout()
plt.show()