# 导入必要的库
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import ndimage
from scipy.ndimage import gaussian_filter

#设置路径
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib显示参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)

# 设置seaborn样式
sns.set_style("whitegrid")

# 定义英文标题映射
TITLES = {
    'original': 'Original Image',
    'original_gray': 'Original Grayscale Image',
    'gaussian_blur': 'Gaussian Blur',
    'gradient_x': 'Gradient X (Gx)',
    'gradient_y': 'Gradient Y (Gy)', 
    'gradient_magnitude': 'Gradient Magnitude',
    'gradient_direction': 'Gradient Direction',
    'non_max_suppression': 'Non-Maximum Suppression',
    'double_threshold': 'Double Threshold',
    'edge_tracking': 'Edge Tracking by Hysteresis',
    'canny_result': 'Canny Edge Detection Result',
    'canny_comparison': 'Canny Parameter Comparison',
    'sobel_x': 'Sobel X',
    'sobel_y': 'Sobel Y',
    'sobel_combined': 'Sobel Combined',
    'harris_response': 'Harris Response',
    'harris_corners': 'Harris Corners',
    'corner_comparison': 'Corner Detection Comparison',
    'structure_tensor': 'Structure Tensor',
    'eigenvalues': 'Eigenvalues Analysis',
    'corner_classification': 'Corner Classification'
}

# 读取和预处理图像
img = cv2.imread('Lenna.jpg')

if img is None:
    print("Error: Cannot read image file 'Lenna.jpg'. Please ensure the file exists in the current directory.")
else:
    print("Image loaded successfully!")
    
    # 转换为RGB和灰度图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度变换：对比度增强
def contrast_stretch(image):
    """对比度拉伸"""
    low_val = np.percentile(image, 2)
    high_val = np.percentile(image, 98)
    stretched = np.clip((image - low_val) * 255.0 / (high_val - low_val), 0, 255)
    return stretched.astype(np.uint8)

# 手动实现直方图均衡化
def manual_histogram_equalization(image):
    """手动实现直方图均衡化"""
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    equalized_image = cdf_normalized[image]
    return equalized_image

# 应用变换
contrast_img = contrast_stretch(img_gray)
equalized_img = manual_histogram_equalization(img_gray)

# 只显示三张图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原图
axes[0].imshow(img_rgb, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

# 灰度变换图
axes[1].imshow(img_gray, cmap='gray')
axes[1].set_title('Contrast Enhanced Image')
axes[1].axis('off')

# 直方图均衡化图
axes[2].imshow(equalized_img, cmap='gray')
axes[2].set_title('Histogram Equalized Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()