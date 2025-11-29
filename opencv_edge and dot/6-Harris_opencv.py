# 导入必要的库
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import ndimage
from scipy.ndimage import gaussian_filter

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

# OpenCV Harris实现对比和参数影响分析
if img is not None:
    # OpenCV Harris角点检测
    harris_opencv = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    harris_k_low = cv2.cornerHarris(img_gray, blockSize=2, ksize=5, k=0.04)
    harris_k_high = cv2.cornerHarris(img_gray, blockSize=5, ksize=3, k=0.04)
    harris_block_large = cv2.cornerHarris(img_gray, blockSize=5, ksize=5, k=0.04)

    # 标记角点
    img_corners_opencv1 = img_rgb.copy()
    img_corners_opencv1[harris_opencv > 0.01 * harris_opencv.max()] = [255, 0, 0]
    img_corners_opencv2 = img_rgb.copy()
    img_corners_opencv2[harris_k_low > 0.01 * harris_k_low.max()] = [255, 0, 0]
    img_corners_opencv3 = img_rgb.copy()
    img_corners_opencv3[harris_k_high > 0.01 * harris_k_high.max()] = [255, 0, 0]
    img_corners_opencv4 = img_rgb.copy()
    img_corners_opencv4[harris_block_large > 0.01 * harris_block_large.max()] = [255, 0, 0]

    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 第二行：参数影响
    axes[0,0].imshow(img_corners_opencv1)
    axes[0,0].set_title('Harris blockSize=2, ksize=3', fontsize=14)
    axes[0,0].axis('off')

    axes[0,1].imshow(img_corners_opencv2)
    axes[0,1].set_title('Harris blockSize=2, ksize=5', fontsize=14)
    axes[0,1].axis('off')
    
    axes[1,0].imshow(img_corners_opencv3, cmap='hot')
    axes[1,0].set_title('Harris blockSize=5, ksize=3', fontsize=14)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img_corners_opencv4, cmap='hot')
    axes[1,1].set_title('Harris blockSize=5, ksize=5', fontsize=14)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\\n=== Harris Parameter Analysis ===")
    print(f"OpenCV Harris max response: {harris_opencv.max():.6f}")
    print(f"Harris k=0.02 max response: {harris_k_low.max():.6f}")
    print(f"Harris k=0.08 max response: {harris_k_high.max():.6f}")
    print(f"Harris blockSize=5 max response: {harris_block_large.max():.6f}")
