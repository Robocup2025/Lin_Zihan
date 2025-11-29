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
    
# 手工实现Canny边缘检测的各个步骤
def manual_canny_edge_detection(image, sigma=1.0, low_threshold=50, high_threshold=150):
    
    # 步骤1: 高斯滤波降噪
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    
    # 步骤2: 计算梯度
    # Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # 卷积计算梯度
    grad_x = ndimage.convolve(blurred, sobel_x)
    grad_y = ndimage.convolve(blurred, sobel_y)
    
    # 梯度幅值和方向
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = np.arctan2(grad_y, grad_x)
    
    return {
        'blurred': blurred,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'gradient_magnitude': gradient_magnitude,
        'gradient_direction': gradient_direction,
    }

# 执行手工实现
if img is not None:
    canny_steps = manual_canny_edge_detection(img_gray)

# 可视化Canny边缘检测的各个步骤
if img is not None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 第一行：原始图像和预处理
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title(TITLES['original_gray'], fontsize=14)
    axes[0,0].axis('off')
    
    # 第二行：梯度分量
    axes[1,0].imshow(canny_steps['grad_x'], cmap='gray')
    axes[1,0].set_title(TITLES['gradient_x'], fontsize=14)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(canny_steps['grad_y'], cmap='gray')
    axes[1,1].set_title(TITLES['gradient_y'], fontsize=14)
    axes[1,1].axis('off')
    
    # 梯度方向可视化
    gradient_direction_deg = canny_steps['gradient_direction'] * 180 / np.pi
    axes[0,1].imshow(gradient_direction_deg, cmap='hsv')
    axes[0,1].set_title(TITLES['gradient_direction'], fontsize=14)
    axes[0,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    