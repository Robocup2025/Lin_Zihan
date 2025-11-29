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

# 手工实现Harris角点检测算法
def manual_harris_corner_detection(image, k=0.04, threshold=0.01, window_size=3):
    """
    手工实现Harris角点检测算法
    """
    print("=== Manual Harris Corner Detection Steps ===")
    
    # 步骤1: 计算图像梯度
    # 使用Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Ix = ndimage.convolve(image.astype(np.float32), sobel_x)
    Iy = ndimage.convolve(image.astype(np.float32), sobel_y)
    
    print("Step 1: Gradient calculation - completed")
    
    # 步骤2: 计算结构张量矩阵的元素
    Ixx = Ix * Ix
    Ixy = Ix * Iy  
    Iyy = Iy * Iy
    
    print("Step 2: Structure tensor elements calculation - completed")
    
    # 步骤3: 高斯加权
    sigma = 1.0
    Sxx = gaussian_filter(Ixx, sigma)
    Sxy = gaussian_filter(Ixy, sigma)
    Syy = gaussian_filter(Iyy, sigma)
    
    print("Step 3: Gaussian weighting - completed")
    
    # 步骤4: 计算Harris响应函数
    det_M = Sxx * Syy - Sxy * Sxy  # 行列式
    trace_M = Sxx + Syy            # 迹
    
    harris_response = det_M - k * (trace_M ** 2)
    
    print(f"Step 4: Harris response calculation (k={k}) - completed")
    
    # 步骤5: 非极大值抑制和阈值化
    # 局部最大值检测
    local_maxima = harris_response.copy()
    
    # 应用阈值
    threshold_value = threshold * harris_response.max()
    local_maxima[harris_response < threshold_value] = 0
    
    # 非极大值抑制
    corners = []
    h, w = local_maxima.shape
    
    for i in range(window_size, h - window_size):
        for j in range(window_size, w - window_size):
            if local_maxima[i, j] > 0:
                # 检查是否为局部最大值
                local_region = harris_response[i-window_size:i+window_size+1, 
                                             j-window_size:j+window_size+1]
                if harris_response[i, j] == local_region.max():
                    corners.append((j, i))  # (x, y) 格式
    
    print(f"Step 5: Non-maximum suppression and thresholding - completed")
    print(f"Found {len(corners)} corner points")
    
    return {
        'Ix': Ix,
        'Iy': Iy,
        'Ixx': Ixx,
        'Ixy': Ixy,
        'Iyy': Iyy,
        'Sxx': Sxx,
        'Sxy': Sxy,
        'Syy': Syy,
        'det_M': det_M,
        'trace_M': trace_M,
        'harris_response': harris_response,
        'corners': corners,
        'threshold_value': threshold_value
    }

# 执行手工实现
if img is not None:
    print("Starting manual Harris corner detection...")
    harris_results = manual_harris_corner_detection(img_gray)
    print("Manual implementation completed!\\n")

# 可视化Harris角点检测的各个步骤
if img is not None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 第一行：原始图像和梯度
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title(TITLES['original_gray'], fontsize=14)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(harris_results['Ix'], cmap='gray')
    axes[0,1].set_title(TITLES['gradient_x'], fontsize=14)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(harris_results['Iy'], cmap='gray')
    axes[0,2].set_title(TITLES['gradient_y'], fontsize=14)
    axes[0,2].axis('off')
    
    # 第二行：结构张量元素
    axes[1,0].imshow(harris_results['Sxx'], cmap='hot')
    axes[1,0].set_title('Structure Tensor Sxx', fontsize=14)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(harris_results['Sxy'], cmap='hot')
    axes[1,1].set_title('Structure Tensor Sxy', fontsize=14)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(harris_results['Syy'], cmap='hot')
    axes[1,2].set_title('Structure Tensor Syy', fontsize=14)
    axes[1,2].axis('off')
    
    # 第三行：Harris响应和结果
    axes[2,0].imshow(harris_results['det_M'], cmap='hot')
    axes[2,0].set_title('Determinant of M', fontsize=14)
    axes[2,0].axis('off')
    
    axes[2,1].imshow(harris_results['harris_response'], cmap='hot')
    axes[2,1].set_title(TITLES['harris_response'], fontsize=14)
    axes[2,1].axis('off')
    
    # 显示检测到的角点
    axes[2,2].imshow(img_gray, cmap='gray')
    for corner in harris_results['corners']:
        circle = Circle((corner[0], corner[1]), 3, color='red', fill=False, linewidth=2)
        axes[2,2].add_patch(circle)
    axes[2,2].set_title(f'{TITLES["harris_corners"]} ({len(harris_results["corners"])} points)', fontsize=14)
    axes[2,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 显示Harris响应函数的统计信息
    print("\\n=== Harris Response Statistics ===")
    print(f"Harris response range: [{harris_results['harris_response'].min():.6f}, {harris_results['harris_response'].max():.6f}]")
    print(f"Threshold value: {harris_results['threshold_value']:.6f}")
    print(f"Positive response pixels: {np.sum(harris_results['harris_response'] > 0)}")
    print(f"Negative response pixels: {np.sum(harris_results['harris_response'] < 0)}")
    print(f"Corner points detected: {len(harris_results['corners'])}")
