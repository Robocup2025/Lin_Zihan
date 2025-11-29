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
    
    print(f"Original image shape: {img_rgb.shape}")
    print(f"Grayscale image shape: {img_gray.shape}")
    print(f"Pixel value range: {img_gray.min()} - {img_gray.max()}")
    
# 手工实现Canny边缘检测的各个步骤
def manual_canny_edge_detection(image, sigma=1.0, low_threshold=50, high_threshold=150):
    """
    手工实现Canny边缘检测算法
    """
    print("=== Manual Canny Edge Detection Steps ===")
    
    # 步骤1: 高斯滤波降噪
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)
    print(f"Step 1: Gaussian blur with σ={sigma} - completed")
    
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
    
    print("Step 2: Gradient calculation - completed")
    
    # 步骤3: 非极大值抑制
    nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)
    print("Step 3: Non-maximum suppression - completed")
    
    # 步骤4: 双阈值检测
    strong_edges, weak_edges = double_threshold(nms_result, low_threshold, high_threshold)
    print(f"Step 4: Double threshold (low={low_threshold}, high={high_threshold}) - completed")
    
    # 步骤5: 边缘连接
    final_edges = edge_tracking_by_hysteresis(strong_edges, weak_edges)
    print("Step 5: Edge tracking by hysteresis - completed")
    
    return {
        'blurred': blurred,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'gradient_magnitude': gradient_magnitude,
        'gradient_direction': gradient_direction,
        'nms_result': nms_result,
        'strong_edges': strong_edges,
        'weak_edges': weak_edges,
        'final_edges': final_edges
    }

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """非极大值抑制"""
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    # 将角度转换为0-180度
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            try:
                q = 255
                r = 255
                
                # 角度0度
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # 角度45度
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # 角度90度
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # 角度135度
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]
                
                if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                    suppressed[i,j] = gradient_magnitude[i,j]
                else:
                    suppressed[i,j] = 0
                    
            except IndexError:
                pass
                
    return suppressed

def double_threshold(image, low_threshold, high_threshold):
    """双阈值检测"""
    strong_edges = np.zeros_like(image)
    weak_edges = np.zeros_like(image)
    
    strong_edges[image >= high_threshold] = 255
    weak_edges[(image >= low_threshold) & (image < high_threshold)] = 75
    
    return strong_edges, weak_edges

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    """边缘连接"""
    final_edges = strong_edges.copy()
    
    # 8-连通性检查
    for i in range(1, strong_edges.shape[0]-1):
        for j in range(1, strong_edges.shape[1]-1):
            if weak_edges[i,j] == 75:
                # 检查8邻域是否有强边缘
                if np.any(strong_edges[i-1:i+2, j-1:j+2] == 255):
                    final_edges[i,j] = 255
    
    return final_edges

# 执行手工实现
if img is not None:
    print("Starting manual Canny edge detection...")
    canny_steps = manual_canny_edge_detection(img_gray)
    print("Manual implementation completed!\\n")

# 可视化Canny边缘检测的各个步骤
if img is not None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 第一行：原始图像和预处理
    axes[0,0].imshow(img_gray, cmap='gray')
    axes[0,0].set_title(TITLES['original_gray'], fontsize=14)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(canny_steps['blurred'], cmap='gray')
    axes[0,1].set_title(TITLES['gaussian_blur'], fontsize=14)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(canny_steps['gradient_magnitude'], cmap='gray')
    axes[0,2].set_title(TITLES['gradient_magnitude'], fontsize=14)
    axes[0,2].axis('off')
    
    # 第二行：梯度分量
    axes[1,0].imshow(canny_steps['grad_x'], cmap='gray')
    axes[1,0].set_title(TITLES['gradient_x'], fontsize=14)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(canny_steps['grad_y'], cmap='gray')
    axes[1,1].set_title(TITLES['gradient_y'], fontsize=14)
    axes[1,1].axis('off')
    
    # 梯度方向可视化
    gradient_direction_deg = canny_steps['gradient_direction'] * 180 / np.pi
    axes[1,2].imshow(gradient_direction_deg, cmap='hsv')
    axes[1,2].set_title(TITLES['gradient_direction'], fontsize=14)
    axes[1,2].axis('off')
    
    # 第三行：边缘检测步骤
    axes[2,0].imshow(canny_steps['nms_result'], cmap='gray')
    axes[2,0].set_title(TITLES['non_max_suppression'], fontsize=14)
    axes[2,0].axis('off')
    
    # 双阈值结果（合并显示）
    double_threshold_result = canny_steps['strong_edges'] + canny_steps['weak_edges']
    axes[2,1].imshow(double_threshold_result, cmap='gray')
    axes[2,1].set_title(TITLES['double_threshold'], fontsize=14)
    axes[2,1].axis('off')
    
    axes[2,2].imshow(canny_steps['final_edges'], cmap='gray')
    axes[2,2].set_title(TITLES['canny_result'], fontsize=14)
    axes[2,2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 显示统计信息
    print("\\n=== Canny Edge Detection Statistics ===")
    print(f"Original image pixels: {img_gray.size}")
    print(f"Strong edge pixels: {np.sum(canny_steps['strong_edges'] == 255)}")
    print(f"Weak edge pixels: {np.sum(canny_steps['weak_edges'] == 75)}")
    print(f"Final edge pixels: {np.sum(canny_steps['final_edges'] == 255)}")
    print(f"Edge density: {np.sum(canny_steps['final_edges'] == 255) / img_gray.size * 100:.2f}%")