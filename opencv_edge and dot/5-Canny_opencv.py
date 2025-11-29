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

 # OpenCV实现
if img is not None:

    fig,axes = plt.subplots(1, 4, figsize=(18, 12))

    #1.高斯滤波
    img_blur =cv2.GaussianBlur(img_gray, (5,5), 1.5)
    
    #2.计算梯度
    Gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    direction = np.arctan2(Gy, Gx)            #方向

    axes[0].imshow(magnitude, cmap='gray')
    axes[0].set_title("Gradient Magnitude", fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(direction, cmap='hsv')
    axes[1].set_title("Gradient Direction", fontsize=14)
    axes[1].axis('off')

    #3.非极大抑制
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
    nms_result = non_maximum_suppression(magnitude,direction)

    axes[2].imshow(nms_result, cmap='gray')
    axes[2].set_title("nms_result", fontsize=14)
    axes[2].axis('off')

    #4.双边缘检测与边缘连接
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
    
    # 双阈值结果
    strong_edges, weak_edges = double_threshold(nms_result, low_threshold=20, high_threshold=80)
    final_edges = edge_tracking_by_hysteresis(strong_edges, weak_edges)
    
    axes[3].imshow(final_edges, cmap='gray')
    axes[3].set_title("final_edges", fontsize=14)
    axes[3].axis('off')

    opencv_canny = cv2.Canny(img_gray, 50, 150)

    plt.tight_layout()
    plt.show()
