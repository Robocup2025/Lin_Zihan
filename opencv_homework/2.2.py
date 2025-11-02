import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(15,10)

#opencv库直接生成高斯滤波核
kernel_size=5 
sigma=1
kernel1=cv2.getGaussianKernel(kernel_size,sigma)
kernel2=np.outer(kernel1,kernel1.T)
plt.imshow(kernel2,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel2[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=20)
plt.colorbar()
plt.title('5x5 Gaussian Kernel (sigma=1)')
plt.show()

kernel_size=15
sigma=3
kernel1=cv2.getGaussianKernel(kernel_size,sigma)
kernel2=np.outer(kernel1,kernel1.T)
plt.imshow(kernel2,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel2[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=10)
plt.colorbar()
plt.title('15x15 Gaussian Kernel (sigma=3)')
plt.show()

kernel_size=31 
sigma=10
kernel1=cv2.getGaussianKernel(kernel_size,sigma)
kernel2=np.outer(kernel1,kernel1.T)
plt.imshow(kernel2,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel2[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=5)
plt.colorbar()
plt.title('31x31 Gaussian Kernel (sigma=10)')
plt.show()
