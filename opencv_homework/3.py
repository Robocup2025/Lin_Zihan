import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(15,10)

#只有图像读取和RGB处理调用了cv库
image=cv2.imread('Einstein.jpg')
image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_h,image_w,color_c=image_RGB.shape

#1.生成高斯滤波核
def gaussian_core(size,sigma):
    kernel=np.zeros((size,size))
    generate=lambda x,y:(1/(2*np.pi*sigma**2))*np.exp(-((x-(size-1)/2)**2+(y-(size-1)/2)**2)/(2*sigma**2))
    for i in range(0,size):
        for j in range(0,size):
            kernel[i][j]=generate(i,j)
    return kernel/np.sum(kernel)

#2.copy edge扩充图像
def copy_edge_padding(image,image_h,image_w,color_c,pad_h,pad_w):
    image_new=np.zeros((image_h+2*pad_h,image_w+2*pad_w,color_c),dtype=image.dtype)
    for i in range(image_h+2*pad_h):
        for j in range(image_w+2*pad_w):
            for c in range(color_c):
                if(pad_h<=i<pad_h+image_h and j<pad_w):
                    image_new[i,j,c]=image[i-pad_h,0,c]
                elif(pad_h<=i<pad_h+image_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[i-pad_h,image_w-1,c]
                elif(i<pad_h and pad_w<=j<image_w+pad_w):
                    image_new[i,j,c]=image[0,j-pad_w,c]
                elif(i>=pad_h+image_h and pad_w<=j<image_w+pad_w):
                    image_new[i,j,c]=image[image_h-1,j-pad_w,c]
                elif(i<pad_h and j<pad_w):
                    image_new[i,j,c]=image[0,0,c]
                elif(i>=pad_h+image_h and j<pad_w):
                    image_new[i,j,c]=image[image_h-1,0,c]
                elif(i<pad_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[0,image_w-1,c]
                elif(i>=pad_h+image_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[image_h-1,image_w-1,c]
                else:
                    image_new[i,j,c]=image[i-pad_h,j-pad_w,c]
    return image_new

#3.高斯滤波
def convolve(image,kernel):
    image_h,image_w,color_c=image.shape
    kernel_h,kernel_w=kernel.shape
    edge_h=int(kernel_h/2)
    edge_w=int(kernel_w/2)
    image_new=image_new=np.zeros((image_h-2*edge_h,image_w-2*edge_w,color_c),dtype=image.dtype)
    for i in range(edge_h,image_h-edge_h):
        for j in range(edge_w,image_w-edge_w):
            for c in range(color_c):
                region=image[i-edge_h:i+edge_h+1,j-edge_w:j+edge_w+1,c]
                image_new[i-edge_h,j-edge_w,c]=np.sum(region*kernel)
    return image_new

#4.滤波操作与原图比较
plt.imshow(image)
plt.title('Original Image')
plt.show()

kernel_size=5
sigma=1
kernel=gaussian_core(kernel_size,sigma)
image_copy=copy_edge_padding(image_RGB,image_h,image_w,color_c,int(kernel_size/2),int(kernel_size/2))
image_gaussian=convolve(image_copy,kernel)
plt.imshow(image_gaussian)
plt.title('Gaussian1(5x5,sigma=1) Processing Image')
plt.show()

kernel_size=15
sigma=3
kernel=gaussian_core(kernel_size,sigma)
image_copy=copy_edge_padding(image_RGB,image_h,image_w,color_c,int(kernel_size/2),int(kernel_size/2))
image_gaussian=convolve(image_copy,kernel)
plt.imshow(image_gaussian)
plt.title('Gaussian2(15x15,sigma=3) Processing Image')
plt.show()

kernel_size=31
sigma=10
kernel=gaussian_core(kernel_size,sigma)
image_copy=copy_edge_padding(image_RGB,image_h,image_w,color_c,int(kernel_size/2),int(kernel_size/2))
image_gaussian=convolve(image_copy,kernel)
plt.imshow(image_gaussian)
plt.title('Gaussian3(31x31,sigma=10) Processing Image')
plt.show()
