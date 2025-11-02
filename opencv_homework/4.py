import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(15,10)

#只有图像读取和RGB处理调用了cv库
image=cv2.imread('scene.jpg')
image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_h,image_w,color_c=image_RGB.shape

#1.Copy Edge填充
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
image_copy=copy_edge_padding(image_RGB,image_h,image_w,color_c,int(image_h/4),int(image_w/4))
plt.imshow(image_copy)
plt.title('Copy edge')
plt.show()

#2.Reflect Across Edge 
def reflect_across_edge_padding(image,image_h,image_w,color_c,pad_h,pad_w):
    image_new=np.zeros((image_h+2*pad_h,image_w+2*pad_w,color_c),dtype=image.dtype)
    for i in range(image_h+2*pad_h):
        for j in range(image_w+2*pad_w):
            for c in range(color_c):
                if(pad_h<=i<pad_h+image_h and j<pad_w):
                    image_new[i,j,c]=image[i-pad_h,pad_w-j,c]
                elif(pad_h<=i<pad_h+image_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[i-pad_h,image_w-1-(j-image_w-pad_w+1),c]
                elif(i<pad_h and pad_w<=j<image_w+pad_w):
                    image_new[i,j,c]=image[pad_h-i,j-pad_w,c]
                elif(i>=pad_h+image_h and pad_w<=j<image_w+pad_w):
                    image_new[i,j,c]=image[image_h-1-(i-image_h-pad_h+1),j-pad_w,c]
                elif(i<pad_h and j<pad_w):
                    image_new[i,j,c]=image[pad_h-i,pad_w-j,c]
                elif(i>=pad_h+image_h and j<pad_w):
                    image_new[i,j,c]=image[image_h-1-(i-image_h-pad_h+1),pad_w-j,c]
                elif(i<pad_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[pad_h-i,image_w-1-(j-image_w-pad_w+1),c]
                elif(i>=pad_h+image_h and j>=pad_w+image_w):
                    image_new[i,j,c]=image[image_h-1-(i-image_h-pad_h+1),image_w-1-(j-image_w-pad_w+1),c]
                else:
                    image_new[i,j,c]=image[i-pad_h,j-pad_w,c]
    return image_new                
image_reflect=reflect_across_edge_padding(image_RGB,image_h,image_w,color_c,int(image_h/4),int(image_w/4))
plt.imshow(image_reflect)
plt.title('Reflect Across edge')
plt.show()
