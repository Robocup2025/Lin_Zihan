import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(15,10)

#手动实现高斯滤波核
def gaussian_core(size,sigma):
    kernel=np.zeros((size,size))
    generate=lambda x,y:(1/(2*np.pi*sigma**2))*np.exp(-((x-(size-1)/2)**2+(y-(size-1)/2)**2)/(2*sigma**2))
    for i in range(0,size):
        for j in range(0,size):
            kernel[i][j]=generate(i,j)
    return kernel/np.sum(kernel)

kernel_size=5
sigma=1
kernel=gaussian_core(kernel_size,sigma)
plt.imshow(kernel,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=20)
plt.colorbar()
plt.title('5x5 Gaussian Kernel (sigma=1)')
plt.show()

kernel_size=15
sigma=3
kernel=gaussian_core(kernel_size,sigma)
plt.imshow(kernel,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=10)
plt.colorbar()
plt.title('15x15 Gaussian Kernel (sigma=3)')
plt.show()

kernel_size=31
sigma=10
kernel=gaussian_core(kernel_size,sigma)
plt.imshow(kernel,cmap='viridis',interpolation='nearest')
for i in range(kernel_size):
    for j in range(kernel_size):
        text_color='white'
        plt.text(j,i,f'{kernel[i,j]:.3f}',ha='center',va='center',color=text_color,fontsize=5)
plt.colorbar()
plt.title('31x31 Gaussian Kernel (sigma=10)')
plt.show()