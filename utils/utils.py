import cv2
import numpy as np
import random
from scipy.signal import convolve2d


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def syn(input1,input2):
    input1 = np.float32(input1) / 255.
    input2 = np.float32(input2) / 255.    
    
    sigma = random.uniform(2,5)
    R_blur = input2
    kernel = cv2.getGaussianKernel(11,sigma)
    kernel2d = np.dot(kernel,kernel.T)
    for i in range(3):
        R_blur[...,i] = convolve2d(R_blur[...,i],kernel2d,mode='same')
    M_ = input1 + R_blur
    if np.max(M_)>1:
        m = M_[M_>1]
        m = (np.mean(m)-1)*1.3
        R_blur = np.clip(R_blur-m,0,1)
        M_ = np.clip(R_blur+input1,0,1)
    
    return np.float32(input1),np.float32(R_blur),np.float32(M_)







