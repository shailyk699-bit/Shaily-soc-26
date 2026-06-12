import numpy as np
import matplotlib.pyplot as plt

# ====================== Common Kernels ======================
kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    
    "Edge_Laplacian": np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]),
    
    "Sobel_Horizontal": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),
    
    "Sobel_Vertical": np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]]),
    
    "Sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    
    "Gaussian_Blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]) / 16.0,
    
    "Emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]])
}

# ====================== YOUR IMPLEMENTATIONS ======================

def convolve2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')

    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    out_h = (i_h - k_h) // stride + 1
    out_w = (i_w - k_w) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = np.sum(region * kernel)

    return output


def max_pool2d(image, pool_size=2, stride=2):
    i_h, i_w = image.shape

    out_h = (i_h - pool_size) // stride + 1
    out_w = (i_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)

    return output


def avg_pool2d(image, pool_size=2, stride=2):
    i_h, i_w = image.shape

    out_h = (i_h - pool_size) // stride + 1
    out_w = (i_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)

    return output