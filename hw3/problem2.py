from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import pandas as pd

path = "hw3_sample_images/"

def part_a():
    img = np.array(Image.open(path+"sample2.png").convert("L"))

    H1 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 36
    H2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) / 12
    H3 = np.array([[-1,2,-1],[-2,4,-2],[-1,2,-1]]) / 12
    H4 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) / 12
    H5 = np.array([[1,0,-1],[0,0,0],[-1,0,1]]) / 4
    H6 = np.array([[-1,2,-1],[0,0,0],[1,-2,1]]) / 4
    H7 = np.array([[-1,-2,-1],[2,4,2],[-1,-2,-1]]) / 12
    H8 = np.array([[-1,0,1],[2,0,-2],[-1,0,1]]) / 4
    H9 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]) / 4
    
    masks = np.array([H1,H2,H3,H4,H5,H6,H7,H8,H9])

    features = []

    for i, mask in enumerate(masks):
        # Step 1: Convolution
        M = signal.convolve2d(img, mask, mode='same')

        # Step 2: Energy Computation
        size = np.ones(shape=(13,13))
        T = signal.convolve2d(M * M, size, mode='same')

        T = T / np.max(T) * 255

        features.append(T)

        # result = Image.fromarray(T.astype(np.uint8))
        # result.save("mask_"+str(i+1)+".png")
    
    features = np.array(features)
    # print(features.reshape(-1,9).shape)

    return features

def part_b():
    features = part_a()
    features = np.moveaxis(features, 0, -1)

    # reference: https://gist.github.com/tvwerkhoven/4fdc9baad760240741a09292901d3abd

    X = features.reshape(-1,9)
    K = 5
    maxIters = 110
    C = []

    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Ensure we have K clusters, otherwise reset centroids and start over
        # If there are fewer than K clusters, outcome will be nan.
        if (len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]

    cluster = C.reshape(features.shape[0], features.shape[1])
    clustered_img = np.array(Image.new("RGB",(cluster.shape[1], cluster.shape[0])))

    for i, index in enumerate(np.unique(cluster)):
        pos = np.argwhere(cluster==index)
        clustered_img[pos[:,0], pos[:,1], :] = np.array((plt.cm.tab10(i)[0], plt.cm.tab10(i)[1], plt.cm.tab10(i)[2])) * 255

    result = Image.fromarray(clustered_img)
    # result.save("clustered_iters"+str(maxIters)+"_k"+str(K)+".png")
    result.save("result6.png")

def part_c():
    img = np.array(Image.open(path+"sample2.png").convert("L"))

    # # Power Law
    # power_img = np.zeros(shape=img.shape)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         power_img[i,j] = ((img[i,j]/255)**2)*255
    # img = power_img

    # Edge Crispening
    filtered_img = np.zeros(shape=img.shape)
    edgecrisp_img = np.zeros(shape=img.shape)

    b = 2
    mask = np.array(([1,b,1],[b,b**2,b],[1,b,1]))
    mask = mask / ((b+2)**2)

    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            window = img[x:x+3, y:y+3]
            value = np.sum(mask * window)

            filtered_img[x+1,y+1] = value

    c = 3/5
    edgecrisp_img = (c/(2*c-1)) * img - ((1-c)/(2*c-1)) * filtered_img
    img = edgecrisp_img

    # result = Image.fromarray(img).convert("L")
    # result.save("afteredge.png")

    H1 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 36
    H2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) / 12
    H3 = np.array([[-1,2,-1],[-2,4,-2],[-1,2,-1]]) / 12
    H4 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) / 12
    H5 = np.array([[1,0,-1],[0,0,0],[-1,0,1]]) / 4
    H6 = np.array([[-1,2,-1],[0,0,0],[1,-2,1]]) / 4
    H7 = np.array([[-1,-2,-1],[2,4,2],[-1,-2,-1]]) / 12
    H8 = np.array([[-1,0,1],[2,0,-2],[-1,0,1]]) / 4
    H9 = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]) / 4
    
    masks = np.array([H1,H2,H3,H4,H5,H6,H7,H8,H9])

    features = []

    for i, mask in enumerate(masks):
        # Step 1: Convolution
        M = signal.convolve2d(img, mask, mode='same')

        # Step 2: Energy Computation
        size = np.ones(shape=(13,13))
        T = signal.convolve2d(M * M, size, mode='same')

        T = T / np.max(T) * 255

        features.append(T)

        # result = Image.fromarray(T.astype(np.uint8))
        # result.save("edge_mask_"+str(i+1)+".png")
    
    features = np.array(features)
    features = np.moveaxis(features, 0, -1)

    X = features.reshape(-1,9)
    K = 5
    maxIters = 110
    C = []

    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Ensure we have K clusters, otherwise reset centroids and start over
        # If there are fewer than K clusters, outcome will be nan.
        if (len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]

    cluster = C.reshape(features.shape[0], features.shape[1])
    clustered_img = np.array(Image.new("RGB",(cluster.shape[1], cluster.shape[0])))

    for i, index in enumerate(np.unique(cluster)):
        pos = np.argwhere(cluster==index)
        clustered_img[pos[:,0], pos[:,1], :] = np.array((plt.cm.tab10(i)[0], plt.cm.tab10(i)[1], plt.cm.tab10(i)[2])) * 255

    result = Image.fromarray(clustered_img)
    # result.save("clustered_power_iters"+str(maxIters)+"_k"+str(K)+".png")
    result.save("result7.png")

part_a()
part_b()
part_c()