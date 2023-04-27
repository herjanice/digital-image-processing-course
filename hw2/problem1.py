from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

path = "imgs_2022/"

def part_a():
    img = np.array(Image.open(path+"sample1.png"))
    final = np.zeros(shape=img.shape)
    sobel_filtered_x = np.zeros(shape=img.shape)
    sobel_filtered_y = np.zeros(shape=img.shape)

    mask_x = np.array([[1,2,1],[0,0,0,],[-1,-2,-1]])
    mask_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
    # for x in range(1):
    #     for y in range(1):
            window = img[x:x+3, y:y+3]
            Gx = np.sum(mask_x * window)
            Gy = np.sum(mask_y * window)

            sobel_filtered_x[x+1, y+1] = Gx
            sobel_filtered_y[x+1, y+1] = Gy

            #Calculating the magnitude
            magnitude = np.sqrt(Gx**2 + Gy**2)
            final[x+1, y+1] = magnitude if magnitude > 150 else 0

    # plt.hist(final.flatten(), cumulative=True, bins=range(256), range=[0,256])
    # plt.show()
    # plt.savefig("sobel_cdf.png")

    # plt.figure(figsize=(6.25, 8.33))
    plt.imshow(final, cmap='gray', aspect='auto')
    plt.axis('off')
    # plt.show()
    plt.savefig("result1.png", bbox_inches='tight', pad_inches=0)

    result = Image.fromarray(final).convert("L")
    # result.show()
    result.save("result2.png")

def part_b():
    img = np.array(Image.open(path+"sample1.png"))
    final = np.zeros(shape=img.shape)

    # step 1 : Noise Reduction (Gaussian Filtering)
    # filter = np.array([[1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
    # filter = np.divide(filter, 256)
    filter = np.array([[1,4,7,4,1], [4,16,26,16,4], [7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]])
    filter = np.divide(filter, 273)
    blur_img = np.zeros(shape=img.shape)

    for step in range(1):
        for x in range(img.shape[0]-4):
            for y in range(img.shape[1]-4):
                window = img[x:x+5, y:y+5]
                value = np.sum(filter * window)
                blur_img[x+2,y+2] = np.floor(value)
        img = blur_img

    # step 2 : Compute gradient magnitude and orientation
    mask_x = np.array([[1,2,1],[0,0,0,],[-1,-2,-1]])
    mask_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    sobel_filtered_x = np.zeros(shape=img.shape)
    sobel_filtered_y = np.zeros(shape=img.shape)
    sobel_filtered_img = np.zeros(shape=img.shape)
    orientation = np.zeros(shape=img.shape)

    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            window = img[x:x+3, y:y+3]
            Gx = np.sum(mask_x * window)
            Gy = np.sum(mask_y * window)


            sobel_filtered_x[x+1, y+1] = Gx
            sobel_filtered_y[x+1, y+1] = Gy

            magnitude = np.sqrt(Gx**2 + Gy**2)
            sobel_filtered_img[x+1, y+1] = magnitude

    orientation = np.arctan2(sobel_filtered_x, sobel_filtered_y) * 180/np.pi

    # Step 3 : Non-maximal Suppression
    suppressed_img = np.zeros(shape=img.shape)
    one = 255
    two = 255
    for x in range(1,img.shape[0]-1):
        for y in range(1,img.shape[1]-1):

            # angle 0
            if (0 <= orientation[x,y] < 22.5) or (157.5 <= orientation[x,y] <= 180):
                one = sobel_filtered_img[x, y+1]
                two = sobel_filtered_img[x, y-1]
            # angle 45
            elif (22.5 <= orientation[x,y] < 67.5):
                one = sobel_filtered_img[x+1, y-1]
                two = sobel_filtered_img[x-1, y+1]
            # angle 90
            elif (67.5 <= orientation[x,y] < 112.5):
                one = sobel_filtered_img[x+1, y]
                two = sobel_filtered_img[x-1, y]
            # angle 135
            elif (112.5 <= orientation[x,y] < 157.5):
                one = sobel_filtered_img[x-1, y-1]
                two = sobel_filtered_img[x+1, y+1]
            
            if (sobel_filtered_img[x,y] >= one) and (sobel_filtered_img[x,y] >= two):
                suppressed_img[x,y] = sobel_filtered_img[x,y]
            else:
                suppressed_img[x,y] = 0
    
    # Step 4 : Hysteretic Thresholding

    # 0 = Edge pixel, 25 = Candidate pixel, 255 = Non-edge pixel
    hysteretic_img = np.zeros(shape=img.shape) # for labelling

    # low_threshold_ratio = 0.03
    # high_threshold_ratio = 0.08

    low_threshold_ratio = 0.05
    high_threshold_ratio = 0.09

    T_h = suppressed_img.max() * high_threshold_ratio
    T_l = T_h * low_threshold_ratio

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if suppressed_img[x,y] >= T_h:
                hysteretic_img[x,y] = 255
            elif  (suppressed_img[x,y] >= T_l) and (suppressed_img[x,y] <= T_h):
                hysteretic_img[x,y] = 25
            elif  suppressed_img[x,y] < T_l:
                hysteretic_img[x,y] = 0

    # Step 5 : Connected component labeling method

    connected_img = np.zeros(shape=img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if hysteretic_img[x,y] == 25:
                if (hysteretic_img[x+1, y-1] == 255) or (hysteretic_img[x+1, y] == 255) or  (hysteretic_img[x+1, y+1] == 255) or (hysteretic_img[x, y-1] == 255) or (hysteretic_img[x,y+1] == 255) or (hysteretic_img[x-1,y-1] == 255) or (hysteretic_img[x-1,y] == 255) or (hysteretic_img[x-1,y+1] == 255):
                    connected_img[x,y] = 255
                else:
                    connected_img[x,y] = 0
            else:
                connected_img[x,y] = hysteretic_img[x,y]

    final = connected_img

    # result = Image.fromarray(blur_img).convert('L')
    # result.save("canny_blur.png")

    # result = Image.fromarray(sobel_filtered_img).convert('L')
    # result.save("canny_sobel.png")

    # result = Image.fromarray(suppressed_img).convert('L')
    # result.show()
    # result.save("canny_suppression.png")

    # result = Image.fromarray(hysteretic_img).convert('L')
    # result.show()
    # result.save("canny_hysteretic.png")

    # result = Image.fromarray(connected_img).convert('L')
    # result.save("canny_connected.png")

    result = Image.fromarray(final).convert('L')
    result.save("result3.png")

def part_c():
    img = np.array(Image.open(path+"sample1.png"))
    final = np.zeros(shape=img.shape)

    gaussian_filter = np.array([[1,4,7,4,1], [4,16,26,16,4], [7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]])
    gaussian_filter = np.divide(gaussian_filter, 273)

    blur_img = np.zeros(shape=img.shape)

    for step in range(1):
        for x in range(img.shape[0]-4):
            for y in range(img.shape[1]-4):
                window = img[x:x+5, y:y+5]
                value = np.sum(gaussian_filter * window)
                blur_img[x+2,y+2] = np.floor(value)
        img = blur_img

    log_img = np.zeros(shape=img.shape)

    # mask = np.array([[0,-1,0],[-1,4,-1,],[0,-1,0]]) # four-neighbour
    # mask = np.divide(mask, 4)

    mask = np.array([[-1,-1,-1],[-1,8,-1,],[-1,-1,-1]]) # eight-neighbour, non-seperable
    mask = np.divide(mask, 8)

    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            window = blur_img[x:x+3, y:y+3]
            value = np.sum(mask * window)

            log_img[x+1, y+1] = value
    
    thresh_img = np.zeros(shape=img.shape)
    T = 5

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if np.abs(log_img[x,y]) <= T:
                thresh_img[x,y] = 0
            else:
                thresh_img[x,y] = log_img[x,y]
    thresh_img = np.sign(thresh_img) # returns -1 if value is negative, returns +1 if value positive


    edge_img = np.zeros(shape=img.shape)
    for x in range(1,img.shape[0]-1):
        for y in range(1,img.shape[1]-1):
            if thresh_img[x,y] == 0:
                window = thresh_img[x-1:x+2, y-1:y+2]
                cross = np.unique(window)

                if (-1 in cross) and (1 in cross):
                    edge_img[x,y] = 255
                else:
                    edge_img[x,y] = 0
            else:
                edge_img[x,y] = 0


    # cnt, intensity = np.histogram(log_img)
    # plt.plot(intensity[:-1], cnt)
    # plt.title("Histogram of Laplacian")
    # plt.show()
    # plt.savefig("laplacian_histogram.png")

    # plt.imshow(final, cmap='gray')
    # plt.show()
    # plt.savefig("laplacian_gradient_map_8.png")

    result = Image.fromarray(edge_img).convert("L")
    # result.show()
    result.save("result4.png")

def part_d():
    img = np.array(Image.open(path+"sample2.png"))
    filtered_img = np.zeros(shape=img.shape)
    final = np.zeros(shape=img.shape)

    b = 2
    mask = np.array(([1,b,1],[b,b**2,b],[1,b,1]))
    mask = mask / ((b+2)**2)

    for x in range(img.shape[0]-2):
        for y in range(img.shape[1]-2):
            window = img[x:x+3, y:y+3]
            value = np.sum(mask * window)

            filtered_img[x+1,y+1] = value

    c = 3/5
    final = (c/(2*c-1)) * img - ((1-c)/(2*c-1)) * filtered_img

    result = Image.fromarray(final).convert("L")
    result.save("result5.png")

    # 2 = 20/30
    # 3 = 23/30

# def bonus():

    
            



part_a()
part_b()
part_c()
part_d()