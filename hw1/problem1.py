from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt

np.seterr(invalid='ignore')

path = "SampleImage/"

def part_a():
    img = Image.open(path+"sample2.png")

    img_array = np.array(img)

    # Dividing all values in the array by 2
    final = img_array / 2
    final = final.astype(np.uint8) # to prevent float type

    result = Image.fromarray(final)
    result.save("result3.png")
    # result.show()

def part_b():
    img = Image.open("result3.png")

    img_array = np.array(img)

    # If intensity value exceed 255 when multiplied by 3, intensity value = 255 (max uint8 value) 
    final = np.where(img_array < 85, img_array*3, 255)
    
    result = Image.fromarray(final)
    result.save("result4.png")
    # result.show()

def part_c():
    img1 = np.array(Image.open(path+"sample2.png"))
    img2 = np.array(Image.open("result3.png"))
    img3 = np.array(Image.open("result4.png"))

    img1_flatten = img1.flatten()
    img2_flatten = img2.flatten()
    img3_flatten = img3.flatten()

    # plt.hist(img1_flatten, bins=range(256), range=[0,256])
    # plt.title("sample2.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_sample2.png')

    # plt.hist(img2_flatten, bins=range(256), range=[0,256])
    # plt.title("result3.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result3.png')

    # plt.hist(img3_flatten, bins=range(256), range=[0,256])
    # plt.title("result4.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result4.png')

def part_d():
    img1 = np.array(Image.open("result3.png"))
    img2 = np.array(Image.open("result4.png"))

    img1_histogram = np.bincount(img1.flatten(), minlength=256)
    num_pixels = np.sum(img1_histogram)
    img1_histogram = img1_histogram/num_pixels # normalize

    img1_cdf = np.cumsum(img1_histogram)

    img1_transform = np.floor(255*img1_cdf).astype(np.uint8)

    img1_list = list(img1.flatten())
    img1_final = [img1_transform[i] for i in img1_list]
    img1_result = np.reshape(np.asarray(img1_final), img1.shape)

    # plt.hist(img1_final, bins=range(256), range=[0,256])
    # plt.title("result5.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result5.png')

    result1 = Image.fromarray(img1_result)
    result1.save("result5.png")
    # result1.show()

    #================================================================================
    img2_histogram = np.bincount(img2.flatten(), minlength=256)
    num_pixels = np.sum(img2_histogram)
    img2_histogram = img2_histogram/num_pixels # normalize

    img2_cdf = np.cumsum(img2_histogram) 

    img2_transform = np.floor(255*img2_cdf).astype(np.uint8)

    img2_list = list(img2.flatten())
    img2_final = [img2_transform[i] for i in img2_list]
    img2_result = np.reshape(np.asarray(img2_final), img2.shape)

    # plt.hist(img2_final, bins=range(256), range=[0,256])
    # plt.title("result6.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result6.png')

    result2 = Image.fromarray(img2_result)
    result2.save("result6.png")
    # result2.show()

def part_e():
    img1 = np.array(Image.open("result3.png"))
    img2 = np.array(Image.open("result4.png"))

    h,w = img1.shape

    img1_final = np.zeros((h,w), dtype = np.uint8)
    parts = []

    cycle = 0
    for i in range(h-200):
        for j in range(w-200):
            x1 = i
            x2 = i+200
            y1 = j
            y2 = j+200

            local = img1[x1:x2,y1:y2]
            parts.append(local)

            img_histogram = np.bincount(img1[x1:x2,y1:y2].flatten(), minlength=256)
            num_pixels = np.sum(img_histogram)
            img_histogram = img_histogram/num_pixels # normalize

            img_cdf = np.cumsum(img_histogram)

            img_transform = np.floor(255*img_cdf).astype(np.uint8)

            img_list = list(img1[x1:x2,y1:y2].flatten())
            img_final = [img_transform[i] for i in img_list]
            img_result = np.reshape(np.asarray(img_final), parts[cycle].shape)

            parts[cycle] = img_result
            
            img1_final[x1:x2,y1:y2] = parts[cycle]

            cycle += 1
    
    result1 = Image.fromarray(img1_final)
    result1.save("result7.png")

    # plt.hist(img1_final.flatten(), bins=range(256), range=[0,256])
    # plt.title("result7.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result7.png')

    #================================================================================

    img2_final = np.zeros((h,w), dtype = np.uint8)
    parts = []

    cycle = 0
    for i in range(h-200):
        for j in range(w-200):
            x1 = i
            x2 = i+200
            y1 = j
            y2 = j+200

            local = img2[x1:x2,y1:y2]
            parts.append(local)

            img_histogram = np.bincount(img2[x1:x2,y1:y2].flatten(), minlength=256)
            num_pixels = np.sum(img_histogram)
            img_histogram = img_histogram/num_pixels # normalize

            img_cdf = np.cumsum(img_histogram)

            img_transform = np.floor(255*img_cdf).astype(np.uint8)

            img_list = list(img2[x1:x2,y1:y2].flatten())
            img_final = [img_transform[i] for i in img_list]
            img_result = np.reshape(np.asarray(img_final), parts[cycle].shape)

            parts[cycle] = img_result
            
            img2_final[x1:x2,y1:y2] = parts[cycle]

            cycle += 1
    
    result2 = Image.fromarray(img2_final)
    result2.save("result8.png")

    # plt.hist(img2_final.flatten(), bins=range(256), range=[0,256])
    # plt.title("result8.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result8.png')

def part_f():
    # power law transform
    img = np.array(Image.open(path+"sample2.png"))

    final =255*(img/255)**(1/2)
    final = final.astype(np.uint8)

    result = Image.fromarray(final)
    result.save("result9.png")

    # plt.hist(final.flatten(), bins=range(256), range=[0,256])
    # plt.title("result9.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result9.png')





part_a()
part_b()
part_c()
part_d()
part_e()
part_f()

