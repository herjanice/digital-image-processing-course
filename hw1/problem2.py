from pickletools import read_unicodestringnl
from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math

path = "SampleImage/"

def part_a():
    # low-pass filtering for uniform noise
    b = 2
    mask = np.array(([1,b,1],[b,b**2,b],[1,b,1]))
    mask = mask / ((b+2)**2)

    img1 = np.array(Image.open(path+"sample4.png"))
    h,w = img1.shape

    img1_final = np.zeros((h, w), dtype=np.uint8)

    for round in range(5):
        for i in range(h):
            for j in range(w):
                pixel_sum = 0
                mask_sum = 0 
                for x in range(3):
                    for y in range(3):
                        if (i+x-1)>=0 and (i+x-1)<h and (j+y-1)>=0 and (j+y-1)<w:
                            pixel_sum += img1[i+x-1][j+y-1] * mask[x][y]
                            mask_sum += mask[x][y]
                img1_final[i][j] = np.round(pixel_sum/mask_sum)

        img1 = img1_final

        # result1 = Image.fromarray(img1_final)
        # result1.save(path+"result10_"+str(round+1)+".png")

        # plt.hist(img1_final.flatten(), bins=range(256), range=[0,256])
        # plt.title("result10_"+str(round+1)+".png Histogram")
        # plt.xlabel("Intensity")
        # plt.ylabel("Number of pixels")    
        # # plt.show()
        # plt.savefig('histogram_result10_'+str(round+1)+'.png')

    result1 = Image.fromarray(img1_final)
    result1.save("result10.png")

    # plt.hist(img1_final.flatten(), bins=range(256), range=[0,256])
    # plt.title("result10.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result10.png')

    #================================================================================

    # outlier detection for impulse noise
    img2 = np.array(Image.open(path+"sample5.png"))
    h,w = img2.shape

    img2_final = np.zeros((h,w),dtype=np.uint8)

    epsilon = 60

    for round in range(5):
        for i in range(h):
            for j in range(w):
                pixel_sum = 0
                amt = 0
                for x in range(3):
                    for y in range(3):
                        if (i+x-1)>=0 and (i+x-1)<h and (j+y-1)>=0 and (j+y-1)<w:
                            if x!=1 or y!=1:
                                pixel_sum += img2[i+x-1][j+y-1]
                                amt += 1

                avg = np.round(pixel_sum/amt)
                gap = int(img2[i][j]) - int(avg)

                if gap > epsilon or gap < -1*epsilon:
                    img2_final[i][j] = avg
                else:
                    img2_final[i][j]  = img2[i][j]

        img2 = img2_final
        epsilon -= 10

        # result2 = Image.fromarray(img2_final)
        # result2.save(path+"result11_"+str(round+1)+".png")

        # plt.hist(img2_final.flatten(), bins=range(256), range=[0,256])
        # plt.title("result11_"+str(round+1)+".png Histogram")
        # plt.xlabel("Intensity")
        # plt.ylabel("Number of pixels")    
        # # plt.show()
        # plt.savefig('histogram_result11_'+str(round+1)+'.png')

    result2 = Image.fromarray(img2_final)
    result2.save("result11.png")

    # plt.hist(img2_final.flatten(), bins=range(256), range=[0,256])
    # plt.title("result11.png Histogram")
    # plt.xlabel("Intensity")
    # plt.ylabel("Number of pixels")
    # # plt.show()
    # plt.savefig('histogram_result11.png')

def part_b():
    img1 = np.array(Image.open(path+"sample3.png"))
    img2 = np.array(Image.open("result10.png"))

    h,w = img1.shape

    MSE = 0
    for i in range(h):
        for j in range(w):
            MSE += (int(img1[i][j])-int(img2[i][j]))**2

    MSE /= h*w

    PSNR = 10 * (math.log((255**2/MSE),10))
    print("PSNR of sample4.png and result10.png:")
    print(PSNR)

    '''
    PSNR values:
    18.469622616299684
    18.483988826826085
    18.420922917838702
    18.35009504572367
    18.28340486523969
    '''
    #================================================================================

    img3 = np.array(Image.open(path+"sample3.png"))
    img4 = np.array(Image.open("result11.png"))

    h,w = img3.shape

    MSE = 0
    for i in range(h):
        for j in range(w):
            MSE += (int(img3[i][j])-int(img4[i][j]))**2

    MSE /= int(h*w)

    PSNR = 10 * (math.log((255**2/MSE),10))
    print("PSNR of sample5.png and result11.png:")
    print(PSNR)

    '''
    PSNR values:
    24.66317486266037
    26.524057113450183
    27.76462542551307
    28.543822146914305
    29.084904889197208
    '''

                                                     
part_a()
part_b()