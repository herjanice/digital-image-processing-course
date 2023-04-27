from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "hw4_sample_images/"

def part_a():
    img = np.array(Image.open(path+"sample2.png").convert("L"))

    h,w = img.shape

    # Down-sampling the image
    ds_img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

    # Reconstruct the image
    final = cv2.resize(ds_img, (h,w))
    
    # img_sample = img[300:350,300:350]
    # final_sample = final[300:350, 300:350]

    # result = Image.fromarray(cv2.resize(img_sample, (h,w)))
    # result.save("img_sample.png")
    # result = Image.fromarray(cv2.resize(final_sample, (h,w)))
    # result.save("final_sample.png")

    result = Image.fromarray(final)
    result.save("result5.png")


def part_b():
    img = np.array(Image.open(path+"sample3.png").convert("L"))

    # Converting it into frequency domain by Fourier Transform
    ft_img = np.fft.fft2(img)
    ft_img = np.fft.fftshift(ft_img)

    # Gaussian Low Pass
    h,w = img.shape
    D0 = 50
    D = np.fromfunction(lambda u,v: np.sqrt((u-h//2)**2 + (v-w//2)**2), (h,w))
    H = np.exp(-D**2/(2*D0**2))

    # Applying filter or frequency domain
    G = H * ft_img

    # Unsharp masking
    final = ft_img + 2 * (ft_img - G)

    # Inverse Fourier Transform
    ift_img = np.fft.ifftshift(final)
    ift_img = np.abs(np.fft.ifft2(ift_img))

    result = Image.fromarray(ift_img.astype(np.uint8))
    result.save("result6.png")


part_a()
part_b()