from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

path = "imgs_2022/"

def rotate(angle):
    angle = np.radians(angle)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return np.round(R)

def translate(tx, ty):
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    return T

def scale(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S

def part_a():
    img = np.array(Image.open(path+"sample3.png").convert("L"))
    h,w = img.shape

    final = np.zeros(shape=img.shape)

    # decrease brightness of the lower part of the image
    for round in range(6,14):
        for i in range(int(np.round(round*h/24)), h):
            for j in range(w):
                if img[i,j]!=0:
                    img[i,j] = np.floor(img[i,j]/1.2) if img[i,j]/1.2 >0 else 0
    
    # result = Image.fromarray(img)
    # result.show()
    # result.save("after_decrease_brightness.png")

    # increase brightness of the image
    for i in range(h):
        for j in range(w):
            if img[i,j]!=0:
                img[i,j] = np.floor(img[i,j]*2.5) if img[i,j]*2.5 <= 255 else 255

    # result = Image.fromarray(img)
    # result.show()
    # result.save("after_increase_brightness.png")

    # img = final
    filtered_img = np.zeros(shape=img.shape)

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

    # result = Image.fromarray(final).convert("L")
    # result.show()
    # result.save("after_edgecrispening.png")

    for i in range(h):
        for j in range(w):
            if final[i,j] >= 220:
                final[i,j] = 0
    
    # result = Image.fromarray(final).convert("L")
    # result.show()
    # result.save("after_deleting_whites.png")

    img = final
    epsilon = 80

    for round in range(3):
        for i in range(h):
            for j in range(w):
                pixel_sum = 0
                amt = 0
                for x in range(3):
                    for y in range(3):
                        if (i+x-1)>=0 and (i+x-1)<h and (j+y-1)>=0 and (j+y-1)<w:
                            if x!=1 or y!=1:
                                pixel_sum += img[i+x-1][j+y-1]
                                amt += 1

                avg = np.round(pixel_sum/amt)
                gap = int(img[i][j]) - int(avg)

                if gap > epsilon or gap < -1*epsilon:
                    final[i][j] = avg
                else:
                    final[i][j]  = img[i][j]
        img = final
        epsilon-=10


    # result = Image.fromarray(final).convert("L")
    # result.show()
    # result.save("after_smoothing.png")

def part_b():
    img = np.array(Image.open(path+"sample3.png").convert("L"))
    final = np.zeros(shape=img.shape)

    size = img.shape[0]

    angle = 90
    for steps in range(4):
        rotated_img = np.zeros(shape=img.shape)
        R = rotate(angle * steps)

        centre = (img.shape[0]//2, img.shape[1]//2)
        for x in range(size):
            for y in range(size):
                coordinates = np.array([x-centre[0],y-centre[1],1])
                x_pos,y_pos,_ = R @ coordinates # Matrix multiply

                x_pos = round(x_pos) + centre[0]
                y_pos = round(y_pos) + centre[1]

                if x_pos >=0 and x_pos < size and y_pos >=0 and y_pos < size:
                    rotated_img[x_pos,y_pos] = img[x,y]
        
        # result = Image.fromarray(rotated_img).convert("L")
        # result.show()
        # result.save("after_rotating"+str(steps)+".png")

        scale_factor = 0.4
        scaled_img = np.zeros(shape=img.shape)
        S = scale(scale_factor,scale_factor)
        for x in range(size):
            for y in range(size):
                coordinates = np.array([x,y,1])
                x_pos,y_pos,_ = S @ coordinates

                x_pos = int(abs(x_pos))
                y_pos = int(abs(y_pos))

                if x_pos >= 0 and x_pos < size and y_pos >=0 and y_pos < size:
                    scaled_img[x_pos, y_pos] = rotated_img[x,y]
        
        # result = Image.fromarray(scaled_img).convert("L")
        # result.show()
        # result.save("after_scaling"+str(steps)+".png")
        
        x_tr = 0
        y_tr = 0
        if steps == 0:
            x_tr = (img.shape[0]/2) - (scaled_img.shape[0]*scale_factor)
            y_tr = (img.shape[1]/2) - (scaled_img.shape[1]*scale_factor/2)
        elif steps == 1:
            x_tr = (img.shape[0]/2) - (scaled_img.shape[0]*scale_factor/2)
            y_tr = (img.shape[1]/2) - (scaled_img.shape[1]*scale_factor)
        elif steps == 2:
            x_tr = img.shape[0]/2
            y_tr = (img.shape[1]/2) - (scaled_img.shape[1]*scale_factor/2)
        elif steps == 3:
            x_tr = (img.shape[0]/2) - (scaled_img.shape[0]*scale_factor/2)
            y_tr = img.shape[1]/2

        T = translate(x_tr,y_tr)
        for x in range(size):
            for y in range(size):
                coordinates = np.array([x,y,1])
                x_pos,y_pos,_ = T @ coordinates

                x_pos = int(abs(x_pos))
                y_pos = int(abs(y_pos))

                if x_pos >= 0 and x_pos < size and y_pos >=0 and y_pos < size and final[x_pos,y_pos]==0:
                    final[x_pos, y_pos] = scaled_img[x,y]

    result = Image.fromarray(final).convert("L")
    # result.show()
    # result.save("after_translate"+str(steps)+".png")
    result.save("result7.png")

def part_c():
    img = np.array(Image.open(path+"sample5.png").convert("L"))
    final = np.zeros(shape=img.shape)
 
    size = img.shape[0]

    x_ = 0.0; y_=0.0

    for x in range(size):
        for y in range(size):

            x_ = x + 20 * np.sin((8*np.pi * y/size))
            x_pos = int(np.round(x_))

            y_ = y + 20 * np.sin((8*np.pi * x/size))
            y_pos = int(np.round(y_))

            if x_pos>=0 and x_pos<size and y_pos>=0 and y_pos<size:
                final[x_pos,y_pos] = img[x,y]
    
    result = Image.fromarray(final).convert("L")
    # result.show()
    result.save("result8.png")
    
part_a()
part_b()
part_c()