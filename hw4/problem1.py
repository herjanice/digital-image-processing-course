from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path = "hw4_sample_images/"

def part_a():
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    final = np.zeros(shape=img.shape, dtype=np.uint8)

    N = 2
    dither = np.array([[1,2],[3,0]])
    threshold = 255 * ((dither+0.5) / (N**2))
    
    for i in range(0,img.shape[0],N):
        for j in range(0,img.shape[1],N):

            window = np.array(img[i:i+N, j:j+N])

            final[i:i+N, j:j+N] = window > threshold

    final = np.where(final==1, 255, 0)
    result = Image.fromarray(final.astype(np.uint8))
    result.save("result1.png")

def part_b():
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    final = np.zeros(shape=img.shape, dtype=np.uint8)

    N = 256
    dither_2 = np.array([[1,2],[3,0]])

    # Creating 256x256 dither matrix
    matrix = dither_2
    for i in range(int(np.log2(N))-1):
        A = 4 * matrix + 1
        B = 4 * matrix + 2
        C = 4 * matrix + 3 
        D = 4 * matrix + 4
        matrix = np.block([[A,B],[C,D]])

    dither_256 = matrix
    threshold = 255 * ((dither_256+0.5) / (N**2))

    for i in range(0,img.shape[0],N):
        for j in range(0, img.shape[1],N):

            window = np.array(img[i:i+N, j:j+N])

            final[i:i+N, j:j+N] = window > threshold
    
    final = np.where(final==1, 255, 0)
    result = Image.fromarray(final.astype(np.uint8))
    result.save("result2.png")

def part_c():
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    floyd = np.zeros(shape=img.shape, dtype=np.uint8)
    jarvis = np.zeros(shape=img.shape, dtype=np.uint8)

    # Floyd Steinberg
    floyd = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            old_value = floyd[i,j]
            new_value = None
            if old_value >= 128:
                new_value = 255
            else:
                new_value = 0
            floyd[i,j] = new_value

            error = old_value - new_value

            if j < img.shape[1]-1:
                new_value = floyd[i,j+1] + error * 7/16
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                floyd[i,j+1] = new_value

            if j < img.shape[1]-1 and i < img.shape[0]-1:
                new_value = floyd[i+1,j+1] + error * 1/16
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                floyd[i+1,j+1] = new_value
            
            if i < img.shape[0]-1:
                new_value = floyd[i+1,j] + error * 5/16
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                floyd[i+1,j] = new_value

            if j > 0 and i < img.shape[0]-1:
                new_value = floyd[i+1,j-1] + error * 3/16
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                floyd[i+1,j-1] = new_value

    result = Image.fromarray(floyd.astype(np.uint8))
    result.save("result3.png")

    # Jarvis et al
    jarvis = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            old_value = jarvis[i,j]
            new_value = None
            if old_value >= 128:
                new_value = 255
            else:
                new_value = 0
            jarvis[i,j] = new_value

            error = old_value - new_value

            if j < img.shape[1]-1:
                new_value = jarvis[i,j+1] + error * 7/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i,j+1] = new_value
            
            if j < img.shape[1]-2:
                new_value = jarvis[i,j+2] + error * 5/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i,j+1] = new_value

            if j < img.shape[1]-1 and i < img.shape[0]-1:
                new_value = jarvis[i+1,j+1] + error * 5/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+1,j+1] = new_value

            if j < img.shape[1]-2 and i < img.shape[0]-1:
                new_value = jarvis[i+1,j+2] + error * 3/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+1,j+2] = new_value

            if j < img.shape[1]-1 and i < img.shape[0]-2:
                new_value = jarvis[i+2,j+1] + error * 3/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+2,j+1] = new_value

            if j < img.shape[1]-2 and i < img.shape[0]-2:
                new_value = jarvis[i+2,j+2] + error * 1/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+2,j+2] = new_value
            
            if i < img.shape[0]-1:
                new_value = jarvis[i+1,j] + error * 7/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+1,j] = new_value

            if i < img.shape[0]-2:
                new_value = jarvis[i+2,j] + error * 5/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+2,j] = new_value

            if j > 0 and i < img.shape[0]-1:
                new_value = jarvis[i+1,j-1] + error * 5/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+1,j-1] = new_value

            if j > 0 and i < img.shape[0]-2:
                new_value = jarvis[i+2,j-1] + error * 3/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+2,j-1] = new_value

            if j > 1 and i < img.shape[0]-1:
                new_value = jarvis[i+1,j-2] + error * 3/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+1,j-2] = new_value

            if j > 1 and i < img.shape[0]-2:
                new_value = jarvis[i+2,j-2] + error * 1/48
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                jarvis[i+2,j-2] = new_value

    result = Image.fromarray(jarvis.astype(np.uint8))
    result.save("result4.png")

    # Stucki
    stucki = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            old_value = stucki[i,j]
            new_value = None
            if old_value >= 128:
                new_value = 255
            else:
                new_value = 0
            stucki[i,j] = new_value

            error = old_value - new_value

            if j < img.shape[1]-1:
                new_value = stucki[i,j+1] + error * 8/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i,j+1] = new_value
            
            if j < img.shape[1]-2:
                new_value = stucki[i,j+2] + error * 4/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i,j+1] = new_value

            if j < img.shape[1]-1 and i < img.shape[0]-1:
                new_value = stucki[i+1,j+1] + error * 4/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+1,j+1] = new_value

            if j < img.shape[1]-2 and i < img.shape[0]-1:
                new_value = stucki[i+1,j+2] + error * 2/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+1,j+2] = new_value

            if j < img.shape[1]-1 and i < img.shape[0]-2:
                new_value = stucki[i+2,j+1] + error * 2/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+2,j+1] = new_value

            if j < img.shape[1]-2 and i < img.shape[0]-2:
                new_value = stucki[i+2,j+2] + error * 1/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+2,j+2] = new_value
            
            if i < img.shape[0]-1:
                new_value = stucki[i+1,j] + error * 8/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+1,j] = new_value

            if i < img.shape[0]-2:
                new_value = stucki[i+2,j] + error * 4/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+2,j] = new_value

            if j > 0 and i < img.shape[0]-1:
                new_value = stucki[i+1,j-1] + error * 4/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+1,j-1] = new_value

            if j > 0 and i < img.shape[0]-2:
                new_value = stucki[i+2,j-1] + error * 2/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+2,j-1] = new_value

            if j > 1 and i < img.shape[0]-1:
                new_value = stucki[i+1,j-2] + error * 2/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+1,j-2] = new_value

            if j > 1 and i < img.shape[0]-2:
                new_value = stucki[i+2,j-2] + error * 1/42
                if new_value > 255:
                    new_value = 255
                elif new_value < 0:
                    new_value = 0
                stucki[i+2,j-2] = new_value

    # result = Image.fromarray(stucki.astype(np.uint8))
    # result.save("stucki.png")

    





part_a()
part_b()
part_c()