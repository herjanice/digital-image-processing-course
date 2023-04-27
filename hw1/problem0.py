from PIL import Image
import cv2 
import numpy as np

path = "SampleImage/"

def part_a():
    img = cv2.imread(path+"sample1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.array(img)

    h,w,d = img_array.shape
    final = np.zeros((h,w,d), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            final[y][w-x-1] = img_array[y][x]

    result = Image.fromarray(final, 'RGB')
    result.save("result1.png")
    # result.show()

def part_b():
    # FORMULA: 0.2989 * R + 0.5870 * G + 0.1140 * B
    img = cv2.imread("result1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.array(img)

    h,w,d = img_array.shape
    final = np.zeros((h,w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            r,g,b = img_array[y][x]
            final[y][x] = 0.2989 * r + 0.5870 * g + 0.1140 * b

    result = Image.fromarray(final)
    result.save("result2.png")
    # result.show()


part_a()
part_b()
