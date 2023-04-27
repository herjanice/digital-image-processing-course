from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "hw3_sample_images/"

def part_a():
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    binary_img = np.where(img==255, 1, 0)
    # Erosion
    eroded_img = np.zeros(shape=img.shape)
    mask = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            window = binary_img[i-1:i+2, j-1:j+2]
            
            if (window==mask).all():
                eroded_img[i,j] = 255

    # Boundary Extraction
    extracted_img = np.zeros(shape=img.shape)

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         extracted_img[i,j] = img[i,j] - eroded_img[i,j]
    extracted_img = img - eroded_img

    result = Image.fromarray(extracted_img).convert("L")
    result.save("result1.png")

def part_b():
    # The Smiley Face boundaries:
    # [[93, 230], [153, 285]]

    # Banner boundaries:
    # [[179, 42], [429, 112]]

    # Math signs boundaries: (Plus, Minus, Times, Divide)
    # [[179, 407], [238, 456], [241, 390], [300, 421], [204, 474], [266, 524], [262, 452], [335, 501]]
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    binary_img = np.where(img==255, 1, 0)
    inverted_img = np.where(binary_img==1, 0, 1)

    mask = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    G = np.zeros(shape=img.shape)
    dilated_img = np.zeros(shape=img.shape)
    
    x1,x2,y1,y2 = 0,0,0,0
    coordinates = [[93, 230], [153, 285], [179, 42], [429, 112], [179, 407], [238, 456], [241, 390], [300, 421], [204, 474], [266, 524], [262, 450], [328, 501]]
    for n in range(0,len(coordinates),2):
        x1,x2 = coordinates[n][1], coordinates[n+1][1]
        y1,y2 = coordinates[n][0], coordinates[n+1][0]
        for i in range(x1,x2):
            for j in range(y1,y2):

                if binary_img[i,j]==1:
                    continue
                
                if G[i,j] == 1:
                    continue
                G[i,j] = 1
                # Dilation
                for x in range(x1,x2):
                    for y in range(y1,y2):
                        window = np.array(G[x-1:x+2, y-1:y+2])

                        if np.sum(window*mask) >= 1:
                            dilated_img[i,j] = 1
                
                G = dilated_img * inverted_img

    G = np.where(G==1, 255, 0)

    hole_filled = G + img
    result = Image.fromarray(hole_filled.astype(np.uint8))
    result.save("result2.png")

            

def part_c(mode="normal"):
    img = np.array(Image.open(path+"sample1.png").convert("L"))

    if mode == "reversed":
        reversed_img = np.where(img==255, 0, 255)
        img = reversed_img

    binary_img = np.where(img==255, 1, 0)

    stage1_mask = np.array([
        # Bond = 4
        [[0,1,0],[0,1,1],[0,0,0]],
        [[0,1,0],[1,1,0],[0,0,0]],
        [[0,0,0],[1,1,0],[0,1,0]],
        [[0,0,0],[0,1,1],[0,1,0]],
        [[0,0,1],[0,1,1],[0,0,1]],
        [[1,1,1],[0,1,0],[0,0,0]],
        [[1,0,0],[1,1,0],[1,0,0]],
        [[0,0,0],[0,1,0],[1,1,1]],
        # Bond = 6
        [[1,1,1],[0,1,1],[0,0,0]],
        [[0,1,1],[0,1,1],[0,0,1]],
        [[1,1,1],[1,1,0],[0,0,0]],
        [[1,1,0],[1,1,0],[1,0,0]],
        [[1,0,0],[1,1,0],[1,1,0]],
        [[0,0,0],[1,1,0],[1,1,1]],
        [[0,0,0],[0,1,1],[1,1,1]],
        [[0,0,1],[0,1,1],[0,1,1]],
        # Bond = 7
        [[1,1,1],[0,1,1],[0,0,1]],
        [[1,1,1],[1,1,0],[1,0,0]],
        [[1,0,0],[1,1,0],[1,1,1]],
        [[0,0,1],[0,1,1],[1,1,1]],
        # Bond = 8
        [[0,1,1],[0,1,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,0,0]],
        [[1,1,0],[1,1,0],[1,1,0]],
        [[0,0,0],[1,1,1],[1,1,1]],
        # Bond = 9
        [[1,1,1],[0,1,1],[0,1,1]],
        [[0,1,1],[0,1,1],[1,1,1]],
        [[1,1,1],[1,1,1],[1,0,0]],
        [[1,1,1],[1,1,1],[0,0,1]],
        [[1,1,1],[1,1,0],[1,1,0]],
        [[1,1,0],[1,1,0],[1,1,1]],
        [[1,0,0],[1,1,1],[1,1,1]],
        [[0,0,1],[1,1,1],[1,1,1]],
        # Bond = 10
        [[1,1,1],[0,1,1],[1,1,1]],
        [[1,1,1],[1,1,1],[1,0,1]],
        [[1,1,1],[1,1,0],[1,1,1]],
        [[1,0,1],[1,1,1],[1,1,1]],
        # Bond = 11
        [[1,1,1],[1,1,1],[0,1,1]],
        [[1,1,1],[1,1,1],[1,1,0]],
        [[1,1,0],[1,1,1],[1,1,1]],
        [[0,1,1],[1,1,1],[1,1,1]],
    ])


    for iterations in range(100):
        stage1_img = np.zeros(shape=img.shape)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if binary_img[i,j] == 0: # black part, not candidate for erasure
                    continue

                window = np.array(binary_img[i-1:i+2, j-1:j+2])

                for mask in stage1_mask:
                    if (window==mask).all():
                        stage1_img[i,j] = 1
                        break

        # print(np.sum(stage1_img))
        # final = np.where(stage1_img==1,255,0)
        # result = Image.fromarray(final.astype(np.uint8))
        # result.save("skeletonize1_"+str(mode)+".png")
        # result.show()

        final_img = binary_img - stage1_img # Remove the pixels
        binary_img = final_img
    
    skeletonize_img = np.where(binary_img==1, 255, 0)
    result = Image.fromarray(skeletonize_img.astype(np.uint8))
    # result.save("skeletonize1_"+str(mode)+"_"+str(iterations)+".png")
        
    if mode == "normal":
        result.save("result3.png")
    else:
        result.save("result4.png")

def part_d(): 
    img = np.array(Image.open(path+"sample1.png").convert("L"))
    binary_img = np.where(img==255, 1, 0)

    # 2-Pass Connected Component Algorithm
    component = 1
    labeled_img = np.zeros(shape=img.shape)

    differences = {}
    kernel = 15
    for i in range(kernel,img.shape[0]-kernel):
        for j in range(kernel,img.shape[1]-kernel):
            window = np.array(labeled_img[i-kernel:i+kernel+1, j-kernel:j+kernel+1])

            if binary_img[i,j] == 0:
                continue

            indexes = np.nonzero(window)
            non_zero = np.array(window[indexes])

            # New component
            if len(non_zero)==0:
                labeled_img[i,j] = component
                component += 1
            # checking if neighbors all have the same label
            elif np.all(non_zero == non_zero[0]):
                labeled_img[i,j] = non_zero[0]
            # if neighbors have different component marking
            else:
                labeled_img[i,j] = np.amin(non_zero)
                
                labels = []
                for label in non_zero:
                    if label not in labels and label!=np.amin(non_zero):
                        labels.append(label)
                        differences[label] = np.amin(non_zero)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if labeled_img[x,y] == 0:
                continue

            while labeled_img[x,y] in differences:
                labeled_img[x,y] = differences[labeled_img[x,y]]

    colored_img = np.array(Image.new("RGB", img.shape))
    colors = np.random.randint(0,256,(component,3),dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if labeled_img[i,j]==0:
                continue
            label = int(labeled_img[i,j]-1)
            colored_img[i,j] = colors[label]
        
    result = Image.fromarray(colored_img)
    # result.save("labeledimages_"+str(kernel*2+1)+"x"+str(kernel*2+1)+"_2.png")
    result.save("result5.png")




part_a()
part_b()
part_c(mode="normal")
part_c(mode="reversed")
part_d()