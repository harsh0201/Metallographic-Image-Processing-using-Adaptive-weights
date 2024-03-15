import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

img = cv2.imread(r"image.jpeg",cv2.IMREAD_GRAYSCALE)
img1 = img.copy()
row,col = img.shape
flag = np.zeros((row,col))
cv2.imshow("Original",img)
#print(row,col)

mask = np.array([[1/9,1/99,1/9],
                 [1/9,1/9,1/9],
                 [1/9,1/9,1/9]])

def add_noise(img,prob):
    output = np.zeros(img.shape,dtype="uint8")
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn<prob:
                output[i][j] = 0
            elif rdn>thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output

img = add_noise(img,0.1)

mean = cv2.filter2D(img,kernel=mask,ddepth=-1)

median = cv2.medianBlur(img,3)

cv2.imshow("Input",img)
cv2.imshow("Mean",mean)
cv2.imshow("Median",median)

for i in range(row):
    for j in range(col):
        if img[i][j]==255 :
            flag[i][j]=1
        elif img[i][j]==0 :
            flag[i][j]=-1


def calculate_miu(start_r,start_c,size):
    sum = 0
    for i in range(start_r,start_r+size):
        for j in range(start_c,start_c+size):
            sum = sum + img[i][j]
    miu = sum / (size*size)

    return miu,sum

def check_extreme_are_noise(start_r,start_c,size):
    miu,sum = calculate_miu(start_r,start_c,size)
    w = miu / (size*size)
    c = (1/3)*size*size
    #print("Sum : ",sum)
    if size==3:
        if w<2:
            return True

        else:
            return False

    elif size==5:
        if w<4:
            return True
        else:
            return False

    else:
        if w<11:
            return True
        else:
            return False


def remove_noise_pixcels(start_r,start_c,size):
    Qij = []
    x = 0
    for i in range(start_r,start_r+size):
        for j in range(start_c,start_c+size):
            if flag[i][j]!=1 or flag[i][j]!=-1:
                Qij.append(img[i][j])
                x = x + 1
    return Qij


def calculate_weights(start_r,start_c,size):
    weights = []
    Qij = remove_noise_pixcels(start_r,start_c,size)
    miu,sum = calculate_miu(start_r,start_c,size)

    for i in Qij:
        weight = np.exp(-((miu - i) * 2) / (2 * 70 * 2))
        weights.append(weight)
    #print("\nWeight")
    #print(weights)
    return weights

def mult_weight(start_r,start_c,size):
    Qij = remove_noise_pixcels(start_r,start_c,size)
    weights = calculate_weights(start_r,start_c,size)
    mult = []

    for i in range(size*size):
        mult.append(Qij[i])

    return mult


def calculate_filtering_result(mult):
    mult = sorted(mult)
    len_mult = len(mult)
    sum = 0
    for i in range(len_mult):
        sum = sum + mult[i]

    mean = sum / (len_mult)

    median = 0

    if len_mult%2==0:
        center = int(len_mult/2)
        median = (mult[center] + mult[center+1])/2

    elif len_mult%2!=0:
        center = len_mult/2
        center = int(center + 0.5)
        median = mult[center]
    #print()
    #print("Mean : ",mean)
    #print("Median : ",median)
    if mean>median :
        return mean

    elif mean<median:
        return median

    else:
        return mean

def final_output(start_r,start_c,size):
    mult = mult_weight(start_r,start_c,size)
    value = int(calculate_filtering_result(mult))
    for i in range(start_r,start_r+size):
        for j in range(start_c,start_c+size):
            if flag[i][j]==1 or flag[i][j]==-1:
                img[i][j] = value

    return img

start_r,start_c = 0,0
sizef = 3
sizes = 5
sizet = 7

count = 0
flagc = "a"

while True:
    #print()
    #print("Start_r",start_r)
    #print("Start_c",start_c)
    flagc = "a"

    size = sizef

    if start_c + size > col :
        start_c = 0
        start_r = start_r + 3

    if start_r + size > row:
        break

    check = check_extreme_are_noise(start_r,start_c,size)
    if check==True:
        final_output(start_r,start_c,size)

    elif check==False:
        size = sizes
        if start_c + size <= col and start_r + size <= row:
            check = check_extreme_are_noise(start_r,start_c,size)
            if check==True:
                final_output(start_r,start_c,size)

            elif check==False:
                size = sizet
                if start_c + size <= col and start_r + size <= row:
                    check = check_extreme_are_noise(start_r,start_c,size)
                    if check==True:
                        final_output(start_r,start_c,size)
    size = 3
    if start_r + size > row:
        break

    if start_c + size > col and start_r + size <= row:
        start_c = 0
        flagc = "b"
        start_r = start_r + 3

    if flagc == "a" :
        if start_c + size <= col and start_r + size <= row:
            start_c = start_c + 3

count,countm,countme = 0,0,0,
for i in range(row):
    for j in range(col):
        if img[i][j] == img1[i][j]:
            count = count + 1
        if img[i][j] == mean[i][j]:
            countm = countm + 1
        if img[i][j]  == median[i][j]:
            countme = countme + 1

accuracy = (count / (row*col))*100
print("Our Method : ",accuracy,"%")
accuracym = (countm / (row*col))*100
print("Mean Average : ",accuracym,"%")
accuracyme = (countme / (row*col))*100
print("Median Accuracy : ",accuracyme,"%")

cv2.imshow("output",img)
cv2.waitKey(0)