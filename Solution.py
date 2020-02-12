#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans





#AUXILIARY FUNCTIONS

def prepare_image(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)


    mask = mask0+mask1

    return mask

def get_cluster(mask):
    #this function runs KMeans on the filtered points (i.e. the mask)
    rows = np.where(mask == 255)[0]
    cols =  np.where(mask == 255)[1]
    X = np.hstack((cols.reshape(len(cols),1), rows.reshape(len(rows),1)))
    alg = KMeans(n_clusters = 4)
    res = alg.fit(X)
    plt.plot(cols, rows, "o", label = "red pixels", color = "red")
    plt.plot(res.cluster_centers_[:,0],res.cluster_centers_[:,1], "o", color = "blue", label = "centers")
    plt.legend()

    return X, res.cluster_centers_

def stack_images(X0, X1):
    #this function stacks the two sets of points and runs KMeans on them
    X3 = np.vstack((X0,X1))
    alg = KMeans(n_clusters = 4)
    res = alg.fit(X3)
    plt.plot(X3[:,0], X3[:,1], "o", label = "red pixels", color = "red")
    plt.plot(res.cluster_centers_[:,0],res.cluster_centers_[:,1], "o", color = "blue", markersize = 10, label = "centers")
    plt.legend()
    return X3, res.cluster_centers_, res



def get_index_2(a, X):
    #This function returns the position of a inside X
    return np.where(np.sum(X == a, axis = 1) == 2)


def get_pairings(X_centers, X2_centers, X3, res):
    pairings = []
    X_centers = np.round(X_centers)
    X2_centers = np.round(X2_centers)
    for i in range(len(X_centers)):
        for j in range(len(X2_centers)):
            ind1 = get_index_2(X_centers[i], X3)
            ind2 = get_index_2(X2_centers[j], X3)
            label1 = res.labels_[ind1]
            label2 = res.labels_[ind2]
            if label1 == label2:
                plt.plot(np.array([X3[ind1][0][0], X3[ind2][0][0]]), np.array([X3[ind1][0][1], X3[ind2][0][1]]))
                pairings.append([X3[ind1], X3[ind2]])
    return pairings


def coordinates_converter(pairings, shape0, shape1):
    for pair in pairings:
        pair[0] = pair[0].astype(float)
        pair[1] = pair[1].astype(float)
        pair[0] *= np.array([shape0[1] /1000, shape0[0] /1000])
        pair[1] *= np.array([shape1[1]/1000, shape1[0]/1000])
        pair[0] = np.round(pair[0])
        pair[1] = np.round(pair[1])
    return pairings


#FUNCTION TO SOLVE TASK1

def approximate_centers(name_image0):
    plt.clf()
    image0 = cv2.imread(name_image0)
    
    plt.imshow(image0)
    
    mask0 = prepare_image(image0)
    
    X0, X0_centers = get_cluster(mask0)
    
    return np.round(X0_centers)

#FUNCTION TO SOLVE BONUS QUESTION

def pair_points(name_image0, name_image1):
    plt.clf()
    image0 = cv2.imread(name_image0)
    image1 = cv2.imread(name_image1)
    shape0 = image0.shape
    shape1 = image1.shape
    image0 = cv2.resize(image0, (1000,1000))
    image1 = cv2.resize(image1, (1000,1000))
    
    plt.imshow(image0)
    plt.imshow(image1)

    
    
    mask0 = prepare_image(image0)
    mask1 = prepare_image(image1)
    
    X0, X0_centers = get_cluster(mask0)
    X1, X1_centers = get_cluster(mask1)
    
    X3, X3_centers, resX3 = stack_images(X0, X1)
    return coordinates_converter(get_pairings(X0_centers, X1_centers, X3, resX3), shape0, shape1)
  




#Uncomment the function calls to test on the 4 images in the folder

#TASK1
#print(approximate_centers("image0.png"))
#print(approximate_centers("image2.png"))

###BONUS QUESTION
#SAME IMAGE
#print(pair_points("image0.png", "image1.png"))
#IMAGES WITH DIFFERENT SIZES (in this case the second image is actually shown)
#print(pair_points("image0.png", "image2.png"))




