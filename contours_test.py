# Final Project: Convolutional Neural Network
import os
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib

X_test=[]
def get_List_bounding_img(file):
    alphabets=[]
    # Read the input image
    img = cv2.imread(file)  
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # Threshold the image to create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # Sort the contours by their x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
    for contour in contours:    
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        alphabet = img[y:y+h, x:x+w]
        # Resize the alphabet image to (32, 32)
        #alphabet = cv2.resize(alphabet, (32, 32), interpolation=cv2.INTER_AREA)
        # Convert the resized image to grayscale
        alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY)
        X_test.append(alphabet)
        alphabets.append(alphabet)
    return alphabets
# Load the image dataset into memory
image_dir = "font"
image_files = os.listdir(image_dir)
X = []
Y = []
Z = "0123456789"
Z += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Z += "abcdefghijklmnopqrstuvwxyz"

# Predict the labels of new images using the trained SVM classifier
image_path = os.path.join("test2.png")
X_new_imgs = get_List_bounding_img(image_path)
plt.imshow(X_new_imgs[3], cmap='gray')
plt.show()










