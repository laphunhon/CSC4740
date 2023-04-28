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

def get_List_bounding_img(file, training):
    alphabets=[]
    # Read the input image
    img = cv2.imread(file)  
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    # Threshold the image to create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       
    if not training:    
        # take average to imulate outlier
        average =0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if average ==0:
                average = w
            else:
                if w/average <1.5:
                    average = (average + w)/2
        # Slipt outlier contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)          
            if w/average>=1.5:
                n_slipt = int(w/average)                 
                gap = int(w/n_slipt)
                # draw a slipt line
                for row in range(img.shape[0]):
                    for k in range(1,n_slipt):
                        img[row, x+gap*k] = 255    
        # renew data counours        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)           
   
    # Sort the contours by their x-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
    for contour in contours:    
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        alphabet = img[y:y+h, x:x+w]
        # Resize the alphabet image to (32, 32)
        alphabet = cv2.resize(alphabet, (32, 32), interpolation=cv2.INTER_AREA)
        # Convert the resized image to grayscale
        alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY)      
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

# load traning data
for file_name in image_files:
    if file_name.endswith(".png"):
        image_path = os.path.join(image_dir, file_name)
        images = get_List_bounding_img(image_path,True)     
        for image in images:            
            image = image.astype(np.float32) / 255.0
            X.append(image.flatten())
            Y.append(Z.index(chr(int(file_name.split("_")[0]))))

# sufffer data
xy_zip = list(zip(X, Y))
random.shuffle(xy_zip)
X_shuff = [x for x, y in xy_zip]
Y_shuff = [y for x, y in xy_zip]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_shuff, Y_shuff, test_size=0.2, random_state=42)
svm = SVC(kernel='rbf',probability=True)
svm.fit(X_train, y_train)

# Evaluate the SVM classifier on the validation set
y_pred = svm.predict(X_val)
val_acc = accuracy_score(y_val, y_pred)
print("Validation accuracy:", val_acc)

# Evaluate the SVM classifier on Confusion
# classification report
print(classification_report(y_val, y_pred, target_names=Z))

# Confusion matrix
matplotlib.use('Agg')
print("Confusion matrix: Confusion.png") 
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_val, y_pred), display_labels=Z)
fig, ax = plt.subplots(figsize=(50, 50))
disp.plot(ax=ax, cmap=plt.cm.Blues)
ax.set_title("Confusion Matrix")
plt.savefig("Confusion.png")

# Predict the labels of new images using the trained SVM classifier
image_path = os.path.join("test2.png")
X_new_imgs = get_List_bounding_img(image_path,False)
Y_new_imgs =[]
for X_new in X_new_imgs:
    X_new = np.array(X_new)
    X_new = X_new.astype(np.float32) / 255.0
    X_new = X_new.flatten()   
    predicted_prob = svm.predict_proba([X_new])
    # Find the index of the class with the highest probability
    predicted_label = predicted_prob.argmax()
    # Get the probability estimate for the predicted class
    predicted_prob = predicted_prob[0, predicted_label]
    percentage = round(predicted_prob*100,2)
    print(f"Predicted label: {Z[int(predicted_label)]} in {percentage}%")
    Y_new_imgs.append(Z[int(predicted_label)])
# show output
matplotlib.use('TkAgg')
if len(X_new_imgs) == 1:
    plt.imshow(X_new_imgs[0], cmap='gray')
    plt.title(f"Prediction: {Y_new_imgs[0]}")
else:
    _, axes = plt.subplots(nrows=1, ncols=len(X_new_imgs), figsize=(10, 6))
    for ax, image, prediction in zip(axes, X_new_imgs, Y_new_imgs):
        ax.set_axis_off()
        image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{prediction}")
        plt.subplots_adjust(wspace=1)
plt.show()








