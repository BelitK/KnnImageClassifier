from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./dataset/") ## Folder structure: datasets --> sub-folders with labels name
#print(imagePaths)


# encode the labels as integer
data = np.array(data)
print(lables)
lables = np.array(lables)

le = LabelEncoder()
lables = le.fit_transform(lables)


myset = set(lables)
print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size,-1)

print("data shape: ",data.shape)
print("label shape: ",lables.shape)
print("dataset buyuklugu",dataset_size)

(trainX, testX, trainY, testY ) = train_test_split(data, lables, test_size= 0.20, random_state=42)

model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
model.fit(trainX, trainY)


print(classification_report(testY, model.predict(testX), target_names=le.classes_))
y_pred = model.predict(testX)
print(confusion_matrix(testY,y_pred))

# testt = testX[1]
# print(testt)
# testt = testt.reshape(1,-1)
img = cv2.imread("dataset/fire_images/14.png")
img = cv2.resize(img,(128,128))
img = np.array(img)
imgg = img.reshape(1,-1)

predicted = model.predict(imgg)
if predicted==0:
    print("ates yok :", predicted)
else:
    print("ates :",predicted)
