#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import sys
import string
# Get user supplied values
imgName = 'abba.png'
imgNameToSave = imgName.translate(imgName.maketrans('', '', string.punctuation)).lower()
imagePath = './'+imgName
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('./'+imgNameToSave+'{0}.jpg'.format(x), image[y:y+h, x:x+w])
    cv2.imshow("Gray",cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY))
    

cv2.imshow("Faces found", image)
cv2.waitKey(0)


# In[ ]:




