import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from shutil import copy2, copyfile
import os

imageSrc = "/Users/tanzil/Data/ENSF619.2/Face_Gender_Data/github-profile-detection/"
imageDst_Face = "/Users/tanzil/Data/ENSF619.2/Face_Gender_Data/github-profiles-faces/"
imageDst_NoFace = "/Users/tanzil/Data/ENSF619.2/Face_Gender_Data/github-profiles-no-faces/"

imagePaths = list(paths.list_images(imageSrc))
cascPath = "/Users/tanzil/Documents/UCalgary/ENSF619.2/Face_Gender/haarcascade_frontalface_default.xml"


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

def detectFace(imagePath, faceCasecade):
    # Read the image
    image = cv2.imread(imagePath)
    
    # Convert to grayscale
    test_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(test_image_gray, cmap='gray')

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        test_image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    faceCount = len(faces)
    
    '''
    print("Found {0} faces!".format(faceCount))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    '''
    
    return faceCount

totalfaces = 0
totalnonfaces = 0
print("Detection starting. Total faces {0}, non faces {1}", totalfaces, totalnonfaces)

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    faceCount = detectFace(imagePath, faceCascade)
    print("Image", name, "Face count", faceCount)
    if faceCount > 0:
        copy2(imagePath, imageDst_Face)
        totalfaces = totalfaces+1
    else:
        copy2(imagePath, imageDst_NoFace)
        totalnonfaces = totalnonfaces+1

print("Detection finished. Total faces {0}, non faces {1}", totalfaces, totalnonfaces)