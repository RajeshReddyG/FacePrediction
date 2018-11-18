'''import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('/home/rajesh/Desktop/Face_Completion/Imgs/2.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/rajesh/Desktop/Face_Completion/Imgs/3.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)
#img.append(img2)'''

from os import listdir
from PIL import Image as PImage
import numpy as np
import pandas as pd

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = np.asarray(PImage.open(path + image).convert('LA'))
        loadedImages.append(img)

    return loadedImages

path = "/home/rajesh/Desktop/Face_Completion/Imgs/" #Specify Path Of Images as Images Numbered from 1.jpg, 2.jpg etc., 

# your images in an array
imgs = loadImages(path)
#print(imgs)
d={'images':imgs,'target': [ 0,  0,  0,  0,  0,  0]}
DF=pd.DataFrame(d)
print(DF)

'''for img in imgs:
    # you can show every image
    print(img)
for img in imgs:
    # you can show every image
    img.show()
'''
