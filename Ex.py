import numpy as np
from PIL import Image

'''import cv2
img = cv2.imread('Imgs/Im1.jpg',0)
print(img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''


def load_image(path):
    img = Image.open(path).convert('LA')
    img.load()
    data = np.asarray(img, dtype="int32")
    print(data)
    print(len(data))


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


# load_image("Imgs/Im4.png")
load_image("Imgs/Im1.jpg")
