"""
==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.

"""
print(__doc__)

# Required Imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_random_state


# Function to load Image
def load_image(path):
    img = Image.open(path)
    img.load()
    ImgData = np.asarray(img, dtype="int32")
    # print(len(ImgData))
    return ImgData


# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target

######## Debugging Purpose
print(data.images)
print(type(data.images))
ex = load_image("Imgs/Im1.jpg")  # HardCoded Image Path - Change Image Path to predict
exa = []
exa.append(ex)
exanp = np.array(exa)
print(exanp)
print(type((exanp)))
########

data = data.images.reshape((len(data.images), -1))

######## Debugging Purpose
exanp = exanp.reshape((len(exanp), -1))
print(data)
print(len(data))
print(exanp)
print(len(exanp))
########

train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

################ Debugging Purpose
print(test)
print(len(test))
test = exanp
################

# Test on a subset of people
n_faces = 1
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces,))
test = test[face_ids, :]

'''###
# test=load_image().reshape((400, -1))
print(test)
print(len(test[0]))
###'''

n_pixels = data.shape[1]
X_train = train[:, :int(np.ceil(0.5 * n_pixels))]  # Upper half of the faces
y_train = train[:, int(np.floor(0.5 * n_pixels)):]  # Lower half of the faces
X_test = test[:, :int(np.ceil(0.5 * n_pixels))]
y_test = test[:, int(np.floor(0.5 * n_pixels)):]

'''###
print(X_test)
print(y_test)
print(len(X_test[0]))
print(len(y_test[0]))
###'''

# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")

    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

print('Plot Here ')
plt.show()
