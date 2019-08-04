import numpy as np
from keras.preprocessing import image
import cv2 as cv
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
import pandas as pd
from builtins import range
# Ignore warnings
import warnings

warnings.filterwarnings('ignore')
print("Files imported successfully")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30 * 30 + 1)

# This function is different from the one that exists in knn_cat_dog as this takes input in BGR format, rather than
# converting into grayscale.


def load_image_files(container_path, chanel, dimension=(64, 64)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    count = 0
    train_img = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            count += 1
            img = imread(file)
            img_pred = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
            img_pred = img_pred[:, :, chanel]
            img_pred = image.img_to_array(img_pred)
            img_pred = img_pred / 255
            train_img.append(img_pred)

    X = np.array(train_img)
    return X


# final vector X
X = np.zeros((4134, 50, 50, 3))

# loads B channel of images
X0 = load_image_files("data/train", 0)

# loads G channel of images
X1 = load_image_files("data/train", 1)

# loads R channel of images
X2 = load_image_files("data/train", 2)

X[:, :, :, 0] = X0[:, :, :, 0]
X[:, :, :, 1] = X1[:, :, :, 0]
X[:, :, :, 2] = X2[:, :, :, 0]
print(X.shape)

# 2000 images for Class 0, 2134 for Class 1.
y0 = np.zeros(2000)
y1 = np.ones(2134)
# concatenate y0 and y1 to form y
y = []
y = np.concatenate((y0, y1), axis=0)

# Using train_test_split from the sklearn library.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=0.5)
print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("X_val: " + str(X_val.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))
print("y_val: " + str(y_val.shape))

####Forming X_test, X_train, y_train, y_test####
num_training = X_train.shape[0]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = X_test.shape[0]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

num_val = X_val.shape[0]
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
print(X_train.shape, X_test.shape, X_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

# Getting data to zero mean, i.e that centred around zero.
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_val -= mean_image

# append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print(X_train.shape, X_test.shape, X_val.shape)
print("Data ready")

##########################

from classifiers.linear_classifier import LinearSVM

svmd = LinearSVM()
loss_hist = svmd.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                       num_iters=1500, verbose=True)

y_train_pred = svmd.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
y_val_pred = svmd.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths.
learning_rates = [5e-7, 5e-6, 5e-5, 5e-4, 5e-3]
regularization_strengths = [2.5e4, 5e4, 2.5e3, 2.5e2, 5e2, 1e1]

results = {}
best_val = -1  # The highest validation accuracy that we have seen so far.
best_svm = None  # The LinearSVM object that achieved the highest validation rate.

# Declare blr, brg to store the best LinearSVM object's learning rate and regularization.
blr = None
brg = None

grid_search = [(lr, rg) for lr in learning_rates for rg in regularization_strengths]
for lr, rg in grid_search:
    svmd = LinearSVM()
    train_loss = svmd.train(X_train, y_train, learning_rate=lr, reg=rg,
                            num_iters=2000, verbose=False)
    # Predict values for training set
    y_train_pred = svmd.predict(X_train)
    # Calculate accuracy
    train_accuracy = np.mean(y_train_pred == y_train)
    # Predict values for validation set
    y_val_pred = svmd.predict(X_val)
    # Calculate accuracy
    val_accuracy = np.mean(y_val_pred == y_val)
    # Save results
    results[(lr, rg)] = (train_accuracy, val_accuracy)
    if best_val < val_accuracy:
        blr = lr
        brg = rg
        best_val = val_accuracy
        best_svm = svmd

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)
svmd = LinearSVM()
loss_hist = svmd.train(X_train, y_train, learning_rate=blr, reg=brg,
                       num_iters=2000, verbose=False)
y_svmd = svmd.predict(X_test)
print("SVMD")
print(y_svmd[0:10])
print("actual")
print(y_test[0:10])
print("best_svm")
print(y_test_pred[0:10])
import os


pred = []
svmd = LinearSVM()
loss_hist = svmd.train(X_train, y_train, learning_rate=blr, reg=brg,
                       num_iters=2000, verbose=False)
svmdpred = []
ccat = 0
cdog = 0
for i in range(1, 50):
    file = os.path.join(
        'data/test/Dog/',
        str(i) + ".jpg")
    img = cv.imread(file)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_pred = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
    img_pred = image.img_to_array(img_pred)
    img_pred = img_pred / 255
    img_pred = np.reshape(img_pred, (1, 3 * img_pred.shape[0] * img_pred.shape[1]))
    img_pred -= mean_image
    img_pred = np.hstack([img_pred, np.ones((img_pred.shape[0], 1))])
    res = svmd.predict(img_pred)
    if res[0] == 0:
        ccat += 1
    else:
        cdog += 1
    svmdpred.append(res[0])

print("Dog testset with custom implementation: Cats and Dogs")
print(ccat, cdog)
#############STANDARD

from sklearn import svm, metrics

classifier = svm.LinearSVC(penalty='l2', loss='squared_hinge', max_iter=2000)
classifier.fit(X_train, y_train)
print(classifier.score(X_train, y_train))
print(classifier.score(X_val, y_val))
print(classifier.score(X_test, y_test))
predx = []
ccat = cdog = 0
for i in range(1, 50):
    file = os.path.join(
        'data/test/Dog/',
        str(i) + ".jpg")
    img = cv.imread(file)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_pred = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
    img_pred = image.img_to_array(img_pred)
    img_pred = img_pred / 255
    img_pred = np.reshape(img_pred, (1, 3 * img_pred.shape[0] * img_pred.shape[1]))
    img_pred = np.hstack([img_pred, np.ones((img_pred.shape[0], 1))])
    res = classifier.predict(img_pred)
    if res[0] == 0:
        ccat += 1
    else:
        cdog += 1
    predx.append(res[0])
print("Using sklearn lib: Cats and Dogs")
print(ccat, cdog)
