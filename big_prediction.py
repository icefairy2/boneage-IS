import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

# load the datasets
train_dataset = pd.read_csv("data/boneage-training-dataset.csv")
test_dataset = pd.read_csv("data/boneage-test-dataset.csv")

print('Datasets read.')

# read in the corresponding image as grayscale
train_dataset['img'] = train_dataset['path'].map(lambda x: imread(x, as_grey=True))

print('Images read.')

n_samples = len(train_dataset['img'])

# put all images in an array
images = []
for i in range(0, n_samples):
    images.append(train_dataset['img'][i])

data_arr = np.array(images)
print(data_arr.shape)
print('Arrayed Images.')

#shape data and target
data = data_arr.reshape((n_samples, -1))
target = np.array(train_dataset['boneage'])

# prepare the test dataset also
# read in the corresponding image as grayscale
test_dataset['img'] = test_dataset['path'].map(lambda x: imread(x, as_grey=True))

print('Images read.')

n_test = len(test_dataset['img'])

# put all images in an array
images = []
for i in range(0, n_test):
    images.append(test_dataset['img'][i])

data_arr = np.array(images)

print('Arrayed Images.')

# shape data and target test
data_test = data_arr.reshape((n_test, -1))
target_test = np.array(test_dataset['boneage'])

print('Reshaped Array.')

# running the prediction
# Fit regression model
svr_rbf = SVR(kernel='rbf', gamma=0.1)
y_rbf = svr_rbf.fit(data, target).predict(data_test)
print('RBF learned.')

errors = []
for i in range(0, n_test):
    errors.append(target_test[i] - y_rbf[i])

plt.plot(range(0, n_test), errors, 'RBF')
plt.show()

cv_results = cross_val_score(svr_rbf, data, target, scoring='neg_mean_absolute_error', cv=10)
print("RBF: %f (%f)" % (cv_results.mean(), cv_results.std()))



