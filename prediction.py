import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn import metrics

# utils
from sklearn.model_selection import cross_val_score

# algorithms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load the datasets
train_dataset = pd.read_csv("data/boneage-training-dataset.csv")
test_dataset = pd.read_csv("data/boneage-test-dataset.csv")

print('Datasets read.')

# this is for algorithm testing purposes, after a solution was found, apply it integrally
train_test_dataset = train_dataset.head(1000).copy()
test_test_dataset = test_dataset.head(100).copy()
# read in the corresponding image as grayscale
train_test_dataset['img'] = train_test_dataset['path'].map(lambda x: imread(x, as_grey=True))
test_test_dataset['img'] = test_test_dataset['path'].map(lambda x: imread(x, as_grey=True))

print('Images read.')
n_samples = len(train_test_dataset['img'])
n_test = len(test_test_dataset['img'])

# put all images in an array
images = []
for i in range(0, n_samples):
    images.append(train_test_dataset['img'][i])

data_arr = np.array(images)

images_test = []
for i in range(0, n_test):
    images_test.append(test_test_dataset['img'][i])

data_arr_test = np.array(images_test)

print('Arrayed Images.')

#shape data and target
data = data_arr.reshape((n_samples, -1))
target = np.array(train_test_dataset['boneage'])

data_test = data_arr_test.reshape((n_test, -1))
target_test = np.array(test_test_dataset['boneage'])

print('Reshaped Array.')

# Regression
models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('NN', MLPRegressor()))
models.append(('GP',  GaussianProcessRegressor()))
models.append(('SVM', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	cv_results = cross_val_score(model, data, target, scoring='neg_mean_absolute_error', cv=5)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# visualizing algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Create a regressor
regressor = SVR(gamma=0.001, kernel='linear')

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', regressor)
])

errors = cross_val_score(pipeline, data, target, scoring='neg_mean_absolute_error', cv=10)

print("SVM mean of errors", np.mean(errors))

# We learn the boneage on the dataset except the last 4
regressor.fit(data[:-4], target[:-4])
print('Learned.')

# Now predict the value of the boneage on the last 4
expected = target[-4:]
predicted = regressor.predict(data[-4:])

print('Predicted.')

print("Regression report for regressor %s:\n%s\n"
      % (regressor, metrics.mean_absolute_error(expected, predicted)))

# Show an image of the 4 predicted ages
images_and_predictions = list(zip(train_test_dataset.img[-4:], predicted, target))
for index, (image, prediction, target) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Pred: {:.2f}\n Act: {:}'.format(prediction, target))

plt.show()

# running the prediction
# Fit regression model
svr_rbf = SVR(kernel='rbf', gamma=0.1)
svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly', degree=2)
y_rbf = svr_rbf.fit(data, target).predict(data_test)
print('RBF learned.')
y_lin = svr_lin.fit(data, target).predict(data_test)
print('Linear learned.')
y_poly = svr_poly.fit(data, target).predict(data_test)
print('Polynomial learned.')

def my_plot(ax, err, title):
    ax.plot(range(0, n_test), err)
    ax.set_ylabel('error')
    ax.set_title(title)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

errors = []
for i in range(0, n_test):
    errors.append(target_test[i] - y_rbf[i])

my_plot(ax1, errors, 'RBF')

errors = []
for i in range(0, n_test):
    errors.append(target_test[i] - y_lin[i])

my_plot(ax2, errors, 'Linear')

errors = []
for i in range(0, n_test):
    errors.append(target_test[i] - y_poly[i])

my_plot(ax3, errors, 'Polynomial')
plt.tight_layout()
plt.show()

cv_results = cross_val_score(svr_rbf, data, target, scoring='neg_mean_absolute_error', cv=10)
print("RBF: %f (%f)" % (cv_results.mean(), cv_results.std()))
cv_results = cross_val_score(svr_lin, data, target, scoring='neg_mean_absolute_error', cv=10)
print("Linear: %f (%f)" % (cv_results.mean(), cv_results.std()))
cv_results = cross_val_score(svr_poly, data, target, scoring='neg_mean_absolute_error', cv=10)
print("Polynomial: %f (%f)" % (cv_results.mean(), cv_results.std()))
