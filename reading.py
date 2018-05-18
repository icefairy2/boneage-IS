import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
import os

# load the dataset
dataset = pd.read_csv("data/boneage-original-dataset.csv")

print('Datasets read.')

# create another field to map the image into the dataset also
dataset['path'] = dataset['id'].map(lambda x: os.path.join('data',
                                                        #change this to boneage-original-dataset if you want the big images
                                                         'boneage-resized-dataset',
                                                         '{}.png'.format(x)))

print('Paths created.')

dataset['years'] = dataset['boneage'].map(lambda x: int(x/12))

#split the dataset into a training and a validation part by 80-20
train_dataset, test_dataset = model_selection.train_test_split(dataset, test_size = 0.01, random_state = 21)

print('Dataset split.')

#write the splits into two .csv files
train_dataset.to_csv("data/boneage-training-dataset.csv")
test_dataset.to_csv("data/boneage-test-dataset.csv")

print('CSV-s written.')