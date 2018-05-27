import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os

# load the dataset
dataset = pd.read_csv("data/boneage-original-dataset.csv")

# summarize the dataset

# dimensions of table
print("Dimensions:", dataset.shape)
# peek the data
print(dataset.head(15))
print(dataset.describe())

# see column names
print('Default columns:', dataset.columns)

# create another field to map the image into the dataset also
dataset['path'] = dataset['id'].map(lambda x: os.path.join('data',
                                                         'boneage-original-dataset',
                                                         '{}.png'.format(x)))

# this new field is for safety measures, it tells us if the requested image was found
dataset['exists'] = dataset['path'].map(os.path.exists)
print('found', dataset['exists'].sum(), 'images found of', dataset.shape[0], 'total')

# map the boolean in the last column to the gender name
dataset['gender'] = dataset['male'].map(lambda x: 'male' if x else 'female')

# names of new columns
print('New columns:', dataset.columns)

# construct a histogram to visualize data
dataset[['boneage', 'male']].hist(figsize = (10, 5))
plt.show()

# print occurences of every age
print(dataset.groupby(by=['boneage', 'gender']).count())

# we can divide the data into some groups by age
age_groups = 8
dataset['age_class'] = pd.qcut(dataset['boneage'], age_groups)
age_overview_df = dataset.groupby(['age_class',
                                  'gender']).apply(lambda x: x.sample(1)
                                                             ).reset_index(drop = True)

# show an example image from each subgroup
fig, m_axs = plt.subplots( age_groups//2, 4, figsize = (12, age_groups))
for c_ax, (_, c_row) in zip(m_axs.flatten(),
                            age_overview_df.sort_values(['age_class', 'gender']).iterrows()):
    c_ax.imshow(imread(c_row['path']),
                cmap = 'viridis')
    c_ax.axis('off')
    c_ax.set_title('{boneage} months, {gender}'.format(**c_row))
plt.show(fig)
