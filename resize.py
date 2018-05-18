from PIL import Image
import os, sys

path = "D:/PycharmProjects/Bone_age/data/boneage-original-dataset"
newpath = "D:/PycharmProjects/Bone_age/data/boneage-resized-dataset"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + '/' + item):
            im = Image.open(path + '/' + item)
            f = str(item).split('.')[0]
            new_img = im.resize((100, 100))
            new_img.save(newpath + '/' + f + ".png")


resize()