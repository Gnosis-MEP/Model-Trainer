import os
from fastai.vision.all import ImageDataLoaders, Resize, get_image_files



def label_func(f):
    return f[0].isupper()


def get_loader(path):
    files = get_image_files(path)
    loader = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
    return loader