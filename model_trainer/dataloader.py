import json
from collections import Counter

import os
from fastai.vision.all import ImageDataLoaders, Resize, get_image_files


class AnnotationLabel():
    def __init__(self, annotations_path, isdetection=False):
        self.annotations_path = annotations_path
        self.annotations = {}
        # self.annotations = {
        #     5906 : {
        #         'category': 'book'
        #     }
        # }
        self.setup_annotations()


    def setup_annotations(self):
        data = None
        with open(self.annotations_path, 'r') as f:
            data = json.load(f)
        self.annotations = data

    # def setup_annotations(self):
    #     data = None
    #     with open(self.annotations_path, 'r') as f:
    #         data = json.load(f)

    #     cat_map = {}
    #     for cd in data['categories']:
    #         cat_map[cd['id']] = cd['name']

    #     prev_img_id = None
    #     prev_c_list = []
    #     for d in data['annotations']:
    #         if prev_img_id is None:
    #             prev_img_id = d['image_id']
    #         if prev_img_id != d['image_id']:
    #             most_common_cat = Counter(prev_c_list).most_common(1)[0][0]
    #             self.annotations[prev_img_id] = {
    #                 'category': most_common_cat
    #             }
    #             prev_img_id = d['image_id']
    #         prev_c_list.append(cat_map[d['category_id']])

    #     most_common_cat = Counter(prev_c_list).most_common(1)[0][0]
    #     self.annotations[prev_img_id] = {
    #         'category': most_common_cat
    #     }

    def label_func(self, filename):
        image_id = int(filename.split('.jpg')[0])

        return self.annotations[image_id]['category']


def get_loader(path, annotation_file='train.json', images_dir='train'):
    annotations_path = os.path.join(path, annotation_file)
    annotation = AnnotationLabel(annotations_path)
    images_path = os.path.join(path, images_dir)
    files = get_image_files(images_path)
    loader = ImageDataLoaders.from_name_func(path, files, annotation.label_func, item_tfms=Resize(224))
    return loader