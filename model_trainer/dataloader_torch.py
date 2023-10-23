import json

import os

from PIL import Image

from sklearn.model_selection import train_test_split

import torch


from torch.utils.data import (
    Dataset,
)

from fastai.vision.all import get_image_files


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


    def get_ids(self):
        return self.annotations['annotations'].keys()

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

    def image_id_from_filename(self, filename):
        return filename.split('.')[0]

    def label_func(self, filename):
        image_id = self.image_id_from_filename(filename)

        return self.annotations['annotations'][image_id]['category']




class ContextualizedDataset(Dataset):
    def __init__(self, path, annotation_file='train.json', images_dir='train', ids=None, transform=None, device=None):
        self.annotations_path = os.path.join(path, annotation_file)
        self.annotation = AnnotationLabel(self.annotations_path)
        self.images_path = os.path.join(path, images_dir)
        self.device = device

        if ids is None:
            self.ids = list(self.annotation.get_ids())
        else:
            self.ids = ids

        self.setup_files()

        self.transform = transform

    def setup_files(self):
        self.files = []
        for f in get_image_files(self.images_path):
            img_id = self.annotation.image_id_from_filename(f.name)
            if '_fg' not in f.name and img_id in self.ids:
                self.files.append(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.files[index]
        image = Image.open(img_path)
        label = self.annotation.label_func(img_path.name)
        y_label = torch.tensor(int(label))

        if self.transform:
            if self.device is not None:
                self.transform.to(self.device)
            image = self.transform(image)

        return (image, y_label)


def get_train_test_split(path, annotation_file='train.json', images_dir='train', train_transform=None, val_transform=None):
    full_dataset = ContextualizedDataset(path, annotation_file, images_dir, ids=None)

    train_ids, val_ids = train_test_split(full_dataset.ids, train_size=0.8, shuffle=True, random_state=123)
    train_dataset = ContextualizedDataset(path, ids=train_ids, transform=train_transform)
    val_dataset = ContextualizedDataset(path, ids=val_ids, transform=val_transform)
    return train_dataset, val_dataset