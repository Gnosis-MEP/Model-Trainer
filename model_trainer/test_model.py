import os
import sys

from matplotlib import pyplot

import torch
import torchvision.models as models

from fastai.basics import default_device
from fastai.vision.all import load_learner, Path, PILImage, vision_learner, accuracy, show_image

from model_trainer.dataloader import get_loader
from model_trainer.conf import DATASET_PATH


def model_eval(image_path):
    dls = get_loader(DATASET_PATH)
    learn = vision_learner(dls, models.mobilenet_v3_large, metrics=accuracy)
    learn.load('cls_model')

    learn.model.cuda()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the learner to the GPU
    # learn.model.to(device)

    # Load and preprocess the image
    img = PILImage.create(image_path)

    # Move the image to the GPU
    # img = img.to(device)

    # Make predictions on the image
    predicted_class, _, class_probs = learn.predict(image_path)

    # Print the predicted class
    print(f'Predicted class: {predicted_class}')

    # Convert class_probs to a list
    # class_probs = class_probs.tolist()

    # Print the confidence scores for each class
    for class_name, prob in zip(learn.dls.vocab, class_probs):
        print(f'Class: {class_name}, Confidence Score: {prob}')

    show_image(img)
    pyplot.show()


if __name__ == '__main__':
    image_path = sys.argv[1]
    # python model_trainer/test_model.py /home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-Q-1/frame_5176.png
    # python model_trainer/test_model.py /home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-Q-1/frame_5280.png
    # python model_trainer/test_model.py /home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-Q-1/frame_7076.png
    model_eval(image_path)
