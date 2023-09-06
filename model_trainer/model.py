from matplotlib import pyplot

import torchvision.models as models

from fastai.vision.all import cnn_learner, accuracy, vision_learner, error_rate

from model_trainer.dataloader import get_loader
from model_trainer.conf import DATASET_PATH


def fine_tune():
    dls = get_loader(DATASET_PATH)
    learn = vision_learner(dls, models.mobilenet_v3_large, metrics=error_rate)
    learn.fine_tune(1)
    # learn.fine_tune(2, 3e-3)
    learn.show_results()
    pyplot.show()

if __name__ == '__main__':
    fine_tune()
