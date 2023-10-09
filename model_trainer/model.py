from matplotlib import pyplot

import torchvision.models as models

from fastai.vision.all import cnn_learner, accuracy, vision_learner, error_rate

from model_trainer.dataloader import get_loader
from model_trainer.conf import DATASET_PATH, MODEL_ID


def fine_tune():
    dls = get_loader(DATASET_PATH)

    import ipdb; ipdb.set_trace()
    learn = vision_learner(dls, models.mobilenet_v3_large, metrics=accuracy)
    # print(learn.lr_find())
    # SuggestedLRs(valley=0.0006918309954926372)█████████████████------------------------------------------------------| 46.15% [12/26 00:06<00:07 1.4979]
    # learn.fine_tune(1)
    # learn.fine_tune(2, 0.00069)
    # learn.save_model()
    # learn.show_results()
    # dls.show_batch()
    # print(len(dls.train_ds), len(dls.valid_ds))
    learn.fine_tune(1)
    learn.save(MODEL_ID)
    # pyplot.show()

if __name__ == '__main__':
    fine_tune()
