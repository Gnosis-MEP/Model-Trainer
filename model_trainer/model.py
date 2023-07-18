import torchvision.models as models

from fastai.vision.all import cnn_learner, accuracy

from model_trainer.dataloader import get_loader


def fine_tune():
    get_loader()


if __name__ == '__main__':
    main()
