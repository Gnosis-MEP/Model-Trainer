
from torchvision import transforms
from torchvision.transforms.transforms import InterpolationMode


def get_transforms(mode='TRAIN'):
    transform = None
    if mode.upper() == 'TRAIN':
        transform = transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transform