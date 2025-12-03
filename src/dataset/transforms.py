import torch
import torchvision.transforms as transforms
from src.utils.tensor.kernel import GaussianBlur

def make_jepa_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    train_transform = transforms.Compose(transform_list)
    
    test_transform = transforms.Compose([   
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization[0], std=normalization[1]),
    ])
    return train_transform, test_transform

def make_probing_transforms(
    crop_size=224,
    img_size=256,
    interpolation=3,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization[0], std=normalization[1]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization[0], std=normalization[1]),
    ])
    return train_transform, test_transform

def make_inverse_normalize(normalization=((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))):
    """Inverse normalization transform.
    Usage:
        inv_normalize = make_inverse_normalize(normalization)
        img = inv_normalize(tensor_img)
    """
    mean = torch.tensor(normalization[0])
    std = torch.tensor(normalization[1])
    inv_normalize = transforms.Normalize(
        mean=(-mean / std).tolist(),
        std=(1.0 / std).tolist()
    )
    return inv_normalize