import torch
from PIL import ImageFilter

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
        Usage:
            >>> transform = GaussianBlur()
            >>> transformed_img = transform(img)
        :param p: probability of applying the Gaussian blur
        :param radius_min: minimum radius for Gaussian blur
        :param radius_max: maximum radius for Gaussian blur
        """
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        radius_value = radius.item() if hasattr(radius, "item") else float(radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius_value))

