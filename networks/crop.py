import torch.nn as nn
from utils.Transforms import RandomCropWb
from torchvision.transforms import Resize
import numpy as np


class Crop(nn.Module):
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, img_clean, img_noisy):
        h, w = img_clean.shape[1:3]
        new_h, new_w = self.crop_size
        target_clean = img_clean
        target_noisy = img_noisy
        if new_h > h:
            target_clean = Resize([new_h, w])(target_clean)
            target_noisy = Resize([new_h, w])(target_noisy)
        if new_w > w:
            target_clean = Resize([h, new_w])(target_clean)
            target_noisy = Resize([h, new_w])(target_noisy)

        top = np.random.randint(0, h - new_h) if h - new_h > 0 else 0
        left = np.random.randint(0, w - new_w) if w - new_w > 0 else 0

        target_clean = target_clean[:, top: top + new_h, left: left + new_w]
        target_noisy = target_noisy[:, top: top + new_h, left: left + new_w]
        return target_clean, target_noisy
