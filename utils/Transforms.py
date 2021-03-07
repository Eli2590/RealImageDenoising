import numpy as np
import torch
from utils.dataset_utils import Augment
from skimage.transform import resize


augment = Augment()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


class Augment(object):

    def __call__(self, sample):
        clean_img, noise_img = sample
        indx = np.random.randint(0, len(transforms_aug))
        apply_trans = transforms_aug[indx]
        clean = getattr(augment, apply_trans)(clean_img)
        noisy = getattr(augment, apply_trans)(noise_img)

        return clean, noisy


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        clean, noisy = sample

        h, w = clean.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h:
            clean = resize(clean, [new_h, w])
            noisy = resize(noisy, [new_h, w])

        if new_w > w:
            clean = resize(clean, [h, new_w])
            noisy = resize(noisy, [h, new_w])

        top = np.random.randint(0, h - new_h) if h - new_h > 0 else 0
        left = np.random.randint(0, w - new_w) if w - new_w > 0 else 0

        clean = clean[top: top + new_h, left: left + new_w]
        noisy = noisy[top: top + new_h, left: left + new_w]

        return clean, noisy


class RandomCropWb(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, target):

        h, w = target.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h:
            target = resize(target, [new_h, w])

        if new_w > w:
            target = resize(target, [h, new_w])

        top = np.random.randint(0, h - new_h) if h - new_h > 0 else 0
        left = np.random.randint(0, w - new_w) if w - new_w > 0 else 0

        target = target[top: top + new_h, left: left + new_w]

        return target


class ToTensor(object):

    def __call__(self, sample):

        clean_img, noise_img = sample
        clean = clean_img.transpose((2, 0, 1))
        noisy = noise_img.transpose((2, 0, 1))

        return torch.from_numpy(clean), torch.from_numpy(noisy)