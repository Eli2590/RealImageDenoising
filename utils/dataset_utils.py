import numpy as np


def crop(rgb: np.ndarray, offset: list, crop_size: list = None) -> np.ndarray:
    """
    Crop RGB image.

    Parameters
    ----------
    rgb : np.ndarray in shape (H, W)
        RGB image to be cropped.'
    crop_size : shape of the cropped image [h_size, w_size]
    offset: offset list of the starting pixel for crop [h_offset, w_offset]
    """
    h_size, w_size = crop_size if crop_size is not None else [128, 128]
    h, w = rgb.shape[:2]
    h_offset, w_offset = offset

    if not isinstance(rgb, np.ndarray) or len(rgb.shape) != 3:
        raise ValueError('rgb should be a 3-dimensional numpy.ndarray!')
    if h_offset + h_size >= h:
        raise IndexError('Cropping height out of the bounds of the image')
    if w_offset + w_size >= h:
        raise IndexError('Cropping width out of the bounds of the image')

    out = rgb[h_offset:h_offset + h_size, w_offset:w_offset + w_size, :]
    return out


def aug(rgb: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool) -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.

    Parameters
    ----------

    rgb : np.ndarray in shape (H, W)
        Bayer rgb image to be augmented. H and W must be even numbers.
    flip_h : bool
        If True, do vertical flip.
    flip_w : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    """

    if not isinstance(rgb, np.ndarray) or len(rgb.shape) != 3:
        raise ValueError('rgb should be a 3-dimensional numpy.ndarray')
    if rgb.shape[0] % 2 == 1 or rgb.shape[1] % 2 == 1:
        raise ValueError('rgb should have even number of height and width!')

    out = rgb
    if flip_h:
        out = out[::-1, :, :]
    if flip_w:
        out = out[:, ::-1, :]
    if transpose:
        out_1 = out[:, :, 0].T
        out_2 = out[:, :, 1].T
        out_3 = out[:, :, 2].T
        out = np.dstack((out_1, out_2, out_3))

    return out


class Augment:
    """
    Inputs:
        rgb: shape (H,W,3)
    Outputs:
        rgb: shape (H,W,3)
    """
    def __init__(self):
        pass

    def transform0(self, rgb):
        return rgb.copy()

    def transform1(self, rgb):
        rgb_flip_v = aug(rgb, flip_h=True, flip_w=False, transpose=False)
        return rgb_flip_v.copy()

    def transform2(self, rgb):
        rgb_flip_h = aug(rgb, flip_h=False, flip_w=True, transpose=False)
        return rgb_flip_h.copy()

    def transform3(self, rgb):
        rgb_flip_h = aug(rgb, flip_h=False, flip_w=False, transpose=True)
        return rgb_flip_h.copy()
