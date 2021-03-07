from torch.utils.data import Dataset
import torch
from utils.image_utils import is_png_file, load_img, is_image_file
from utils.GaussianBlur import get_gaussian_kernel
from utils.dataset_utils import Augment
import re
from WBEmulator import *

import torch.nn.functional as F


augment = Augment()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

GT = "_GT_SRGB"
NOISY = "_NOISY_SRGB"


class DataLoaderSiddCrop(Dataset):
    def __init__(self, rgb_dir, transform=None):
        super(DataLoaderSiddCrop, self).__init__()

        self.clean_filenames = []
        self.noisy_filenames = []
        dirs = sorted(os.listdir(rgb_dir))
        for directory in dirs:
            files = sorted(os.listdir(os.path.join(rgb_dir, directory)))
            for file in files:
                if re.search(GT, file, re.IGNORECASE):
                    self.clean_filenames.append(os.path.join(rgb_dir, directory, file))
                elif re.search(NOISY, file, re.IGNORECASE):
                    self.noisy_filenames.append(os.path.join(rgb_dir, directory, file))

        self.rgb_size = len(self.clean_filenames)  # get the size of input
        self.transform = transform

    def __len__(self):
        return self.rgb_size

    def __getitem__(self, index):
        tar_index = index % self.rgb_size

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean_filename = re.sub(GT, "", clean_filename, flags=re.I)
        noisy_filename = re.sub(NOISY, "", noisy_filename, flags=re.I)

        ## Load Images
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))

        if self.transform is not None:
            clean, noisy = self.transform([clean, noisy])

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, p=0.2, transform=None):
        super(DataLoaderTrain, self).__init__()

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'clean')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))

        self.clean_filenames = [os.path.join(rgb_dir, 'clean', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]

        self.rgb_size = len(self.clean_filenames)  # get the size of input
        self.transform = transform
        self.p = p

    def __len__(self):
        return self.rgb_size

    def __getitem__(self, index):
        tar_index = index % self.rgb_size

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        ## Load Images
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))

        noisy = addWB(noisy, self.p)

        if self.transform is not None:
            clean, noisy = self.transform([clean, noisy])

        return clean, noisy, clean_filename


##################################################################################################

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'clean')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))

        self.clean_filenames = [os.path.join(rgb_dir, 'clean', x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'noisy')))

        self.noisy_filenames = [os.path.join(rgb_dir, 'noisy', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2, 0, 1)

        return noisy, noisy_filename


##################################################################################################

MAX_SIZE = 512    


def divisible_by(img, factor=16):
    h, w, _ = img.shape
    img = img[:int(np.floor(h/factor)*factor), :int(np.floor(w/factor)*factor), :]
    return img


class DataLoader_NoisyData(Dataset):
    def __init__(self, rgb_dir, transform=None):
        super(DataLoader_NoisyData, self).__init__()

        rgb_files = sorted(os.listdir(rgb_dir))

        self.target_filenames = []

        #print("number of images:", len(rgb_files))
        for path in rgb_files:
            if os.path.isfile(os.path.join(rgb_dir, path)):
                if is_png_file(path) or is_image_file(path):
                    self.target_filenames.append(os.path.join(rgb_dir, path))
            else:
                files = sorted(os.listdir(os.path.join(rgb_dir, path)))
                for file in files:
                    if is_image_file(file) or is_png_file(file):
                        self.target_filenames.append(os.path.join(rgb_dir, path, file))
        
        self.tar_size = len(self.target_filenames)  # get the size of target
        self.blur, self.pad = get_gaussian_kernel(kernel_size=5, sigma=1)   # preprocessing to remove noise from the input rgb image
        self.transform = transform

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        target = np.float32(load_img(self.target_filenames[tar_index]))
        
        target = divisible_by(target, 16)

        tar_filename = os.path.split(self.target_filenames[tar_index])[-1]

        if self.transform:
            target = self.transform(target)

        target = torch.Tensor(target)
        target = target.permute(2, 0, 1)
        
        target = F.pad(target.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        target = self.blur(target).squeeze(0)

        return target, tar_filename
