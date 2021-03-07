import os
import argparse
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader

from networks.crop import Crop
from dataloaders.dataset_rgb import DataLoaderSiddCrop
import utils
import cv2
from skimage import img_as_ubyte
from utils.Transforms import *

parser = argparse.ArgumentParser(description='From RGB images, generate cropped RGB images')
parser.add_argument('--input_dir', default='./datasets/flickr/',
                    type=str, help='Directory of clean RGB images')
parser.add_argument('--result_dir', default='./results/sidd_train/rgb/',
                    type=str, help='Directory for results')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--crop_size', default=128, type=int, help='crop_size')
parser.add_argument('--num_of_crops', default=250, type=int, help='Number of crops per image')
parser.add_argument('--use_gpu', action='store_true', help='use gpu or not')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')

args = parser.parse_args()

dataset = DataLoaderSiddCrop(args.input_dir)
dataloarder = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=args.num_workers)

use_cuda = args.use_gpu and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(os.path.join(args.result_dir, 'clean'))
utils.mkdir(os.path.join(args.result_dir, 'noisy'))

model = Crop((args.crop_size, args.crop_size)).to(device)

if use_cuda:
    model = nn.DataParallel(model)

for i, data in enumerate(tqdm(dataloarder)):
    clean_img = data[0]
    noisy_img = data[1]
    clean_filenames = data[2]
    noisy_filenames = data[3]
    for j in range(args.num_of_crops):
        clean_patch, noisy_patch = model(clean_img, noisy_img)
        for k in range(clean_img.shape[0]):
            clean_img_crop = img_as_ubyte(clean_patch[k, :, :, :].squeeze().cpu().detach().numpy())
            noisy_img_crop = img_as_ubyte(noisy_patch[k, :, :, :].squeeze().cpu().detach().numpy())

            cv2.imwrite(os.path.join(args.result_dir, 'clean', clean_filenames[k][:-4]+f'_{j}.png'), clean_img_crop)
            cv2.imwrite(os.path.join(args.result_dir, 'noisy', noisy_filenames[k][:-4]+f'_{j}.png'), noisy_img_crop)


