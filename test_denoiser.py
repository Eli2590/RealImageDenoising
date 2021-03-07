"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_validation_data
import utils
import cv2
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--test_dataset', default='sidd',
                    type=str, help='Directory of validation images')
parser.add_argument('--model_to_train', default='sidd_train', type=str, help='Model to train')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers')
parser.add_argument('--use_flickr',  action='store_true', help='Use pretrained model on flickr dataset')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

MODEL_NAME = f'flickr_{args.model_to_train}' if args.use_flickr else args.model_to_train
DATASETS_PATH = f'./datasets/{args.test_dataset}/'
RESULTS_PATH = f'./results/{MODEL_NAME}_test/denoise/'
PRETRAIND_PATH = f'./pretrained_models/{args.model_to_train}/{MODEL_NAME}.pth'

print(PRETRAIND_PATH)

use_cuda = args.use_gpu and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
if use_cuda:
    torch.cuda.manual_seed(42)

utils.mkdir(RESULTS_PATH)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


test_dataset = get_validation_data(DATASETS_PATH)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                         drop_last=False)

model_restoration = DenoiseNet()

utils.load_checkpoint(model_restoration, PRETRAIND_PATH, map_location=device)
print("===>Testing using weights: ", PRETRAIND_PATH)

model_restoration.to(device)

model_restoration = nn.DataParallel(model_restoration)

model_restoration.eval()

psnr_val_rgb = []
ssim_val_rgb = []

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].to(device)
        rgb_noisy = data_test[1].to(device)
        filenames = data_test[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored, 0, 1)

        psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, rgb_gt, 1.))
        ssim_val_rgb.append(utils.batch_SSIM(rgb_restored.permute(0, 2, 3, 1), rgb_gt.permute(0, 2, 3, 1)))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                cv2.imwrite(RESULTS_PATH + filenames[batch][:-4] + '.png', denoised_img)

psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)
print("PSNR: %.2f " % psnr_val_rgb)
print("SSIM: %.2f " % ssim_val_rgb)
results = np.asarray([[psnr_val_rgb, ssim_val_rgb]])
np.savetxt(f"{MODEL_NAME}_test.csv", results, delimiter=',', header='PSNR,SSIM', comments='')