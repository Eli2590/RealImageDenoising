import os
import argparse
from tqdm import tqdm

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize
from utils.Transforms import *

from networks.denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_training_data
import utils
import cv2
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description='RGB denoising training on the train set of SIDD')
parser.add_argument('--dataset',  default='flickr',
                    type=str, help='dataset to use for training')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--lr', default=10e-4, type=float, help='Initial learning rate')
parser.add_argument('--b1', default=0.9, type=float, help='beta1 for adam optimizer')
parser.add_argument('--b2', default=0.999, type=float, help='beta2 for adam optimizer')
parser.add_argument('--epochs', default=65, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--use_gpu', action='store_true', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
parser.add_argument('--use_test_set', action='store_true', help='If provided splits the dataset by 90:5:5 else 90:10')
parser.add_argument('--use_flickr',  action='store_true', help='Use pretrained model on flickr dataset')
parser.add_argument('--p', default=0.4, type=float, help='probability to add wb noise')

args = parser.parse_args()

MODEL_NAME = f'flickr_{args.dataset}' if args.use_flickr else args.dataset
DATASETS_PATH = f'./datasets/{args.dataset}/'
RESULTS_PATH = f'./results/{args.dataset}/denoise/'
PRETRAIND_PATH = f'./pretrained_models/{args.dataset}/'


use_cuda = args.use_gpu and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

np.random.seed(42)
torch.manual_seed(42)
if use_cuda:
    torch.cuda.manual_seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdirs([RESULTS_PATH, PRETRAIND_PATH])

model_restoration = DenoiseNet()
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)

dataset = get_training_data(DATASETS_PATH, args.p, transforms=Compose([RandomCrop(64), Augment(), ToTensor()]))
dataset_size = len(dataset)

train_dataset_size = int(dataset_size*0.9)
val_dataset_size = (len(dataset) - train_dataset_size) // 2 if args.use_test_set else len(dataset) - train_dataset_size
test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_dataset_size, val_dataset_size, test_dataset_size],
    torch.Generator().manual_seed(42)
)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, drop_last=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers, drop_last=False)

if use_cuda:
    torch.cuda.empty_cache()

optimizer = Adam(model_restoration.parameters(), lr=args.lr, betas=(args.b1, args.b2))
criterion = nn.MSELoss().to(device)

cur_epoch = 0
total_train_losses = []
validation_loss = []
checkpoint = None

if os.path.exists(os.path.join(PRETRAIND_PATH, f"{MODEL_NAME}.pth")):
    checkpoint = torch.load(os.path.join(PRETRAIND_PATH, f"{MODEL_NAME}.pth"))
    cur_epoch = checkpoint["epoch"] + 1
elif args.use_flickr:
    model_path = os.path.join("./pretrained_models/flickr/flickr.pth")
    print(model_path)
    assert os.path.exists(model_path)
    checkpoint = torch.load(model_path)
    cur_epoch = 0

if checkpoint:
    model_restoration.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    total_train_losses = checkpoint["train_loss"]
    validation_loss = checkpoint["val_loss"]


def test(model_parameters_path, criterion, data):
    print("===>Testing denoiser:")
    model_path = os.path.join(model_parameters_path, f"{MODEL_NAME}.pth")
    print(model_path)
    assert os.path.exists(model_path)
    params = torch.load(model_path)
    model = DenoiseNet().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(params["model_state_dict"])
    model.eval()
    losses = []
    with torch.no_grad():
        for i, test_data in enumerate(tqdm(data), 0):
            rgb_gt = test_data[0].to(device)
            rgb_noisy = test_data[1].to(device)
            rgb_restored = model(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            loss = criterion(rgb_restored, rgb_gt)
            losses.append(loss.data.item())
            if args.save_images:
                for j in range(test_data[0].shape[0]):
                    dirs = ["restored", "noisy", "clean"]
                    paths = [os.path.join(RESULTS_PATH, x) for x in dirs]
                    utils.mkdirs(paths)
                    cv2.imwrite(os.path.join(paths[0], test_data[2][j]), img_as_ubyte(rgb_restored[j].permute(1, 2, 0)
                                                                                      .cpu().detach().numpy()))
                    cv2.imwrite(os.path.join(paths[1], test_data[2][j]), img_as_ubyte(rgb_noisy[j].permute(1, 2, 0).cpu().detach().numpy()))
                    cv2.imwrite(os.path.join(paths[2], test_data[2][j]), img_as_ubyte(rgb_gt[j].permute(1, 2, 0).cpu().detach().numpy()))

    return np.mean(losses)


def evaluate(model, criterion, data):
    losses = []
    model.eval()
    for i, train_data in enumerate(tqdm(data), 0):
        with torch.no_grad():
            rgb_gt = train_data[0].to(device)
            rgb_noisy = train_data[1].to(device)
            rgb_restored = model(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            loss = criterion(rgb_restored, rgb_gt)
            losses.append(loss.data.item())
    return np.mean(losses)


def train(model, optimizer, criterion, data, total_train_losses, validation_loss, cur_epoch):
    print("===>Training denoiser:")
    for epoch in range(cur_epoch, args.epochs):
        losses = []
        for i, train_data in enumerate(tqdm(data), 0):
            model.train()

            rgb_gt = train_data[0].to(device)
            rgb_noisy = train_data[1].to(device)

            rgb_restored = model(rgb_noisy)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            loss = criterion(rgb_restored, rgb_gt)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.data.item())

        total_train_losses.append(np.mean(losses))
        validation_loss.append(evaluate(model, criterion, val_loader))

        if (epoch + 1) % 5 == 0:
            save_params = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_train_losses,
                'val_loss': validation_loss
            }

            torch.save(save_params, os.path.join(PRETRAIND_PATH, f"{MODEL_NAME}.pth"))

        print(f"Epoch: {epoch+1}/{args.epochs}\ttrain loss: {round(total_train_losses[-1], 4)}"
              f"\tval loss: {round(validation_loss[-1], 4)}")
        utils.plot_loss(f'{MODEL_NAME}_', args.epochs, [total_train_losses, validation_loss])

        if (epoch + 1) % 25 == 0:
            for g in optimizer.param_groups:
                g['lr'] /= 10

    return total_train_losses, validation_loss


total_train_losses, validation_loss = train(model_restoration, optimizer, criterion, train_loader,
                                            total_train_losses, validation_loss, cur_epoch)
if args.use_test_set:
    test_loss = test(PRETRAIND_PATH, criterion, test_loader)
    print(f"Test Loss:\t{test_loss}")
losses = np.asarray([total_train_losses, validation_loss]).transpose()
utils.plot_loss(f'{args.dataset}_', args.epochs, [total_train_losses, validation_loss])
np.savetxt(f"{args.dataset}.csv", losses, delimiter=',', header='train,val', comments='')
