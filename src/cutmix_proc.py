import argparse
import numpy as np
import torch
import os
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import csv

from dataloaders import CustomDataset, load_waterbirds, load_urbancars, load_cocogb

DATASET_PATHS = {
    "waterbirds": "/media/exx/HDD/rwiddhic/waterbird_complete95_forest2water2",
    "urbancars": "/media/exx/HDD/rwiddhic/urbancars_new",
    "cocogb": "/media/exx/HDD/rwiddhic/coco/images/train2017"
}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def load_raw_data(data_dir, batch_size, n_workers):
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist"
    dataset = datasets.ImageFolder(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    return dataloader

def process_cutmix_data(dataloader, save_path, alpha, beta, cutmix_prob):
    os.makedirs(save_path, exist_ok=True)
    metadata_file = os.path.join(save_path, 'metadata.csv')

    with open(metadata_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img_filename', 'split', 'y', 'place'])

        for batch_id, (batch, label, group) in tqdm(enumerate(dataloader)):
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(batch.size()[0])
                bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(), lam)
                batch[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]
                for image_id, image in enumerate(batch):
                    save_image_path = os.path.join(save_path, f'batch_{batch_id}_image_{image_id}.jpg')
                    save_image(image, save_image_path)
                
                    writer.writerow([save_image_path, 0, label[image_id].item(), group[image_id].item()])

def main():
    parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str, 
                        choices=['waterbirds', 'urbancars', 'cocogb'])
    parser.add_argument('--save_dir', default='/media/exx/HDD/rwiddhic/aug_datasets/CutMix', type=str,
                        help='directory to save cutmix images')
    parser.add_argument('--alpha', default=300, type=float,
                        help='number of new channel increases per depth (default: 300)')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=1.0, type=float,
                        help='cutmix probability')
    args = parser.parse_args()

    train_loader = None
    save_path = None
    if args.dataset == 'waterbirds':
        train_loader, _, _ = load_waterbirds(data_dir=DATASET_PATHS['waterbirds'], batch_size=args.batch_size, num_workers=args.workers, shuffle=True, augmentation=False, method='raw')
        save_path = os.path.join(args.save_dir, 'Waterbirds')
    elif args.dataset == 'urbancars':
        train_loader, _, _ = load_urbancars(data_dir=DATASET_PATHS['urbancars'], batch_size=args.batch_size, num_workers=args.workers, shuffle=True, augmentation=False, method='raw')
        save_path = os.path.join(args.save_dir, 'UrbanCars')
    elif args.dataset == 'cocogb':
        train_loader, _, _ = load_cocogb(data_dir=DATASET_PATHS['cocogb'], batch_size=args.batch_size, num_workers=args.workers, shuffle=True, augmentation=False, method='raw')
        save_path = os.path.join(args.save_dir, 'COCOGB')

    process_cutmix_data(train_loader, save_path, args.alpha, args.beta, args.cutmix_prob)
    
if __name__ == '__main__':
    main()