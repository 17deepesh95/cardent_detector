import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Normalizer, Augmenter, Resizer, CSVDataset, AspectRatioBasedSampler, \
    collater

assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations.')
    parser.add_argument('--csv_classes', help='Path to file containing class list.')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations')

    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO')
        if parser.csv_classes in None:
            raise ValueError('Must provide --csv_classes when training on COCO')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco)')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
