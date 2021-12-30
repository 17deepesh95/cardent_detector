import csv
import random
import sys

import numpy as np
import os

import skimage
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        :param root_dir: COCO directory
        :param set_name:
        :param transform: (optional) transforms to be applied on sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)

        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class CSVDataset(Dataset):
    def __init__(self, train_file, class_list, transform=None):
        """
        :param train_file: CSV file with training annotations
        :param class_list: CSV file with class list
        :param transform:
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError(f'invalid CSV class file: {self.class_list}: {e}')

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise ValueError('invalid csv annotations file')

        self.image_names = list(self.image_data.keys())

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        :param path:
        :return:
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError(f'line {line}: format should be "class_name,class_id"')
            class_id = self._parse(class_id, int, f'line {line}:malformed class ID: {line}')

            if class_name in result:
                raise ValueError(f'line {line}: duplicate class name : {class_name}')

            result[class_name] = class_id

        return result

    def _parse(self, value, function, fmt):
        """
        parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is caught and new `ValueError` is raised with message 'fmt.format(e),
        where message `fmt.format(e)`, where e is the caught ValueError
        :param value:
        :param function:
        :param fmt:
        :return:
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise ValueError(f'line {line}: format should be "img_file,x1,y1,x2,y2,class_name')

            if img_file not in result:
                result[img_file] = []

            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, f'line {line}: malformed x1')
            y1 = self._parse(y1, int, f'line {line}: malformed y1')
            x2 = self._parse(x2, int, f'line {line}: malformed x2')
            y2 = self._parse(y2, int, f'line {line}: malformed y2')

            if x2 <= x1:
                raise ValueError(f'line {line}: x2 must be higher than x1')
            if y2 <= y1:
                raise ValueError(f'line {line}: y2 must be higher than y1')

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class -- {class_name}')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name})
        return result

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size,
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
