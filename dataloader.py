import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms


class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])

            data_enhance = ImageEnhance.Brightness(data_image)
            label_enhance = ImageEnhance.Brightness(label_image)

            if self.mode == 'train':
                # Apply various combinations of transformations as necessary

                # data_image = data_image.transpose(Image.FLIP_LEFT_RIGHT) #horiz
                # data_image = data_image.transpose(Image.FLIP_TOP_BOTTOM) #vert
                # data_image = data_image.rotate(10)
                data_image = data_enhance.enhance(0.9)  # gamma

                # label_image = label_image.transpose(Image.FLIP_LEFT_RIGHT)  # horiz
                # label_image = label_image.transpose(Image.FLIP_TOP_BOTTOM)  # vert
                # label_image = label_image.rotate(10)

            data_image = data_image.resize((128, 128))
            data_image = np.asarray(data_image)

            label_image = label_image.resize((128, 128))
            label_image = np.asarray(label_image)

            data_image = data_image / 255.0
            # label_image = label_image/255.0
            current += 1

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
