# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os.path import join
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence

MAP_ACROSOME = {'intact': 0, 'intermediaire': 1, 'perdu': 2}


class Sequence(Sequence):
    def __init__(self, data_dir, batch_size, is_training):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.labels = load_labels(join(data_dir, "annotations.xml"))
        self.labels = filter_labels(self.labels, is_training)

        self.on_epoch_end()

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, idx):
        current_filenames = self.batch_filenames[idx]
        images = []
        labels = []
        for filename in current_filenames:
            image = load_image(join(self.data_dir, "images", filename))
            label = self.labels[filename]
            images.append(image)
            labels.append(label)
        return images, labels

    def on_epoch_end(self):
        self.batch_filenames = np.random.choice(
            [key for key in self.labels.keys()],
            size=[len(self), self.batch_size],
            replace=False,
        )


def load_labels(annotations_path):
    annotations = xml_parse(annotations_path)
    root = annotations.getroot()
    labels = {}
    for image in root.findall('image'):
        boxes = []
        attributes = []
        for box in image:
            boxes.append(
                [
                    box.get('xtl'),
                    box.get('ytl'),
                    box.get('xbr'),
                    box.get('ybr'),
                ]
            )
            attribute = next(box.iter('attribute'))
            attributes.append(MAP_ACROSOME[attribute.text])
        boxes = np.array(boxes, dtype=np.float16)
        attributes = np.array(attributes, dtype=int)
        labels[image.attrib.get('name')] = (boxes, attributes)
    return labels


def filter_labels(labels, is_training):
    keep_names = []
    for name, label in labels.items():
        boxes, _ = label
        has_labels = boxes.size != 0
        do_keep = (
            (has_labels and is_training)
            or (not has_labels and not is_training)
        )
        if do_keep:
            keep_names.append(name)
    return {name: labels[name] for name in keep_names}


def load_image(filename):
    image = plt.imread(filename)
    return image
