# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os.path import join, exists, split
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence

MAP_ACROSOME = {'intact': 0, 'intermediaire': 1, 'perdu': 2}
IMAGE_SHAPE = [1024, 1024, 3]
MAX_PER_IMAGE = 100
QTY_CLASSES = 3


class Sequence(Sequence):
    def __init__(self, data_dir, batch_size, is_training):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_training = is_training

        self.labels = load_labels(join(data_dir, "annotations.xml"))
        self.labels = filter_labels(self.labels, is_training)

        self.on_epoch_end()

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, idx):
        current_filenames = self.batch_filenames[idx]
        images = []
        boxes = []
        classes = []
        for i, filename in enumerate(current_filenames):
            image = load_image(join(self.data_dir, "images", filename))
            images.append(image)
            if self.is_training:
                height, width, _ = image.shape
                current_boxes, attributes = self.labels[filename]
                current_boxes[:, [0, 2]] /= height
                current_boxes[:, [1, 3]] /= width
                current_classes = np.zeros([len(attributes), QTY_CLASSES])
                for j, class_ in enumerate(attributes):
                    current_classes[j, class_] = 1
                current_boxes = tf.convert_to_tensor(
                    current_boxes, dtype=tf.float32,
                )
                boxes.append(np.array(current_boxes))
                current_classes = tf.convert_to_tensor(
                    current_classes, dtype=tf.float32,
                )
                classes.append(current_classes)
        images = np.array(images)
        images[..., 0] = 0
        images /= 255
        images = [
            tf.expand_dims(
                tf.convert_to_tensor(image, dtype=tf.float32),
                axis=0
            )
            for image in images
        ]
        return images, (boxes, classes)

    def on_epoch_end(self):
        self.batch_filenames = np.random.choice(
            [key for key in self.labels.keys()],
            size=[len(self), self.batch_size],
            replace=False,
        )


def load_labels(annotations_path):
    data_dir, _ = split(annotations_path)
    annotations = xml_parse(annotations_path)
    root = annotations.getroot()
    labels = {}
    for image in root.findall('image'):
        name = image.attrib.get('name')
        if not exists(join(data_dir, "images", name)):
            continue
        boxes = []
        attributes = []
        for box in image:
            boxes.append(
                [
                    box.get('ytl'),
                    box.get('xtl'),
                    box.get('ybr'),
                    box.get('xbr'),
                ]
            )
            attribute = next(box.iter('attribute'))
            attributes.append(MAP_ACROSOME[attribute.text])
        boxes = np.array(boxes, dtype=np.float16)
        attributes = np.array(attributes, dtype=int)
        labels[name] = (boxes, attributes)
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
