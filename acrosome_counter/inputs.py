# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os.path import join, exists, split
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as aug
from imgaug.parameters import Normal, TruncatedNormal

from acrosome_counter.bounding_box_interface import BoundingBoxes

MAP_ACROSOME = {'intact': 0, 'intermediaire': 1, 'perdu': 2}
QTY_CLASSES = len(MAP_ACROSOME)


class Sequence:
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
        try:
            current_filenames = next(self.batch_filenames)
        except StopIteration:
            self.on_epoch_end()
            current_filenames = next(self.batch_filenames)
        images = []
        boxes = []
        classes = []
        for i, filename in enumerate(current_filenames):
            image = load_image(join(self.data_dir, "images", filename))
            images.append(image)
            if self.is_training:
                current_boxes, current_classes = self.labels[filename]
                boxes.append(current_boxes)
                classes.append(current_classes)

        if self.is_training:
            images, boxes, classes = augment(images, boxes, classes)

        for i, (image, current_boxes) in enumerate(zip(images, boxes)):
            height, width, _ = image.shape
            boxes[i][:, [0, 2]] /= height
            boxes[i][:, [1, 3]] /= width

        images = np.array(images, dtype=np.float32)
        images[..., 0] = 0
        classes = [
            one_hot_encode(current_classes, QTY_CLASSES)
            for current_classes in classes
        ]
        return images, (boxes, classes)

    def on_epoch_end(self):
        self.batch_filenames = np.random.choice(
            [key for key in self.labels.keys()],
            size=[len(self), self.batch_size],
            replace=False,
        )
        self.batch_filenames = iter(self.batch_filenames)


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


def augment(images, boxes, classes):
    sequential = aug.Sequential(
        [
            aug.Add(Normal(0, 10), per_channel=True),
            aug.Multiply(TruncatedNormal(1, .1, low=.5, high=1.5)),
            aug.GaussianBlur((0, 2)),
            aug.Fliplr(.5),
            aug.Flipud(.5),
            aug.Affine(
                scale={
                    'x': TruncatedNormal(1, .1, low=.8, high=1.2),
                    'y': TruncatedNormal(1, .1, low=.8, high=1.2),
                },
                translate_percent={
                    'x': TruncatedNormal(0, .1, low=-.2, high=.2),
                    'y': TruncatedNormal(0, .1, low=-.2, high=.2),
                },
                rotate=(-180, 180),
                shear={
                    'x': TruncatedNormal(0, 10, low=-30, high=30),
                    'y': TruncatedNormal(0, 10, low=-30, high=30),
                },
                cval=(0, 255),
            ),
            aug.CoarseSaltAndPepper((.01, .1), size_percent=(5E-3, 5E-2)),
        ]
    )

    boxes = BoundingBoxes(images, boxes, classes)
    with boxes:
        images, boxes[:] = sequential(images=images, bounding_boxes=boxes)
        boxes.clip()
    return images, boxes, boxes.classes


def one_hot_encode(classes, qty_classes):
    encoded = np.zeros([len(classes), qty_classes])
    for i, class_ in enumerate(classes):
        encoded[i, class_] = 1
    return encoded
