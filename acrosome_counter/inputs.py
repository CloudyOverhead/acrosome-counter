# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os.path import join, exists, split
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as aug
from imgaug.parameters import Normal, TruncatedNormal
from detectron2.structures.BoxMode import XYXY_ABS
from detectron2.data import MetadataCatalog, DatasetCatalog

from acrosome_counter.bounding_box_interface import BoundingBoxes

MAP_ACROSOME = {'intact': 0, 'intermediaire': 1, 'perdu': 2}
QTY_CLASSES = len(MAP_ACROSOME)


class Dataset:
    def __init__(self, data_dir, is_training):
        self.data_dir = data_dir

        self.labels = load_labels(join(data_dir, "annotations.xml"))
        self.labels = filter_labels(self.labels, is_training)
        self.filenames = [filename for filename in self.labels.keys()]

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = plt.imread(join(self.data_dir, "images", filename))
        height, width, _ = image.shape

        annotations = []
        if self.is_training:
            boxes, classes = self.labels[filename]
            for box, class_ in zip(boxes, classes):
                annotation = {
                    "bbox": box,
                    "bbox_mode": XYXY_ABS,
                    "category_id": class_,
                }
                annotations.append(annotation)

        return {
            "file_name": filename,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": annotations,
        }

    def register(self):
        name = "train" if self.is_training else "test"
        DatasetCatalog.register(name, lambda: self)
        MetadataCatalog.get(name).set(thing_classes=MAP_ACROSOME.keys())
        return MetadataCatalog.get(name)


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
