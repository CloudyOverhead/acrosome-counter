# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os import walk
from os.path import join, exists, split, relpath
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from acrosome_counter.visualize import visualize

MAP_NAMES = ['intact', 'intermediaire', 'perdu']
MAP_IDS = {name: i for i, name in enumerate(MAP_NAMES)}
QTY_CLASSES = len(MAP_NAMES)


class Dataset:
    def __init__(self, data_dir, is_training):
        self.data_dir = data_dir
        self.is_training = is_training

        if is_training:
            self.images_dir = join(data_dir, "images")
            self.labels = load_labels(join(data_dir, "annotations.xml"))
            self.labels = filter_labels(self.labels, is_training)
            self.filenames = [filename for filename in self.labels.keys()]
        else:
            self.images_dir = data_dir
            self.filenames = []
            for root, _, files in walk(data_dir):
                for file in files:
                    if 'tif' in file.split(".")[-1]:
                        filepath = relpath(join(root, file), data_dir)
                        self.filenames.append(filepath)

        self.metadata = self.register()

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = join(self.images_dir, filename)
        image = plt.imread(filepath)
        height, width, _ = image.shape

        annotations = []
        if self.is_training:
            boxes, classes = self.labels[filename]
            for box, class_ in zip(boxes, classes):
                annotation = {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_,
                }
                annotations.append(annotation)

        return {
            "file_name": filepath,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": annotations,
        }

    def register(self):
        name = "train" if self.is_training else "test"
        DatasetCatalog.register(name, lambda: self)
        metadata = MetadataCatalog.get(name)
        metadata.set(thing_classes=MAP_NAMES)
        return metadata

    def quality_control(self, dataset):
        for image_info in dataset:
            image_path = image_info["file_name"]
            image = plt.imread(image_path).copy()
            annotations = image_info["annotations"]
            class_ids = [
                annotation["category_id"] for annotation in annotations
            ]
            _, quantities = np.unique(class_ids, return_counts=True)
            text_info = ", ".join(
                f"{class_}: {quantity}"
                for class_, quantity in zip(MAP_NAMES, quantities)
            )
            plt.text(0, 0, text_info)
            visualize(image, annotations, self.metadata)


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
                    box.get('xtl'),
                    box.get('ytl'),
                    box.get('xbr'),
                    box.get('ybr'),
                ]
            )
            attribute = next(box.iter('attribute'))
            attributes.append(MAP_IDS[attribute.text])
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
