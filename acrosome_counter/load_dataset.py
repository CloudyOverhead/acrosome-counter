# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os import walk
from os.path import join, exists, split, relpath
from xml.etree.ElementTree import parse as xml_parse

import numpy as np
from matplotlib import pyplot as plt
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.instances import Instances

from .visualize import visualize

MAP_NAMES = ['intact', 'intermediaire', 'perdu']
MAP_IDS = {name: i for i, name in enumerate(MAP_NAMES)}
QTY_CLASSES = len(MAP_NAMES)

if isinstance(DatasetCatalog, type):
    DatasetCatalog = DatasetCatalog()
if isinstance(MetadataCatalog, type):
    MetadataCatalog = MetadataCatalog()


class Dataset:
    def __init__(self, data_dir, is_training):
        self.data_dir = data_dir
        self.is_training = is_training

        if is_training:
            self.images_dir = join(data_dir, "images")
            self.labels = self.load_labels("annotations.xml")
            self.labels = self.filter_labels(self.labels, is_training)
            self.filenames = [filename for filename in self.labels.keys()]
        else:
            self.images_dir = data_dir
            if exists(join(data_dir, "predictions.xml")):
                self.labels = self.load_labels("predictions.xml")
            else:
                self.labels = None
            self.filenames = []
            for root, _, files in walk(data_dir):
                for file in files:
                    if 'tif' in file.split(".")[-1]:
                        filename = relpath(join(root, file), data_dir)
                        self.filenames.append(filename)

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
        try:
            boxes, classes, scores = self.labels[filename]
            for box, class_, score in zip(boxes, classes, scores):
                annotation = {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_,
                    "score": score,
                }
                annotations.append(annotation)
        except KeyError:
            pass

        return {
            "filename": filename,
            "filepath": filepath,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": annotations,
        }

    def load_labels(self, annotations_file):
        annotations_path = join(self.data_dir, annotations_file)
        data_dir, _ = split(annotations_path)
        annotations = xml_parse(annotations_path)
        root = annotations.getroot()
        labels = {}
        for image in root.findall('image'):
            name = image.attrib.get('name')
            if not exists(join(self.images_dir, name)):
                continue
            boxes = []
            classes = []
            scores = []
            for box in image:
                boxes.append(
                    [
                        box.get('xtl'),
                        box.get('ytl'),
                        box.get('xbr'),
                        box.get('ybr'),
                    ]
                )
                attributes = box.iter('attribute')
                attribute = next(attributes)
                classes.append(MAP_IDS[attribute.text])
                try:
                    attribute = next(attributes)
                    scores.append(float(attribute.text))
                except StopIteration:
                    scores.append(1.)
            boxes = np.array(boxes, dtype=np.float16)
            classes = np.array(classes, dtype=int)
            scores = np.array(scores, dtype=np.float16)
            labels[name] = (boxes, classes, scores)
        return labels

    def filter_labels(self, labels, is_training):
        keep_names = []
        for name, label in labels.items():
            boxes, _, _ = label
            has_labels = boxes.size != 0
            do_keep = (
                (has_labels and is_training)
                or (not has_labels and not is_training)
            )
            if do_keep:
                keep_names.append(name)
        return {name: labels[name] for name in keep_names}

    def register(self):
        name = "train" if self.is_training else "test"
        if name in DatasetCatalog.keys():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda: self)
        metadata = MetadataCatalog.get(name)
        metadata.set(thing_classes=MAP_NAMES)
        return metadata

    def review(self):
        for image_info in self:
            filepath = image_info["filepath"]
            image = plt.imread(filepath).copy()
            annotations = image_info["annotations"]
            boxes = [annotation["bbox"] for annotation in annotations]
            ids = [annotation["category_id"] for annotation in annotations]
            _, quantities = np.unique(ids, return_counts=True)
            text_info = ", ".join(
                f"{class_}: {quantity}"
                for class_, quantity in zip(MAP_NAMES, quantities)
            )
            scores = [annotation["score"] for annotation in annotations]
            plt.title(text_info)
            image_size = (image_info["height"], image_info["width"])
            instances = Instances(
                image_size=image_size,
                pred_boxes=boxes,
                pred_classes=ids,
                scores=scores,
            )
            visualize(image, instances, self.metadata)
