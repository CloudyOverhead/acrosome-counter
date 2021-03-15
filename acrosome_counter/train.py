# -*- coding: utf-8 -*-

from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from imgaug import augmenters as aug
from imgaug.parameters import Normal, TruncatedNormal
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

AUGMENTER = aug.Sequential(
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

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=augment)


def augment(record):
    record = deepcopy(record)
    image = plt.imread(record["file_name"])
    annotations = record["annotations"]

    boxes = [annotation["bbox"] for annotation in annotations]
    classes = [annotation["category_id"] for annotation in annotations]
    boxes = BoundingBoxesOnImage(
        [
            BoundingBox(*box, label=class_)
            for box, class_ in zip(boxes, classes)
        ],
        shape=image.shape,
    )
    image, boxes = AUGMENTER(image=image, bounding_boxes=boxes)
    classes = [bbox.label for bbox in boxes.bounding_boxes]
    boxes = np.array([[box.x1, box.y1, box.x2, box.y2] for box in boxes.items])
    image = image[..., [1, 2]]
    image = image.transpose(2, 0, 1).astype(np.float32)

    annotations = [
        {"bbox": box, "bbox_mode": BoxMode.XYXY_ABS, "category_id": class_}
        for box, class_ in zip(boxes, classes)
    ]
    record["image"] = torch.as_tensor(image)
    instances = utils.annotations_to_instances(annotations, image.shape[:2])
    record["instances"] = utils.filter_empty_instances(instances)
    return record
