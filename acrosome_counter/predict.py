# -*- coding: utf-8 -*-
"""Run inference and save predictions to disk."""

from os.path import join
from xml.etree.ElementTree import ElementTree, Element, SubElement

import matplotlib as mpl
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
import pandas as pd
import numpy as np

from .load_dataset import MAP_NAMES, MetadataCatalog
from .visualize import visualize

mpl.use('TkAgg')

DEFAULT_ZOOM = 20


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = MetadataCatalog.get("test")
        self.results = {}
        self.dataset = None

    def __call__(self, dataset, iou_threshold=.3, plot=True):
        self.dataset = dataset
        self.results = {}
        for image_info in dataset:
            filename = image_info["filename"]
            filepath = image_info["filepath"]
            image = plt.imread(filepath).copy()
            input_image = image[..., [2, 1]]
            outputs = super().__call__(input_image)
            instances = outputs['instances'].to("cpu")
            instances = filter_iou(instances, threshold=.3)
            self.results[filename] = instances
            if plot:
                visualize(image, instances, self.metadata)

    def export_xml(self, dest_path=None):
        if dest_path is None:
            dest_path = join(self.dataset.data_dir, "predictions.xml")

        root = Element("annotations")

        SubElement(root, "version").text = "1.1"

        for id, (filepath, outputs) in enumerate(self.results.items()):
            height, width = outputs.image_size
            image_element = SubElement(
                root,
                "image",
                height=str(height),
                width=str(width),
                name=filepath,
                id=str(id),
            )
            boxes = outputs.pred_boxes
            classes = outputs.pred_classes
            scores = outputs.scores
            for box, class_, score in zip(boxes, classes, scores):
                box = box.data.cpu().numpy()
                class_ = class_.data.item()
                score = score.data.item()
                x1, y1, x2, y2 = box
                box_element = SubElement(
                    image_element,
                    "box",
                    z_order="0",
                    ybr=str(y2),
                    xbr=str(x2),
                    ytl=str(y1),
                    xtl=str(x1),
                    source="manual",
                    occluded="0",
                    label="acrosome",
                )
                attribute = SubElement(
                    box_element, "attribute", name="acrosome",
                )
                attribute.text = MAP_NAMES[class_]
                attribute = SubElement(
                    box_element, "attribute", name="score",
                )
                attribute.text = str(score)

        file = ElementTree(root)
        file.write(dest_path)

    def export_csv(self, dest_path=None):
        if dest_path is None:
            dest_path = join(self.dataset.data_dir, "predictions.csv")

        quantities = pd.DataFrame(
            [], columns=["intact", "intermediaire", "perdu"],
        )
        for filename, outputs in self.results.items():
            quantities.loc[filename] = [0, 0, 0]
            classes = outputs.pred_classes
            scores = outputs.scores
            for class_, score in zip(classes, scores):
                class_ = class_.data.item()
                score = score.data.item()
                class_name = MAP_NAMES[class_]
                quantities.loc[filename, class_name] += 1
        quantities.to_csv(dest_path, sep=';')


def adjust_zoom(cfg, target_zoom):
    cfg.INPUT.MIN_SIZE_TEST *= DEFAULT_ZOOM / target_zoom
    cfg.INPUT.MAX_SIZE_TEST *= DEFAULT_ZOOM / target_zoom


def filter_iou(instances, threshold=.3):
    boxes = np.array([box.data.cpu().numpy() for box in instances.pred_boxes])
    scores = np.array([score.data.item() for score in instances.scores])
    ious = iou(boxes)
    assert np.allclose(ious, ious.T)
    ious[np.tri(*ious.shape, dtype=bool)] = 0
    b1, b2 = np.array(np.nonzero(ious > threshold))
    b1, b2 = np.append(b1, b1), np.append(b2, b1)
    b1, b2 = b1[np.argsort(b1)], b2[np.argsort(b1)]
    mask = np.ones_like(scores, dtype=bool)
    for matches in group_by(b2, b1):
        current_scores = scores[matches]
        sort_idx = np.argsort(current_scores)
        mask[matches[sort_idx][:-1]] = False
    instances = instances[mask]
    return instances


def iou(boxes):
    """Compute the Intersection over Union (IoU) of two bounding boxes.

    Modified from https://stackoverflow.com/a/42874377/8376138 and
    https://stackoverflow.com/a/58108241/8376138.
    """
    x1, y1, x2, y2 = boxes.T[..., None]
    assert (x1 <= x2).all()
    assert (y1 <= y2).all()

    x1i = np.maximum(x1, x1.T)
    y1i = np.maximum(y1, y1.T)
    x2i = np.minimum(x2, x2.T)
    y2i = np.minimum(y2, y2.T)

    intersection_area = (x2i-x1i+1) * (y2i-y1i+1)

    bb1_area = (x2-x1+1) * (y2-y1+1)
    bb2_area = (x2.T-x1.T+1) * (y2.T-y1.T+1)

    iou = intersection_area / (bb1_area+bb2_area-intersection_area)
    iou[(x2i < x1i) | (y2i < y1i)] = 0

    assert (iou >= 0.0).all()
    assert (iou <= 1.0).all()
    return iou


def group_by(a, groups):
    """Group array by a the first column.

    Modified from https://stackoverflow.com/a/43094244/8376138.
    """
    return np.split(a, np.unique(groups, return_index=True)[1][1:])
