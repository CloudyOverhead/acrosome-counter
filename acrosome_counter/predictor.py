# -*- coding: utf-8 -*-
"""Run inference and save predictions to disk."""

from os.path import join, split
from xml.etree.ElementTree import ElementTree, Element, SubElement

import matplotlib as mpl
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import pandas as pd

from acrosome_counter.inputs import MAP_NAMES
from acrosome_counter.visualize import visualize

mpl.use('TkAgg')


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = MetadataCatalog.get("test")
        self.results = {}
        self.dataset = None

    def __call__(self, dataset, plot=True):
        self.dataset = dataset
        self.results = {}
        for image_info in dataset:
            image_path = image_info["file_name"]
            image = plt.imread(join(dataset.images_dir, image_path)).copy()
            input_image = image[..., [2, 1]]
            outputs = super().__call__(input_image)
            self.results[image_path] = outputs['instances']
            if plot:
                visualize(image, outputs["instances"].to("cpu"), self.metadata)

    def export_xml(self, dest_path=None):
        if dest_path is None:
            dest_path = join(self.dataset.data_dir, "predictions.xml")

        root = Element("annotations")

        SubElement(root, "version").text = "1.1"

        for id, (image_path, outputs) in enumerate(self.results.items()):
            height, width = outputs.image_size
            image_element = SubElement(
                root,
                "image",
                height=str(height),
                width=str(width),
                name=image_path,
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
        for image_path, outputs in self.results.items():
            _, image_name = split(image_path)
            quantities.loc[image_name] = [0, 0, 0]
            classes = outputs.pred_classes
            scores = outputs.scores
            for class_, score in zip(classes, scores):
                class_ = class_.data.item()
                score = score.data.item()
                class_name = MAP_NAMES[class_]
                quantities.loc[image_name, class_name] += 1
        quantities.to_csv(dest_path)
