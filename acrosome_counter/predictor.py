# -*- coding: utf-8 -*-
"""Run inference and save predictions to disk."""

from os.path import join, exists, split
from xml.etree.ElementTree import parse as xml_parse

import matplotlib as mpl
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from acrosome_counter.inputs import MAP_IDS

mpl.use('TkAgg')

CLASSES_COLORS = [(0, 1, 0), (1, 1, 1), (0, 0, 1)]


class Visualizer(Visualizer):
    def overlay_instances(self, **kwargs):
        labels = [label.split(" ")[0] for label in kwargs['labels']]
        percentages = [label.split(" ")[1] for label in kwargs['labels']]
        category_ids = [MAP_IDS[label] for label in labels]
        kwargs['assigned_colors'] = [
            self.metadata.thing_colors[c] for c in category_ids
        ]
        kwargs['labels'] = percentages
        return super().overlay_instances(**kwargs)


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = MetadataCatalog.get("test")
        self.metadata.thing_colors = CLASSES_COLORS

    def __call__(self, dataset, plot=True):
        for image_info in dataset:
            image_path = image_info["file_name"]
            image = plt.imread(image_path).copy()
            input_image = image[..., [2, 1]]
            outputs = super().__call__(input_image)
            visualizer = Visualizer(
                image,
                metadata=self.metadata,
                scale=3.0,
            )
            out = visualizer.draw_instance_predictions(
                outputs["instances"].to("cpu")
            )
            annotated_image = out.get_image()
            if plot:
                plt.imshow(annotated_image)
                plt.show()

    def export(self):
        pass
