# -*- coding: utf-8 -*-
"""Run inference and save predictions to disk."""

from os.path import join, exists, split
from xml.etree.ElementTree import parse as xml_parse

import matplotlib as mpl
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

mpl.use('TkAgg')


class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.metadata = MetadataCatalog.get("test")

    def __call__(self, dataset, plot=True):
        for image_info in dataset:
            image_path = image_info["file_name"]
            image = plt.imread(image_path).copy()
            input_image = image[..., [2, 1]]
            outputs = super().__call__(input_image)
            visualizer = Visualizer(
                image,
                metadata=self.metadata,
                scale=.5,
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
