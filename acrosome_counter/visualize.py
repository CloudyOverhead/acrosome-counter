# -*- coding: utf-8 -*-
"""Provide visualization utilities."""

from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer

from acrosome_counter.inputs import MAP_IDS

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


def visualize(image, outputs, metadata):
    metadata.thing_colors = CLASSES_COLORS
    visualizer = Visualizer(
        image,
        metadata=metadata,
        scale=3.0,
    )
    out = visualizer.draw_instance_predictions(
        outputs["instances"].to("cpu")
    )
    annotated_image = out.get_image()
    plt.imshow(annotated_image)
    plt.show()
