# -*- coding: utf-8 -*-
"""Pack and unpack bounding boxes for use with `imgaug`."""

import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class BoundingBoxes(list):
    def __init__(self, images, boxes, classes):
        assert len(boxes) == len(classes)
        self.images = images
        self.boxes = boxes
        self.classes = classes
        self._in_context = False
        super().__init__(boxes)

    def __enter__(self):
        self._in_context = True
        for i, (image, boxes, classes) in self.enumerate_all():
            self[i] = BoundingBoxesOnImage(
                [
                    BoundingBox(x1, y1, x2, y2, label=class_)
                    for (y1, x1, y2, x2), class_ in zip(boxes, classes)
                ],
                shape=image.shape,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        for i, (image, boxes, _) in self.enumerate_all():
            self[i] = np.array(
                [[box.y1, box.x1, box.y2, box.x2] for box in boxes.items]
            )

    def enumerate_all(self):
        for all_contents in enumerate(zip(self.images, self, self.classes)):
            yield all_contents

    def clip(self):
        if not self._in_context:
            raise RuntimeError(
                "Use `BoudingBoxes.clip` within `with` context."
            )

        for i, (image, boxes, _) in self.enumerate_all():
            self[i] = boxes.remove_out_of_image().clip_out_of_image()
            self.classes[i] = [box.label for box in self[i]]
