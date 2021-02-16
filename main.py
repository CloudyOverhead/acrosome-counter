# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os.path import join

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from object_detection.utils.visualization_utils import (
    visualize_boxes_and_labels_on_image_array as visualize
)

from acrosome_counter.inputs import Sequence, MAP_ACROSOME
from acrosome_counter.build_model import build_model, restore, initialize_modeL
from acrosome_counter.train import train

PRETRAINED_CHECKPOINT = "faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8"
PRETRAINED_CHECKPOINT = join(PRETRAINED_CHECKPOINT, "saved_model")

mpl.use('TkAgg')


def main(args):
    is_training = not args.infer
    sequence = Sequence(args.data_dir, args.batch_size, is_training)
    log_dir = join(".", "logs")
    model = build_model(not args.infer)
    if is_training:
        restore_from = PRETRAINED_CHECKPOINT
    else:
        restore_from = join(log_dir, "ckpt-1")
    model, checkpoint = restore(model, restore_from)
    initialize_modeL(model)
    if not args.infer:
        manager = tf.train.CheckpointManager(
            checkpoint, log_dir, max_to_keep=1,
        )
        train(model, sequence, args.epochs)
        manager.save()
    else:
        images, _ = sequence[0]
        for image in images:
            preprocessed_image, shapes = model.preprocess(image)
            predictions = model.predict(preprocessed_image, shapes)
            predictions = model.postprocess(predictions, shapes)
            predictions = {
                key: value[0].numpy() for key, value in predictions.items()
            }
            image = image[0]
            boxes = predictions['detection_boxes']
            classes = predictions['detection_classes']
            scores = predictions['detection_scores']
            category_index = {
                id: {'id': id, 'name': name}
                for name, id in MAP_ACROSOME.items()
            }
            annotated_image = image.numpy().copy()
            visualize(
                annotated_image,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.8
            )
            plt.imshow(annotated_image)
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    main(args)
