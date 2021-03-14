# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os.path import join

import matplotlib as mpl
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.compat.v2.train import Checkpoint
from object_detection.utils.visualization_utils import (
    visualize_boxes_and_labels_on_image_array as visualize
)

from acrosome_counter.inputs import Sequence, MAP_ACROSOME
from acrosome_counter.build_model import build_model, restore

PRETRAINED_CHECKPOINT = "faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8"
PRETRAINED_CHECKPOINT = join(PRETRAINED_CHECKPOINT, 'checkpoint', 'ckpt-0')

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
    restore(model, restore_from, is_training)
    if is_training:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    else:
        images, _ = sequence[0]
        for image in images:
            predictions = model.call(image)
            predictions = {
                key: value[0].numpy() for key, value in predictions.items()
            }
            boxes = predictions['detection_boxes']
            classes = predictions['detection_classes'].astype(int).tolist()
            scores = predictions['detection_scores']
            category_index = {
                id: {'id': id, 'name': name}
                for name, id in MAP_ACROSOME.items()
            }
            annotated_image = image[0].numpy().copy()
            visualize(
                annotated_image,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.8,
                max_boxes_to_draw=None,
            )
            annotated_image /= 255
            plt.imshow(annotated_image)
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1E-3, type=float)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    main(args)
