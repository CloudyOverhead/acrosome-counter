# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os.path import join

import matplotlib as mpl
from matplotlib import pyplot as plt

from acrosome_counter.inputs import Sequence, MAP_ACROSOME
from acrosome_counter.build_model import build_cfg

mpl.use('TkAgg')


def main(args):
    is_training = not args.infer
    sequence = Sequence(args.data_dir, args.batch_size, is_training)
    cfg = build_cfg(is_training, args.learning_rate, args.qty_iters)
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
    parser.add_argument('-bs', '--batch_size', default=1, type=int)
    parser.add_argument('-it', '--qty_iters', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=2.5E-4, type=float)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    main(args)
