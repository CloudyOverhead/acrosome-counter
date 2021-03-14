# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os import makedirs
from os.path import join

import matplotlib as mpl
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from acrosome_counter.inputs import Dataset, MAP_IDS
from acrosome_counter.build_model import build_cfg
from acrosome_counter.train import Trainer

mpl.use('TkAgg')


def main(args):
    is_training = not args.infer
    dataset = Dataset(args.data_dir, is_training)
    cfg = build_cfg(
        is_training, args.batch_size, args.learning_rate, args.qty_iters,
    )
    if is_training:
        makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    else:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .7
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get("test")
        for image_info in dataset:
            image_path = image_info["file_name"]
            image = plt.imread(image_path).copy()
            image[..., 0] = 0
            image = image.transpose(2, 0, 1).astype(np.float32)
            outputs = predictor(image)
            visualizer = Visualizer(
                image,
                metadata=metadata,
                scale=.5,
            )
            out = visualizer.draw_instance_predictions(
                outputs["instances"].to("gpu")
            )
            annotated_image = out.get_image()
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
