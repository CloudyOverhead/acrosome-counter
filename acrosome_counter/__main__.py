# -*- coding: utf-8 -*-
"""Launch training and inference."""

from os import makedirs
from argparse import ArgumentParser

from acrosome_counter.inputs import Dataset
from acrosome_counter.build_model import build_cfg
from acrosome_counter.train import Trainer
from acrosome_counter.predictor import Predictor


def main(args):
    if args.train:
        is_training = True
        dataset = Dataset(args.data_dir, is_training)
        cfg = build_cfg(
            is_training, args.batch_size, args.learning_rate, args.qty_iters,
        )
        makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    if args.infer:
        is_training = False
        dataset = Dataset(args.data_dir, is_training)
        cfg = build_cfg(
            is_training, args.batch_size, args.learning_rate, args.qty_iters,
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .3
        predictor = Predictor(cfg)
        predictor(dataset, plot=args.plot)
        predictor.export_xml()
        predictor.export_csv()

    if args.review:
        dataset = Dataset(args.data_dir, is_training=False)
        dataset.review()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('-bs', '--batch_size', default=1, type=int)
    parser.add_argument('-it', '--qty_iters', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=2.5E-4, type=float)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--review', action='store_true')
    args = parser.parse_args()
    main(args)
