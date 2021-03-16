# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os import makedirs
from os.path import join

from acrosome_counter.inputs import Dataset
from acrosome_counter.build_model import build_cfg
from acrosome_counter.train import Trainer
from acrosome_counter.predictor import Predictor


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
        predictor = Predictor(cfg)
        predictor(dataset, plot=args.plot)
        predictor.export(join(args.data_dir, "predictions.xml"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('-bs', '--batch_size', default=1, type=int)
    parser.add_argument('-it', '--qty_iters', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=2.5E-4, type=float)
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
