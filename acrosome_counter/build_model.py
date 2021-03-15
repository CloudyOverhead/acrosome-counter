# -*- coding: utf-8 -*-
"""Build the network."""

from os.path import join

from detectron2 import model_zoo
from detectron2.config import get_cfg

from acrosome_counter.inputs import QTY_CLASSES

LOG_DIR = join(".", "logs")
PRETRAINED_CHECKPOINT = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)


def build_cfg(is_training, batch_size, learning_rate, qty_iters):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.FORMAT = "RGB"
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.OUTPUT_DIR = LOG_DIR
    if is_training:
        cfg.MODEL.WEIGHTS = PRETRAINED_CHECKPOINT
    else:
        cfg.MODEL.WEIGHTS = join(LOG_DIR, "model_final.pth")

    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = qty_iters
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16]]
    cfg.MODEL.PIXEL_MEAN = [116.28, 103.53]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = QTY_CLASSES
    cfg.MODEL.RPN.IN_FEATURES = ['p5', 'p6']

    cfg.TEST.DETECTIONS_PER_IMAGE = 256

    return cfg
