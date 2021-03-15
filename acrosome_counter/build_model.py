# -*- coding: utf-8 -*-
"""Build the network."""

from os.path import join

from detectron2 import model_zoo
from detectron2.config import get_cfg

from acrosome_counter.inputs import QTY_CLASSES

IMAGE_SHAPE = [1024, 1024, 3]
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
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.FORMAT = "RGB"
    cfg.OUTPUT_DIR = LOG_DIR
    if is_training:
        cfg.MODEL.WEIGHTS = PRETRAINED_CHECKPOINT
    else:
        cfg.MODEL.WEIGHTS = join(LOG_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = qty_iters
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = QTY_CLASSES
    return cfg
