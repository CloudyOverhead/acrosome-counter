# -*- coding: utf-8 -*-
"""Build the network."""

import tensorflow as tf
from tensorflow.compat.v2.train import Checkpoint
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.builders.model_builder import build

QTY_CLASSES = 3
CONFIG_PATH = (
    'models\\research\\object_detection\\configs\\tf2'
    '\\faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.config'
)
RESTORE_PATH = 'models/research/object_detection/test_data/checkpoint/ckpt-0'


def build_model(is_training):
    configs = get_configs_from_pipeline_file(CONFIG_PATH)
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = QTY_CLASSES
    model = build(model_config=model_config, is_training=True)
    return model


def restore(model, checkpoint_path):
    checkpoint = Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()
    return model, checkpoint


def initialize_modeL(model):
    model.provide_groundtruth(
        groundtruth_boxes_list=tf.zeros([1, 1, 4]),
        groundtruth_classes_list=tf.zeros([1, 1, 3]),
    )
    image, shapes = model.preprocess(tf.zeros([1, 1024, 1024, 3]))
    prediction_dict = model.predict(image, shapes)
    _ = model.postprocess(prediction_dict, shapes)
