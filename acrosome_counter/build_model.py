# -*- coding: utf-8 -*-
"""Build the network."""

import tensorflow as tf
from tensorflow.compat.v2.train import Checkpoint
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.builders.model_builder import build

from acrosome_counter.inputs import QTY_CLASSES

IMAGE_SHAPE = [1024, 1024, 3]
CONFIG_PATH = (
    'models\\research\\object_detection\\configs\\tf2'
    '\\faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.config'
)


def build_model(is_training):
    configs = get_configs_from_pipeline_file(CONFIG_PATH)
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = QTY_CLASSES
    model = build(model_config=model_config, is_training=True)
    initialize_model(model)
    return model


def restore(model, checkpoint_path, is_training):
    if is_training:
        heads_checkpoint = Checkpoint(
            _first_stage_box_predictor_first_conv=(
                model._first_stage_box_predictor_first_conv
            ),
            _feature_extractor_for_proposal_features=(
                model._feature_extractor_for_proposal_features
            ),
            _feature_extractor_for_box_classifier_features=(
                model._feature_extractor_for_box_classifier_features
            ),
            # _mask_rcnn_box_predictor=model._mask_rcnn_box_predictor,
            _first_stage_box_predictor=model._first_stage_box_predictor,
        )
        checkpoint = Checkpoint(model=heads_checkpoint)
        checkpoint.restore(checkpoint_path).expect_partial()
    else:
        checkpoint = Checkpoint(model=model)
        checkpoint.restore(checkpoint_path).expect_partial()


def initialize_model(model):
    model.provide_groundtruth(
        groundtruth_boxes_list=tf.zeros([1, 1, 4]),
        groundtruth_classes_list=tf.zeros([1, 1, QTY_CLASSES]),
    )
    image, shapes = model.preprocess(tf.zeros([1, *IMAGE_SHAPE]))
    prediction_dict = model.predict(image, shapes)
    _ = model.postprocess(prediction_dict, shapes)
