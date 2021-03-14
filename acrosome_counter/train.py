# -*- coding: utf-8 -*-
"""Launch training on a TensorFlow Model Garden model.

This is modified from Model Garden's "Eager Few Shot Object Detection Colab"
(github.com/tensorflow/models/blob/master/research/object_detection
/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb).
"""

import tensorflow as tf


def get_model_train_step_function(
    model, optimizer, vars_to_fine_tune, batch_size,
):
    """Get a tf.function for training step."""
    @tf.function
    def train_step_fn(images, shapes):
        """A single training iteration.

        :param images: A list of [1, height, width, 3] Tensor of type
            tf.float32. Note that the height and width can vary across images,
            as they are reshaped within this function to be 320x320.
        :param boxes: A list of Tensors of shape [N_i, 4] with type tf.float32
            representing groundtruth boxes for each image in the batch.
        :param classes: A list of Tensors of shape [N_i, num_classes] with type
            tf.float32 representing groundtruth boxes for each image in the
            batch.

        :return: The total loss for the input batch.
        """
        with tf.GradientTape() as tape:
            prediction_dict = model.predict(images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = sum(losses_dict.values())
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn


def train(model, sequence, qty_epochs, learning_rate):
    tf.keras.backend.set_learning_phase(True)

    trainable_variables = model.trainable_variables
    prefixes_to_train = ['RPNConv', 'FirstStageBoxPredictor', 'mask_rcnn']
    to_fine_tune = []
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_fn = get_model_train_step_function(
        model, optimizer, to_fine_tune, sequence.batch_size,
    )

    for epoch in range(qty_epochs):
        images, (boxes, classes) = sequence[epoch]
        model.provide_groundtruth(
            groundtruth_boxes_list=boxes,
            groundtruth_classes_list=classes,
        )
        inputs = [model.preprocess(image) for image in images]
        images = tf.concat([input[0] for input in inputs], axis=0)
        shapes = tf.concat([input[1] for input in inputs], axis=0)
        total_loss = train_step_fn(images, shapes)
        print(
            f"Epoch {epoch+1} of {qty_epochs}, loss={total_loss.numpy()}",
            flush=True,
        )
