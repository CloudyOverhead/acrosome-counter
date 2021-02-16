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
    def train_step_fn(images, boxes, classes):
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
        shapes = tf.constant(batch_size * [[1024, 1024, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=boxes,
            groundtruth_classes_list=classes,
        )
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [model.preprocess(image)[0] for image in images],
                axis=0,
            )
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = sum(losses_dict.values())
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn


def train(model, sequence, qty_epochs):
    tf.keras.backend.set_learning_phase(True)

    trainable_variables = model.trainable_variables
    prefix = "FirstStageBoxPredictor/ConvolutionalBoxHead"
    to_fine_tune = []
    for var in trainable_variables:
        if var.name.startswith(prefix):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.Adam(learning_rate=.00004)
    train_step_fn = get_model_train_step_function(
        model, optimizer, to_fine_tune, sequence.batch_size,
    )

    for epoch in range(qty_epochs):
        images, (boxes, classes) = sequence[epoch]
        total_loss = train_step_fn(images, boxes, classes)
        print(
            f"Epoch {epoch+1} of {qty_epochs}, loss={total_loss.numpy()}",
            flush=True,
        )
