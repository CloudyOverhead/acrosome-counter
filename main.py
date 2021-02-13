# -*- coding: utf-8 -*-
"""Launch training and inference."""

from argparse import ArgumentParser
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model

from acrosome_counter.inputs import Sequence
from acrosome_counter.build_model import build_model


def main(args):
    sequence = Sequence(args.data_dir, args.batch_size)
    log_dir = join(".", "logs")
    if not args.infer:
        model = build_model([340, 340])
        model.compile(optimizer='Adam', loss='')
        callbacks = [
            TensorBoard(log_dir=log_dir, profile_batch=0),
            ModelCheckpoint(log_dir, save_freq='epoch'),
        ]
        model.fit(
            sequence,
            epochs=args.epochs,
            callbacks=callbacks,
            steps_per_epoch=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
    else:
        model = load_model(log_dir, compile=False)
        scans, labels = sequence[0]
        predictions = model.predict(scans)
        predictions = np.argmax(predictions, axis=-1)
        labels = np.argmax(labels, axis=-1)
        for scan, prediction, label in zip(scans, predictions, labels):
            fig, axs = plt.subplots(ncols=2, figsize=[8, 4])
            axs[0].set_title("Actual labels")
            axs[0].imshow(scan, cmap='Greys')
            axs[1].set_title("Predictions")
            axs[1].imshow(scan, cmap='Greys')
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    main(args)
