# -*- coding: utf-8 -*-
"""Fetch and process inputs to the network."""

from os import listdir
from os.path import join, isdir

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence


class Sequence(Sequence):
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
        # return data, labels

    def on_epoch_end(self):
        pass


if __name__ == "__main__":
    pass
