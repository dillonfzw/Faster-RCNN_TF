# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.an_imdb import an_imdb
import numpy as np
import os

def get_imdb(data_path, image_set):
    assert os.path.exists(data_path), 'Path does not exist: {}'.format(data_path)
    
    return an_imdb(data_path, image_set)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
