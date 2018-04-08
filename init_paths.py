# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(1, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
add_path('/Users/fuzhiwen/workspace/Faster-RCNN_TF/lib')
add_path('/usr/local/opt/opencv@2/lib/python2.7/site-packages')
add_path('/Users/fuzhiwen/workspace/bluemind/tools/parameter_mgr')
add_path('/Users/fuzhiwen/workspace/bluemind/dlpd/src/dists/bin/common_utils')
