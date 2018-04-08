#! /usr/bin/env python2

import init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import tf_parameter_mgr, inference_helper
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import json

def inference(sess, net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    return scores, boxes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN inference')

    parser.add_argument('--net', dest='test_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        required=True)
    parser.add_argument('--input_dir', dest='image_path', help='Directory where the inference images are',
                        default=None)
    parser.add_argument('--image_file', dest='image_file', help='Particuarl image file to run inference upon',
                        default=None)
    parser.add_argument('--output_dir', dest='output_dir', help='Where to put the image with bbox drawed',
                        required=True)
    parser.add_argument('--label_file', dest='label_file', help='Where to put the image with bbox drawed',
                        required=True)
    parser.add_argument('--output_file', dest='output_file', help='A .txt file contains the inference result with format as: sampleid label score xmin:ymin:xmax:ymax',
                        default=None)
    parser.add_argument('--prob_thresh', dest='prob_thresh', help='Threashold to control the minimal probility in the result',
                        default=0.8, type=float)
    parser.add_argument('--validate', dest='validate', help='Evaluating this model with validation dataset or not',
                        default=False, type=bool)
    """
    format of the output_file for object detection
    {
         "type": "det",
         "result" : [
         {
         "sampleid": "11_01.jpg",
         "label": "cat",
         "prob": 0.99,
         "bbox": {"xmin": 11, "ymin": 13, "xmax": 21, "ymax": 23}
         }
         ]

     }
    """

    args = parser.parse_args()

    return args

def getAllImagesInPath(path):
    files = []
    for file in os.listdir(path):
        files.append(os.path.join(path,file))
    return files

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    if args.validate:
        valPath = tf_parameter_mgr.getValData()
        dataPath = os.path.abspath(os.path.join(os.path.split(valPath[0])[0], "..", "..", 'JPEGImages'))
        im_files = [os.path.join(dataPath, fname+'.jpg') for fname in np.loadtxt(valPath[0], str)]
    elif args.image_path is not None:
        im_files = getAllImagesInPath(args.image_path)
    elif args.image_file is not None:
        im_files = [args.image_file]
    else:
        print "Either --image_path or --image_file must be specified"

    if tf.gfile.Exists(args.output_dir):
        tf.gfile.DeleteRecursively(args.output_dir)
    tf.gfile.MakeDirs(args.output_dir)

    output_dir = args.output_dir
    output_file = args.output_file
    prob_thresh = args.prob_thresh
    label_file = args.label_file

    #CLASSES = np.loadtxt(args.label_file, str, delimiter='\t')
    CLASSES = [line.rstrip('\n') for line in open(args.label_file).readlines()]
    net = get_network(args.test_net, len(CLASSES) + 1)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    ckpt = tf.train.get_checkpoint_state(args.model)
    print "ckpt", ckpt
    if ckpt and ckpt.model_checkpoint_path:
        print "restore checkpoint"
        saver.restore(sess, ckpt.model_checkpoint_path)

    print '\n\nLoaded network {:s}'.format(args.model)

    json_list = []
    samples = []
    prediction = []
    all_box = []
    for im_file in im_files:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Run inference for image {}'.format(im_file)
        scores, boxes = inference(sess, net, im_file)
        samples.append(im_file)
        prediction.append(scores)
        all_box.append(boxes)

    NMS_THRESH = 0.3
    rest = list()
    im_detect_res = np.ndarray([len(samples),0,6])
    for i in range(len(samples)):
        im_file = samples[i]
        im_name = os.path.basename(im_file)
        scores = prediction[i]
        boxes = all_box[i]
        #im = cv2.imread(im_file)
        detects_per_img = np.ndarray([0, 6])
        for cls_ind, cls in enumerate(CLASSES):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= prob_thresh)[0]
            if len(inds) < 1:
                continue
            
            dets = dets[inds, :]

            labels = np.ndarray([len(inds)]).astype(str)
            labels[:] = cls
            dets = np.hstack((dets, labels[:,np.newaxis]))
            """            
            im_files = np.ndarray([len(inds)]).astype(str)
            im_files[:] = im_file
            dets = np.hstack((dets, im_files[:,np.newaxis]))
            """
            detects_per_img = np.append(detects_per_img, dets, 0)

        rest += [detects_per_img.tolist()]     


    inference_helper.writeObjectDetectionResult(output_dir, output_file, samples, rest, prob_thresh)


