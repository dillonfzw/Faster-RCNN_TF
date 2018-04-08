#! /usr/bin/env python2

import init_paths
from fast_rcnn.train import get_training_roidb
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import argparse
import pprint
import numpy as np
import sys
import pdb
import tf_parameter_mgr
import os
import tensorflow as tf
from monitor_cb import CMonitor


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--job_name', dest='job_name',
                        help='One of "ps", "worker"',required=False,type=str)
    parser.add_argument('--ps_hosts', dest='ps_hosts',
                        help='Comma-separated list of hostname:port for the parameter server jobs',default=None,type=str)
    parser.add_argument('--worker_hosts', dest='worker_hosts',
                        help='Comma-separated list of hostname:port for the worker jobs',required=False,type=str)
    parser.add_argument('--task_id', dest='task_id',
                        help='Task ID of the worker/replica running the training',required=False,default=0,type=int)

    parser.add_argument('--train_dir', dest='train_dir',
                        help='Directory where to write event logs and checkpoint',required=False,default='train',type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=None, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--data_path', dest='data_path',
                        help='dataset to work on',
                        default=None, type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='VGGnet_train', type=str)

    args = parser.parse_args()
    return args

def _get_ckpt_v1(weights):
    files = []
    for file in os.listdir(weights):
        files.append(os.path.join(weights,file))
    if len(files) == 0:
        return None
    else:
        return files[0]

def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
    ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
    SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                  |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul

def train_model(net, server, task_id, train_roidb, train_imdb, test_roidb, test_imdb, saver, output_dir,max_iters, is_chief=False, pretrained_model=None):
    """Network training loop."""
    #with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % self.task_id, cluster=self.cluster)):
    print 'Computing bounding-box regression targets...'
    if cfg.TRAIN.BBOX_REG:
        bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(train_roidb)
        rdl_roidb.add_bbox_regression_targets(test_roidb)
    print 'done'
    train_data_layer = get_data_layer(train_roidb, train_imdb.num_classes)
    test_data_layer = get_data_layer(test_roidb, test_imdb.num_classes)

    global_step = tf.contrib.framework.get_or_create_global_step()

    # RPN
    # classification loss
    rpn_cls_score = tf.reshape(net.get_output('rpn_cls_score_reshape'),[-1,2])
    rpn_label = tf.reshape(net.get_output('rpn-data')[0],[-1])
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
    rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # bounding box regression L1 loss
    rpn_bbox_pred = net.get_output('rpn_bbox_pred')
    rpn_bbox_targets = tf.transpose(net.get_output('rpn-data')[1],[0,2,3,1])
    rpn_bbox_inside_weights = tf.transpose(net.get_output('rpn-data')[2],[0,2,3,1])
    rpn_bbox_outside_weights = tf.transpose(net.get_output('rpn-data')[3],[0,2,3,1])

    rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 
    # R-CNN
    # classification loss
    cls_score = net.get_output('cls_score')
    label = tf.reshape(net.get_output('roi-data')[1],[-1])
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

    # bounding box regression L1 loss
    bbox_pred = net.get_output('bbox_pred')
    bbox_targets = net.get_output('roi-data')[2]
    bbox_inside_weights = net.get_output('roi-data')[3]
    bbox_outside_weights = net.get_output('roi-data')[4]

    smooth_l1 = _modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

    # final loss
    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

    # optimizer and learning rate
    #lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
    #                            cfg.TRAIN.STEPSIZE, cfg.TRAIN.LEARNING_RATE_DECAY_RATE, staircase=True)
    lr = tf_parameter_mgr.getLearningRate(global_step)
    train_op = tf_parameter_mgr.getOptimizer(lr).minimize(loss, global_step=global_step)    

    variables_to_restore = []
    variables_to_restore_all = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if var.op.name.startswith('conv'):
            variables_to_restore.append(var)
    rst_saver = tf.train.Saver(var_list=variables_to_restore)

    log_dir = os.path.join(cfg.DATA_DIR, 'log')
    if is_chief:
        monitor = CMonitor(log_dir, tf_parameter_mgr.getTestInterval(), tf_parameter_mgr.getMaxSteps())        
        graph = tf.get_default_graph()
        all_ops = graph.get_operations()
        for op in all_ops:
            if op.type == 'VariableV2':
                output_tensor = graph.get_tensor_by_name(op.name+':0')
                if op.name.endswith('/weights'):
                    monitor.SummaryHist("weight", output_tensor, op.name.replace('/weights', ''))
                    monitor.SummaryNorm2("weight", output_tensor, op.name.replace('/weights', ''))
                elif op.name.endswith('/biases'):
                    monitor.SummaryHist("bias", output_tensor, op.name.replace('/biases', ''))
            elif op.type == 'Relu':
                output_tensor = graph.get_tensor_by_name(op.name+':0')
                monitor.SummaryHist("activation", output_tensor, op.name[:op.name.find('/')])
        monitor.SummaryScalar("train loss", loss)
        monitor.SummaryGradient("weight", loss)
        monitor.SummaryGWRatio()
        merged = tf.summary.merge_all()
        summaryWriter = tf.summary.FileWriter(log_dir)

    last_snapshot_iter = -1
    timer = Timer() #chief_only_hooks
    class _CKHook(tf.train.SessionRunHook):
        def __init__(self, isChief, net):
            self._isChief = isChief
            self.net = net
            self._next_trigger_step = cfg.TRAIN.TEST_INTERVAL
            self._trigger = False
        def before_run(self, run_context):
            args = {'global_step': global_step}
            if self._trigger:
                self._trigger = False
                args['summary'] = merged
            return tf.train.SessionRunArgs(args)
        def after_run(self, run_context, run_values):
            #m_gs = global_step.eval(session=run_context.session)
            u_gs = run_values.results['global_step']
            if u_gs >= self._next_trigger_step:
                self._trigger = True
                self._next_trigger_step += cfg.TRAIN.TEST_INTERVAL
            summary = run_values.results.get('summary', None)
            if summary is not None:
                summaryWriter.add_summary(summary, gs)
                total_test_loss = 0;            
                for i in range(cfg.TRAIN.TEST_ITERS):
                    test_blobs = test_data_layer.forward()
                    feed_dict={net.data: test_blobs['data'], net.im_info: test_blobs['im_info'], net.keep_prob: 0.5, \
                       net.gt_boxes: test_blobs['gt_boxes']}
                    test_rpn_loss_cls_value, test_rpn_loss_box_value,test_loss_cls_value, test_loss_box_value = run_context.session.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box],
                                                                                            feed_dict=feed_dict,
                                                                                            options=run_options,
                                                                                            run_metadata=run_metadata)
                    total_test_loss = total_test_loss + test_rpn_loss_cls_value + test_rpn_loss_box_value + test_loss_cls_value + test_loss_box_value
                #print "[./log/test] Iteration tag loss, simple_value: %.4f" % (total_test_loss/int(cfg.TRAIN.TEST_ITERS))
                test_summary = tf.Summary(value=[tf.Summary.Value(tag='test loss', simple_value=total_test_loss/int(cfg.TRAIN.TEST_ITERS))])
                summaryWriter.add_summary(test_summary, gs)
                 
        def end(self, session):
            self.snapshot(session, max_iters)

        def snapshot(self, sess, iter):
            #Take a snapshot of the network after unnormalizing the learned
            #bounding-box regression weights. This enables easy use at test-time.
       
            net = self.net

            if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
                    # save original values
                with tf.variable_scope('bbox_pred', reuse=True):
                    weights = tf.get_variable("weights")
                    biases = tf.get_variable("biases")

                orig_0 = weights.eval(session=sess)
                orig_1 = biases.eval(session=sess)

                # scale and shift with bbox reg unnormalization; then save snapshot
                weights_shape = weights.get_shape().as_list()
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(bbox_stds, (weights_shape[0], 1))})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * bbox_stds + bbox_means})

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
             if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
            filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                '_iter_{:d}'.format(iter) + '.ckpt')
            filename = os.path.join(output_dir, filename)

            saver.save(sess, filename, global_step=iter)
            print 'Wrote snapshot to: {:s}'.format(filename)

            if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
                with tf.variable_scope('bbox_pred', reuse=True):
                    # restore net to original state
                    sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                    sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})
    class _RestoreHook(tf.train.SessionRunHook):
        def after_create_session(self, session, coord):
            if pretrained_model is None:
                return
            ckpt=tf.train.latest_checkpoint(pretrained_model)
            if ckpt is None:
               ckpt = _get_ckpt_v1(pretrained_model)
            print "Restore pre-trained checkpoint detail:", ckpt
            try:
                saver.restore(session, ckpt)
                print "Restore pre-trained checkpoint succeed"
            except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
                print "Restore all trainable variables failed, Try to only restore cnn layers variables"
                rst_saver.restore(session, ckpt)
                print "Restore pre-trained weight succeed"
            
    hooks=[tf.train.StopAtStepHook(last_step=max_iters), _RestoreHook()]
    #with tf.Session(server.target,config=tf.ConfigProto(allow_soft_placement=True)) as sess: #checkpoint_dir
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, hooks=hooks, chief_only_hooks=[_CKHook(is_chief, net)]) as sess:
        iter = 0
        while not sess.should_stop():
            # get one batch
            blobs = train_data_layer.forward()
            print task_id, 'blobs shape', blobs['data'].shape

            # Make one SGD update
            feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 0.5, \
                                   net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value,gs, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, global_step, train_op],
                                                                                    feed_dict=feed_dict,
                                                                                    options=run_options,
                                                                                    run_metadata=run_metadata)

            timer.toc()
            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'step: %d (global_step %d),, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.6f'%\
                    (iter+1, gs, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            iter = iter + 1


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

'''
setup TensorFlow distribute environment
NOTE: we only do this if there are more than 1 workers had been requested
'''
def setup_distribute(TF_FLAGS):
    worker_hosts = []
    ps_hosts = []
    spec = {}
    if TF_FLAGS.worker_hosts is not None:
        worker_hosts = TF_FLAGS.worker_hosts.split(',')
        spec.update({'worker': worker_hosts})

    if TF_FLAGS.ps_hosts is not None:
        ps_hosts = TF_FLAGS.ps_hosts.split(',')
        spec.update({'ps': ps_hosts})

    if len(worker_hosts) > 0:
        print('Cluster spec: ', spec)
        cluster = tf.train.ClusterSpec(spec)

        # Create and start a server for the local task.
        server = tf.train.Server(cluster, job_name=TF_FLAGS.job_name, task_index=TF_FLAGS.task_id)
        if TF_FLAGS.job_name == "ps":
            server.join()
    else:
        cluster = None
        server = tf.train.Server.create_local_server()
        # enforce a task_id for single node mode
        TF_FLAGS.task_id = 0

    return cluster, server


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    cfg.EXP_DIR = 'faster_rcnn_end2end'
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1 #RPN currently only support batch size as 1, tf_parameter_mgr.getTrainBatchSize()
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BG_THRESH_LO = 0.0

    cfg.TRAIN.LEARNING_RATE = tf_parameter_mgr.getBaseLearningRate()
    cfg.TRAIN.LEARNING_RATE_DECAY_RATE = tf_parameter_mgr.getLearningRateDecay()
    cfg.TRAIN.TEST_INTERVAL=tf_parameter_mgr.getTestInterval()

    print('Using config:')
    pprint.pprint(cfg)

    if args.max_iters is None:
        maxStep = tf_parameter_mgr.getMaxSteps()
    else:
        maxStep = args.max_iters

    trainPath = tf_parameter_mgr.getTrainData()
    trainSet = trainPath[0].split("/")[-1].split(".")[0]

    testPath = tf_parameter_mgr.getTestData()
    testSet = testPath[0].split("/")[-1].split(".")[0]

    if args.data_path is None:
        dataPath = os.path.join(os.path.split(os.path.abspath(trainPath[0]))[0], "..", "..")
    else:
        dataPath = args.data_path
        trainSet = "trainval"
        testSet = "test"

    # flexible for standalone or distribute run
    cluster, server = setup_distribute(args)

    is_chief = (args.task_id == 0)

    cfg.DATA_DIR = args.train_dir
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % args.task_id, cluster=cluster)):
        train_imdb = get_imdb(dataPath, trainSet)
        print 'Loaded Train dataset `{:s}` for training'.format(train_imdb.name)
        train_roidb = get_training_roidb(train_imdb)

        test_imdb = get_imdb(dataPath, testSet)
        print 'Loaded Test dataset `{:s}` for testing'.format(test_imdb.name)
        test_roidb = get_training_roidb(test_imdb)

        output_dir = args.train_dir #get_output_dir(imdb, None)  #train_dir
        print 'Output will be saved to `{:s}`'.format(output_dir)

        train_roidb = filter_roidb(train_roidb)
        test_roidb = filter_roidb(test_roidb)

        print 'Use network `{:s}` in training'.format(args.network_name)
        net = get_network(args.network_name, train_imdb.num_classes)
        saver = tf.train.Saver(max_to_keep=100)
        print 'server targe:',server.target
        train_model(net, server, args.task_id, train_roidb, train_imdb, test_roidb, test_imdb, saver, output_dir, maxStep, is_chief=is_chief, pretrained_model=args.pretrained_model)
