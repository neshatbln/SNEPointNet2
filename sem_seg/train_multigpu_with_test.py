import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import pc_util
from pointnet2_sem_seg import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=2, help='How many gpus to use [default: 1]') #Multi
parser.add_argument('--log_dir', default='log2', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=12288, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=51, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=72, help='Batch Size during training [default: 24]')  
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=807, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.70, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--perform_test', type=bool, default=True, help='performing test after training is over [default: False]')

FLAGS = parser.parse_args()


NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE // NUM_GPUS

NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
PERFORM_TEST = FLAGS.perform_test

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 16384
NUM_CLASSES = 3  #cracks,spalls,nodefects

BN_INIT_DECAY = 0.50
BN_DECAY_DECAY_RATE = 0.50
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

############################################################################################
#                            Loading the train, eval, and test data                        #
############################################################################################

ALL_FILES = provider.getDataFiles('bridge_npy_hdf5_data/all_files.txt')

##### Loading ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

##### Reading the train, eval, and test indices
root_path = './bridge_npy_hdf5_data/room_filelist_'
train_idxs = [int(line.split(":")[1].strip()) for line in open(root_path + 'train.txt', 'r')]
eval_idxs = [int(line.split(":")[1].strip()) for line in open(root_path + 'eval.txt', 'r')]
test_idxs = [int(line.split(":")[1].strip()) for line in open(root_path + 'test.txt', 'r')]

print(f"train size: {len(train_idxs)}, eval size: {len(eval_idxs)}, test size: {len(test_idxs)}")
##### Creating the final train, eval, and test data
train_data = data_batches[train_idxs, ...]
train_label = label_batches[train_idxs]

eval_data = data_batches[eval_idxs, ...]
eval_label = label_batches[eval_idxs]

test_data = data_batches[test_idxs, ...]
test_label = label_batches[test_idxs]

total_num_points = eval_label.shape[0] * eval_label.shape[1]
print(total_num_points)

############################################################################################
#                                                                                          #
############################################################################################

class Dataset():
    def __init__(self, npoints=NUM_POINT, split='train'):
        self.npoints = npoints
        self.train_data_list = train_data
        self.train_label_list = train_label
        self.test_data_list = test_data
        self.test_label_list = test_label
        if split=='train':
            labelweights = np.zeros(NUM_CLASSES)
            for seg in self.train_label_list:
                tmp, _ = np.histogram(seg, range(NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.05 + labelweights)
        if split == 'test':
            self.labelweights = np.ones(NUM_CLASSES)

    def __getitem__(self, index):
        point_set = self.train_data_list[index]
        semantic_seg = self.train_label_list[index].astype(np.int32)
        sample_weight = self.labelweights[semantic_seg]
        return point_set, semantic_seg, sample_weight
    def __len__(self):
        return len(self.train_data_list)

TRAIN_DATASET = Dataset(npoints=NUM_POINT, split='train')
EVAL_DATASET = Dataset(npoints=NUM_POINT, split='test')
TEST_DATASET = Dataset(npoints=NUM_POINT, split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def average_gradients(tower_grads):
    """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        #for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            pointclouds_pl, labels_pl, smpws_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.

            batch = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # -------------------------------------------
            # Get model and loss on multiple GPU devices
            # -------------------------------------------
            # Allocating variables on CPU first will greatly accelerate multi-gpu training.
            # Ref: https://github.com/kuza55/keras-extras/issues/21
            get_model(pointclouds_pl, is_training_pl, NUM_CLASSES,  bn_decay=bn_decay)

            tower_grads = []
            pred_gpu = []
            total_loss_gpu = []
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:
                        # Evenly split input data to each GPU
                        
                        pc_batch = tf.slice(pointclouds_pl, [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                        smpws_batch = tf.slice(smpws_pl, [i * DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])
                        label_batch = tf.slice(labels_pl, [i * DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])

                        pred, end_points = get_model(pc_batch, is_training_pl, NUM_CLASSES,  bn_decay=bn_decay)

                        losses = get_loss(pred, label_batch, smpws_batch) #NEW-NB
                        tf.summary.scalar('losses', losses)

                        grads = optimizer.compute_gradients(losses)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(losses)

            # Merge pred and losses from multiple GPUs
            pred = tf.concat(pred_gpu, 0)
            losses = tf.reduce_mean(total_loss_gpu)

            # Get training operator 
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=batch)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            accuracy_1 = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy = correct / (BATCH_SIZE*NUM_POINT) =', accuracy)
            tf.summary.scalar('accuracy_1 = correct / (BATCH_SIZE) =', accuracy_1)
            
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'))
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': losses,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            start_time = time.time()
            train_one_epoch(sess, ops, train_writer)
            train_time = time.time()
            log_string(f"Train time: {train_time - start_time}")

            eval_one_epoch(sess, ops, eval_writer, epoch)
            eval_time = time.time()
            log_string(f"Train time: {eval_time - train_time}")

            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
        
        if PERFORM_TEST:
            test_model(sess, ops, test_writer)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 10))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 10))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)

    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
        
def eval_one_epoch(sess, ops, eval_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    eval_idxs = np.arange(0, len(EVAL_DATASET))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    h = 0 #MAJID
    label = [0 for _ in range(total_num_points)]
    predicted = [0 for _ in range(total_num_points)]
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = eval_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(eval_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(EVAL_DATASET, eval_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        eval_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                label[h] = current_label[i, j]
                predicted[h] = pred_val[i - start_idx, j]
                h += 1
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string('all room eval accuracy-Crack: %f'% (total_correct_class[0] / float(total_seen_class[0])))
    log_string('all room eval accuracy-Spalling: %f'% (total_correct_class[1] / float(total_seen_class[1])))
    log_string('all room eval accuracy-No-Defect: %f'% (total_correct_class[2] / float(total_seen_class[2])))
    
    if epoch == 50:
        confusion = tf.confusion_matrix(labels=label[:h], predictions=predicted[:h], num_classes=NUM_CLASSES)
        conf = confusion.eval(session=sess)

        log_string('crack pre.: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[0, 1]+conf[0, 2])))
        log_string('crack recall.: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[1, 0]+conf[2, 0])))
        log_string('crack IoU: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[0, 1]+conf[0, 2]+conf[1, 0]+conf[2, 0])))
        log_string('Spal Pre.: %f' % (conf[1, 1]/ float(conf[1, 0]+conf[1, 1]+conf[1, 2])))
        log_string('Spal recall.: %f' % (conf[1, 1]/ float(conf[1, 1]+conf[0, 1]+conf[2, 1])))
        log_string('Spal IoU: %f' % (conf[1, 1]/ float(conf[1, 0]+conf[1, 1]+conf[1, 2]+conf[0, 1]+conf[2, 1])))
        log_string('No-Defect Pre.: %f' % (conf[2, 2]/ float(conf[2, 0]+conf[2, 1]+conf[2, 2])))
        log_string('No-Defect Recal.: %f' % (conf[2, 2] / float(conf[0, 2] + conf[1, 2] + conf[2, 2])))
        log_string('No-defect IoU: %f' % (conf[2,2]/float(conf[0, 2]+conf[1, 2]+conf[2, 2]+conf[2, 0]+conf[2, 1])))

        log_string('Confusion Matrix: \n  %d  %d  %d \n  %d  %d  %d \n  %d  %d  %d' % (
        conf[0, 0], conf[0, 1], conf[0, 2], conf[1, 0], conf[1, 1], conf[1, 2], conf[2, 0], conf[2, 1], conf[2, 2]))

        print(confusion.eval(session=sess))

def test_model(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    test_idxs = np.arange(0, len(TEST_DATASET))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    h = 0

    total_num = test_label.shape[0] * test_label.shape[1]
    label = [0 for _ in range(total_num)]
    predicted = [0 for _ in range(total_num)]    
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                label[h] = current_label[i, j]
                predicted[h] = pred_val[i - start_idx, j]
                h += 1
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string('all room eval accuracy-Crack: %f'% (total_correct_class[0] / float(total_seen_class[0])))
    log_string('all room eval accuracy-Spalling: %f'% (total_correct_class[1] / float(total_seen_class[1])))
    log_string('all room eval accuracy-No-Defect: %f'% (total_correct_class[2] / float(total_seen_class[2])))
    
    confusion = tf.confusion_matrix(labels=label[:h], predictions=predicted[:h], num_classes=NUM_CLASSES)
    conf = confusion.eval(session=sess)

    log_string('crack pre.: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[0, 1]+conf[0, 2])))
    log_string('crack recall.: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[1, 0]+conf[2, 0])))
    log_string('crack IoU: %f' % (conf[0, 0]/ float(conf[0, 0]+conf[0, 1]+conf[0, 2]+conf[1, 0]+conf[2, 0])))
    log_string('Spal Pre.: %f' % (conf[1, 1]/ float(conf[1, 0]+conf[1, 1]+conf[1, 2])))
    log_string('Spal recall.: %f' % (conf[1, 1]/ float(conf[1, 1]+conf[0, 1]+conf[2, 1])))
    log_string('Spal IoU: %f' % (conf[1, 1]/ float(conf[1, 0]+conf[1, 1]+conf[1, 2]+conf[0, 1]+conf[2, 1])))
    log_string('No-Defect Pre.: %f' % (conf[2, 2]/ float(conf[2, 0]+conf[2, 1]+conf[2, 2])))
    log_string('No-Defect Recal.: %f' % (conf[2, 2] / float(conf[0, 2] + conf[1, 2] + conf[2, 2])))
    log_string('No-defect IoU: %f' % (conf[2,2]/float(conf[0, 2]+conf[1, 2]+conf[2, 2]+conf[2, 0]+conf[2, 1])))

    log_string('Confusion Matrix: \n  %d  %d  %d \n  %d  %d  %d \n  %d  %d  %d' % (
    conf[0, 0], conf[0, 1], conf[0, 2], conf[1, 0], conf[1, 1], conf[1, 2], conf[2, 0], conf[2, 1], conf[2, 2]))

    print(confusion.eval(session=sess))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
