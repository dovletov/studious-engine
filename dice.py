from config import *
from utils.helper import *
from utils.models import unet

import argparse
from pathlib import Path

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_id', 
    default = '0', 
    help = """Id number of graphical card used to run %(prog)s. 
            By default uses '0'.""")
parser.add_argument('-m', '--mode')
parser.add_argument('-n', '--configuration_name')

# parsing
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# create main folder
configuration_path = os.path.join(RESULTS_DIR, args.configuration_name)
Path(configuration_path).mkdir(parents=True, exist_ok=True)

# load data
x_batch_orig = loadFromHdf5(NP_TRAIN_DIR, 'patient001_ED.hdf5', 'train')
y_batch_orig = loadFromHdf5(NP_TRAIN_DIR, 'patient001_ED_gt.hdf5', 'train_gt')
print(x_batch_orig.shape)
print(y_batch_orig.shape)
print('='*80)

# input sizes
input_h = 128
input_w = 128

# preprocess data
x_batch = resizedArray(x_batch_orig, input_h, input_w, output_mode='DHW1')
y_batch = resizedArray(y_batch_orig, input_h, input_w, output_mode='DHW')
print(x_batch.shape)
print(y_batch.shape)
print('='*80)



# network params
batch_size = x_batch_orig.shape[0]
lr = 1e-3
max_iter = 1000000

# placeholders
x = tf.placeholder(tf.float32, shape=[None, input_h, input_w, 1])
y_ = tf.placeholder(tf.int32, shape=[None, input_h, input_w])

y = unet(x)

# prediction
prediction = tf.nn.softmax(y)

# loss
def dice_loss(y_, y):
    epsilon = tf.constant(0.00001)
    
    depth = y.get_shape()[3].value
    one_hot_y_ = tf.one_hot(y_, depth, axis=-1, dtype=tf.float32)

    intersection = tf.reduce_sum(one_hot_y_ * y)
    dice = (2.*intersection+epsilon) / (tf.reduce_sum(one_hot_y_) + \
        tf.reduce_sum(y)+epsilon)
    loss = 1 - dice

    return loss
loss = tf.reduce_mean(dice_loss(y_, prediction))

# train
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# init
initializer = tf.global_variables_initializer()

# config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# session
with tf.Session(config=config) as sess:
    sess.run(initializer)

    # training loop
    for step_id in range(max_iter):
        feed_train = {x: x_batch, y_: y_batch}
        _, loss_tr = sess.run([train_step, loss], feed_dict=feed_train)

        if step_id % 10 == 0:
            print("Step: %s loss_tr: %s" % (step_id, loss_tr))

            if loss_tr < 0.1:
                break;
    
    # get prediction arrays
    pr = sess.run(prediction, feed_dict=feed_train)
    pr = np.argmax(pr, axis=3)
    print(pr.shape)

    # return gt
    gt = sess.run(y_, feed_dict=feed_train)
    print(gt.shape)
    
# save prediction and gt images
for i in range(batch_size):
    gt_slice = gt[i,:,:]
    plt.imshow(gt_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_gt.png'))

    pr_slice = pr[i,:,:]
    plt.imshow(pr_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_pr.png'))
