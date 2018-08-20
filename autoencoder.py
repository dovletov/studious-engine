from config import *
from utils.helper import *
from utils.models import autoencoder

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

# input sizes
input_h = 128
input_w = 128



# network params
batch_size = 10
lr = 1e-3
max_iter = 1000000

# placeholders
x = tf.placeholder(tf.float32, shape=[None, input_h, input_w, 1])
y_ = tf.placeholder(tf.int32, shape=[None, input_h, input_w])

y = autoencoder(x)

# prediction
prediction = tf.nn.softmax(y)

# loss
def rms_loss(y_, prediction):
    depth = prediction.get_shape()[3].value
    one_hot_y_ = tf.one_hot(y_, depth, axis=-1, dtype=tf.float32)
    loss = tf.square(prediction - one_hot_y_)

    return loss
loss = tf.reduce_mean(rms_loss(y_, prediction))

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
        x_batch, y_batch = formRandBatch(batch_size, input_h, input_w, 'tr')
        x_batch = np.expand_dims(y_batch, axis=3)
        feed_train = {x: x_batch, y_: y_batch}
        _, loss_tr = sess.run([train_step, loss], feed_dict=feed_train)

        if step_id % 10 == 0:
            x_batch_vl, y_batch_vl = formRandBatch(batch_size, input_h, input_w, 'vl')
            x_batch_vl = np.expand_dims(y_batch_vl, axis=3)
            feed_valid = {x: x_batch_vl, y_: y_batch_vl}
            loss_vl = sess.run(loss, feed_dict=feed_valid)

            print("Step: %s loss_tr: %s loss_vl %s" % (step_id, loss_tr, loss_vl))

            if loss_vl < 0.01:
                break;
    # get prediction arrays
    pr = sess.run(y, feed_dict=feed_valid)
    pr = np.argmax(pr, axis=3)
    print(pr.shape)

    # return gt
    gt = sess.run(y_, feed_dict=feed_valid)
    gt = np.squeeze(gt)
    print(gt.shape)
    
# save prediction and gt images
for i in range(batch_size):
    gt_slice = gt[i,:,:]
    plt.imshow(gt_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_gt.png'))

    pr_slice = pr[i,:,:]
    plt.imshow(pr_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_pr.png'))
