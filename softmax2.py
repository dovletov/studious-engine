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

y = unet(x)

# loss
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(y_, y))

# prediction
prediction = tf.nn.softmax(y)

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
    for step_id in range(200):
        x_batch, y_batch = formRandBatch(batch_size, input_h, input_w, 'tr')
        feed_train = {x: x_batch, y_: y_batch}
        _, loss_tr = sess.run([train_step, loss], feed_dict=feed_train)

        if step_id % 10 == 0:
            x_batch_vl, y_batch_vl = formRandBatch(batch_size, input_h, input_w, 'vl')
            feed_valid = {x: x_batch_vl, y_: y_batch_vl}
            loss_vl = sess.run(loss, feed_dict=feed_valid)

            print("Step: %s loss_tr: %s loss_vl %s" % (step_id, loss_tr, loss_vl))

            if loss_tr < 0.1:
                break;
    
    x_ev = loadFromHdf5(NP_TEST_DIR, 'patient101_ED.hdf5', 'test')
    x_batch_ev = resizedArray(x_ev, input_h, input_w, output_mode='DHW1')

    feed_eval = {x: x_batch_ev}

    # get prediction arrays
    pr = sess.run(prediction, feed_dict=feed_eval)
    pr = np.argmax(pr, axis=3)
    print(pr.shape)

# save prediction and gt images
for i in range(batch_size):
    pr_slice = pr[i,:,:]
    plt.imshow(pr_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_pr.png'))
