from config import *
from utils.helper import *
from utils.models import unet

import argparse
from pathlib import Path

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import tensorflow as tf
import tensorflow.contrib.slim as slim


def unet3d(inputs):
    """
    unet model without softmax.
    """
    print(inputs.shape)
    conv1 = slim.repeat(inputs=inputs,
                        repetitions=2,
                        layer=slim.layers.conv3d,
                        num_outputs=64,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)
    pool1 = slim.max_pool3d(inputs=conv1, kernel_size=2)
    print(pool1.shape)

    conv2 = slim.repeat(inputs=pool1,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=128,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)
    pool2 = slim.max_pool3d(inputs=conv2, kernel_size=2)
    print(pool2.shape)

    conv3 = slim.repeat(inputs=pool2,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)
    pool3 = slim.max_pool3d(inputs=conv3, kernel_size=2)
    print(pool3.shape)

    conv4 = slim.repeat(inputs=pool3,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv4.shape)
    # pool4 = slim.max_pool3d(inputs=conv4, kernel_size=2)
    # print(pool4.shape)

    # conv5 = slim.repeat(inputs=pool4,
    #                     repetitions=2,
    #                     layer=slim.conv3d,
    #                     num_outputs=1024,
    #                     activation_fn=tf.nn.relu,
    #                     kernel_size=3,
    #                     normalizer_fn=slim.batch_norm)
    # print(conv5.shape)

    # upsampling1 = slim.conv3d_transpose(inputs=conv5,
    #                                     kernel_size=3,
    #                                     num_outputs=1024,
    #                                     stride=2,
    #                                     activation_fn=tf.nn.relu,
    #                                     normalizer_fn=slim.batch_norm)
    # print(upsampling1.shape)
    # upconv1 = slim.conv3d(inputs=upsampling1,
    #                       kernel_size=2,
    #                       num_outputs=512,
    #                       activation_fn=tf.nn.relu,
    #                       normalizer_fn=slim.batch_norm)
    # print(upconv1.shape)
    # concat1 = tf.concat([conv4, upconv1], 3)
    # print(concat1.shape)

    # conv4 = slim.repeat(inputs=concat1,
    #                     repetitions=2,
    #                     layer=slim.conv3d,
    #                     num_outputs=512,
    #                     activation_fn=tf.nn.relu,
    #                     kernel_size=3,
    #                     normalizer_fn=slim.batch_norm)
    # print(conv4.shape)

    upsampling2 = slim.conv3d_transpose(inputs=conv4,
                                        kernel_size=3,
                                        num_outputs=512,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling2.shape)
    upconv2 = slim.conv3d(inputs=upsampling2,
                          kernel_size=2,
                          num_outputs=256,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv2.shape)
    concat2 = tf.concat([conv3, upconv2], 4)
    print(concat2.shape)
    conv3 = slim.repeat(inputs=concat2,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)

    upsampling3 = slim.conv3d_transpose(inputs=conv3,
                                        kernel_size=3,
                                        num_outputs=256,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling3.shape)
    upconv3 = slim.conv3d(inputs=upsampling3,
                          kernel_size=2,
                          num_outputs=128,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv3.shape)
    concat3 = tf.concat([conv2, upconv3], 4)
    print(concat3.shape)
    conv2 = slim.repeat(inputs=concat3,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=128,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)

    upsampling4 = slim.conv3d_transpose(inputs=conv2,
                                        kernel_size=3,
                                        num_outputs=128,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling4.shape)
    upconv4 = slim.conv3d(inputs=upsampling4,
                          kernel_size=2,
                          num_outputs=64,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv4.shape)
    concat4 = tf.concat([conv1, upconv4], 4)
    print(concat4.shape)
    conv1 = slim.repeat(inputs=concat4,
                        repetitions=2,
                        layer=slim.conv3d,
                        num_outputs=64,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)

    output = slim.repeat(inputs=conv1,
                         repetitions=1,
                         layer=slim.conv3d,
                         num_outputs=4,
                         activation_fn=tf.identity,
                         kernel_size=1,
                         normalizer_fn=slim.batch_norm)
    print(output.shape)

    return output


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
x_batch_orig = loadFromHdf5(NP_TRAIN_DIR, 'patient001_ED.hdf5', 'train')[2:,:,:]
y_batch_orig = loadFromHdf5(NP_TRAIN_DIR, 'patient001_ED_gt.hdf5', 'train_gt')[2:,:,:]
print(x_batch_orig.shape)
print(y_batch_orig.shape)
print('='*80)

# input sizes
input_d = 8
input_h = 256
input_w = 216


x_batch = np.expand_dims(x_batch_orig, axis=0)
x_batch = np.expand_dims(x_batch, axis=4)
y_batch = np.expand_dims(y_batch_orig, axis=0)
print(x_batch.shape)
print(y_batch.shape)
print('='*80)



# network params
batch_size = x_batch_orig.shape[0]
lr = 1e-2
max_iter = 1000000

# placeholders
x = tf.placeholder(tf.float32, shape=[None, input_d, input_h, input_w, 1])
y_ = tf.placeholder(tf.int32, shape=[None, input_d, input_h, input_w])


y = unet3d(x)

# prediction
prediction = tf.nn.softmax(y)

# loss
def dice_loss(y_, y):
    epsilon = tf.constant(0.00001)
    
    depth = y.get_shape()[4].value
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

            if loss_tr < 0.05:
                break;
    
    # get prediction arrays
    pr = sess.run(prediction, feed_dict=feed_train)
    pr = np.argmax(pr, axis=4)
    print(pr.shape)

    # return gt
    gt = sess.run(y_, feed_dict=feed_train)
    print(gt.shape)

for i in range(input_d):
    gt_slice = gt[0,i,:,:]
    plt.imshow(gt_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_gt.png'))

    pr_slice = pr[0,i,:,:]
    plt.imshow(pr_slice, cmap=mcolors.ListedColormap(colors))
    plt.savefig(os.path.join(configuration_path, str(i).zfill(2)+'_pr.png'))