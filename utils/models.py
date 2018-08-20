import tensorflow as tf
import tensorflow.contrib.slim as slim


def unet(inputs):
    """
    unet model without softmax.
    """
    print(inputs.shape)
    conv1 = slim.repeat(inputs=inputs,
                        repetitions=2,
                        layer=slim.layers.conv2d,
                        num_outputs=64,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)
    pool1 = slim.max_pool2d(inputs=conv1, kernel_size=2)
    print(pool1.shape)

    conv2 = slim.repeat(inputs=pool1,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=128,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)
    pool2 = slim.max_pool2d(inputs=conv2, kernel_size=2)
    print(pool2.shape)

    conv3 = slim.repeat(inputs=pool2,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)
    pool3 = slim.max_pool2d(inputs=conv3, kernel_size=2)
    print(pool3.shape)

    conv4 = slim.repeat(inputs=pool3,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv4.shape)
    pool4 = slim.max_pool2d(inputs=conv4, kernel_size=2)
    print(pool4.shape)

    conv5 = slim.repeat(inputs=pool4,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=1024,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv5.shape)

    upsampling1 = slim.conv2d_transpose(inputs=conv5,
                                        kernel_size=3,
                                        num_outputs=1024,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling1.shape)
    upconv1 = slim.conv2d(inputs=upsampling1,
                          kernel_size=2,
                          num_outputs=512,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv1.shape)
    concat1 = tf.concat([conv4, upconv1], 3)
    print(concat1.shape)

    conv4 = slim.repeat(inputs=concat1,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv4.shape)

    upsampling2 = slim.conv2d_transpose(inputs=conv4,
                                        kernel_size=3,
                                        num_outputs=512,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling2.shape)
    upconv2 = slim.conv2d(inputs=upsampling2,
                          kernel_size=2,
                          num_outputs=256,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv2.shape)
    concat2 = tf.concat([conv3, upconv2], 3)
    print(concat2.shape)
    conv3 = slim.repeat(inputs=concat2,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)

    upsampling3 = slim.conv2d_transpose(inputs=conv3,
                                        kernel_size=3,
                                        num_outputs=256,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling3.shape)
    upconv3 = slim.conv2d(inputs=upsampling3,
                          kernel_size=2,
                          num_outputs=128,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv3.shape)
    concat3 = tf.concat([conv2, upconv3], 3)
    print(concat3.shape)
    conv2 = slim.repeat(inputs=concat3,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=128,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)

    upsampling4 = slim.conv2d_transpose(inputs=conv2,
                                        kernel_size=3,
                                        num_outputs=128,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling4.shape)
    upconv4 = slim.conv2d(inputs=upsampling4,
                          kernel_size=2,
                          num_outputs=64,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv4.shape)
    concat4 = tf.concat([conv1, upconv4], 3)
    print(concat4.shape)
    conv1 = slim.repeat(inputs=concat4,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=64,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)

    output = slim.repeat(inputs=conv1,
                         repetitions=1,
                         layer=slim.conv2d,
                         num_outputs=4,
                         activation_fn=tf.identity,
                         kernel_size=1,
                         normalizer_fn=slim.batch_norm)
    print(output.shape)

    return output
def unet3d(inputs):
    """
    unet3D model without softmax.
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
def autoencoder(inputs):
    """
    Autoencoder model.
    """
    print(inputs.shape)
    conv1 = slim.repeat(inputs=inputs,
                        repetitions=2,
                        layer=slim.layers.conv2d,
                        num_outputs=64,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)
    pool1 = slim.max_pool2d(inputs=conv1, kernel_size=2)
    print(pool1.shape)

    conv2 = slim.repeat(inputs=pool1,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=128,
                        kernel_size=3,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)
    pool2 = slim.max_pool2d(inputs=conv2, kernel_size=2)
    print(pool2.shape)

    conv3 = slim.repeat(inputs=pool2,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)
    pool3 = slim.max_pool2d(inputs=conv3, kernel_size=2)
    print(pool3.shape)

    conv4 = slim.repeat(inputs=pool3,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv4.shape)
    pool4 = slim.max_pool2d(inputs=conv4, kernel_size=2)
    print(pool4.shape)

    conv5 = slim.repeat(inputs=pool4,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=1024,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv5.shape)

    upsampling1 = slim.conv2d_transpose(inputs=conv5,
                                        kernel_size=3,
                                        num_outputs=1024,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling1.shape)
    upconv1 = slim.conv2d(inputs=upsampling1,
                          kernel_size=2,
                          num_outputs=512,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv1.shape)
    
    conv4 = slim.repeat(inputs=upconv1,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=512,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv4.shape)

    upsampling2 = slim.conv2d_transpose(inputs=conv4,
                                        kernel_size=3,
                                        num_outputs=512,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling2.shape)
    upconv2 = slim.conv2d(inputs=upsampling2,
                          kernel_size=2,
                          num_outputs=256,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv2.shape)
    conv3 = slim.repeat(inputs=upconv2,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=256,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv3.shape)

    upsampling3 = slim.conv2d_transpose(inputs=conv3,
                                        kernel_size=3,
                                        num_outputs=256,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling3.shape)
    upconv3 = slim.conv2d(inputs=upsampling3,
                          kernel_size=2,
                          num_outputs=128,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv3.shape)
    conv2 = slim.repeat(inputs=upconv3,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=128,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv2.shape)

    upsampling4 = slim.conv2d_transpose(inputs=conv2,
                                        kernel_size=3,
                                        num_outputs=128,
                                        stride=2,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm)
    print(upsampling4.shape)
    upconv4 = slim.conv2d(inputs=upsampling4,
                          kernel_size=2,
                          num_outputs=64,
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm)
    print(upconv4.shape)
    conv1 = slim.repeat(inputs=upconv4,
                        repetitions=2,
                        layer=slim.conv2d,
                        num_outputs=64,
                        activation_fn=tf.nn.relu,
                        kernel_size=3,
                        normalizer_fn=slim.batch_norm)
    print(conv1.shape)

    output = slim.repeat(inputs=conv1,
                         repetitions=2,
                         layer=slim.conv2d,
                         num_outputs=4,
                         activation_fn=tf.nn.relu,
                         kernel_size=1,
                         normalizer_fn=slim.batch_norm)
    print(output.shape)

    return output
