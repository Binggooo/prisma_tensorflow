import numpy as np
import tensorflow as tf


class Vgg16:

    WIDTH = 224
    HEIGHT = 224
    CHANNELS = 3
    _VGG_MEAN = [103.939, 116.779, 123.68]

    model = None
    inputRGB = None

    def __init__(self, model_path):
        self.model = np.load(model_path, encoding='latin1').item()
        self.inputRGB = tf.Variable(np.zeros((1, self.WIDTH, self.HEIGHT, self.CHANNELS)), dtype='float32', trainable=True)

        # Size: 224x224x3
        red, green, blue = tf.split(self.inputRGB, 3, 3)
        self._inputBGR = tf.concat([blue, green, red], 3)
        self._inputNormalizedBGR = tf.concat([blue - self._VGG_MEAN[0], green - self._VGG_MEAN[1], red - self._VGG_MEAN[2]], 3)

        # Size: 224x224x64
        self._conv1_1 = self._conv_layer(self._inputNormalizedBGR, "conv1_1")
        self._conv1_2 = self._conv_layer(self._conv1_1, "conv1_2")
        # Size: 112x112x64
        self._pool1 = self._max_pool(self._conv1_2, 'pool1')

        # Size: 112x112x128
        self._conv2_1 = self._conv_layer(self._pool1, "conv2_1")
        self._conv2_2 = self._conv_layer(self._conv2_1, "conv2_2")
        # Size: 56x56x128
        self._pool2 = self._max_pool(self._conv2_2, 'pool2')

        # Size: 56x56x256
        self._conv3_1 = self._conv_layer(self._pool2, "conv3_1")
        self._conv3_2 = self._conv_layer(self._conv3_1, "conv3_2")
        self._conv3_3 = self._conv_layer(self._conv3_2, "conv3_3")
        # Size: 28x28x256
        self._pool3 = self._max_pool(self._conv3_3, 'pool3')

        # Size: 28x28x512
        self._conv4_1 = self._conv_layer(self._pool3, "conv4_1")
        self._conv4_2 = self._conv_layer(self._conv4_1, "conv4_2")
        self._conv4_3 = self._conv_layer(self._conv4_2, "conv4_3")
        # Size: 14x14x512
        self._pool4 = self._max_pool(self._conv4_3, 'pool4')

        # Size: 14x14x512
        self._conv5_1 = self._conv_layer(self._pool4, "conv5_1")
        # self._conv5_2 = self._conv_layer(self._conv5_1, "conv5_2")
        # self._conv5_3 = self._conv_layer(self._conv5_2, "conv5_3")
        # # Size: 7x7x512
        # self._pool5 = self._max_pool(self._conv5_3, 'pool5')
        #
        # # Size: 25088(=7x7x512)x4096
        # self._fc6 = self._fc_layer(self._pool5, "fc6")
        # self._relu6 = tf.nn.relu(self._fc6)
        #
        # # Size: 4096x4096
        # self._fc7 = self._fc_layer(self._relu6, "fc7")
        # self._relu7 = tf.nn.relu(self._fc7)
        #
        # # Size: 4096x1000
        # self._fc8 = self._fc_layer(self._relu7, "fc8")
        # self._class = tf.nn.softmax(self._fc8, name="classification")

    @property
    def contentRGB(self):
        return self._conv4_2

    @property
    def stylesRGB(self):
        return [self._conv1_1, self._conv2_1, self._conv3_1, self._conv4_1, self._conv5_1]

    @property
    def outputRGB(self):
        red, green, blue = tf.split(self._inputBGR, 3, 3)
        outputRGB = tf.concat([red + self._VGG_MEAN[2], green + self._VGG_MEAN[1], blue + self._VGG_MEAN[0]], 3)
        return outputRGB

    def _avg_pool(self, value, name):
        return tf.nn.avg_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _max_pool(self, value, name):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, value, name):
        with tf.variable_scope(name):
            filt = self._get_conv_filter(name)
            conv = tf.nn.conv2d(value, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self._get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, value, name):
        with tf.variable_scope(name):
            shape = value.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(value, [-1, dim])
            weights = self._get_fc_weight(name)
            biases = self._get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def _get_conv_filter(self, name):
        return tf.constant(self.model[name][0], name="filter", dtype='float32')

    def _get_bias(self, name):
        return tf.constant(self.model[name][1], name="biases", dtype='float32')

    def _get_fc_weight(self, name):
        return tf.constant(self.model[name][0], name="weights", dtype='float32')
