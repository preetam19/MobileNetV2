import tensorflow as tf
from functools import partial
import numpy as np
from tensorflow.keras import initializers


class MobileNetV2():
    def __init__(self, num_class, is_training, num_bits=None, depth_multiplier=1, quant_mode='tensorflow',
                 conv2d_regularizer=None):
        self.add_fake_quant = quant_mode == 'custom' and num_bits is not None
        self.num_class = num_class
        self.num_bits = num_bits
        self.conv2d_regularizer = conv2d_regularizer
        self.relu6 = partial(self._relu6, num_bits)
        self.conv2d = partial(self._conv2d, num_bits)
        self.depthwise_conv2d = partial(self._depthwise_conv2d, num_bits)
        self.is_training = is_training
        self.depth_multiplier = depth_multiplier
        tf.logging.info("Creating graph. is_training=%s, width multiplier=%.2f add_fake_quant=%s" % (
        self.is_training, self.depth_multiplier, self.add_fake_quant))

    # Define ReLU6 activation
    def _relu6(self, num_bits, x):
        with tf.variable_scope("act"):
            x = tf.nn.relu6(x)
            if self.add_fake_quant:
                x = tf.fake_quant_with_min_max_vars(x, 0.0, 6.0, num_bits)
        return x

    def _conv2d(self,
                num_bits,
                x,
                filters,
                kernels,
                strides=1,
                bias=False,
                padding='SAME',
                name='conv2d'):
        with tf.variable_scope(name):
            n_input_plane = x.get_shape().as_list()[3]
            w_dim = [kernels, kernels, n_input_plane, filters]
            w = tf.get_variable("weight", w_dim,
                                initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                                regularizer=self.conv2d_regularizer)
            if self.add_fake_quant:
                w_min = tf.reduce_min(w)
                w_max = tf.reduce_max(w)
                w = tf.fake_quant_with_min_max_vars(w, w_min, w_max, num_bits)

            output = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding)

            if bias:
                b = tf.get_variable('bias', [filters])
                output = tf.nn.bias_add(output, b)
        # tf.logging.info('conv2d output tensor: %s'%x.get_shape())

        return output

    def _depthwise_conv2d(self,
                          num_bits,
                          x,
                          filters,
                          kernels,
                          strides=1,
                          bias=False,
                          padding='SAME',
                          name='depthwise_conv2d'):

        with tf.variable_scope(name):
            n_input_plane = x.get_shape().as_list()[3]
            w_dim = [kernels, kernels, filters, 1]
            w = tf.get_variable("weight", w_dim,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            if self.add_fake_quant:
                w_min = tf.reduce_min(w)
                w_max = tf.reduce_max(w)
                w = tf.fake_quant_with_min_max_vars(w, w_min, w_max, num_bits)

            output = tf.nn.depthwise_conv2d(x, w, [1, strides, strides, 1], padding)

            if bias:
                b = tf.get_variable('bias', [filters])
                output = tf.nn.bias_add(output, b)
        # tf.logging.info('depthwise output tensor: %s'%x.get_shape())

        return output

    def _conv_block(self, x, filters, kernel, strides):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
        # Returns
            Output tensor.
        """

        y = self.conv2d(filters, kernel, padding='same', strides=strides)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        return self.relu6(y)

    def _bottleneck(self, x, filters, kernel, t, s, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            r: Boolean, Whether to use the residuals.
        # Returns
            Output tensor.
        """

        tchannel = x.shape[-1] * t

        y = self._conv_block(x, tchannel, (1, 1), (1, 1))

        y = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = self.relu6(y)

        y = self.conv2d(filters, (1, 1), strides=(1, 1), padding='same')(y)
        y = tf.keras.layers.BatchNormalization()(y)

        if r:
            y = tf.keras.layers.add([y, x])
        return y

    def _inverted_residual_block(self, x, filters, kernel, t, strides, n):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        y = self._bottleneck(x, filters, kernel, t, strides)

        for i in range(1, n):
            y = self._bottleneck(y, filters, kernel, t, 1, True)

        return y

    def forward_pass(self,input_shape, k=10 ):
        """MobileNetv2
        This function defines a MobileNetv2 architecture.
        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            k: Integer, number of classes.
            plot_model: Boolean, whether to plot model architecture or not
        # Returns
            MobileNetv2 model.
        """

        inputs = tf.keras.layers.Input(shape=input_shape, name='input')
        y = self._conv_block(inputs, 32, (3, 3), strides=(2, 2))

        y = self._inverted_residual_block(y, 16, (3, 3), t=1, strides=1, n=1)
        y = self._inverted_residual_block(y, 24, (3, 3), t=6, strides=2, n=2)
        y = self._inverted_residual_block(y, 32, (3, 3), t=6, strides=2, n=3)
        y = self._inverted_residual_block(y, 64, (3, 3), t=6, strides=2, n=4)
        y = self._inverted_residual_block(y, 96, (3, 3), t=6, strides=1, n=3)
        y = self._inverted_residual_block(y, 160, (3, 3), t=6, strides=2, n=3)
        y = self._inverted_residual_block(y, 320, (3, 3), t=6, strides=1, n=1)

        y = self._conv_block(y, 1280, (1, 1), strides=(1, 1))
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Reshape((1, 1, 1280))(y)
        y = tf.keras.layers.Dropout(0.3, name='Dropout')(y)
        y = tf.keras.layers.Conv2D(k, (1, 1), padding='same')(y)
        y = tf.keras.layers.Activation('softmax', name='final_activation')(y)
        output = tf.keras.layers.Reshape((k,), name='output')(y)
        model = tf.keras.models.Model(inputs, output)
        model.summary()
