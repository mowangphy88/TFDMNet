import tensorflow as tf
import numpy as np
import scipy.io as scio
from tensorflow.keras import regularizers


class RandomFFT2dInitReal(tf.keras.initializers.Initializer):

    def __init__(self, convfilterSize, featureSize, outChannels):
        self.convfilterSize = convfilterSize
        self.featureSize = featureSize
        self.outChannels = outChannels

        if self.convfilterSize >= self.featureSize[1]:
            self.convfilterSize = self.featureSize[1]

    def __call__(self, shape, dtype=None, **kwargs):
        # TODO: add initializer of fft2d
        randFilter = np.random.randn(
                self.convfilterSize, self.convfilterSize, self.featureSize[3], self.outChannels)
        realFeatureSize = self.featureSize[3]

        if self.convfilterSize < self.featureSize[1]:
            paddingLeft = 0
            paddingRight = self.featureSize[1] - self.convfilterSize
            randFilterPad = np.pad(randFilter, ((paddingLeft, paddingRight), (paddingLeft, paddingRight), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            randFilterPad = randFilter
        fftFilter = np.zeros_like(randFilterPad)
        for i in range(realFeatureSize):
            for j in range(self.outChannels):
                fftFilter[:, :, i, j] = np.fft.fft2(randFilterPad[:, :, i, j]).real
        return fftFilter
    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}

class RandomFFT2dInitImag(tf.keras.initializers.Initializer):

    def __init__(self, convfilterSize, featureSize, outChannels):
        self.convfilterSize = convfilterSize
        self.featureSize = featureSize
        self.outChannels = outChannels

        if self.convfilterSize >= self.featureSize[1]:
            self.convfilterSize = self.featureSize[1]

    def __call__(self, shape, dtype=None, **kwargs):
        # TODO: add initializer of fft2d
        randFilter = np.random.randn(
                self.convfilterSize, self.convfilterSize, self.featureSize[3], self.outChannels)
        realFeatureSize = self.featureSize[3]

        if self.convfilterSize < self.featureSize[1]:
            paddingLeft = 0
            paddingRight = self.featureSize[1] - self.convfilterSize
            randFilterPad = np.pad(randFilter, ((paddingLeft, paddingRight), (paddingLeft, paddingRight), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        else:
            randFilterPad = randFilter

        fftFilter = np.zeros_like(randFilterPad)

        for i in range(realFeatureSize):
            for j in range(self.outChannels):
                fftFilter[:, :, i, j] = np.fft.fft2(randFilterPad[:, :, i, j]).imag
        return fftFilter
    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}



class ComplexDotLayer(tf.keras.layers.Layer):
    def __init__(self, featureSize, outChannels, use_bias=False):
        super(ComplexDotLayer, self).__init__()
        # weight_decay = 5e-4
        self.outChannels = outChannels
        self.featureSize = featureSize
        self.use_bias = use_bias

        self.mulReal = tf.keras.layers.Multiply()
        self.mulImaginary = tf.keras.layers.Multiply()

        self.bnorm_Real = tf.keras.layers.BatchNormalization()
        self.relu_Real = tf.keras.layers.LeakyReLU(alpha=0.2)  # Default: alpha=0.3
        self.bnorm_Imaginary = tf.keras.layers.BatchNormalization()
        self.relu_Imaginary = tf.keras.layers.LeakyReLU(alpha=0.2)
        # self.relu = ComplexBNormReLULayer(self.droprate, self.isPooling, self.pooling_window_size)

        self.initMethod = 1  # 0: regular init; 1: small filter -> zero padding -> 2dfft
        self.convfilterSize = 5
        self.convfilterSizeIdentitySize = 5
        
        if self.convfilterSizeIdentitySize >= self.featureSize[1]:
            convfilterIdentityPad = np.ones((self.featureSize[1], self.featureSize[1]))
        else:
            convfilterIdentity = np.ones((self.convfilterSizeIdentitySize, self.convfilterSizeIdentitySize))
            paddingLeft = 0
            paddingRight = self.featureSize[1] - self.convfilterSizeIdentitySize # + self.featureSize[1] # L >= N + M - 1
            convfilterIdentityPad = np.pad(convfilterIdentity,
                                           ((paddingLeft, paddingRight), (paddingLeft, paddingRight)),
                                           'constant', constant_values=(0, 0)) # , self.realFeatureSize, self.outChannels

        convfilterIdentityPad = tf.expand_dims(convfilterIdentityPad, 2)
        convfilterIdentityPad = tf.expand_dims(convfilterIdentityPad, 3)
        convfilterIdentityPad = tf.tile(convfilterIdentityPad, [1, 1, self.featureSize[3], self.outChannels])
        
        # fft 2d
        convfilterIdentityPad = tf.transpose(convfilterIdentityPad, [2, 3, 0, 1])
        # fft 3d
        # convfilterIdentityPad = tf.transpose(convfilterIdentityPad, [3, 0, 1, 2])
        
        convfilterIdentityPad = tf.cast(convfilterIdentityPad, tf.float32)
        self.convfilterIdentityPad = convfilterIdentityPad
        # convfilterIdentityPad_freq = np.fft.fft2(convfilterIdentityPad)       
        # convfilterIdentityPad_freq = np.fft.fftshift(convfilterIdentityPad_freq) # do fft-shift for feat fix       
        # self.convfilterIdentityPad_freq_real = tf.cast(convfilterIdentityPad_freq.real, dtype=tf.float32)
        # self.convfilterIdentityPad_freq_imag = tf.cast(convfilterIdentityPad_freq.imag, dtype=tf.float32)

        if self.initMethod == 0:
            weight_shape = tf.TensorShape(
                (self.featureSize[1], self.featureSize[2], self.featureSize[3], self.outChannels))

            bias_shape = tf.TensorShape(
                (self.featureSize[1], self.featureSize[2], self.featureSize[3], self.outChannels))  # H-W-C-C

            self.weightReal = self.add_weight(name='weightReal',
                                              shape=weight_shape,
                                              initializer=tf.keras.initializers.HeNormal(),
                                              regularizer='l2',
                                              trainable=True)
            self.weightImaginary = self.add_weight(name='weightImaginary',
                                                   shape=weight_shape,
                                                   initializer=tf.keras.initializers.HeNormal(),
                                                   regularizer='l2',
                                                   trainable=True)
            if self.use_bias:
                self.biasReal = self.add_weight(name='biasReal',
                                                  shape=bias_shape,
                                                  initializer=tf.keras.initializers.Constant(),
                                                  trainable=True)
                self.biasImaginary = self.add_weight(name='biasImaginary',
                                                       shape=bias_shape,
                                                       initializer=tf.keras.initializers.Constant(),
                                                       trainable=True)

        else:
            weight_shape = tf.TensorShape(
                    (self.featureSize[1], self.featureSize[2], self.featureSize[3], self.outChannels))
                    
            bias_shape = tf.TensorShape(
                (self.featureSize[1], self.featureSize[2], self.featureSize[3], self.outChannels))  # H-W-C-C

            self.weightReal = self.add_weight(name='weightReal',
                                              shape=weight_shape,
                                              initializer=RandomFFT2dInitReal(self.convfilterSize, self.featureSize,
                                                                              self.outChannels),
                                              trainable=True)
            self.weightImaginary = self.add_weight(name='weightImaginary',
                                                   shape=weight_shape,
                                                   initializer=RandomFFT2dInitImag(self.convfilterSize,
                                                                                   self.featureSize, self.outChannels),
                                                   trainable=True)
            if self.use_bias:                                       
                self.biasReal = self.add_weight(name='biasReal',
                                                  shape=bias_shape,
                                                  initializer=tf.keras.initializers.Constant(),
                                                  trainable=True)
                self.biasImaginary = self.add_weight(name='biasImaginary',
                                                       shape=bias_shape,
                                                       initializer=tf.keras.initializers.Constant(),
                                                       trainable=True)

    def call(self, inputs, training, isWeightFix=False):
        x_Real = inputs[0]  # N-by-length-by-Channels
        x_Imaginary = inputs[1]

        # batchSize = self.featureSize[0]
        batchSize = x_Real.shape[0]

        # TODO: do weight fix here (multiply with weights or feature?)
        if isWeightFix:
            weightComplex = tf.dtypes.complex(self.weightReal, self.weightImaginary)
            
            # ifft2d and fft 2d
            weightComplex_trans = tf.transpose(weightComplex, [2, 3, 0, 1]) # H-W-C_in-C_out -> C_in-C_out-H-W
            weightTime = tf.signal.ifft2d(weightComplex_trans)
            weightTime = tf.math.real(weightTime)
            weightTime_fixed = tf.math.multiply(weightTime, self.convfilterIdentityPad)
            weightTime_fixed = tf.cast(weightTime_fixed, tf.complex64)
            weightFreq_fixed = tf.signal.fft2d(weightTime_fixed)
            weightFreq_fixed = tf.transpose(weightFreq_fixed, [2, 3, 0, 1])
            
            
            weightRealFixed = tf.math.real(weightFreq_fixed)
            weightImagFixed = tf.math.imag(weightFreq_fixed)
            
            self.weightReal.assign(weightRealFixed)
            self.weightImaginary.assign(weightImagFixed)

            curWeightReal = tf.expand_dims(self.weightReal, 0)
            curWeightImaginary = tf.expand_dims(self.weightImaginary, 0)
            if self.use_bias:
                curBiasReal = tf.expand_dims(self.biasReal, 0)
                curBiasImaginary = tf.expand_dims(self.biasImaginary, 0)
        else:
            curWeightReal = tf.expand_dims(self.weightReal, 0)
            curWeightImaginary = tf.expand_dims(self.weightImaginary, 0)
            if self.use_bias:
                curBiasReal = tf.expand_dims(self.biasReal, 0)
                curBiasImaginary = tf.expand_dims(self.biasImaginary, 0)

        curWeightReal = tf.tile(curWeightReal, [batchSize, 1, 1, 1, 1])
        curWeightImaginary = tf.tile(curWeightImaginary, [batchSize, 1, 1, 1, 1])
        if self.use_bias:
            curBiasReal = tf.tile(curBiasReal, [batchSize, 1, 1, 1, 1])
            curBiasImaginary = tf.tile(curBiasImaginary, [batchSize, 1, 1, 1, 1])

        curXReal = tf.expand_dims(x_Real, -1)
        curXReal = tf.tile(curXReal, [1, 1, 1, 1, self.outChannels])
        curXImaginary = tf.expand_dims(x_Imaginary, -1)
        curXImaginary = tf.tile(curXImaginary, [1, 1, 1, 1, self.outChannels])

        out_R_WR = self.mulReal([curXReal, curWeightReal])
        out_I_WI = self.mulImaginary([curXImaginary, curWeightImaginary])
        out_R_WI = self.mulImaginary([curXReal, curWeightImaginary])
        out_I_WR = self.mulReal([curXImaginary, curWeightReal])

        # using correlation instead of conv (conv layer in CNN is correlation in signal processing)
        cur_out_Real = out_R_WR + out_I_WI
        cur_out_Imaginary = out_R_WI - out_I_WR
        if self.use_bias:
            cur_out_Real = cur_out_Real + curBiasReal
            cur_out_Imaginary = cur_out_Imaginary + curBiasImaginary

        # Firstly finish all calculation, and then do bnorm and relu
        out_Real = tf.math.reduce_sum(cur_out_Real, 3)
        out_Imaginary = tf.math.reduce_sum(cur_out_Imaginary, 3)

        out_Real = self.bnorm_Real(out_Real, training=training)
        # out_Real = self.relu_Real(out_Real)
        out_Imaginary = self.bnorm_Imaginary(out_Imaginary, training=training)
        # out_Imaginary = self.relu_Imaginary(out_Imaginary)

        return out_Real, out_Imaginary


class ComplexDropoutLayer_approx(tf.keras.layers.Layer): # Leaky ReLU included
    def __init__(self, droprate):
        super(ComplexDropoutLayer_approx, self).__init__()

        self.droprate = droprate

        self.mulRealLayer = tf.keras.layers.Multiply()
        self.mulImagLayer = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        [batchSize, H, W, C] = x_Real.shape

        relu_ratio_Real = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))  # small droprate means small variance of normal distribution
        relu_ratio_Imag = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))

        if training:
            drop_ratio_Real = np.random.normal(1.0, self.droprate/2, size=(batchSize, H, W, C))# small droprate means small variance of normal distribution
            drop_ratio_Imag = np.random.normal(1.0, self.droprate/2, size=(batchSize, H, W, C))

            out_Real = self.mulRealLayer([x_Real, relu_ratio_Real])
            out_Real = self.mulRealLayer([out_Real, drop_ratio_Real])
            out_Imaginary = self.mulRealLayer([x_Imaginary, relu_ratio_Imag])
            out_Imaginary = self.mulRealLayer([out_Imaginary, drop_ratio_Imag])
        else:
            out_Real = self.mulRealLayer([x_Real, relu_ratio_Real])
            out_Imaginary = self.mulRealLayer([x_Imaginary, relu_ratio_Imag])

        return out_Real, out_Imaginary
    
class ComplexDropoutLayer_approx_fc(tf.keras.layers.Layer):
    def __init__(self, droprate):
        super(ComplexDropoutLayer_approx_fc, self).__init__()

        self.droprate = droprate

        self.mulRealLayer = tf.keras.layers.Multiply()
        self.mulImagLayer = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        if training:
            [batchSize, L] = x_Real.shape
            drop_ratio_Real = np.random.normal(1.0, self.droprate/2, size=(batchSize, L))# small droprate means small variance of normal distribution
            drop_ratio_Imag = np.random.normal(1.0, self.droprate/2, size=(batchSize, L))

            out_Real = self.mulRealLayer([x_Real, drop_ratio_Real])
            out_Imaginary = self.mulRealLayer([x_Imaginary, drop_ratio_Imag])
        else:
            out_Real = x_Real #/(1-self.droprate/2)
            out_Imaginary = x_Imaginary #/(1-self.droprate/2)

        return out_Real, out_Imaginary
        

class ComplexPoolLayer(tf.keras.layers.Layer):
    # TODO: Modify the pooling layer: (1) ifft2D; (2) Pooling; (3) Moving back to frequency domain or not

    def __init__(self, pooling_window_size, featureSize):
        super(ComplexPoolLayer, self).__init__()  # using average pooling
        # weight_decay = 5e-4
        self.pooling_window_size = pooling_window_size
        self.featureSize = featureSize
        self.pool = tf.keras.layers.MaxPooling2D((self.pooling_window_size, self.pooling_window_size))

    def call(self, inputs, movingback, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        t = tf.dtypes.complex(x_Real, x_Imaginary)
        if movingback:
            # ifft 2d and fft 2d
            t = tf.transpose(t, [3, 0, 1, 2]) # N-H-W-C
            t_time = tf.signal.ifft2d(t)
            t_time = tf.math.real(t_time)
            x_time = tf.transpose(t_time, [1, 2, 3, 0]);
            
        else:
            t = tf.transpose(t, [3, 0, 1, 2])
            t_time = tf.signal.ifft2d(t)
            t_time = tf.math.real(t_time)
            x_time = tf.transpose(t_time, [1, 2, 3, 0]);

        out_time = self.pool(x_time, training=training)

        if movingback:
            out_time = tf.cast(out_time, dtype=tf.complex64)
            # ifft 2d and fft 2d            
            out_time = tf.transpose(out_time, [3, 0, 1, 2])
            out_freq = tf.signal.fft2d(out_time)
            t_freq = tf.transpose(out_freq, [1, 2, 3, 0]);
                
            out_Real = tf.math.real(t_freq)
            out_Imaginary = tf.math.imag(t_freq)
        else:
            out_Real = out_time
            out_Imaginary = out_time
        return out_Real, out_Imaginary
    
class complexfcLayer(tf.keras.layers.Layer):
    def __init__(self, neurNum):
        super(complexfcLayer, self).__init__()
        self.neurNum = neurNum
        self.fc_real = tf.keras.layers.Dense(self.neurNum, activation=None)
        self.fc_imag = tf.keras.layers.Dense(self.neurNum, activation=None)

        self.bnorm_Real = tf.keras.layers.BatchNormalization()
        self.relu_Real = tf.keras.layers.LeakyReLU(alpha=0.2)  # Default: alpha=0.3
        self.bnorm_Imaginary = tf.keras.layers.BatchNormalization()
        self.relu_Imaginary = tf.keras.layers.LeakyReLU(alpha=0.2)

    # @tf.function
    def call(self, inputs, training):
        x_Real = inputs[0]  # [batchSize, H, W, C]
        x_Imaginary = inputs[1]

        out_Real = self.fc_real(x_Real)
        out_Imaginary = self.fc_imag(x_Imaginary)

        out_Real = self.bnorm_Real(out_Real, training=training)
        out_Real = self.relu_Real(out_Real)
        out_Imaginary = self.bnorm_Imaginary(out_Imaginary, training=training)
        out_Imaginary = self.relu_Imaginary(out_Imaginary)

        return out_Real, out_Imaginary


class complexconcatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(complexconcatLayer, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=1)

    # @tf.function
    def call(self, inputs):
        x_Real = inputs[0]  # [batchSize, H, W, C]
        x_Imaginary = inputs[1]

        out = self.concat([x_Real, x_Imaginary])

        return out


class LeNet_CEMNet(tf.keras.models.Model):

    def __init__(self, batchSize):
        """

        :param input_shape: [32, 32, 3]
        """
        super(LeNet_CEMNet, self).__init__()

        weight_decay = 5e-4
        self.num_classes = 10
        self.droprate1 = 0.4
        self.droprate2 = 0.5

        self.out_channel = [6, 16]

        self.batchSize = batchSize

        # Input layers: real data and imaginary data
        self.Input_Real = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
        self.Input_Imaginary = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))

        # Complex Layers:
        # Block 1:
        self.complex1 = ComplexDotLayer([self.batchSize, 28, 28, 1], self.out_channel[0], use_bias=False)
        self.drop1 = ComplexDropoutLayer_approx(self.droprate1) # (self, inChannels, droprate):
        self.pool1 = ComplexPoolLayer(2, [self.batchSize, 28, 28, self.out_channel[0]])

        # Block 2:
        self.complex2 = ComplexDotLayer([self.batchSize, 14, 14, self.out_channel[0]], self.out_channel[1], use_bias=False)
        self.drop2 = ComplexDropoutLayer_approx(self.droprate1)
        self.pool2 = ComplexPoolLayer(2, [self.batchSize, 14, 14, self.out_channel[1]])


        # Flatten
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()

        # complex fc
        self.fc1 = complexfcLayer(120)
        self.drop3 = ComplexDropoutLayer_approx_fc(self.droprate2)
        self.fc2 = complexfcLayer(84)
        self.drop4 = ComplexDropoutLayer_approx_fc(self.droprate2)
        self.concat = complexconcatLayer()
        self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=None, isWeightFix=False):
        # Inputs
        input_Real, input_Imaginary = inputs
        x_real = self.Input_Real(input_Real)
        x_imaginary = self.Input_Imaginary(input_Imaginary)

        # Block 1:
        x_real, x_imaginary = self.complex1([x_real, x_imaginary], training=training, isWeightFix=isWeightFix)  # include bnorm and relu
        x_real, x_imaginary = self.drop1([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.pool1([x_real, x_imaginary], movingback=True, training=training)

        # Block 2:
        x_real, x_imaginary = self.complex2([x_real, x_imaginary], training=training, isWeightFix=isWeightFix)  # include bnorm and relu
        x_real, x_imaginary = self.drop2([x_real, x_imaginary], training=training)
        x, _ = self.pool2([x_real, x_imaginary], movingback=False, training=training)

        # complex fc layers
        x_real = self.flatten1(x_real)
        x_imaginary = self.flatten2(x_imaginary)

        x_real, x_imaginary = self.fc1([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.drop3([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.fc2([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.drop4([x_real, x_imaginary], training=training)
        xb1 = self.concat([x_real, x_imaginary])
        ClsResults = self.output1(xb1)

        return ClsResults
