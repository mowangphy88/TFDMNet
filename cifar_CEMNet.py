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

class RandomDropMaskInit(tf.keras.initializers.Initializer):
    # From 0-1 real mask to FFT freqency mask
    # Size: featureSize
    # TODO: initialize random 0-1 mask based on drop rate and feature size in time domain
    # TODO: transfer to freqency domain and return the real and imaginary masks for depth-wise conv

    def __init__(self, featureSize, droprate):
        self.featureSize = featureSize # batchsize, height, width, channels 0-3
        self.droprate = droprate

        tmp_randMat = np.random.rand(self.featureSize[1], self.featureSize[2], self.featureSize[3], 1)
        self.randFilter_FFT_real = np.zeros_like(tmp_randMat, dtype=np.float)
        self.randFilter_FFT_imag = np.zeros_like(tmp_randMat, dtype=np.float)
        # self.randFilter_FFT_imaginary = np.zeros_like(tmp_randMat, dtype=np.float)

    def __call__(self, dtype=None, **kwargs):
        # TODO: generate 0-1 mask, and do fft2d on the mask
        randMat = np.random.rand(self.featureSize[1], self.featureSize[2], self.featureSize[3])
        randFilter = (randMat >= self.droprate).astype(float)
        randFilter = randFilter * (1/(1-self.droprate))

        log_shift2centerRGB_real = np.zeros_like(randFilter, dtype=np.float)
        log_shift2centerRGB_imaginary = np.zeros_like(randFilter, dtype=np.float)
        for iChannel in range(self.featureSize[3]):
            fft2 = np.fft.fft2(randFilter[:, :, iChannel])
            # shift2center = fft2
            # shift2center = np.fft.fftshift(fft2)
            # log_fft2 = np.log(1 + np.abs(fft2))
            log_shift2centerRGB_real[:, :, iChannel] = fft2.real
            # print(log_shift2centerRGB_real)
            log_shift2centerRGB_imaginary[:, :, iChannel] = fft2.imag

            # print(type(log_shift2centerRGB))
        self.randFilter_FFT_real[:, :, :, 0] = log_shift2centerRGB_real
        self.randFilter_FFT_imag[:, :, :, 0] = log_shift2centerRGB_imaginary
        return self.randFilter_FFT_real, self.randFilter_FFT_imag

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}

class ComplexDotLayer(tf.keras.layers.Layer):
    def __init__(self, featureSize, outChannels, use_bias=False):
        super(ComplexDotLayer, self).__init__()
        # weight_decay = 5e-4
        self.outChannels = outChannels
        self.featureSize = featureSize # b-H-W-C(in the frequency domain)
        self.use_bias = use_bias

        self.mulReal = tf.keras.layers.Multiply()
        self.mulImaginary = tf.keras.layers.Multiply()

        self.bnorm_Real = tf.keras.layers.BatchNormalization()
        self.relu_Real = tf.keras.layers.LeakyReLU(alpha=0.2)  # Default: alpha=0.3
        self.bnorm_Imaginary = tf.keras.layers.BatchNormalization()
        self.relu_Imaginary = tf.keras.layers.LeakyReLU(alpha=0.2)
        # self.relu = ComplexBNormReLULayer(self.droprate, self.isPooling, self.pooling_window_size)

        self.initMethod = 0  # 0: regular init; 1: small filter -> zero padding -> 2dfft
        self.convfilterSize = 3
        self.convfilterSizeIdentitySize = 3

        # TODO: generate indentity filter and its corresponding fft (using the same size as feature?)
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

        onesPad = tf.ones_like(convfilterIdentityPad, dtype=tf.float32)
        convfilterIdentityPad_supp = onesPad - convfilterIdentityPad

        self.convfilterIdentityPad = convfilterIdentityPad
        self.convfilterIdentityPad_supp = convfilterIdentityPad_supp

        if self.initMethod == 0:
            if self.convfilterSize >= self.featureSize[1]:
                weight_shape = tf.TensorShape(
                    (self.featureSize[1], self.featureSize[2], self.featureSize[3], self.outChannels))


                self.weightTime = self.add_weight(name='weightTime',
                                              shape=weight_shape,
                                              initializer=tf.keras.initializers.HeNormal(),
                                              regularizer='l2',
                                              trainable=True)        
            else:          
                weight_shape = tf.TensorShape(
                    (self.convfilterSize, self.convfilterSize, self.featureSize[3], self.outChannels))
 

                self.weightTime = self.add_weight(name='weightTime',
                                              shape=weight_shape,
                                              initializer=tf.keras.initializers.HeNormal(),
                                              regularizer='l2',
                                              trainable=True)


    def call(self, inputs, training, isFeatFix=False):
        x_Real = inputs[0]  # N-by-length-by-Channels
        x_Imaginary = inputs[1]

        # batchSize = self.featureSize[0]
        batchSize = x_Real.shape[0]
        
        # TODO: Transfer weightTime to the frequency domain
        if self.convfilterSize < self.featureSize[1]:
            paddingLeft = 0
            paddingRight = self.featureSize[1] - self.convfilterSize
            paddings = tf.constant([[paddingLeft, paddingRight], [paddingLeft, paddingRight], [0, 0], [0, 0]])
            weightPad = tf.pad(self.weightTime, paddings, 'CONSTANT', constant_values=0)
        else:
            weightPad = self.weightTime
            
        weightPad_trans = tf.transpose(weightPad, [2, 3, 0, 1]) # H-W-C_in-C_out -> C_in-C_out-H-W
        weightPad_trans = tf.cast(weightPad_trans, tf.complex64)
        weightFreq = tf.signal.fft2d(weightPad_trans)
        weightFreq = tf.transpose(weightFreq, [2, 3, 0, 1])
        
        weightReal = tf.math.real(weightFreq)
        weightImaginary = tf.math.imag(weightFreq)
        

        # TODO: do weight fix here (multiply with weights or feature?)
        if isFeatFix:
            weightComplex = tf.dtypes.complex(weightReal, weightImaginary)
            
            # ifft2d and fft 2d
            weightComplex_trans = tf.transpose(weightComplex, [2, 3, 0, 1]) # H-W-C_in-C_out -> C_in-C_out-H-W
            weightTime = tf.signal.ifft2d(weightComplex_trans)
            weightTime = tf.math.real(weightTime)

            weightTime_supp = tf.math.multiply(weightTime, self.convfilterIdentityPad_supp)
            weightTime_supp_mean0 = tf.math.reduce_sum(weightTime_supp, axis=2, keepdims=True)
            weightTime_supp_mean1 = tf.math.reduce_sum(weightTime_supp_mean0, axis=3, keepdims=True)
            weightTime_supp_mean1 = weightTime_supp_mean1 / (self.convfilterSize*self.convfilterSize)
            weightTime = weightTime + weightTime_supp_mean1

            weightTime_fixed = tf.math.multiply(weightTime, self.convfilterIdentityPad)
            weightTime_fixed = tf.cast(weightTime_fixed, tf.complex64)
            weightFreq_fixed = tf.signal.fft2d(weightTime_fixed)
            weightFreq_fixed = tf.transpose(weightFreq_fixed, [2, 3, 0, 1])
            

            weightReal = tf.math.real(weightFreq_fixed)
            weightImaginary = tf.math.imag(weightFreq_fixed)

            curWeightReal = tf.expand_dims(weightReal, 0)
            curWeightImaginary = tf.expand_dims(weightImaginary, 0)

        else:
            curWeightReal = tf.expand_dims(weightReal, 0)
            curWeightImaginary = tf.expand_dims(weightImaginary, 0)


        curWeightReal = tf.tile(curWeightReal, [batchSize, 1, 1, 1, 1])
        curWeightImaginary = tf.tile(curWeightImaginary, [batchSize, 1, 1, 1, 1])


        curXReal = tf.expand_dims(x_Real, -1)
        curXReal = tf.tile(curXReal, [1, 1, 1, 1, self.outChannels])
        curXImaginary = tf.expand_dims(x_Imaginary, -1)
        curXImaginary = tf.tile(curXImaginary, [1, 1, 1, 1, self.outChannels])

        out_R_WR = self.mulReal([curXReal, curWeightReal])
        out_I_WI = self.mulImaginary([curXImaginary, curWeightImaginary])
        out_R_WI = self.mulImaginary([curXReal, curWeightImaginary])
        out_I_WR = self.mulReal([curXImaginary, curWeightReal])

        # print(out_R_WR)
        # using correlation instead of conv (conv layer in CNN is correlation in signal processing)
        cur_out_Real = out_R_WR + out_I_WI # + curBiasReal
        cur_out_Imaginary = out_R_WI - out_I_WR # + curBiasImaginary # N-H-W-C_in-C_out, needs a summation over the last channel

        # Firstly finish all calculation, and then do bnorm and relu
        out_Real = tf.math.reduce_sum(cur_out_Real, 3)
        out_Imaginary = tf.math.reduce_sum(cur_out_Imaginary, 3)

        out_Real = self.bnorm_Real(out_Real, training=training)
        # out_Real = self.relu_Real(out_Real)
        out_Imaginary = self.bnorm_Imaginary(out_Imaginary, training=training)
        # out_Imaginary = self.relu_Imaginary(out_Imaginary)
        # out_Real, out_Imaginary = self.relu([out_Real, out_Imaginary], training=training)

        return out_Real, out_Imaginary



class ComplexBNormReLULayer(tf.keras.layers.Layer):
    def __init__(self, droprate, isPooling, pooling_window_size):
        super(ComplexBNormReLULayer, self).__init__()
        self.bnormLayer = tf.keras.layers.BatchNormalization() 
        self.reluLayer = tf.keras.layers.ReLU()
        
        self.droprate = droprate
        if self.droprate > 0:
            self.dropoutLayer = tf.keras.layers.Dropout(self.droprate)
        
        self.isPooling = isPooling
        self.pooling_window_size = pooling_window_size
        if self.isPooling:
            self.pool = tf.keras.layers.MaxPooling2D((self.pooling_window_size, self.pooling_window_size))

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]
    
        x_Freq = tf.dtypes.complex(x_Real, x_Imaginary) # N-H-W-C
        x_Freq_trans = tf.transpose(x_Freq, [3, 0, 1, 2])
        x_Time_trans = tf.signal.ifft2d(x_Freq_trans)
        x_Time_trans = tf.math.real(x_Time_trans)
        x_Time       = tf.transpose(x_Time_trans, [1, 2, 3, 0])
        x_Time_relu  = self.reluLayer(x_Time)
        x_Time_bnorm = self.bnormLayer(x_Time_relu, training=training) 
        if self.droprate > 0:        
            x_Time_dropped  = self.dropoutLayer(x_Time_bnorm, training=training)
        else:
            x_Time_dropped = x_Time_bnorm
        
        if self.isPooling:
            x_Time_dropped = self.pool(x_Time_dropped, training=training)
        
        x_Time_dropped_trans  = tf.transpose(x_Time_dropped, [3, 0, 1, 2])
        x_Time_dropped_trans = tf.cast(x_Time_dropped_trans, tf.complex64)
        x_Freq_dropped_trans = tf.signal.fft2d(x_Time_dropped_trans)
        x_Freq_dropped = tf.transpose(x_Freq_dropped_trans, [1, 2, 3, 0])
        out_Real = tf.math.real(x_Freq_dropped)
        out_Imaginary = tf.math.imag(x_Freq_dropped)

        return out_Real, out_Imaginary


class ComplexDropoutLayer(tf.keras.layers.Layer):
    def __init__(self, droprate):
        super(ComplexDropoutLayer, self).__init__()
        # weight_decay = 5e-4
        self.droprate = droprate
        self.droplayer = tf.keras.layers.Dropout(self.droprate)
        self.droptag = 0

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        if training:
            self.droptag = np.random.rand(1);
            if self.droptag > 0.5: 
                # do dropout on real part
                out_Real = self.droplayer(x_Real, training=True)
                out_Imaginary = x_Imaginary
            else:
                # do dropout on imaginary part
                out_Real = x_Real
                out_Imaginary = self.droplayer(x_Imaginary, training=True)
        else:
            out_Real = x_Real
            out_Imaginary = x_Imaginary

        return out_Real, out_Imaginary
        
        
class ComplexDropoutLayer_approx(tf.keras.layers.Layer):
    def __init__(self, droprate):
        super(ComplexDropoutLayer_approx, self).__init__()

        self.droprate = droprate

        self.mulRealLayer = tf.keras.layers.Multiply()
        self.mulImagLayer = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        [batchSize, H, W, C] = x_Real.shape

        relu_ratio_Real = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))  
        relu_ratio_Imag = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))

        if training:
            drop_ratio_Real = np.random.normal(1.0, self.droprate/2, size=(batchSize, H, W, C))
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
            x_time = tf.transpose(t_time, [1, 2, 3, 0])
            
            # ifft 3d and fft 3d
            # t_time = tf.signal.ifft3d(t)
            # x_time = tf.math.real(t_time)
        else:
            t = tf.transpose(t, [3, 0, 1, 2])
            t_time = tf.signal.ifft2d(t)
            t_time = tf.math.real(t_time)
            x_time = tf.transpose(t_time, [1, 2, 3, 0])

        out_time = self.pool(x_time, training=training)

        if movingback:
            out_time = tf.cast(out_time, dtype=tf.complex64)
            # ifft 2d and fft 2d            
            out_time = tf.transpose(out_time, [3, 0, 1, 2])
            out_freq = tf.signal.fft2d(out_time)
            t_freq = tf.transpose(out_freq, [1, 2, 3, 0])
            
            # ifft 3d and fft 3d
            # t_freq = tf.signal.fft3d(out_time)
                
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


class magnitudeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(magnitudeLayer, self).__init__()

    # @tf.function
    def call(self, inputs):
        x_Real = inputs[0]  # [batchSize, H, W, C]
        x_Imaginary = inputs[1]

        out = tf.math.sqrt(tf.math.add(tf.math.square(x_Real), tf.math.square(x_Imaginary)))

        return out

class VGG16_CEMNet(tf.keras.models.Model):

    def __init__(self, batchSize):
        """

        :param input_shape: [32, 32, 3]
        """
        super(VGG16_CEMNet, self).__init__()

        weight_decay = 5e-4
        self.num_classes = 10
        self.droprate1 = 0.5
        self.droprate2 = 0.5
        self.droprate3 = 0.5
        
        self.padding=1
        self.isCut=True

        self.out_channel = [64, 128, 256, 512, 512]

        self.batchSize = batchSize

        # Input layers: real data and imaginary data
        self.Input_Real = tf.keras.layers.InputLayer(input_shape=(32, 32, 3))
        self.Input_Imaginary = tf.keras.layers.InputLayer(input_shape=(32, 32, 3))

        # Complex Layers:
        # Block 1:
        self.complex1 = ComplexDotLayer([self.batchSize, 32, 32, 3], self.out_channel[0], use_bias=False)  
        self.drop1 = ComplexDropoutLayer_approx(self.droprate1) # (self, inChannels, droprate):

        # self.complex2 = ComplexDotLayer([self.batchSize, 32, 32, self.out_channel[0]], self.out_channel[0], use_bias=False) 
        # self.drop2 = ComplexDropoutLayer_approx(self.droprate1)

        self.pool1 = ComplexPoolLayer(2, [self.batchSize, 32, 32, self.out_channel[0]])
        # self.pool1 = ComplexPoolLayer_magnitude(2)

        # Block 2:
        self.complex3 = ComplexDotLayer([self.batchSize, 16, 16, self.out_channel[0]], self.out_channel[1], use_bias=False) 
        self.drop3 = ComplexDropoutLayer_approx(self.droprate1)

        # self.complex4 = ComplexDotLayer([self.batchSize, 16, 16, self.out_channel[1]], self.out_channel[1], use_bias=False) 
        # self.drop4 = ComplexDropoutLayer_approx(self.droprate1)

        self.pool2 = ComplexPoolLayer(2, [self.batchSize, 16, 16, self.out_channel[1]])
        # self.pool2 = ComplexPoolLayer_magnitude(2)

        # Block 3:
        self.complex5 = ComplexDotLayer([self.batchSize, 8, 8, self.out_channel[1]], self.out_channel[2], use_bias=False) 
        self.drop5 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex6 = ComplexDotLayer([self.batchSize, 8, 8, self.out_channel[2]], self.out_channel[2], use_bias=False)  
        # self.drop6 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex7 = ComplexDotLayer([self.batchSize, 8, 8, self.out_channel[2]], self.out_channel[2], use_bias=False) 
        # self.drop7 = ComplexDropoutLayer_approx(self.droprate2)

        self.pool3 = ComplexPoolLayer(2, [self.batchSize, 8, 8, self.out_channel[2]])
        # self.pool3 = ComplexPoolLayer_magnitude(2)

        # Block 4:
        self.complex8 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[2]], self.out_channel[3], use_bias=False) 
        self.drop8 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex9 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[3]], self.out_channel[3], use_bias=False)
        # self.drop9 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex10 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[3]], self.out_channel[3], use_bias=False)
        # self.drop10 = ComplexDropoutLayer_approx(self.droprate2)

        self.pool4 = ComplexPoolLayer(2, [self.batchSize, 4, 4, self.out_channel[3]])
        # self.pool4 = ComplexPoolLayer_magnitude(2)

        # Block 5:
        self.complex11 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[3]], self.out_channel[4], use_bias=False)
        self.drop11 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex12 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[4]], self.out_channel[4], use_bias=False)
        # self.drop12 = ComplexDropoutLayer_approx(self.droprate2)

        # self.complex13 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[4]], self.out_channel[4], use_bias=False)
        # self.drop13 = ComplexDropoutLayer_approx(self.droprate2)
        #
        # self.pool5_real = tf.keras.layers.MaxPooling2D((2, 2))
        # self.pool5_imaginary = tf.keras.layers.MaxPooling2D((2, 2))
        # self.pool5 = ComplexPoolLayer(2)
        # self.concat = magnitudeLayer()
        # self.concat = ifft2DLayer([self.batchSize, 2, 2, self.out_channel[4]])
        # self.pool5 = tf.keras.layers.MaxPooling2D((2, 2))
        self.pool5 = ComplexPoolLayer(2, [self.batchSize, 2, 2, self.out_channel[4]])
        # self.pool5 = ComplexPoolLayer_magnitude(2)

        # Flatten
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()

        # Block 6.2
        # self.fc1 = tf.keras.layers.Dense(512, activation=None)
        # self.relu14 = tf.keras.layers.ReLU()
        # self.bnorm14 = tf.keras.layers.BatchNormalization()
        # self.drop14 = tf.keras.layers.Dropout(self.droprate3)
        # self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        
        # Block 6.2 complex fc
        self.fc1 = complexfcLayer(512)
        self.drop14 = ComplexDropoutLayer_approx_fc(self.droprate3)
        self.concat = complexconcatLayer()
        self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    # @tf.function
    def call(self, inputs, training=None, isFeatFix=False):
        # Inputs
        input_Real, input_Imaginary = inputs
        x_real = self.Input_Real(input_Real)
        x_imaginary = self.Input_Imaginary(input_Imaginary)
        # input_complex = [input_Real, input_Imaginary]

        # x_real, x_imaginary = self.Flatten_Real(input_complex)

        # Block 1:
        # Block 1:
        x_real, x_imaginary = self.complex1([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        x_real, x_imaginary = self.drop1([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex2([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        # x_real, x_imaginary = self.drop2([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.pool1([x_real, x_imaginary], movingback=True, training=training)
        # x_real, x_imaginary = self.pool1([x_real, x_imaginary], training=training)

        # Block 2:
        x_real, x_imaginary = self.complex3([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        x_real, x_imaginary = self.drop3([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex4([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        # x_real, x_imaginary = self.drop4([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.pool2([x_real, x_imaginary], movingback=True, training=training)
        # x_real, x_imaginary = self.pool2([x_real, x_imaginary], training=training)

        # Block 3:
        x_real, x_imaginary = self.complex5([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        x_real, x_imaginary = self.drop5([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex6([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        # x_real, x_imaginary = self.drop6([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex7([x_real, x_imaginary], training=training, isFeatFix=isFeatFix)  # include bnorm and relu
        # x_real, x_imaginary = self.drop7([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.pool3([x_real, x_imaginary], movingback=True, training=training)
        # x_real, x_imaginary = self.pool3([x_real, x_imaginary], training=training)

        # Block 4:
        x_real, x_imaginary = self.complex8([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop8([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex9([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        # x_real, x_imaginary = self.drop9([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex10([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        # x_real, x_imaginary = self.drop10([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.pool4([x_real, x_imaginary], movingback=True, training=training)
        # x_real, x_imaginary = self.pool4([x_real, x_imaginary], training=training)

        # Block 5:
        x_real, x_imaginary = self.complex11([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop11([x_real, x_imaginary], training=training)

        # x_real, x_imaginary = self.complex12([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        # x_real, x_imaginary = self.drop12([x_real, x_imaginary], training=training)
        
        # x_real, x_imaginary = self.complex13([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        # x_real, x_imaginary = self.drop13([x_real, x_imaginary], training=training)
        #
        # x_real = self.pool5_real(x_real)
        # x_imaginary = self.pool5_imaginary(x_imaginary)
        # x_real, x_imaginary = self.pool5([x_real, x_imaginary])
        # concat = self.concat([x_real, x_imaginary])
        # x_real = self.pool5_real(x_real)
        # x_imaginary = self.pool5_imaginary(x_imaginary)
        # concat = self.concat([x_real, x_imaginary])
        # x = self.pool5(concat)
        
        x_real, x_imaginary = self.pool5([x_real, x_imaginary], movingback=True, training=training)
        # x_real, x_imaginary = self.pool5([x_real, x_imaginary], training=training)

        # Flatten
        # xb1 = self.flatten1(x)

        # Block 6:
        # xb1 = self.fc1(xb1)
        # xb1 = self.relu14(xb1)
        # xb1 = self.bnorm14(xb1, training=training)
        # xb1 = self.drop14(xb1, training=training)
        # ClsResults = self.output1(xb1)
        
        # complex fc layers
        x_real = self.flatten1(x_real)
        x_imaginary = self.flatten2(x_imaginary)

        x_real, x_imaginary = self.fc1([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.drop14([x_real, x_imaginary], training=training)
        xb1 = self.concat([x_real, x_imaginary])
        ClsResults = self.output1(xb1)

        return ClsResults
