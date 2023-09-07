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
    def __init__(self, featureSize, outChannels, use_bias=False, **kwargs):
        super(ComplexDotLayer, self).__init__(**kwargs)
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

        self.initMethod = 0 
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
                                              initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=2.0),
                                              regularizer='l2',
                                              trainable=True)        
            else:          
                weight_shape = tf.TensorShape(
                    (self.convfilterSize, self.convfilterSize, self.featureSize[3], self.outChannels))
 

                self.weightTime = self.add_weight(name='weightTime',
                                              shape=weight_shape,
                                              initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=2.0),
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


        # 20230322: try broadcast here
        # curWeightReal = tf.tile(curWeightReal, [batchSize, 1, 1, 1, 1]) # bs, H, W, Cin, Cout
        # curWeightImaginary = tf.tile(curWeightImaginary, [batchSize, 1, 1, 1, 1])
        curWeightReal = tf.transpose(curWeightReal, [1, 0, 2, 3, 4]) # H, 1, W, Cin, Cout
        curWeightImaginary = tf.transpose(curWeightImaginary, [1, 0, 2, 3, 4])


        curXReal = tf.expand_dims(x_Real, -1) # bs, H, W, C-in, 1
        # curXReal = tf.tile(curXReal, [1, 1, 1, 1, self.outChannels])
        curXReal = tf.transpose(curXReal, [1, 0, 2, 3, 4]) # H, bs, W, Cin, 1
        
        curXImaginary = tf.expand_dims(x_Imaginary, -1)
        # curXImaginary = tf.tile(curXImaginary, [1, 1, 1, 1, self.outChannels])
        curXImaginary = tf.transpose(curXImaginary, [1, 0, 2, 3, 4])

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
        
        out_Real = tf.transpose(out_Real, [1, 0, 2, 3])
        out_Imaginary = tf.transpose(out_Imaginary, [1, 0, 2, 3])

        out_Real = self.bnorm_Real(out_Real, training=training)
        out_Real = self.relu_Real(out_Real)
        out_Imaginary = self.bnorm_Imaginary(out_Imaginary, training=training)
        out_Imaginary = self.relu_Imaginary(out_Imaginary)
        # out_Real, out_Imaginary = self.relu([out_Real, out_Imaginary], training=training)

        return out_Real, out_Imaginary



class ComplexBNormReLULayer(tf.keras.layers.Layer):
    def __init__(self, droprate, isPooling, pooling_window_size, **kwargs):
        super(ComplexBNormReLULayer, self).__init__(**kwargs)
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
    def __init__(self, droprate, **kwargs):
        super(ComplexDropoutLayer, self).__init__(**kwargs)
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
    def __init__(self, droprate, **kwargs):
        super(ComplexDropoutLayer_approx, self).__init__(**kwargs)

        self.droprate = droprate

        self.mulRealLayer = tf.keras.layers.Multiply()
        self.mulImagLayer = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        [batchSize, H, W, C] = x_Real.shape

        # relu_ratio_Real = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))  
        # relu_ratio_Imag = np.random.normal(1.0, self.droprate / 6, size=(batchSize, H, W, C))

        if training:
            drop_ratio_Real = np.random.normal(1.0, self.droprate/2, size=(batchSize, H, W, C))
            drop_ratio_Imag = np.random.normal(1.0, self.droprate/2, size=(batchSize, H, W, C))

            # out_Real = self.mulRealLayer([x_Real, relu_ratio_Real])
            out_Real = self.mulRealLayer([x_Real, drop_ratio_Real])
            # out_Real = out_Real * (2/(7.0022*self.droprate))
            # out_Imaginary = self.mulRealLayer([x_Imaginary, relu_ratio_Imag])
            out_Imaginary = self.mulRealLayer([x_Imaginary, drop_ratio_Imag])
            # out_Imaginary = out_Imaginary * (2 / (7.0022 * self.droprate))
        else:
            out_Real = x_Real #/ self.droprate
            out_Imaginary = x_Imaginary #/ self.droprate
            # out_Real = self.mulRealLayer([x_Real, relu_ratio_Real])
            # out_Imaginary = self.mulRealLayer([x_Imaginary, relu_ratio_Imag])

        return out_Real, out_Imaginary
        
class ComplexDropoutLayer_approx_fc(tf.keras.layers.Layer):
    def __init__(self, droprate, **kwargs):
        super(ComplexDropoutLayer_approx_fc, self).__init__(**kwargs)

        self.droprate = droprate

        self.mulRealLayer = tf.keras.layers.Multiply()
        self.mulImagLayer = tf.keras.layers.Multiply()

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        if training:
            [batchSize, L] = x_Real.shape
            # drop_ratio_Real = ( (np.random.rand(batchSize, H, W, C)*(self.droprate)-(self.droprate/2) )+1 )
            # drop_ratio_Imag = ( (np.random.rand(batchSize, H, W, C)*(self.droprate)-(self.droprate/2) )+1 )
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

    def __init__(self, pooling_window_size, featureSize, **kwargs):
        super(ComplexPoolLayer, self).__init__(**kwargs)  # using average pooling
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
        
class ComplexPoolLayer_conv(tf.keras.layers.Layer):
    # TODO: Modify the pooling layer: (1) ifft2D; (2) Pooling; (3) Moving back to frequency domain or not

    def __init__(self, pooling_window_size, featureSize, out_channel, **kwargs):
        super(ComplexPoolLayer_conv, self).__init__(**kwargs)  # using average pooling
        weight_decay = 5e-4
        self.pooling_window_size = pooling_window_size
        self.featureSize = featureSize
        self.out_channel = out_channel
        self.pool_real = tf.keras.layers.AveragePooling2D((self.pooling_window_size, self.pooling_window_size))
        self.pool_imaginary = tf.keras.layers.AveragePooling2D((self.pooling_window_size, self.pooling_window_size))
        # self.pool_real = tf.keras.layers.Conv2D(filters=self.out_channel, kernel_size=(self.pooling_window_size, self.pooling_window_size), padding='valid', strides=(2, 2), kernel_regularizer=regularizers.l2(weight_decay))
        # self.relu_real = tf.keras.layers.LeakyReLU(alpha=0.2)  # Default: alpha=0.3
        # self.bnorm_real = tf.keras.layers.BatchNormalization()
        
        # self.pool_imaginary = tf.keras.layers.Conv2D(filters=self.out_channel, kernel_size=(self.pooling_window_size, self.pooling_window_size), padding='valid', strides=(2, 2), kernel_regularizer=regularizers.l2(weight_decay))
        # self.relu_imaginary = tf.keras.layers.LeakyReLU(alpha=0.2)  # Default: alpha=0.3
        # self.bnorm_imaginary = tf.keras.layers.BatchNormalization()

    def call(self, inputs, movingback, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]       

        # out_R_WR = self.pool_real(x_Real)
        # out_I_WI = self.pool_imaginary(x_Imaginary)
        # out_R_WI = self.pool_imaginary(x_Real)
        # out_I_WR = self.pool_real(x_Imaginary)

        # out_Real = out_R_WR - out_I_WI # + curBiasReal
        # out_Imaginary = out_R_WI + out_I_WR
 
        # out_Real = self.relu_real(out_Real)
        # out_Real = self.bnorm_real(out_Real, training=training)

        # out_Imaginary = self.relu_imaginary(out_Imaginary)
        # out_Imaginary = self.bnorm_imaginary(out_Imaginary, training=training)
        out_Real = self.pool_real(x_Real)
        out_Imaginary = self.pool_imaginary(x_Imaginary)

        
        return out_Real, out_Imaginary
        
        
class ComplexPoolLayer_magnitude(tf.keras.layers.Layer):

    def __init__(self, pooling_window_size, **kwargs):
        super(ComplexPoolLayer_magnitude, self).__init__(**kwargs)  # using average pooling
        # weight_decay = 5e-4
        self.pooling_window_size = pooling_window_size
        # self.pool = tf.keras.layers.MaxPooling2D((self.pooling_window_size, self.pooling_window_size))

    def call(self, inputs, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        # magnitude = tf.math.sqrt(tf.math.add(tf.math.square(x_Real), tf.math.square(x_Imaginary)))
        [out_Real, idx] = tf.nn.max_pool_with_argmax(x_Real, ksize=self.pooling_window_size, strides=2, padding='VALID',
                                                    include_batch_in_index=True)

        [batchSize, H, W, C] = out_Real.shape

        idx_1D = tf.reshape(idx, shape=[-1])
        # x_Real_1D = tf.reshape(x_Real, shape=[-1])
        x_Imaginary_1D = tf.reshape(x_Imaginary, shape=[-1])

        # out_Real_1D = tf.gather(x_Real_1D, idx_1D)
        out_Imaginary_1D = tf.gather(x_Imaginary_1D, idx_1D)

        # out_Real = tf.reshape(out_Real_1D, shape=[batchSize, H, W, C])
        out_Imaginary = tf.reshape(out_Imaginary_1D, shape=[batchSize, H, W, C])

        return out_Real, out_Imaginary
        
class ComplexSpectralPoolLayer(tf.keras.layers.Layer):
    def __init__(self, outSize, featureSize, **kwargs):
        super(ComplexSpectralPoolLayer, self).__init__(**kwargs)  # using average pooling
        # weight_decay = 5e-4
        self.outSize = outSize
        self.featureSize = featureSize
        self.bnormLayer = tf.keras.layers.BatchNormalization()

    def call(self, inputs, movingback, training):
        x_Real = inputs[0]
        x_Imaginary = inputs[1]

        t = tf.dtypes.complex(x_Real, x_Imaginary)
        t_shift = tf.signal.fftshift(t, axes=(1, 2))
        cut_start = np.ceil((self.featureSize[1] - self.outSize)/2)
        cut_end = np.ceil((cut_start + self.outSize))
        cut_start = tf.cast(cut_start, dtype=tf.int32)
        cut_end = tf.cast(cut_end, dtype=tf.int32)
        out_shift = t_shift[:, cut_start:cut_end, cut_start:cut_end, :]

        out = tf.signal.ifftshift(out_shift, axes=(1, 2))
        
        
        out = tf.transpose(out, [3, 0, 1, 2]) # N-H-W-C -> C-N-H-W
        out_time = tf.signal.ifft2d(out)
        out_time = tf.math.real(out_time)
        out_time = tf.transpose(out_time, [1, 2, 3, 0])
            
        out_time = self.bnorm_Real(out_time, training=training)
            
            
        out_time = tf.cast(out_time, dtype=tf.complex64)
        # ifft 2d and fft 2d            
        out_time = tf.transpose(out_time, [3, 0, 1, 2])
        out = tf.signal.fft2d(out_time)
        out = tf.transpose(out, [1, 2, 3, 0])
        
        
        out_Real = tf.math.real(out)
        out_Imaginary = tf.math.imag(out)

        return out_Real, out_Imaginary
        
class complexfcLayer(tf.keras.layers.Layer):
    def __init__(self, neurNum, **kwargs):
        super(complexfcLayer, self).__init__(**kwargs)
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
    def __init__(self, **kwargs):
        super(complexconcatLayer, self).__init__(**kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    # @tf.function
    def call(self, inputs):
        x_Real = inputs[0]  # [batchSize, H, W, C]
        x_Imaginary = inputs[1]

        out = self.concat([x_Real, x_Imaginary])

        return out


class magnitudeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(magnitudeLayer, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs):
        x_Real = inputs[0]  # [batchSize, H, W, C]
        x_Imaginary = inputs[1]

        out = tf.math.sqrt(tf.math.add(tf.math.square(x_Real), tf.math.square(x_Imaginary)))

        return out
        
        
class freqtransferLayer(tf.keras.layers.Layer):
    # TODO: Modify the pooling layer: (1) ifft2D; (2) Pooling; (3) Moving back to frequency domain or not

    def __init__(self, **kwargs):
        super(freqtransferLayer, self).__init__(**kwargs)  # using average pooling

    def call(self, input_time):

        input_time = tf.cast(input_time, dtype=tf.complex64)
        # ifft 2d and fft 2d            
        input_time_trans = tf.transpose(input_time, [3, 0, 1, 2])
        out_freq_trans = tf.signal.fft2d(input_time_trans)
        out_freq = tf.transpose(out_freq_trans, [1, 2, 3, 0])
   
        out_Real = tf.math.real(out_freq)
        out_Imaginary = tf.math.imag(out_freq)
            
        return out_Real, out_Imaginary

class VGG16_mixture(tf.keras.models.Model):

    def __init__(self, batchSize, **kwargs):
        """

        :param input_shape: [32, 32, 3]
        """
        super(VGG16_mixture, self).__init__(**kwargs)

        weight_decay = 5e-4
        self.num_classes = 10
        self.droprate1 = 0.3
        self.droprate2 = 0.4
        self.droprate3 = 0.5
        
        self.padding=1
        self.isCut=True

        self.out_channel = [64, 128, 256, 512, 512]

        self.batchSize = batchSize
        
        # Input layers: time domain data with regular cnn layers
        self.Input = tf.keras.layers.InputLayer(input_shape=(32, 32, 3))
        
        # Time domain layers:
        
        # Block 1
        self.conv1L = tf.keras.layers.Conv2D(64, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform',
                 kernel_regularizer=regularizers.l2(weight_decay)) # , input_shape=(32, 32, 3)
        self.bnorm1L = tf.keras.layers.BatchNormalization()            
        self.relu1L  = tf.keras.layers.ReLU()
        self.drop1L = tf.keras.layers.Dropout(self.droprate1)
                 
        self.conv2L = tf.keras.layers.Conv2D(64, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm2L = tf.keras.layers.BatchNormalization()            
        self.relu2L  = tf.keras.layers.ReLU()

                 
        self.pool1L = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        
        # Block 2
        self.conv3L = tf.keras.layers.Conv2D(128, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm3L = tf.keras.layers.BatchNormalization()            
        self.relu3L  = tf.keras.layers.ReLU()
        self.drop2L = tf.keras.layers.Dropout(self.droprate2)
                 
        self.conv4L = tf.keras.layers.Conv2D(128, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm4L = tf.keras.layers.BatchNormalization()            
        self.relu4L  = tf.keras.layers.ReLU()

                 
        self.pool2L = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        
        # Block 3
        self.conv5L = tf.keras.layers.Conv2D(256, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm5L = tf.keras.layers.BatchNormalization()            
        self.relu5L  = tf.keras.layers.ReLU()
        self.drop3L = tf.keras.layers.Dropout(self.droprate2)
                 
        self.conv6L = tf.keras.layers.Conv2D(256, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm6L = tf.keras.layers.BatchNormalization()            
        self.relu6L  = tf.keras.layers.ReLU()
        self.drop4L = tf.keras.layers.Dropout(self.droprate2)
                 
        self.conv7L = tf.keras.layers.Conv2D(256, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm7L = tf.keras.layers.BatchNormalization()            
        self.relu7L  = tf.keras.layers.ReLU()
               
        self.pool3L = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')
        
        
        # Freqency domain layers:
        self.ftrans = freqtransferLayer() 
        
        # Block 4:
        self.complex8 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[2]], self.out_channel[3], use_bias=False, trainable=True, name="conv8") 
        self.drop8 = ComplexDropoutLayer_approx(self.droprate3, name="drop8")

        self.complex9 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[3]], self.out_channel[3], use_bias=False, trainable=True, name="conv9")
        self.drop9 = ComplexDropoutLayer_approx(self.droprate3, name="drop9")

        self.complex10 = ComplexDotLayer([self.batchSize, 4, 4, self.out_channel[3]], self.out_channel[3], use_bias=False, trainable=True, name="conv10")
        self.drop10 = ComplexDropoutLayer_approx(self.droprate3, name="drop10")

        self.pool4 = ComplexPoolLayer(2, [self.batchSize, 4, 4, self.out_channel[3]], name="pool4")
        # self.pool4 = ComplexPoolLayer_conv(2, [self.batchSize, 4, 4, self.out_channel[3]], self.out_channel[3])
        # self.pool4 = ComplexSpectralPoolLayer(2, [self.batchSize, 4, 4, self.out_channel[3]])

        # Block 5:
        self.complex11 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[3]], self.out_channel[4], use_bias=False, trainable=True, name="conv11")
        self.drop11 = ComplexDropoutLayer_approx(self.droprate3, name="drop11")

        self.complex12 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[4]], self.out_channel[4], use_bias=False, trainable=True, name="conv12")
        self.drop12 = ComplexDropoutLayer_approx(self.droprate3, name="drop12")

        self.complex13 = ComplexDotLayer([self.batchSize, 2, 2, self.out_channel[4]], self.out_channel[4], use_bias=False, trainable=True, name="conv13")
        self.drop13 = ComplexDropoutLayer_approx(self.droprate3, name="drop13")

        self.pool5 = ComplexPoolLayer(2, [self.batchSize, 2, 2, self.out_channel[4]], name="pool5")

        # Flatten
        self.flatten1 = tf.keras.layers.Flatten(name="flat_real")
        self.flatten2 = tf.keras.layers.Flatten(name="flat_imag")
        
        # Block 6 complex fc
        self.fc1 = complexfcLayer(512, name="dense1")
        self.drop14 = ComplexDropoutLayer_approx_fc(self.droprate3, name="drop14")
        self.concat = complexconcatLayer(name="concat1")
        self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax', name="dense_out")

    # @tf.function
    def call(self, inputs, training=None, isFeatFix=False):           
        # Block 1
        x = self.Input(inputs)
        
        x = self.conv1L(x) 
        x = self.bnorm1L(x,training=training)
        x = self.relu1L(x) 
        x = self.drop1L(x,training=training)                
        x = self.conv2L(x) 
        x = self.bnorm2L(x,training=training)
        x = self.relu2L(x) 
                 
        x = self.pool1L(x)
        
        # Block 2
        x = self.conv3L(x) 
        x = self.bnorm3L(x,training=training)
        x = self.relu3L(x)
        x = self.drop2L(x,training=training)           
        x = self.conv4L(x) 
        x = self.bnorm4L(x,training=training)
        x = self.relu4L(x) 
             
        x = self.pool2L(x)
        
        # Block 3
        x = self.conv5L(x)  
        x = self.bnorm5L(x,training=training)
        x = self.relu5L(x) 
        x = self.drop3L(x,training=training)          
        x = self.conv6L(x)  
        x = self.bnorm6L(x,training=training)
        x = self.relu6L(x) 
        x = self.drop4L(x,training=training)         
        x = self.conv7L(x) 
        x = self.bnorm7L(x,training=training)
        x = self.relu7L(x) 
          
        x = self.pool3L(x)
        
        x_real, x_imaginary = self.ftrans(x)
        # Block 4:
        x_real, x_imaginary = self.complex8([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
       
        
        x_real, x_imaginary = self.drop8([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.complex9([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop9([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.complex10([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop10([x_real, x_imaginary], training=training)
        

        x_real, x_imaginary = self.pool4([x_real, x_imaginary], movingback=True, training=training)

        # Block 5:
        x_real, x_imaginary = self.complex11([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        
        x_real, x_imaginary = self.drop11([x_real, x_imaginary], training=training)

        x_real, x_imaginary = self.complex12([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop12([x_real, x_imaginary], training=training)
        
        x_real, x_imaginary = self.complex13([x_real, x_imaginary], training=training, isFeatFix=False)  # include bnorm and relu
        x_real, x_imaginary = self.drop13([x_real, x_imaginary], training=training)
        
        x_real, x_imaginary = self.pool5([x_real, x_imaginary], movingback=True, training=training)
        
        # complex fc layers
        x_real = self.flatten1(x_real)
        x_imaginary = self.flatten2(x_imaginary)

        x_real, x_imaginary = self.fc1([x_real, x_imaginary], training=training)
        x_real, x_imaginary = self.drop14([x_real, x_imaginary], training=training)
        xb1 = self.concat([x_real, x_imaginary])
        ClsResults = self.output1(xb1)

        return ClsResults
