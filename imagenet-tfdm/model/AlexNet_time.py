import tensorflow as tf
import numpy as np
import scipy.io as scio
from tensorflow.keras import regularizers


class AlexNet_Time_ImageNet(tf.keras.models.Model):

    def __init__(self):
        """

        :param input_shape: [224, 224, 3]
        """
        super(AlexNet_Time_ImageNet, self).__init__()

        weight_decay = 5e-4
        self.num_classes = 1000
        self.droprate1 = 0.5
        self.droprate2 = 0.5
        self.droprate3 = 0.5
        
        self.padding=1
        self.isCut=True


        # Input layers: real data and imaginary data
        self.Input = tf.keras.layers.InputLayer(input_shape=(227, 227, 3))

        # Complex Layers:
        self.conv1L = tf.keras.layers.Conv2D(96, (11, 11), activation=None, 
                 padding='valid', strides=4, kernel_initializer='he_uniform',
                 kernel_regularizer=regularizers.l2(weight_decay)) # , input_shape=(32, 32, 3)
        self.bnorm1L = tf.keras.layers.BatchNormalization()            
        self.relu1L  = tf.keras.layers.ReLU()
        # self.drop1L = tf.keras.layers.Dropout(self.droprate1) # 55

                 
        self.pool1L = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid') #27
        
        # Block 2
        self.conv2L = tf.keras.layers.Conv2D(256, (5, 5), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm2L = tf.keras.layers.BatchNormalization()            
        self.relu2L  = tf.keras.layers.ReLU()
        # self.drop2L = tf.keras.layers.Dropout(self.droprate2)

                 
        self.pool2L = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid') #13
        
        # Block 3
        self.conv3L = tf.keras.layers.Conv2D(384, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm3L = tf.keras.layers.BatchNormalization()            
        self.relu3L  = tf.keras.layers.ReLU()
        # self.drop3L = tf.keras.layers.Dropout(self.droprate2)
                 
        self.conv4L = tf.keras.layers.Conv2D(384, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm4L = tf.keras.layers.BatchNormalization()            
        self.relu4L  = tf.keras.layers.ReLU()
        # self.drop4L = tf.keras.layers.Dropout(self.droprate2)
        
        self.conv5L = tf.keras.layers.Conv2D(256, (3, 3), activation=None, 
                 padding='same', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(weight_decay))
        self.bnorm5L = tf.keras.layers.BatchNormalization()            
        self.relu5L  = tf.keras.layers.ReLU()
        # self.drop5L = tf.keras.layers.Dropout(self.droprate2)
               
        self.pool3L = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid')# 6
                

        # Flatten
        self.gp = tf.keras.layers.Flatten()

        # Block 6.2
        # self.fc1 = tf.keras.layers.Dense(512, activation=None)
        # self.relu14 = tf.keras.layers.ReLU()
        # self.bnorm14 = tf.keras.layers.BatchNormalization()
        # self.drop14 = tf.keras.layers.Dropout(self.droprate3)
        # self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        
        # Block 6.2 complex fc
        self.fc1 = tf.keras.layers.Dense(4096, activation=None)
        self.relu14 = tf.keras.layers.ReLU()
        self.bnorm14 = tf.keras.layers.BatchNormalization()
        self.drop9 = tf.keras.layers.Dropout(self.droprate3)
        self.fc2 = tf.keras.layers.Dense(4096, activation=None)
        self.relu15 = tf.keras.layers.ReLU()
        self.bnorm15 = tf.keras.layers.BatchNormalization()
        self.drop10 = tf.keras.layers.Dropout(self.droprate3)
        self.output1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    # @tf.function
    def call(self, inputs, training=None):
        # Inputs
        # Block 1
        x = self.Input(inputs)
        
        x = self.conv1L(x) 
        x = self.bnorm1L(x,training=training)
        x = self.relu1L(x) 
        # x = self.drop1L(x,training=training)                
                 
        x = self.pool1L(x)
        
        # Block 2
        x = self.conv2L(x) 
        x = self.bnorm2L(x,training=training)
        x = self.relu2L(x)
        # x = self.drop2L(x,training=training)           
             
        x = self.pool2L(x)
        
        # Block 3
        x = self.conv3L(x)  
        x = self.bnorm3L(x,training=training)
        x = self.relu3L(x) 
        # x = self.drop3L(x,training=training)          
        x = self.conv4L(x)  
        x = self.bnorm4L(x,training=training)
        x = self.relu4L(x) 
        # x = self.drop4L(x,training=training)      
        x = self.conv5L(x)  
        x = self.bnorm5L(x,training=training)
        x = self.relu5L(x) 
        # x = self.drop5L(x,training=training)         
          
        x = self.pool3L(x)
        
        x = self.gp(x)

        # Block 6:
        # xb1 = self.fc1(xb1)
        # xb1 = self.relu14(xb1)
        # xb1 = self.bnorm14(xb1, training=training)
        # xb1 = self.drop14(xb1, training=training)
        # ClsResults = self.output1(xb1)
        
        # complex fc layers
        x = self.fc1(x)
        x = self.relu14(x)
        x = self.bnorm14(x)
        x = self.drop9(x)
        x = self.fc2(x)
        x = self.relu15(x)
        x = self.bnorm15(x)
        x = self.drop10(x)
        ClsResults = self.output1(x)

        return ClsResults
