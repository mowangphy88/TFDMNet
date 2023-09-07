import os
import tensorflow as tf
import argparse
import numpy as np
# import cv2

from complexNet_cnn_mixturemodel import VGG16_mixture

physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


# tf.config.experimental.set_memory_growth(physical_devices[1], True)

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    mean_test = np.mean(X_test, axis=(0, 1, 2, 3))
    std_test = np.std(X_test, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    print('mean_test:', mean_test, 'std_test:', std_test)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    mean_test = np.mean(X_test, axis=(0, 1, 2, 3))
    std_test = np.std(X_test, axis=(0, 1, 2, 3))
    print('mean_afternormalization:', mean, 'std_afternormalization:', std)
    print('mean_test_afternormalization:', mean_test, 'std_test_afternormalization:', std_test)
    return X_train, X_test


def normalize_fft(fft_train, fft_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    fft_train = fft_train #  / 20.
    fft_test = fft_test #  / 20.

    mean = np.mean(fft_train, axis=(0, 1, 2, 3))
    std = np.std(fft_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    fft_train = (fft_train - mean) / (std + 1e-7)
    fft_test = (fft_test - mean) / (std + 1e-7)
    return fft_train, fft_test

def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    # x = tf.cast(x, tf.float32)
    # xfft = tf.cast(xfft, tf.float32)
    y = tf.cast(y, tf.int32)
    return x,  y


def images_augment(images, N, H, W, C):
    # images = tf.image.resize(images, [H+8, W+8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # images = tf.image.random_crop(images, size=[N, H, W, C])
    # print('Doing data augmentation...')
    images = tf.image.resize_with_crop_or_pad(images, H + 8, W + 8)
    images = tf.image.random_crop(images, size=[N, H, W, C])
    images = tf.image.random_flip_left_right(images)

    return images
    

# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5"])
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])

# Data Preparation

print('loading data...')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# all data: uint8
# N-H-W-C

# Normalization
x_train, x_test = normalize(x_train, x_test)


# Build database
batchSize_per_replica = 50

batchSize = (batchSize_per_replica * mirrored_strategy.num_replicas_in_sync)

train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(batchSize)

dist_train_loader = mirrored_strategy.experimental_distribute_dataset(train_loader)

test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(batchSize)

dist_test_loader = mirrored_strategy.experimental_distribute_dataset(test_loader)
print('done')



# aeMetric  = tf.keras.metrics.MeanSquaredError()

lr_time = 0.005
minimum_learning_rate = 0.000001# 0.0000001
data_augmentation = False

H = x_train.shape[1]
W = x_train.shape[2]
C = x_train.shape[3]

train_num = 50000 # 1281167
test_num = 10000 # 50000
iterations_per_epoch = int(train_num / batchSize)
test_iterations = int(test_num / batchSize) + 1
warm_iterations = iterations_per_epoch
n_of_epochs = 600

learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=lr_time,
                                                    decay_steps=n_of_epochs * iterations_per_epoch - warm_iterations,
                                                    alpha=minimum_learning_rate,
                                                    warm_up_step=warm_iterations)

with mirrored_strategy.scope():
    model = VGG16_mixture(batchSize_per_replica, name="mixture_model")
    
    TimeOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedules, momentum=0.9) #SGD/RMSprop
    # FreqOptimizer = tf.keras.optimizers.RMSprop(learning_rate=initlr_freq, momentum=0.9)
    
    ceLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    clsMetric = tf.keras.metrics.CategoricalAccuracy()
    metric_test = tf.keras.metrics.CategoricalAccuracy()
    # mseLoss   = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(labels, predictions):
        per_example_loss = ceLoss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batchSize)

    
def train_step(inputs):
    x, y, isFeatFix = inputs
    
    # [b, 1] => [b]
    y = tf.squeeze(y, axis=1)
    # [b, 10]
    y = tf.one_hot(y, depth=10)
    
    with tf.GradientTape(persistent=True) as tape: # persistent=True
        ClsResults = model(x, training=True, isFeatFix=isFeatFix)
        # print(ClsResults)
        loss1 = compute_loss(y, ClsResults)
        # loss_regularization = []
        
        # for tv in model.trainable_variables:
        #     loss_regularization.append(tf.nn.l2_loss(tv))
        # loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
        # loss1 = loss1 + 0.00002 * loss_regularization # previous 0.00005
        # print(loss1)
    grads1 = tape.gradient(loss1, model.trainable_variables)
                
    # MUST clip gradient here or it will disconverge!
    grads1 = [tf.clip_by_norm(g, 15) for g in grads1]
    TimeOptimizer.apply_gradients(zip(grads1, model.trainable_variables)) 
    clsMetric.update_state(y, ClsResults)
    return loss1
    
def test_step(inputs):
    x, y = inputs
    # [b, 1] => [b]
    y = tf.squeeze(y, axis=1)
    # [b, 10]
    y = tf.one_hot(y, depth=10)
    
    ClsResults = model(x, training=False, isFeatFix=False)
    metric_test.update_state(y, ClsResults)
    
@tf.function
def distributed_train_step(dist_inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
@tf.function
def distributed_test_step(dist_inputs):
    return mirrored_strategy.run(test_step, args=(dist_inputs,))

for epoch in range(n_of_epochs):

    total_loss = 0.0
    num_batches = 0
    for step, (x, y) in enumerate(dist_train_loader): # xl: real; xr: imaginary

        # if data_augmentation:
            # print(type(xl))
            # N = xl.shape[0]
            # xl = images_augment(xl, N, H, W, C)
            # xr = images_augment(xr, N, H, W, C)

        if step % 1 == 0: # epoch >= 2 and step % 1 == 0: #
            isFeatFix = False
        else:
            isFeatFix = False

        total_loss += distributed_train_step((x, y, isFeatFix))
        num_batches += 1

        # if epoch > 1:
        #    TimeOptimizer.learning_rate = lr_time

        if step % 1 == 0:
            print(epoch, step, 'loss:', float(total_loss / num_batches), 'acc:', clsMetric.result().numpy())
            with open('train_log.txt', 'a') as f:
                f.write('epoch:{}, step:{} '.format(epoch, step))
                f.write('train: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(float(total_loss / num_batches),
                                                                                                   clsMetric.result().numpy()))
                
            
    train_loss = total_loss / num_batches
    train_acc = clsMetric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    clsMetric.reset_states()

    if epoch % 1 == 0:
        # sumacc = 0
        for (x, y) in dist_test_loader: #test_loader:
            distributed_test_step((x, y))
            
        val_acc = metric_test.result()
        metric_test.reset_states()
        # print('Current acc: ', metric_test.result().numpy())
        # print('test acc:', sumacc / 500)
        print("Validation acc: %.4f" % (float(val_acc),))
        with open('test_log.txt', 'a') as f_val:
                f_val.write('epoch:{}, '.format(epoch))
                f_val.write('Validation: accuracy: {:.4f}\n'.format(float(val_acc)))
                
                
model.save_weights('complex_mixture_model_13layers.h5')
# model.load_weights(fname, by_name=True)
