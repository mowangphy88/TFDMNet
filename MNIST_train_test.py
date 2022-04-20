import os
import tensorflow as tf
import argparse
import numpy as np

from MNIST_CEMNet import LeNet_CEMNet

physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

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


def prepare_mnist(xl, xr, y):
    xl = tf.cast(xl, tf.float32)
    xr = tf.cast(xr, tf.float32)
    y = tf.cast(y, tf.int32)
    return xl, xr, y

    

# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5"])
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0"])

# Data Preparation
n_classes = 10

print('loading data...')
path = './mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)
f.close()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_real = np.load("x_train_FFT_mnist_real.npy")
x_test_real = np.load("x_test_FFT_mnist_real.npy")
x_train_imaginary = np.load("x_train_FFT_mnist_imag.npy")
x_test_imaginary = np.load("x_test_FFT_mnist_imag.npy")

# Build database
batchSize_per_replica = 100

batchSize = (batchSize_per_replica * mirrored_strategy.num_replicas_in_sync)

train_loader = tf.data.Dataset.from_tensor_slices((x_train_real, x_train_imaginary, y_train))
train_loader = train_loader.map(prepare_mnist).shuffle(60000).batch(batchSize)
dist_train_loader = mirrored_strategy.experimental_distribute_dataset(train_loader)

test_loader = tf.data.Dataset.from_tensor_slices((x_test_real, x_test_imaginary, y_test))
test_loader = test_loader.map(prepare_mnist).shuffle(10000).batch(batchSize)
dist_test_loader = mirrored_strategy.experimental_distribute_dataset(test_loader)
print('done')


lr_time = 0.004
minimum_learning_rate = 0.00001
data_augmentation = False

H = x_train_real.shape[1]
W = x_train_real.shape[2]
C = x_train_real.shape[3]

train_num = 60000 # 1281167
test_num = 10000 # 50000
iterations_per_epoch = int(train_num / batchSize)
test_iterations = int(test_num / batchSize) + 1
warm_iterations = iterations_per_epoch
n_of_epochs = 1000

learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=lr_time,
                                                    decay_steps=n_of_epochs * iterations_per_epoch - warm_iterations,
                                                    alpha=minimum_learning_rate,
                                                    warm_up_step=warm_iterations)

with mirrored_strategy.scope():
    model = LeNet_CEMNet(batchSize_per_replica)
    FreqOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedules, momentum=0.9) #SGD/RMSprop
    
    ceLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    clsMetric = tf.keras.metrics.CategoricalAccuracy()
    metric_test = tf.keras.metrics.CategoricalAccuracy()
    
    def compute_loss(labels, predictions):
        per_example_loss = ceLoss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batchSize)
        
    
    
def train_step(inputs):
    xl, xr, y, isFeatFix = inputs
    with tf.GradientTape(persistent=True) as tape:
        ClsResults = model([xl, xr], training=True, isFeatFix=isFeatFix)
        loss1 = compute_loss(y, ClsResults)
    grads1 = tape.gradient(loss1, model.trainable_variables) # without bnorm in FC layers: 30; with: 70; no bnorm: 26
    # MUST clip gradient here or it will disconverge!
    grads1 = [tf.clip_by_norm(g, 15) for g in grads1]
    FreqOptimizer.apply_gradients(zip(grads1, model.trainable_variables)) #[20:30]
    clsMetric.update_state(y, ClsResults)
    # del tape
    return loss1
    
def test_step(inputs):
    xl, xr, y = inputs
    ClsResults = model([xl, xr], training=False, isFeatFix=False)
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
    for step, (xl, xr, y) in enumerate(dist_train_loader): # xl: real; xr: imaginary

        if step % 1 == 0:
            isWeightFix = True
        else:
            isWeightFix = False

        total_loss += distributed_train_step((xl, xr, y, isWeightFix))
        num_batches += 1

        if step % 1 == 0:
            print(epoch, step, 'loss:', float(total_loss / num_batches), 'acc:', clsMetric.result().numpy())
            with open('train_log_mnist.txt', 'a') as f:
                f.write('epoch:{}, step:{} '.format(epoch, step))
                f.write('train: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(float(total_loss / num_batches),
                                                                                                   clsMetric.result().numpy()))
                
            
    train_loss = total_loss / num_batches
    train_acc = clsMetric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    clsMetric.reset_states()

    if epoch % 1 == 0:
        for (xl, xr, y) in dist_test_loader: #test_loader:
            distributed_test_step((xl, xr, y))
            
        val_acc = metric_test.result()
        metric_test.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        with open('test_log_mnist.txt', 'a') as f_val:
                f_val.write('epoch:{}, '.format(epoch))
                f_val.write('Validation: accuracy: {:.4f}\n'.format(float(val_acc)))
