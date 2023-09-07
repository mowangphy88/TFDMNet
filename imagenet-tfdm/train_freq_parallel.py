import os
import tensorflow as tf
import config as c
import argparse
from tqdm import tqdm
from tensorflow.keras import optimizers
from utils.data_utils import train_iterator, test_iterator
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet
from model.ResNet_v2 import ResNet_v2
from model.VGG16_mixture import VGG16_MixtureNet_ImageNet
from model.AlexNet_mixture import AlexNet_MixtureNet_ImageNet
from test import test
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3", "GPU:4", "GPU:5", "GPU:6", "GPU:7"]) # , "GPU:4", "GPU:5", "GPU:6"
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])# devices=["GPU:0", "GPU:1"]



batchSize_per_replica = c.batch_size
batchSize = (batchSize_per_replica * mirrored_strategy.num_replicas_in_sync)

iterations_per_epoch = int(c.train_num / batchSize)
iterations_per_epoch_test = int(c.test_num / batchSize)
warm_iterations = iterations_per_epoch

iterations_per_test_epoch = int(c.test_num / batchSize)

# load data
train_data_iterator = train_iterator(batch_size=batchSize)
dist_train_loader = mirrored_strategy.experimental_distribute_dataset(train_data_iterator)

test_data_iterator = test_iterator(batch_size=batchSize)
dist_test_loader = mirrored_strategy.experimental_distribute_dataset(test_data_iterator)

it = dist_train_loader.__iter__()
it_test = dist_test_loader.__iter__()

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


learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate,
                                                    decay_steps=c.epoch_num * iterations_per_epoch - warm_iterations,
                                                    alpha=c.minimum_learning_rate,
                                                    warm_up_step=warm_iterations)


with mirrored_strategy.scope():
    model = AlexNet_MixtureNet_ImageNet(batchSize_per_replica)
    # model = ResNet_v2(18)
    # show
    # model.build(input_shape=(None,) + c.input_shape)
    # model.summary()
    

    TimeOptimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
    ceLoss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    clsMetric = tf.keras.metrics.CategoricalAccuracy()
    metric_test = tf.keras.metrics.CategoricalAccuracy()

    def compute_loss(labels, predictions):
        per_example_loss = ceLoss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batchSize)



def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True) # , isFeatFix=isFeatFix
        loss = compute_loss(labels, prediction)
        # l2 = l2_loss(model)
        # loss = ce + l2    
        gradients = tape.gradient(loss, model.trainable_variables)
    TimeOptimizer.apply_gradients(zip(gradients, model.trainable_variables))
    clsMetric.update_state(labels, prediction)
    return loss
    
def test_step(images, labels):
    prediction = model(images, training=False)
    metric_test.update_state(labels, prediction)
    
    
@tf.function
def distributed_train_step(inputs, labels):
    per_replica_losses = mirrored_strategy.run(train_step, args=(inputs, labels,))
    # print(per_replica_losses)
    reduced_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    # print(reduced_loss)
    return reduced_loss
    
@tf.function
def distributed_test_step(inputs, labels):
    return mirrored_strategy.run(test_step, args=(inputs, labels,))
    

def train(data_iterator, log_file):

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(iterations_per_epoch)):
        inputs, labels = data_iterator.next()
        
        ce = distributed_train_step(inputs, labels)

        sum_ce += ce * batchSize
        print('iter: {}, ce: {:.4f}, accuracy: {:.4f}'.format(i, ce, clsMetric.result().numpy()))

    log_file.write('train: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.train_num, clsMetric.result().numpy()))
    clsMetric.reset_states()

def test(data_iterator, log_file):

    for i in tqdm(range(iterations_per_test_epoch)):
        images, labels = data_iterator.next()
        
        distributed_test_step(images, labels)

    val_acc = metric_test.result()
    metric_test.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    log_file.write('test: accuracy: {:.4f}\n'.format(float(val_acc)))

# train
for epoch_num in range(c.epoch_num):
    with open(c.log_file, 'a') as f:
        f.write('epoch:{}\n'.format(epoch_num))
        # train(model, it, TimeOptimizer, f)
        train(it, f)
        test(it_test, f)
    # save intermediate results
    if epoch_num % 10 == 9:
        model.save_weights(c.save_weight_file, save_format='h5')
        os.system('cp {} {}_epoch_{}.h5'.format(c.save_weight_file, c.save_weight_file.split('.')[0], epoch_num))
