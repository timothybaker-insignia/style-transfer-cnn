#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import errno
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from multiprocessing import cpu_count

from CNN import CNN

THREADS = cpu_count()-1

# Model Specific Parameters
NUM_CLASSES           =    10       # number of output classes

# Flags
ROTATE_BATCHES        =    False    # rotation augmentation
RESTORE_MODEL         =    False    # restore previous model
SAVE_MODEL            =    False    # save model 
TEST_ONLY             =    False    # skips training

# Tunable Parameters
BATCH_SIZE            =    32       # training batch size, ONLY 1 IS IMPLEMENTED
EPOCHS                =    1        # number of epochs to train for
LEARNING_RATE         =    1e-3     # learning rate for gradient descent
KEEP_PROB             =    0.5      # keep probability for dropout layers
LAMBDA_REG            =    0.001    # lambda for kernel regularization

# I/O folders and files
DATA_PATH = "data/"
CHECKPOINT_PATH = "checkpoint/" 
OUTPUT_PATH = "output/"
PICKLE_PATH = DATA_PATH + "pickle/"
TRAINING_PICKLE_PATH = PICKLE_PATH + "training_batch.pkl"
VALIDATION_PICKLE_PATH = PICKLE_PATH + "validation_batch.pkl"
TESTING_PICKLE_PATH = PICKLE_PATH + "testing_batch.pkl"

if TEST_ONLY:
    EPOCHS = 0

def checkEnv():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATA_PATH)

def setupEnv():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(PICKLE_PATH):
        os.makedirs(PICKLE_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

def cifar10Model(keepprob=1.0):
    # create network model for training cifar 10
    cifar10cnn = CNN()
    # convolution layer 1
    cifar10cnn.addLayer('convolution', 2, 3, 32, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('convolution', 2, 32, 32, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('maxpooling')
    cifar10cnn.addLayer('dropout')
    # convolution layer 2
    cifar10cnn.addLayer('convolution', 2, 32, 64, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('convolution', 2, 64, 64, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('maxpooling')
    cifar10cnn.addLayer('dropout')
    # convolution layer 3
    cifar10cnn.addLayer('convolution', 2, 64, 128, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('convolution', 2, 128, 128, activation='relu', keepprob=keepprob)
    cifar10cnn.addLayer('maxpooling')
    cifar10cnn.addLayer('dropout')
    # fully connected layer 1
    cifar10cnn.addLayer('fully connected', 2, 128, 256, activation='none', keepprob=keepprob)
    # fully connected layer 2
    cifar10cnn.addLayer('fully connected', 256, NUM_CLASSES, activation='none', keepprob=1.0)
    return cifar10cnn

def loadDatasets():
    training_batches = loadBatches(TRAINING_LIST_PATH, TRAINING_PICKLE_PATH, 
        CHAINS_PATH, LAYERS, is_chief=is_chief, is_training=True)
    if CULL_OUTLIERS:
        training_batches = removeOutlierChains(training_batches)
    if is_chief:
      printChainStats(training_batches, PLOTS_PATH + "trainingStats.txt")
    if ROTATE_BATCHES:
        training_batches = rotateBatches(training_batches)
    total_training_batches = len(training_batches)
    Random(0).shuffle(training_batches)                                    
    training_batches.reverse() 
    training_iterations = int(len(training_batches) / BATCH_SIZE)

    def gen_training():
        for i in range(training_iterations * BATCH_SIZE):
            batch, label, is_training = training_batches[i]
            yield (batch, label, is_training)

    def self_map(batch, label, is_training):
        return batch, label, is_training

    training_dataset = tf.data.Dataset.from_generator(gen_training, (tf.float16,tf.float16,tf.bool))
    training_dataset = training_dataset.shuffle(buffer_size=BATCH_SIZE)
    training_dataset = training_dataset.map(self_map, num_parallel_calls=THREADS)
    training_dataset = training_dataset.prefetch(buffer_size=1)
    print("loaded",total_training_batches,"training batches")

    print("loading validation batches...")
    sys.stdout.flush()
    validation_batches = loadBatches(VALIDATION_LIST_PATH, VALIDATION_PICKLE_PATH)
    if CULL_OUTLIERS:
        validation_batches = removeOutlierChains(validation_batches)                
    if is_chief:
      printChainStats(validation_batches, PLOTS_PATH + "validationStats.txt")
    total_validation_batches = len(validation_batches)
    Random(0).shuffle(validation_batches)             
    validation_iterations = int(len(validation_batches) 
                                / BATCH_SIZE)
    def gen_validation():
        for i in range(validation_iterations * BATCH_SIZE):
            chains, batch, coords, orig, is_training, is_rotated = \
                validation_batches[i]
            yield (chains, batch, orig, coords, is_training, is_rotated)
    validation_dataset = tf.data.Dataset.from_generator(gen_validation, (
        (tf.string,tf.string,tf.string),(tf.float32,tf.float32,tf.float32,  
        (tf.int64, tf.int64, tf.int64, tf.int64)),
        (tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64),
        (tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64), 
          tf.bool, tf.bool))
    validation_dataset = validation_dataset.map(
        self_map, num_parallel_calls=1)
    validation_dataset = validation_dataset.prefetch(
        buffer_size=1)
    print("loaded",total_validation_batches,"validation batches")

    print("loading testing batches...")
    sys.stdout.flush()
    testing_batches = loadBatches(TESTING_LIST_PATH, TESTING_PICKLE_PATH, 
        CHAINS_PATH, LAYERS, is_chief=is_chief, is_training=False)
    if CULL_OUTLIERS:
        testing_batches = removeOutlierChains(testing_batches)                
    if is_chief:
      printChainStats(testing_batches, PLOTS_PATH + "testingStats.txt")
    total_testing_batches = len(testing_batches)
    Random(0).shuffle(testing_batches)                                    
    testing_iterations = int(len(testing_batches) 
                            / BATCH_SIZE)
    def gen_testing():
        for i in range(testing_iterations * BATCH_SIZE):
            chains, batch, orig, coords, is_training, is_rotated = \
                testing_batches[i]
            yield (chains, batch, orig, coords, is_training, is_rotated)
    testing_dataset = tf.data.Dataset.from_generator(gen_testing, 
        ((tf.string,tf.string,tf.string),(tf.float32,tf.float32,tf.float32, 
         (tf.int64, tf.int64, tf.int64, tf.int64)),
         (tf.int64,tf.int64,tf.int64, tf.int64,tf.int64,tf.int64), 
         (tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64),
          tf.bool, tf.bool))
    testing_dataset = testing_dataset.map(
            self_map, num_parallel_calls=1)
    testing_dataset = testing_dataset.prefetch(
        buffer_size=1)
    print("loaded",total_testing_batches,"testing batches")
    print("building computational graph...")
    sys.stdout.flush()

    return training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations

def main():
    start_time = time.time()
    print("started at", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))
    sys.stdout.flush()

    checkEnv();
    setupEnv();

    config = tf.ConfigProto(allow_soft_placement=True, 
        log_device_placement=False)

    training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations = loadDatasets()

    with tf.name_scope("computational_graph"):
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)
        testing_init_op = iterator.make_initializer(testing_dataset)
        global_step = tf.train.create_global_step()
        learning_rate = LEARNING_RATE
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        batch, label, is_training = iterator.get_next()
        keep_prob = tf.cond(is_training, lambda: KEEP_PROB, lambda: 1.0)
        model = cifar10Model(keep_prob)
        model_output = model.computationalGraph(batch)
        model_output_reshaped = tf.reshape(model_output,[-1, NUM_CLASSES])
        model_output_argmax = tf.argmax(model_output_reshaped, 1)
        label_reshaped = tf.reshape(label,[-1, NUM_CLASSES])
        label_argmax = tf.argmax(label_reshaped, 1)
        prediction = tf.equal(model_output_argmax, label_argmax)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_reshaped, logits=model_output_reshaped))
        reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = cross_entropy + LAMBDA_REG*reg
        train_step = opt.minimize(loss, global_step=global_step)

    with tf.Session(config=config) as sess:
        print("training session starting...")
        sys.stdout.flush()
        
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()


        epoch_test_loss = 0.0
        epoch_test_acc = 0.0

        sess.run(testing_init_op)
        for iteration in range(testing_iterations):   
            try:   
                lab, pred, loss_, acc = sess.run([label, prediction, loss, accuracy])
            except tf.errors.ResourceExhaustedError as err:
                print(err)
                sys.stdout.flush()
            except tf.errors.OutOfRangeError as err:
                print(error)
                sys.stdout.flush()
            except tf.errors.InternalError as err:
                print(err)
                sys.stdout.flush()
            else:
                epoch_test_loss += loss_
                epoch_test_acc += acc
                print("testing", "iteration", iteration)
            finally:
                pass
                
        epoch_test_loss = str(round(epoch_test_loss / testing_iterations,
            4))
        epoch_test_acc = str(round(epoch_test_acc / testing_iterations,
            4))
        print("testing loss:", 
            epoch_test_loss, "testing acc:", epoch_test_acc)
        sys.stdout.flush()  

    end_time = (time.time() - start_time) / 60
    print("finished training in: ", round(end_time, 2))
    sys.stdout.flush() 

if __name__ == '__main__':
    main();
