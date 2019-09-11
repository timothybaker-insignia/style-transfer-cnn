import warnings
warnings.filterwarnings('ignore')

import errno
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import time

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from multiprocessing import cpu_count
from PIL import Image
from random import Random
from sklearn.model_selection import train_test_split

from CNN import cifar10ModelInception as cifar10Model

THREADS = cpu_count()-1             # number of system threads minus one

# Model Specific Parameters
NUM_CLASSES           =    10       # number of output classes

# Flags
RESTORE_MODEL         =    False    # restore previous model
SAVE_MODEL            =    True     # save model 
TEST_ONLY             =    False    # skips training
SAVEDMODEL            =    False    # saves SavedModel object for deploy

# Tunable Parameters
BATCH_SIZE            =    64      # training batch size
BUFFER_SIZE           =    1       # number of batches to buffer
EPOCHS                =    100        # number of epochs to train for
LEARNING_RATE         =    1e-4     # learning rate for gradient descent
KEEP_PROB             =    0.5      # keep probability for dropout layers
LAMBDA_REG            =    0.1      # lambda for kernel regularization

# I/O folders and files
DATA_PATH = "data/"
CHECKPOINT_PATH = "checkpoint/" 
OUTPUT_PATH = "output/"
SAVEDMODEL_PATH = "savedModel/"

if TEST_ONLY:
    EPOCHS = 0

def checkEnv():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATA_PATH)
    if RESTORE_MODEL:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), CHECKPOINT_PATH)

def setupEnv():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if SAVE_MODEL:
      if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if SAVEDMODEL:
        if not os.path.exists(SAVEDMODEL_PATH):
            os.makedirs(SAVEDMODEL_PATH)
        
def alphaLabelToNumLabel(key):
    label_dict = {"airplane" : 0,
                  "automobile" : 1,
                  "bird" : 2,
                  "cat" : 3,
                  "deer" : 4,
                  "dog" : 5,
                  "frog" : 6, 
                  "horse" : 7,
                  "ship" : 8,
                  "truck" : 9}
    return label_dict[key]

def numLabelToAlphaLabel(key):
    label_dict = {0 : "airplane",
                  1 : "automobile",
                  2 : "bird",
                  3 : "cat",
                  4 : "deer",
                  5 : "dog",
                  6 : "frog", 
                  7 : "horse",
                  8 : "ship",
                  9 : "truck"}
    return label_dict[key]

def loadTrainingAndValidationBatches():
    image_and_label = []    
    for i in range(10):
        filepath = "data/train/"+numLabelToAlphaLabel(i)
        for filename in os.listdir(filepath):
            image_and_label.append((filename, i))

    Random(0).shuffle(image_and_label)
    batches = []
    mini_batch_frame = []
    mini_batch_label = []
    j = 0
    for i in range(len(image_and_label)):
            filename, index = image_and_label[i]
            filepath = "data/train/"+numLabelToAlphaLabel(index)
            if j < BATCH_SIZE:  
              im_frame = Image.open(filepath + "/" + filename)
              np_frame = np.array(im_frame)
              label = np.array(np.zeros(10))
              label[index] = 1
              mini_batch_label.append(label)
              mini_batch_frame.append(np_frame) 
              j = j + 1
            else:
              mini_batch = (mini_batch_frame, mini_batch_label, True)
              batches.append(mini_batch)
              mini_batch_frame = []
              mini_batch_label = []
              j = 0
              im_frame = Image.open(filepath + "/" + filename)
              np_frame = np.array(im_frame)
              label = np.array(np.zeros(10))
              label[index] = 1
    batches.pop()
    Random(0).shuffle(batches)
    training, validation = train_test_split(batches)
    validation = [(i,j,False) for (i,j,k) in validation]
    return training, validation

def loadTestingBatches():
    image_and_label = []    
    for i in range(10):
        filepath = "data/test/"+numLabelToAlphaLabel(i)
        for filename in os.listdir(filepath):
            image_and_label.append((filename, i))

    Random(0).shuffle(image_and_label)
    batches = []
    mini_batch_frame = []
    mini_batch_label = []
    j = 0
    for i in range(len(image_and_label)):
            filename, index = image_and_label[i]
            filepath = "data/test/"+numLabelToAlphaLabel(index)
            if j < BATCH_SIZE:  
              im_frame = Image.open(filepath + "/" + filename)
              np_frame = np.array(im_frame)
              label = np.array(np.zeros(10))
              label[index] = 1
              mini_batch_label.append(label)
              mini_batch_frame.append(np_frame) 
              j = j + 1
            else:
              mini_batch = (mini_batch_frame, mini_batch_label, False)
              batches.append(mini_batch)
              mini_batch_frame = []
              mini_batch_label = []
              j = 0
              im_frame = Image.open(filepath + "/" + filename)
              np_frame = np.array(im_frame)
              label = np.array(np.zeros(10))
              label[index] = 1
    batches.pop()
    Random(0).shuffle(batches)
    return batches

def loadDatasets():
    print("loading training and validation batches")
    sys.stdout.flush()
    training_batches, validation_batches = loadTrainingAndValidationBatches() 
    total_training_batches = len(training_batches)
    Random(0).shuffle(training_batches)                                    
    training_iterations = int(len(training_batches))

    def genBatches():
        for i in range(len(training_batches)):
            batch, label, is_training = training_batches[i]
            yield (batch, label, is_training)

    training_dataset = tf.data.Dataset.from_generator(genBatches, (tf.float32,tf.float32,tf.bool))
    training_dataset = training_dataset.shuffle(buffer_size=BUFFER_SIZE)
    training_dataset = training_dataset.prefetch(buffer_size=BUFFER_SIZE)
    print("loaded",total_training_batches,"training batches")
    sys.stdout.flush()

    total_validation_batches = len(validation_batches)
    Random(0).shuffle(validation_batches)             
    validation_iterations = int(len(validation_batches))
    def gen_validation():
        for i in range(validation_iterations):
            batch, label, is_training = validation_batches[i]
            yield (batch, label, is_training)
    validation_dataset = tf.data.Dataset.from_generator(gen_validation, (tf.float32,tf.float32,tf.bool))
    validation_dataset = validation_dataset.prefetch(buffer_size=BUFFER_SIZE)
    print("loaded",total_validation_batches,"validation batches")
    sys.stdout.flush()

    print("loading testing batches...")
    sys.stdout.flush()
    testing_batches = loadTestingBatches()
    total_testing_batches = len(testing_batches)
    Random(0).shuffle(testing_batches)                                    
    testing_iterations = int(len(testing_batches))
    def gen_testing():
        for i in range(testing_iterations):
            batch, label, is_training = testing_batches[i]
            yield (batch, label, is_training)
    testing_dataset = tf.data.Dataset.from_generator(gen_testing, (tf.float32,tf.float32,tf.bool))
    testing_dataset = testing_dataset.prefetch(buffer_size=BUFFER_SIZE)
    print("loaded",total_testing_batches,"testing batches")
    print("building computational graph...")
    sys.stdout.flush()

    return training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations

def main():
    start_time = time.time()
    print("started at", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))
    sys.stdout.flush()

    setupEnv();
    checkEnv();

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations = loadDatasets()

    if True:
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
            model = cifar10Model(NUM_CLASSES, keep_prob)
            model_output = model.computationalGraph(batch)
            model_output_reshaped = tf.reshape(model_output,[-1, NUM_CLASSES])
            model_output_argmax = tf.argmax(model_output_reshaped, 1)
            label_reshaped = tf.reshape(label,[-1, NUM_CLASSES])
            label_argmax = tf.argmax(label_reshaped, 1)
            prediction = tf.equal(model_output_argmax, label_argmax)
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_reshaped, logits=model_output_reshaped)
            cross_entropy =  tf.reduce_mean(softmax_cross_entropy)
            reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss = cross_entropy + LAMBDA_REG*reg
            train_step = opt.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        print("initializing graph...")
        sys.stdout.flush()
        
        if RESTORE_MODEL:
            try:
                saver.restore(sess, CHECKPOINT_PATH + "model.ckpt")
            except ValueError as err:
                pass
        
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()      

        print("training session starting...")
        sys.stdout.flush()

        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            epoch_train_time = 0.0
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            epoch_validation_loss = 0.0
            epoch_validation_acc = 0.0
            
            sess.run(training_init_op)
            while True:
                try:
                    t_stime = time.time()
                    _, loss_, acc = sess.run([train_step, loss, accuracy])
                    t_etime = time.time()
                    t_rtime = round((t_etime - t_stime) / 60, 4)
                    epoch_train_time += t_rtime
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    break
                except tf.errors.InternalError as err:
                    print(err)
                    sys.stdout.flush()
                else:
                    epoch_train_loss += loss_
                    epoch_train_acc += acc
                finally:
                    pass

            sess.run(validation_init_op)
            while True:   
                try:   
                    loss_, acc = sess.run([loss, accuracy])
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    break
                except tf.errors.InternalError as err:
                    print(err)
                    sys.stdout.flush()
                else:
                    epoch_validation_loss += loss_
                    epoch_validation_acc += acc
                finally:
                    pass
                
            epoch_validation_loss = str(round(
                epoch_validation_loss / validation_iterations, 4))
            epoch_validation_acc = str(round(
                epoch_validation_acc / validation_iterations, 4))
            epoch_run_time = (time.time() - epoch_start_time) / 60
            epoch_run_time = str(round(epoch_run_time, 2))
            epoch_train_time = str(round(epoch_train_time, 2))
            epoch_train_loss = str(round(epoch_train_loss / training_iterations,
                4))
            epoch_train_acc = str(round(epoch_train_acc / training_iterations,
                4))
            print("run-time:", epoch_run_time, "training time:",
                epoch_train_time)
            print("epoch", str(epoch), "validation loss:", 
                epoch_validation_loss, "validation acc:", epoch_validation_acc)
            print("epoch", str(epoch), "training loss:", 
                epoch_train_loss, "training acc:", epoch_train_acc)
            sys.stdout.flush()  

        epoch_test_loss = 0.0
        epoch_test_acc = 0.0
        sess.run(testing_init_op)
        while True:
            try:   
                lab, pred, loss_, acc = sess.run([label, prediction, loss, accuracy])
            except tf.errors.ResourceExhaustedError as err:
                print(err)
                sys.stdout.flush()
            except tf.errors.OutOfRangeError as err:
                break
            except tf.errors.InternalError as err:
                print(err)
                sys.stdout.flush()
            else:
                epoch_test_loss += loss_
                epoch_test_acc += acc
            finally:
                pass

        if SAVE_MODEL:
            saver.save(sess, CHECKPOINT_PATH + "model.ckpt") 
                
        epoch_test_loss = str(round(epoch_test_loss / testing_iterations,4))
        epoch_test_acc = str(round(epoch_test_acc / testing_iterations,4))
        print("testing loss:",epoch_test_loss, "testing acc:", epoch_test_acc)
        sys.stdout.flush() 

    end_time = (time.time() - start_time) / 60
    print("finished training in: ", round(end_time, 2))
    sys.stdout.flush() 

if __name__ == '__main__':
    main();
