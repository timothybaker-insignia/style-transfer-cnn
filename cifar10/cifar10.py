import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.dont_write_bytecode = True
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import memory_saving_gradients
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
from CNN import mariana as cnn
from loadData import loadDatasets
from trainingManager import TrainingManager

# Model Specific Parameters
NUM_CLASSES           =    10       # number of output classes
INPUT_CHANNELS        =    3        # number of input channels

# Tunable Parameters
<<<<<<< HEAD
BATCH_SIZE            =    32      # training batch size
BUFFER_SIZE           =    1      # number of batches to buffer
EPOCHS                =    100        # number of epochs to train for
LEARNING_RATE         =    1e-4     # learning rate for gradient descent
=======
BATCH_SIZE            =    512      # training batch size
ROTATE_BATCHES        =    True     # augment training data by adding rotated samples
ROTATIONS             =    8        # number of times to rotate data if ROTATE_BATCHES
ZOOM_BATCHES          =    True     # augment training data by adding zoomed in samples
ZOOMS                 =    8        # number of times to zoom data if ZOOM_BATCHES
GAUSSIAN_BATCHES      =    False    # augment training data by adding samples with added gaussian noise
GAUSSIANS             =    8        # number of times to apply random gaussian filters if GAUSSSIAN_BATCHES
BUFFER_SIZE           =    2        # number of batches to buffer
SHUFFLE_SIZE          =    10       # number of batches to shuffle (check dataset API for more info)
EPOCHS                =    100       # number of epochs to train for
>>>>>>> af9729adf6c9c0cb80c5393579f33da7c7a62848
KEEP_PROB             =    0.5      # keep probability for dropout layers
LAMBDA_REG            =    1e-4     # lambda for kernel regularization
LEARNING_RATE         =    1e-3     # learning rate for gradient descent

def main():
<<<<<<< HEAD
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
        
        sess.run(tf.global_variables_initializer())

        if RESTORE_MODEL:
            try:
                saver.restore(sess, CHECKPOINT_PATH + "model.ckpt")
            except ValueError as err:
                pass
        
        tf.get_default_graph().finalize()      

        print("training session starting...")
        sys.stdout.flush()

=======
    tm = TrainingManager()
    tm.saveModel(True)
    tm.restoreModel(False)

    with tf.device('/cpu:0'):
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations = \
            loadDatasets(BATCH_SIZE, SHUFFLE_SIZE, BUFFER_SIZE, ROTATE_BATCHES, ROTATIONS, ZOOM_BATCHES, ZOOMS, GAUSSIAN_BATCHES)
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)
        testing_init_op = iterator.make_initializer(testing_dataset)
        global_step = tf.train.create_global_step()
        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        batch, label, is_training = iterator.get_next()
        keep_prob = tf.cond(is_training, lambda: KEEP_PROB, lambda: 1.0)       
    with tf.device('/gpu:1'):
        model = cnn(NUM_CLASSES, INPUT_CHANNELS, keep_prob).computationalGraph(batch)     
        model_reshaped = tf.reshape(model,[-1, NUM_CLASSES])
        model_argmax = tf.argmax(model_reshaped, 1)
        label_reshaped = tf.reshape(label,[-1, NUM_CLASSES])
        label_argmax = tf.argmax(label_reshaped, 1)      
        reg = tf.reduce_sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        prediction = tf.equal(model_argmax, label_argmax) 
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_reshaped, logits=model_reshaped))
        loss = cross_entropy + LAMBDA_REG*reg  
        train_step = opt.minimize(loss, global_step=global_step)      

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session(config=config) as sess:
        tm.initSess(sess, saver=saver)       
>>>>>>> af9729adf6c9c0cb80c5393579f33da7c7a62848
        for epoch in range(EPOCHS):
            sess.run(training_init_op)
            while True:
                try:
                    _, loss_, acc = sess.run([train_step, loss, accuracy])
                except tf.errors.OutOfRangeError as err:
                    break
                else:
                    tm.incrementTrainingLossAndAcc(loss_, acc)
            sess.run(validation_init_op)
            while True:   
                try:   
                    loss_, acc = sess.run([loss, accuracy])
                except tf.errors.OutOfRangeError as err:
                    break
                else:
                    tm.incrementValidationLossAndAcc(loss_, acc)
            tm.printEpochStats(training_iterations, validation_iterations, epoch)
        sess.run(testing_init_op)
        while True:
            try:   
                lab, pred, loss_, acc = sess.run([label, prediction, loss, accuracy])
            except tf.errors.OutOfRangeError as err:
                break
            else:
                tm.incrementTestingLossAndAcc(loss_, acc)    
        tm.printTestStats(testing_iterations)       
        tm.endSess()

if __name__ == '__main__':
    main();
