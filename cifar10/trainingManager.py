import errno
import os
import sys
import time

import tensorflow as tf

from multiprocessing import cpu_count

class Flags:
    restore_model = False
    save_model = False
    def __init__(self):
        pass

class Paths:
    data = "data/"
    checkpoint = "checkpoint/"
    output = "output/"
    def __init__(self):
        pass

class TrainingManager:
    def __init__(self):
        self._sess = None
        self._saver = None
        self._start_time = time.time()
        self._epoch_start_time = 0.0
        self._epoch_train_time = 0.0
        self._epoch_train_loss = 0.0
        self._epoch_train_acc = 0.0
        self._epoch_validation_loss = 0.0
        self._epoch_validation_acc = 0.0
        self._epoch_test_loss = 0.0
        self._epoch_test_acc = 0.0
        self.paths = Paths()
        self.flags = Flags()
        self.threads = cpu_count()-1
        self.startTime()
        self.checkAndSetupEnv()

    def initSess(self, sess, saver=None):
        self._sess = sess
        self._saver = saver
        print("initializing graph...")
        sys.stdout.flush()        
        if self.flags.restore_model and not self._saver == None:
            try:
                self._saver.restore(self._sess, self.paths.checkpoint + "model.ckpt")
            except ValueError as err:
                self._sess.run(tf.global_variables_initializer())
        else:
            self._sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()       
        print("training session starting...")
        sys.stdout.flush()
        self._epoch_start_time = time.time()

    def endSess(self):
        if self.flags.save_model and not self._saver == None:
            self._saver.save(self._sess, self.paths.checkpoint + "model.ckpt")
        self._sess = None
        run_time = (time.time() - self._start_time) / 60
        print("finished training in: ", round(run_time, 2))
        sys.stdout.flush() 

    def saveModel(self, save):
        self.flags.save_model = save

    def restoreModel(self, restore):
        self.flags.restore_model = restore

    def printEpochStats(self, training_iterations, validation_iterations, epoch):
        self._epoch_validation_loss = round(self._epoch_validation_loss / validation_iterations, 4)
        self._epoch_validation_acc = round(self._epoch_validation_acc / validation_iterations, 4)
        self._epoch_run_time = (time.time() - self._epoch_start_time) / 60
        self._epoch_run_time = round(self._epoch_run_time, 2)
        self._epoch_train_loss = round(self._epoch_train_loss / training_iterations,4)
        self._epoch_train_acc = round(self._epoch_train_acc / training_iterations,4)
        print("epoch", epoch, "run-time:", self._epoch_run_time)
        print("epoch", epoch, "validation loss:", self._epoch_validation_loss, "validation acc:", self._epoch_validation_acc)
        print("epoch", epoch, "training loss:", self._epoch_train_loss, "training acc:", self._epoch_train_acc)
        sys.stdout.flush()         
        self._epoch_train_loss = 0.0
        self._epoch_validation_loss = 0.0
        self._epoch_test_loss = 0.0
        self._epoch_train_acc = 0.0
        self._epoch_validation_acc = 0.0
        self._epoch_test_acc = 0.0
        self._epoch_train_time = 0.0
        self._epoch_start_time = time.time()

    def printTestStats(self, testing_iterations):
        self._epoch_test_loss = str(round(self._epoch_test_loss / testing_iterations,4))
        self._epoch_test_acc = str(round(self._epoch_test_acc / testing_iterations,4))
        print("testing loss:",self._epoch_test_loss, "testing acc:", self._epoch_test_acc)
        self._epoch_test_loss = 0.0
        self._epoch_test_acc = 0.0
        sys.stdout.flush() 
        
    def incrementTrainingLossAndAcc(self, loss, acc):
        self._epoch_train_loss += loss
        self._epoch_train_acc += acc

    def incrementValidationLossAndAcc(self, loss, acc):
        self._epoch_validation_loss += loss
        self._epoch_validation_acc += acc
    
    def incrementTestingLossAndAcc(self, loss, acc):
        self._epoch_test_loss += loss
        self._epoch_test_acc += acc

    def checkEnv(self):
        if not os.path.exists(self.paths.data):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.paths.data)
        if self.flags.restore_model:
            if not os.path.exists(paths.checkpoint):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.paths.checkpoint)

    def setupEnv(self):
        if not os.path.exists(self.paths.output):
            os.makedirs(self.paths.output)
        if self.flags.save_model or self.flags.restore_model:
          if not os.path.exists(self.paths.checkpoint):
            os.makedirs(self.paths.checkpoint)

    def checkAndSetupEnv(self):
        self.checkEnv()
        self.setupEnv()
        
    def startTime(self):
        print('\n')
        self.start_time = time.time()
        print("started at", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.start_time)))
        sys.stdout.flush()


