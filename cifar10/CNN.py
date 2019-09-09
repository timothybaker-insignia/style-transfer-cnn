import tensorflow as tf

from Layer import Layer
         
class CNN:
    def __init__(self):
        self.layers = []
        self.finalized = False

    def addLayer(self, layertype):
        if not layertype in ['maxpooling', 'dropout']:
            raise ValueError('layertype must be maxpooling or dropout')
        if not self.finalized:
            layer = Layer(layertype)
        else:
            print('Error: Cannot add layer after graph is finalized!')
        
    def addLayer(self, layertype, dimensions, inputchannels=0, outputchannels=0, activation='none', keepprob=1.0):
        if not self.finalized:
            layer = Layer(layertype, dimensions, inputchannels, outputchannels, activation, keepprob)
            self.layers.append(layer)
        else:
            print('Error: Cannot add layer after graph is finalized!')
        
    def clearLayers(self):
        self.layers = []

    def computationalGraph(self, batch):
        self.finalized = True
        for layer in self.layers:
            with tf.name_scope(layer.layertype + "-" + str(layer.id)):
                batch = layer.compute(batch)
        return batch

    def train(self, sess, training_init_op, epochs):
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_train_time = 0.0
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0          

            sess.run(training_init_op)
            for iteration in range(training_iterations):
                try:
                    t_stime = time.time()
                    _, lab, pred, loss_, acc = sess.run([train_step, label, prediction, total_loss, accuracy])
                    t_etime = time.time()
                    t_rtime = round((t_etime - t_stime) / 60, 4)
                    epoch_train_time += t_rtime
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.InternalError as err:
                    print(err)
                    sys.stdout.flush()
                else:
                    epoch_train_loss += loss_
                    epoch_train_acc += acc
                finally:
                    pass

            epoch_run_time = (time.time() - epoch_start_time) / 60
            epoch_run_time = str(round(epoch_run_time, 2))
            epoch_train_time = str(round(epoch_train_time, 2))
            epoch_train_loss = str(round(epoch_train_loss / training_iterations,
                4))
            epoch_train_acc = str(round(epoch_train_acc / training_iterations,
                4))
            print("run-time:", epoch_run_time, "training time:",
                epoch_train_time)
            print("epoch", str(epoch), "training loss:", 
                epoch_train_loss, "training acc:", epoch_train_acc)
            sys.stdout.flush()  

    def validate(self, sess, validation_init_op, epochs):
        for epoch in range(epochs):
            epoch_validation_loss = 0.0
            epoch_validation_acc = 0.0
            
            sess.run(validation_init_op)
            for iteration in range(validation_iterations):   
                try:   
                    lab, pred, loss_, acc = sess.run([label, prediction, loss, accuracy])
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    print(err)
                    sys.stdout.flush()
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
            print("epoch", str(epoch), "validation loss:", 
                epoch_validation_loss, "validation acc:", epoch_validation_acc)
            sys.stdout.flush()  
        
    def trainWithValidation(self, sess, training_init_op, validation_init_op, epochs):
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_train_time = 0.0
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            epoch_validation_loss = 0.0
            epoch_validation_acc = 0.0
            
            sess.run(training_init_op)
            for iteration in range(training_iterations):
                try:
                    t_stime = time.time()
                    _, lab, pred, loss_, acc = sess.run([train_step, label, prediction, total_loss, accuracy])
                    t_etime = time.time()
                    t_rtime = round((t_etime - t_stime) / 60, 4)
                    epoch_train_time += t_rtime
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.InternalError as err:
                    print(err)
                    sys.stdout.flush()
                else:
                    epoch_train_loss += loss_
                    epoch_train_acc += acc
                finally:
                    pass

            sess.run(validation_init_op)
            for iteration in range(validation_iterations):   
                try:   
                    lab, pred, loss_, acc = sess.run([label, prediction, loss, accuracy])
                except tf.errors.ResourceExhaustedError as err:
                    print(err)
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError as err:
                    print(err)
                    sys.stdout.flush()
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
        
