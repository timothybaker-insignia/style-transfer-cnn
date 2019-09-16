import os
import sys

import numpy as np
import tensorflow as tf 

from itertools import repeat
from multiprocessing import cpu_count, current_process, Pool
from PIL import Image
from random import Random, randint
from scipy.ndimage import rotate, zoom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def modeToChannels(mode):
    modeDict = {
        '1' : 1, # 1-bit pixels, black and white
        'L' : 1, # 8-bit pixels, black and white
        'P' : 1, # 8-bit pixels
        'RGB' : 3, # 3x8-bit pixels, true color
        'RGBA' : 4, # 4x8-bit pixels, true color with transparency mask
        'CMYK' : 4, # 4x8-bit pixels, color separation
        'YCbCr' : 3, # 3x8-bit pixels, color video format
        'LAB' : 3, # 3x8-bit pixels, L*a*b color space
        'HSV' : 3, # 3x8-bit pixels, Hue, Saturation, Value color space
        'I' : 1, # 32-bit signed integer pixels
        'F' : 1} # 32-bit floating point pixels
    return modeDict[mode]

def rotateBatch(batch, angle):
    angle = 1.0 + angle*angle/3.0
    rotated_batch = rotate(batch[0], angle, reshape=False)
    label = batch[1]
    return (rotated_batch, label)

def zoomBatch(batch, seed):
    batch, label = batch
    oy, ox, _ = batch.shape
    zoom_factor = 1.0 + float(seed) // 2
    zoomed = zoom(batch, [zoom_factor, zoom_factor, 1.0])
    x, y, _ = zoomed.shape
    startx = x // 2 - (ox // 2)
    starty = y // 2 - (oy // 2)
    zoomed = np.array(zoomed[starty:starty + oy, startx:startx + ox, :])
    return zoomed, label

def fetchTrainBatch(image_and_label):
    filename, index = image_and_label
    filepath = "data/train/"+numLabelToAlphaLabel(index)
    img_frame = Image.open(filepath + "/" + filename)
    scaled_img_frame = StandardScaler().fit_transform(img_frame.getdata())
    height, width = img_frame.size
    channels = modeToChannels(img_frame.mode)
    np_frame = np.asarray(scaled_img_frame).reshape(height, width, channels)
    label = np.array(np.zeros(10))
    label[index] = 1
    batch = (np_frame, label)
    return batch

def loadTrainingAndValidationBatches(batch_size, rotate_batches, rotations, zoom_batches, zooms, gaussian_batches):
    image_and_label = []    
    for i in range(10):
        filepath = "data/train/"+numLabelToAlphaLabel(i)
        for filename in os.listdir(filepath):
            image_and_label.append((filename, i))    
    Random(0).shuffle(image_and_label)
    pool = Pool(cpu_count())
    batches = pool.map(fetchTrainBatch, iter(image_and_label))
    pool.close()
    pool.join()
    Random(0).shuffle(batches)
    training_batches, validation_batches = train_test_split(batches, test_size=0.10)
    rotated_batches = []
    if rotate_batches:
        for i in range(1,rotations+1):
            pool = Pool(cpu_count())
            results = pool.starmap(rotateBatch, zip(training_batches, repeat(i)))
            pool.close()
            pool.join()
            rotated_batches += results
    Random(0).shuffle(rotated_batches)
    zoomed_batches = []
    if zoom_batches:
        for i in range(1,zooms+1):
            pool = Pool(cpu_count())
            results = pool.starmap(zoomBatch, zip(training_batches, repeat(i)))
            pool.close()
            pool.join()
            zoomed_batches += results
    Random(0).shuffle(zoomed_batches)
    gaussianed_batches = []
    if gaussian_batches:
        pass
        #pool = Pool(cpu_count())
        #results = pool.starmap(zoomBatch, zip(training_batches, repeat(i)))
        #pool.close()
        #pool.join()
        #zoomed_batches += results
    #Random(0).shuffle(zoomed_batches)

    training_batches += zoomed_batches
    Random(0).shuffle(training_batches)
    training_batches += rotated_batches
    Random(0).shuffle(training_batches)
    training_mini_batches = []
    frames = []
    labels = []
    j = 0
    for batch in training_batches:
        if j < batch_size:  
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
        else:
            training_mini_batches.append((frames, labels, True))
            frames = []
            labels = []
            j = 0
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
    Random(0).shuffle(training_mini_batches)
    validation_mini_batches = []
    frames = []
    labels = []
    j = 0
    for batch in validation_batches:
        if j < batch_size:  
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
        else:
            validation_mini_batches.append((frames, labels, False))
            frames = []
            labels = []
            j = 0
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
    return training_mini_batches, validation_mini_batches

def fetchTestBatch(image_and_label):
    filename, index = image_and_label
    filepath = "data/test/"+numLabelToAlphaLabel(index)
    img_frame = Image.open(filepath + "/" + filename)
    scaled_img_frame = StandardScaler().fit_transform(img_frame.getdata())
    height, width = img_frame.size
    channels = modeToChannels(img_frame.mode)
    np_frame = np.asarray(scaled_img_frame).reshape(height, width, channels)
    label = np.array(np.zeros(10))
    label[index] = 1
    batch = (np_frame, label)
    return batch
 
def loadTestingBatches(batch_size):
    image_and_label = []    
    for i in range(10):
        filepath = "data/test/"+numLabelToAlphaLabel(i)
        for filename in os.listdir(filepath):
            image_and_label.append((filename, i))
    Random(0).shuffle(image_and_label)
    pool = Pool(cpu_count())
    batches = pool.map(fetchTestBatch, iter(image_and_label))
    pool.close()
    pool.join()
    Random(0).shuffle(batches)
    mini_batches = []
    frames = []
    labels = []
    j = 0
    for batch in batches:
        if j < batch_size:  
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
        else:
            mini_batches.append((frames, labels, False))
            frames = []
            labels = []
            j = 0
            frame, label = batch
            frames.append(frame)
            labels.append(label)
            j = j + 1
    Random(0).shuffle(mini_batches)
    return mini_batches

def loadDatasets(batch_size, shuffle_size, buffer_size, rotate_batches, rotations, zoom_batches, zooms, gaussian_batches):
    print("loading training and validation batches")
    sys.stdout.flush()
    training_batches, validation_batches = loadTrainingAndValidationBatches(batch_size, rotate_batches, rotations, zoom_batches, zooms, gaussian_batches) 
    total_training_batches = len(training_batches)
    Random(0).shuffle(training_batches)                                    
    training_iterations = int(len(training_batches))

    def genBatches():
        for i in range(len(training_batches)):
            batch, label, is_training = training_batches[i]
            yield (batch, label, is_training)

    training_dataset = tf.data.Dataset.from_generator(genBatches, (tf.float32,tf.float32,tf.bool))
    training_dataset = training_dataset.shuffle(buffer_size=shuffle_size)
    training_dataset = training_dataset.prefetch(buffer_size=buffer_size)
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
    validation_dataset = validation_dataset.prefetch(buffer_size=buffer_size)
    print("loaded",total_validation_batches,"validation batches")
    sys.stdout.flush()

    print("loading testing batches...")
    sys.stdout.flush()
    testing_batches = loadTestingBatches(batch_size)
    total_testing_batches = len(testing_batches)
    Random(0).shuffle(testing_batches)                                    
    testing_iterations = int(len(testing_batches))
    def gen_testing():
        for i in range(testing_iterations):
            batch, label, is_training = testing_batches[i]
            yield (batch, label, is_training)
    testing_dataset = tf.data.Dataset.from_generator(gen_testing, (tf.float32,tf.float32,tf.bool))
    testing_dataset = testing_dataset.prefetch(buffer_size=buffer_size)
    print("loaded",total_testing_batches,"testing batches")
    print("building computational graph...")
    sys.stdout.flush()

    return training_dataset, training_iterations, validation_dataset, validation_iterations, testing_dataset, testing_iterations
