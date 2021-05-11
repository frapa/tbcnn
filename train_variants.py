import os
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import gc

import train_loop

def shuffle(images, labels):
    """Return shuffled copies of the arrays, keeping the indexes of
    both arrays in corresponding places
    """

    cp_images = np.copy(images)
    cp_labels = np.copy(labels)

    rng_state = np.random.get_state()
    np.random.shuffle(cp_images)
    np.random.set_state(rng_state)
    np.random.shuffle(cp_labels)

    return cp_images, cp_labels
    
def split_train_and_test(images, labels, ratio=0.8):
    """Splits the array into two randomly chosen arrays of training and testing data.
    ratio indicates which percentage will be part of the training set."""

    images, labels = shuffle(images, labels)

    split = int(images.shape[0] * ratio)

    training_images = images[:split]
    training_labels = labels[:split]

    test_images = images[split:]
    test_labels = labels[split:]

    return [training_images, training_labels], [test_images, test_labels]

def create_sets(num, images, labels):
    """Splits the array into num equally sized sets."""

    images, labels = shuffle(images, labels)

    set_size = images.shape[0] // num
    remaining = images.shape[0] - set_size * num

    image_sets = []
    label_sets = []
    offset = 0
    for i in range(num):
        extra = 1 if i < remaining else 0

        image_sets.append(images[i*set_size + offset:i*set_size + set_size + offset + extra])
        label_sets.append(labels[i*set_size + offset:i*set_size + set_size + offset + extra])

        offset += extra
    
    return image_sets, label_sets

def get_rotations(num, image_sets, label_sets):
    """Create rotations of the training and test sets for cross validation training
    This means if image_sets = [A, B, C] the output will be [[A, B], [B, C], [A, C]]
    for the training set."""

    training_sets = []
    test_sets = []
    for i in range(num):
        test_sets.append((
            image_sets[i],
            label_sets[i]
        ))

        training_sets.append((
            np.concatenate([s for j, s in enumerate(image_sets) if j != i]),
            np.concatenate([s for j, s in enumerate(label_sets) if j != i])
        ))
    
    return training_sets, test_sets

def train_single(inFile, size=512):
    """Train network a single time using the given files as input.

    inFile => path without extension (more than one file will be read)
    """

    print('Training...')

    # Load data
    images = np.load(inFile + '.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    # Create training and test sets
    training, test = split_train_and_test(images, labels)

    train_loop.train_net(training, test, size=size)

def train_cross_validation(inFile, sets=3, size=512):
    """Train network multiple times in a cross validation fashon, in order to
    cover all the dataset in the tests and avoid the effect of outliers.

    inFile => path without extension (more than one file will be read)
    sets   => number of cross validation sets (training will be repeated this many times
              and the size of the test set will be dataset_size / sets)
    """

    print('Starting {}-fold cross validation study...'.format(sets))

    # Load data
    images = np.load(inFile + '.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    # Create training and test sets for the cross validation study
    image_sets, label_sets = create_sets(sets, images, labels)

    training_sets, test_sets = get_rotations(sets, image_sets, label_sets)
    # import matplotlib.pyplot as plt
    # plt.imshow(training_sets[0][0][0,:,:,0]); plt.show();

    for i in range(sets):
        print('Set {}'.format(i+1))

        train_loop.train_net(
            training_sets[i],
            test_sets[i],
            size=size,
            run_name='Set {} ({})'.format(i+1, datetime.now().strftime(r'%Y-%m-%d_%H:%M')),
        )

        tf.reset_default_graph()
        gc.collect()
