# Libs
import os
import sys

# Own modules
import preprocess
import prepare_input
import train_variants
import progress

# Constants
SIZE = 512

# Helper functions
def relPath(dir):
    "Returns path of directory relative to the executable"
    return os.path.join(os.path.dirname(__file__), dir)

# Crop and resize images
# This expects the images to be saved in the data folder
# Extract 1/4 more for cropping augmentation
print('Preprocessing...')
preprocess.preprocess(relPath('data'), relPath('preprocessed'), size=int(SIZE*1.1))

# Prepare input: convert to float with unit variance and zero mean,
# extract labels and save everything as a big numpy array to be used for training
print('Preparing input...')
prepare_input.prepare(relPath('preprocessed'), relPath('input'))

# print command to start tensorboard
progress.start_tensorboard()

# Train network
if '--cross-validation' in sys.argv:
    train_variants.train_cross_validation(relPath('input'), sets=3, size=SIZE)
else:
    train_variants.train_single(relPath('input'), size=SIZE)