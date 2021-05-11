import os
import numpy as np
import imageio

def prepare(inDir, outFile):
    """Prepare input: convert to float with unit variance and zero mean,
    extract labels and pack everything into a big numpy array to be used for training

    outFile => path without extension (more than one file will be created)
    """

    if os.path.exists(outFile + '.npy'):
        return print("Input was already prepared")
    
    files = sorted(os.listdir(inDir))
    num = len(files)

    name_list = []
    label_list = []
    image_list = []
    for f in files:
        in_path = os.path.join(inDir, f)
        
        filename = os.path.splitext(f)[0]
        pieces = filename.split('_')
        name = pieces[1]
        label = int(pieces[2]) # 1 tbc, 0 nothing

        img = imageio.imread(in_path)

        # Convert to float
        img_float = img.astype(np.float32)
        
        label_list.append(label)
        name_list.append(name)
        image_list.append(img_float)
    
    # Now we have all images in an array
    # First convert it to a single ndarray instead of a list
    images = np.stack(image_list)
    labels = np.array(label_list, dtype=np.int32)

    # Input normalization
    # Remove mean
    images -= np.mean(images)
    # Divide by standard deviation
    images /= np.std(images)

    # Add dummy channel layer
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

    # Write data
    np.save(outFile + '.npy', images)
    np.save(outFile + '_labels.npy', labels)
    np.save(outFile + '_patients.npy', name_list)
