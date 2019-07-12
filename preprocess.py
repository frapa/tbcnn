import os
import sys
import numpy as np
import scipy.misc
import skimage.transform

def remove_border(img, threshold=0):
    "Crop image, throwing away the border below the threshold"
    mask = img > threshold
    return img[np.ix_(mask.any(1), mask.any(0))]

def crop_center(img, size):
    "Crop center sizexsize of the image"
    y, x = img.shape
    startx = (x - size) // 2 
    starty = (y - size) // 2
    return img[starty:starty+size, startx:startx+size]

def bigger_edge(img):
    y, x = img.shape
    return y if y < x else x

def preprocess(inDir, outDir, size=512):
    "Preprocess files, resizing them to sizexsize pixels and removing black borders"

    # Ensure output folder exists
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    files = sorted(os.listdir(inDir))
    num = len(files)

    if num == 0:
        print("Please put the images into the data folder. Download from https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/")
        sys.exit(1)

    for i, f in enumerate(files):
        in_path = os.path.join(inDir, f)
        out_path = os.path.join(outDir, f)

        # Skip files which are not in the correct format
        ext = os.path.splitext(f)[1]
        if ext.lower() != '.png':
            print('Skipping file {}, as it isn\'t a PNG image.'.format(f))


        if os.path.exists(out_path):
            # If the file was already preprocessed, do nothing
            continue

        print('Preprocessing {} - {} %'.format(f, int(i / num * 100)), end='\r')

        img = scipy.misc.imread(in_path)

        # If the image is RGB, compress it
        if len(img.shape) > 2:
            img = img.mean(2)

        # PREPROCESSING
        # Remove black border (sometimes there is a black band)
        img_noborder = remove_border(img)
        # Find bigger edge
        edge = bigger_edge(img_noborder)
        # Crop center
        img_cropped = crop_center(img_noborder, edge)
        # Resize to final size
        img_resized = skimage.transform.resize(img_cropped, (size, size), order=3)

        scipy.misc.imsave(out_path, img_resized)