# Convolutional neural network for tuberculosis diagnosis

This repo contains the implementation of the convolutional neural network
for tuberculosis diagnosis described in [paper is coming, hepefully :-)].
The network uses frontal chest X-Rays images as input.

## How to get it to work

First clone the repo to your preferred location:

```bash
git clone https://github.com/frapa/tbcnn.git
```

We then need to install the dependencies. The network depends on (assuming python3):

```bash
# CUDA and CUDNN: install according to your platform. For ubuntu:
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt update

# Install CUDA and tools. Include optional NCCL 2.x
sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# tensorflow
# CPU only (training will take forever)
pip3 install --user tensorflow
# you need a nvidia GPU with CUDA support
pip3 install --user tensorflow-gpu

# numpy
pip3 install --user numpy

# scipy
pip3 install --user scipy

# skimage
pip3 install --user skimage
```

Once we have installed the needed dependencies, we need to download the
data to train the network on. You can get some from the NIH public dataset
[here](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/).
You can for example download the Montgomery dataset, open the zip and copy the
image files into the `data` directory. Another approach that I prefer is making
a directory `montgomery` copying the images into it and create a symlink called data
to this directory, as it makes swapping the database very easy:

```bash
ln -rs montgomery data
```

Then you can run the network running simply

```bash
python3 train.py
```

If you want to run a cross-validation study (5-fold), you can run:

```bash
python3 train.py --cross-validation
```

You can also open tensorboard at http://localhost:6006 to check graphs reporting
training and test accuracy and AUC.

There are no other options apart from these two, but the source code is
well commented and should be easy to play around with.