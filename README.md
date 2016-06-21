# Ball-Tracking
BSc Thesis - Ball Tracking with Deep Learning

Abstract
--------
Deep learning systems have evolved rapidly in the last years, thanks to the enormous availability of data and the increasingly lower cost of computational power. This kind of systems now produce state-of-the-art results in many areas, such as image classification and object detection, surpassing previously existing systems meticulously hand-crafted by engineers.
We devise a deep learning pipeline to automatically detect and track a ball in a video.
Our system uses a convolutional neural network trained to model the classification problem of determining the class of an image (classifier), which is either sphere or background.
In addition to the classifier, our system uses another convolutional neural network trained to model the regression problem of determining the center and the radius of the sphere in an image (regressor).
Both networks are trained on synthetic datasets of images we generate.
Finally, our system integrates the results from both the classifier and the regressor to locate the ball in each frame of the input video stream.

# Requirements

Ubuntu 14.04
----
* SciPy (http://www.scipy.org/install.html)
* CUDA (https://developer.nvidia.com/cuda-downloads)
After installing, add the following lines to your `.bashrc`:
```
export CUDA_HOME=/usr/local/cuda-7.5 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
```
* Caffe framework (http://caffe.berkeleyvision.org/install_apt.html)
After installation, be sure to set the caffe root environment variable in `.bashrc`:
```
CAFFE_ROOT=<path to CAFFE>
export CAFFE_ROOT
```
* HDF5_Py - `sudo apt-get install libhdf5-dev python-h5py`
* Blender - `sudo apt-get install blender`
* FFmpeg - `sudo apt-get install ffmpeg`
* Python Pillow - `sudo apt-get install python-imaging`

# How to use

Generating the datasets
----

To generate the datasets, set the `generate_datasets.sh` file as executable and then run it.
```
$ chmod +x generate_datasets.sh
$ ./generate_datasets.sh
```
This script will download the backgrounds needed for the dataset generation and some examples to try the system on.
Once the backgrounds are downloaded, the script will render 10000 different spheres with blender.
Finally, the script will execute `generate_dataset_multithreaded.py` to generate all datasets necessary for the training of the networks.
Generating the datasets can take a *very long* time (hours), depending on the system specs.

Training the networks
----

To train the classification network, set the `train_classification_net.py` file as executable and then run it.
```
$ chmod +x train_classification_net.py
$ ./train_classification_net.py
```

To train the regression network, set the `train_regression_net.py` file as executable and then run it.
```
$ chmod +x train_regression_net.py
$ ./train_regression_net.py
```

If the scripts output a CUDA memory error, it means that your GPU has not enough video RAM to train the networks.
You can try setting Caffe to only use the CPU, by deleting the following lines in the scripts `train_classification_net.py`, `train_regression_net.py`, `single_frame.py` and `video.py`.
```
# setup solver
caffe.set_device(0)
caffe.set_mode_gpu()
```

Training can take a long time depending on the system specs.

Image and Video analysis
----
Once the networks are trained, they can be used for detecting and tracking a ball in images and videos.

You can try the system on the images and videos found in the `examples` folder.

For images, set the `single_frame.py` file as executable and run it.
```
$ chmod +x single_frame.py
$ ./single_frame.py lenet_classification_test.prototxt lenet_regression_test.prototxt balltracker_classification.caffemodel balltracker_regression.caffemodel <image file>
```

For videos, set the `video.py` file as executable and run it.
```
$ chmod +x video.py
$ ./video.py lenet_classification_test.prototxt lenet_regression_test.prototxt balltracker_classification.caffemodel balltracker_regression.caffemodel <input video file> <output video file>
```
