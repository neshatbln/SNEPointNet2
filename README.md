# SNEPointNet2
_
This repo is the implementation of an adapted version of PointNet++, called Surface Normal Enhanced PointNet++ (SNEPointNet++), 
for semantic segmentation of two types of concrete surface defects: spalls and cracks. 

The code is tested under TensorFlow-GPU 1.15.1, Cuda 11.0, and python 3.6.

The available dataset should be annotated into three classes spall, crack, and nodefect. 

The raw data is stored in “bridge” and two empty folders “bridge_npy” and “bridge_npy_h5” are created to store numpy and h5 files, respectively.

First, based on some given parameters (e.g., size of the block, stirde size) in indoor_3dutil.py, all the point clouds which are listed in 
“/sem_seg/meta/anno_paths.txt”, are converted into *.npy format using collect_indoor3d_data.py. 
The name of the numpy files should be listed as “/sem_seg/meta/all_data_label.txt”. 
Then, some parameters (e.g. number of points, block size, stride size) are set in “gen_indoor3d_h5.py”, which is used to generate the h5 files based on the npy. 
The list of h5 files should be written in “all_files.txt” file. These files are used as the input of the training model.

“train_and_test_multigpu.py” is used for training the dataset and testing it on multi GPUs. 
The results including a model, the confusion matrix, and the required calculated parameters during training and testing, are stored in “log” folder._
