# SNEPointNet2
_
This repo includes the following:

1) The 3D point cloud dataset was created in Dr. Amin Hammad's research lab at Concordia University. It includes 102 segments of four reinforced concrete bridges in Montreal, which were scanned using a FARO Focus3D scanner.

2) the implementation of an adapted version of PointNet++, called Surface Normal Enhanced PointNet++ (SNEPointNet++), 
for semantic segmentation of two types of concrete surface defects: spalls and cracks. 

This method is developed mainly by Neshat Bolourian, a Ph.D. candidate at Concordia University under the supervision of Dr. Amin Hammad.

The code is tested under TensorFlow-GPU 1.15.1, Cuda 11.0, and python 3.6.

The raw data is stored in “bridge” folder and two empty folders “bridge_npy” and “bridge_npy_h5” are created to store numpy and h5 files, respectively.

First, based on some given parameters (e.g., size of the block, stirde size) in indoor_3dutil.py, all the point clouds which are listed in 
“/sem_seg/meta/anno_paths.txt”, are converted into *.npy format using collect_indoor3d_data.py. 
The name of the numpy files should be listed as “/sem_seg/meta/all_data_label.txt”. 
Then, some parameters (e.g. number of points, block size, stride size) are set in “gen_indoor3d_h5.py”, which is used to generate the h5 files based on the npy. 
The list of h5 files should be written in “all_files.txt” file. These files are used as the input of the training model.

“train_and_test_multigpu.py” is used for training the dataset and testing it on multi GPUs. 
The results including a model, the confusion matrix, and the required calculated parameters during training and testing, are stored in “log” folder._

Related Publications:

Journal Papers
•	Bolourian, N., Nasrollahi, M., Bahreini, F., & Hammad, A. (2022). Point cloud-based concrete surface defect semantic segmentation using modified PointNet++. Journal of Computing in Civil Engineering, ASCE (Submitted).

Conference Papers
•	Bolourian, N., Hammad, A., & Ghelmani, A. (2022). Point cloud-based concrete surface defect semantic segmentation using modified PointNet++. In 29th EG-ICE International Workshop on Intelligent Computing in Engineering, Denmark. 
•	Nasrollahi, M., Bolourian, N., & Hammad, A. (2019). Concrete surface defect detection using deep neural network based on lidar scanning. In Proceedings of the CSCE Annual Conference, Laval, Greater Montreal, QC, Canada (pp. 12-15).

