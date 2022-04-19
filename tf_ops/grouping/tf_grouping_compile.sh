#/bin/sh
nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/ -I /CUDA_HOME/extras/CUPTI/include -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -L /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0

# TF1.8
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/ -I /usr/local/include -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/lib64/  #-L home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I$TF_INC -I $EBROOTCUDA/include -I$TF_INC/external/nsync/public -L /usr/local/cuda-11.4/targets/x86_64-linux/lib -lcudart -L $EBROOTCUDAlib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
