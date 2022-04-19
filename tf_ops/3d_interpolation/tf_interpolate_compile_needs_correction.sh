#/bin/sh
# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/ -I /CUDA_HOME/extras/CUPTI/include -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0

# TF1.8
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/ -I $EBROOTCUDA/include -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L $EBROOTCUDA/lib64/ -L home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/ -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I <path to your python virtualenv>/lib/python3.6/site-packages/tensorflow/include -I <path to your python virtualenv>/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/ -I $EBROOTCUDA/include -I /home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L $EBROOTCUDA/lib64/ -L/home/neshbln/tensorflow/lib/python3.6/site-packages/tensorflow -ltensorflow_framework
