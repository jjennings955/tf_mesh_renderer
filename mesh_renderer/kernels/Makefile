TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())');
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())');

rasterize_triangles_kernel.so:  rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc rasterize_triangles_impl.h
	g++ -std=c++11 -shared rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc -o rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -ltensorflow_framework -O2
