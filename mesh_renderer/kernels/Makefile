TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())');
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())');

rasterize_triangles_kernel.so:  rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc rasterize_triangles_impl.h
	echo $(TF_INC)
	echo $(TF_LIB)
