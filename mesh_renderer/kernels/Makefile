CXX := g++
PYTHON_BIN_PATH = python

SRCS = rasterize_triangles_grad.cc rasterize_triangles_impl.cc rasterize_triangles_op.cc

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = ../rasterize_triangles_kernel.so

.PHONY: op
op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

.PHONY: clean
clean:
	rm -f $(TARGET_LIB)
