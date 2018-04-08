TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_INC="$TF_INC -I $TF_INC/../../external/nsync/public"
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

CUDA_PATH=${CUDA_PATH:-"/usr/local/cuda/"}
export PATH=${CUDA_PATH}/bin:$PATH
CXXFLAGS=''

if [[ "$OSTYPE" =~ ^darwin ]]; then
	CXXFLAGS+='-undefined dynamic_lookup'
fi

cd roi_pooling_layer

gcc_comp=$1

os_arch=`uname -m | grep ^x86`

tf_fwk_so=$TF_LIB/libtensorflow_framework.so

if [ -d "$CUDA_PATH" ]; then
    if [ -e $tf_fwk_so ]; then
 	    nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
		    -I $TF_INC -L $TF_LIB -ltensorflow_framework -O2 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
		    -arch=sm_37
    else
 	    nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
		    -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
		    -arch=sm_37
    fi

    if [ ! "$gcc_comp"x = "same"x ]; then
        if [ -e $tf_fwk_so ]; then
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        roi_pooling_op.cu.o -I $TF_INC -L $TF_LIB -ltensorflow_framework -O2 -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC $CXXFLAGS \
		        -lcudart -L $CUDA_PATH/lib64            
        else
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        roi_pooling_op.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC $CXXFLAGS \
		        -lcudart -L $CUDA_PATH/lib64        
        fi
    
    else
        if [ -e $tf_fwk_so ]; then
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        roi_pooling_op.cu.o -I $TF_INC -L $TF_LIB -ltensorflow_framework -O2 -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
		        -lcudart -L $CUDA_PATH/lib64    
        else
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        roi_pooling_op.cu.o -I $TF_INC -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
		        -lcudart -L $CUDA_PATH/lib64            
        fi

    fi

else
    if [ ! "gcc_comp"x = "same"x ]; then
        if [ -e $tf_fwk_so ]; then
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        -I $TF_INC -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC $CXXFLAGS         
        else
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        -I $TF_INC -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC $CXXFLAGS           
        fi  
    else
        if [ -e $tf_fwk_so ]; then
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        -I $TF_INC -L $TF_LIB -ltensorflow_framework -O2 -fPIC $CXXFLAGS        
        else
            g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
		        -I $TF_INC -fPIC $CXXFLAGS        
        fi

    fi
fi

cd ..

#cd feature_extrapolating_layer

#nvcc -std=c++11 -c -o feature_extrapolating_op.cu.o feature_extrapolating_op_gpu.cu.cc \
#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

#g++ -std=c++11 -shared -o feature_extrapolating.so feature_extrapolating_op.cc \
#	feature_extrapolating_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
#cd ..
