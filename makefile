a: addArray.cu
	nvcc -g -G addArray.cu -L/usr/X11/lib -lX11 -lstdc++