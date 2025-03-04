main: main.o basic.o
	@nvcc -o main *.o -g -m64 -I/usr/local/cuda/samples/Common -gencode arch=compute_86,code=sm_86 --compiler-options -Wall -Xcompiler -fopenmp

basic.o:
	@nvcc -c -g --resource-usage basic.cu -I/usr/local/cuda/samples/Common -gencode arch=compute_86,code=sm_86

main.o:
	@nvcc -c -g --resource-usage main.cu -I/usr/local/cuda/samples/Common -gencode arch=compute_86,code=sm_86
