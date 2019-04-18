## GPU OpenCV Benchmark tests
 
This section shows how to compile, run and measure perfomance of vision kernels on NVIDIA Jetson TX1/TX2 platforms (Pascal GPU or ARM Cortex-A57). 
* Each test code has three different implementations:
	* OpenCV C++ implemenation. (ARM Cortex-A57)
	* CUDA implementation. (Pascal GPU)
	* VisionWork implementation.(Pascal GPU)
 
### Builing source code

To build test code for a vision kernel (i.e *Filter2D*), navigate to the kernel's directory, and create *build* folder:

```commandline
$ cd ./cvBench/GPUTests/Image Filters/testFilter2D/ 
$ mkdir build  
$ cd build 
```

#### (1) OpenCV Implementation. (ARM Cortex-A57):

To compile test code to run on the ARM-57 CPU of Jetson platform, excute:

```commandline
$ cmake ..  
$ make testFilter2D  
```
#### (2) CUDA implementation. (Pascal GPU):

To enable compiling test code with CUDA support, excute:

```commandline
$ cmake .. -DWITH_CUDA=ON
$ make testFilter2D  
``` 
**Note:** make sure that you built your opencv library with -DWITH_CUDA=ON .

#### (3) VisionWork implementation.(Pascal GPU):

To enable compiling tests with VisionWorks support, excute:

```commandline
$ cmake .. -DWITH_OPENVX=ON
$ make testFilter2D  
```

### Maximizing Performance

To maximize performance of Jetson TX1/TX2 platforms, excute: 

```commandline
$ sudo nvpmodel -m 0 
$ sudo ~/jetson_clocks.sh 
```
 
## Running Test Code 







