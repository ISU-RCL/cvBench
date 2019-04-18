## GPU OpenCV Benchmark tests
 
This section shows how to compile, run and measure perfomance of vision kernels on NVIDIA Jetson TX1/TX2 platforms (Pascal GPU or ARM Cortex-A57). 
Each vision kernel has three different implementations:
 1. OpenCV C++ implemenation. (ARM Cortex-A57)
 2. CUDA implementation. (Pascal GPU)
 3. VisionWork implementation.(Pascal GPU)
 
### Builing source code

To build test code for a vision kernel (i.e Filter2D), navigate to the kernel's directory, and create build folder:

```commandline
$ cd ./cvBench/GPUTests/Image Filters/testFilter2D/ 
$ mkdir build  
$ cd build 
```



#### (1) OpenCV Implementation:

To compile test code to run on (ARM Cortex-A57), excute:

```commandline
$ cmake ..  
$ make testName  
```
#### (2) OpenCV Implementation with CUDA Support:

To enable compiling test code with CUDA support, excute:

```commandline
$ cmake .. -DWITH_CUDA=ON
$ make testName  
``` 
Note: make sure that your opencv has been built with -DWITH_CUDA=ON .

#### (3) VisionWorks Implementation:

To enable compiling tests with VisionWorks support:

```commandline
$ cmake .. -DWITH_OPENVX=ON
$ make testName  
```


### Maximizing Performance

To maximize performance of Jetson TX1/TX2 platforms, excute: 

```commandline
$ sudo nvpmodel -m 0 
$ sudo ~/jetson_clocks.sh 
```
 
## Running the tests 







