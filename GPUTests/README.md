## GPU OpenCV Benchmark tests
 
This sections shows how to: (1) compile source code, (2) run tests and (3) measure perfomance of vision kernels on NVIDIA Jetson TX1/TX2 platforms. Jetson platforms features an integrated 256-core NVIDIA Pascal GPU and ARM Cortex-A57.

* Three different implementations of vision kernels can be evaualted: 
(1) OpenCV on the CPU core.
(2) CUDA implementation 
(3) VisionWork implementation.
 
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







