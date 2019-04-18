## GPU OpenCV Benchmark tests
 
This sub-folder contains the test code for benchmarking vision kernels on GPU platforms such as: NVIDIA Jetson TX1/TX2. Three different implementations of vision kernels can be evaualted: (1) OpenCV on the CPU core, (2) CUDA implementation and (3) VisionWork implementation.


### Builing source code

#### (1) OpenCV Implementation:

To compile test code:

```commandline
$ cmake ..  
$ make testName  
```
#### (2) OpenCV Implementation with CUDA Support:

To enable compiling test code with CUDA support:

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







