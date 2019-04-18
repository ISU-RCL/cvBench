## Benchmarking Analysis of Vision Kernels on Embedded CPU, GPU and FPGA :
 
This reposorary can be used to evaluate the perofmance of vision kernels implementation from the commonly used and publicly available vision libraries: OpenCV, Nvidia VisionWorks and xfOpenCV on embedded CPUs, GPUs and FPGAs platofrms.  


* [Repository structure](#Repository_structure) 
* [Hardware and Software Environments](#Hardware_and_Software_Environments)
* [List of Vision Kernels](#List_of_Vision_Kernels)
* [Installation](#Installation) 
* [Building and Running Test Codes](#Building_and_Running_Test_Codes) 
* [Results Summary](#Results_Summary) 
* [References](#references)
* [License](#license) 
 
 

## Repository structure

This repository contains benchmarking data and source code for evaluating the performance of 50+ vision kernels on embedded CPU, GPU and FPGA.
  
This repository consists of two parts:
* FPGA testbenches:
* GPU testbenches:
    
## Hardware and Software Environments
* Software Environments:
	* OpenCV 3.x: https://www.opencv.org/
	* xfOpenCV 2018.x: https://github.com/Xilinx/xfopencv
	* VisionWorks 1.6 : https://developer.nvidia.com/embedded/jetpack
 
* Hardware Platforms:
	* **FPGA board**: Xilinx Zynq UltraScale+ ZCU102/ZCU104, PYNQ-Z1 or Ultra96.
	* **GPU board**: NVIDIA Jetson TX1/ TX2.

 
## List of Vision Kernels

 
| Input Processing | Image Arithmatic | Filters       |  Image Analysis | Geometric Transforms|  Features  | Flow and Depts|
| -------------    | -------------    | ------------- | -------------   |    -------------    | ---------- | ----------    |
| combine          | AbsDiff          |  filter2D     |calcHist         | affine warp         | canny      | OF pyramid    |
| extract          | accumulate       |  box filter   |equalizeHist     |perspective warp     | fast       | stereoBM      | 
| convertTo        |accumulate squared|  dilate   |integral image   | resize              | harris     |               |
| cvtConvert       |accumulate weighted| erode        |mean std dev     | remap               |            |               | 
| table lookup     | add/subtract     |  median       |min/max loc      |                     |            |               | 
|                  |  mulitply        | pyramidUp     |                 |                     |            |               | 
|                  | threshold        | pyramidDown   |                 |                     |            |               | 
|             | bitwise and,or,xor,not|               |                 |                     |            |               | 
|                  | magnitude        |               |                 |                     |            |               | 
|                  | phase            |               |                 |                     |            |               | 


## Installation

Installation

* clone this repository with Xilinx's [PYNQ-ComputerVision](https://github.com/Xilinx/PYNQ-ComputerVision.git) submodules

```
git clone --recursive https://github.com/ISU-RCL/cvBench.git
```
## Building and Running Test Codes 
 

## Results Summary


## Reference 

## License
The source for this project is licensed under the [3-Clause BSD License](LICENSE)
