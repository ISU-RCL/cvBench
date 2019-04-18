## Benchmarking Analysis of Vision Kernels on Embedded CPU, GPU and FPGA :

<p align="justify">

This repository contains benchmark framework for measuring and comparing energy efficiency of different vision kernels on embedded platforms. It aims to provide computer vision community an easy tool to analyze the performance of vision kernels on different hardware architectures and aids with determining which hardware architecture is most suitable for different kind of vision applications.

</p>

Table of Contents:
* [Repository structure](#Repository_structure) 
* [Hardware and Software Environments](#Hardware_and_Software_Environments)
* [List of Vision Kernels](#List_of_Vision_Kernels)
* [Installation](#Installation) 
* [Build Test Codes](#Build_Test_Codes) 
* [Results Summary](#Results_Summary) 
* [References](#references)
* [License](#license) 
 
 

## Repository structure

This repository consists of two parts:
* FPGA testbenches:
* GPU testbenches:
    
## Hardware and Software Environments
* Software Environments:
	* **OpenCV3.x** : https://www.opencv.org/
	* **xfOpenCV 2018.x** : https://github.com/Xilinx/xfopencv
	* **VisionWorks 1.6** : https://developer.nvidia.com/embedded/jetpack
 
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

|Input processing|
---------------------------------------------------------------
| combine |  extract |  convertTo |  cvtConvert | table lookup |

| First Header  | Second Header |
| ------------- | ------------- |
| combine  | extract  | 

## Installation

To clone the repository with [PYNQ-ComputerVision](https://github.com/Xilinx/PYNQ-ComputerVision.git) submodules, open a terminal and execute:

```
git clone --recursive https://github.com/ISU-RCL/cvBench.git
```
## Build Test Codes 
The steps required to build and run unit tests is described in:

+ [GPU Implementation  (GPUTest)](GPUTests/README.md)  
+ [FPGA Implementation (PLTests)](FPGATests/README.md)

## Results Summary

This table summarize the ratios of energy/frame reduction (reference CPU).

| Input Processing | Image Arithmatic | Filters       |


## Reference 

```
@inproceedings{EnergyEfficiencyFCCM2019,
  title={Analyzing the Energy-Efficiency of Vision Kernels on Embedded CPU, GPU and FPGA Platforms},
  author={Qasaimeh, Murad and Kristof, Denolf and Jack, Lo and Kees, Vissers and Zambreno, Joseph and Jones, Phillip H},
  booktitle={2019 IEEE 27th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
  year={2019},
  organization={IEEE}
}
```
## License
The source for this project is licensed under the [3-Clause BSD License](LICENSE)
