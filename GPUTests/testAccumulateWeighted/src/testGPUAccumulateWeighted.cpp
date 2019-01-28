/*****************************************************************************
*
* 	Copyright (c) 2019 Iowa State University
*
* 	Redistribution and use in source and binary forms, with or without
* 	modification, are permitted provided that the following conditions 
* 	are met:
*
* 	1.Redistributions of source code must retain the above copyright 
* 	notice, this list of conditions and the following disclaimer.
* 	2.Redistributions in binary form must reproduce the above copyright 
* 	notice, this list of conditions and the following disclaimer in the
* 	documentation and/or other materials provided with the distribution.
* 	3.Neither the name of the copyright holder nor the names of its 
* 	contributors may be used to endorse or promote products derived 
* 	from this software without specific prior written permission.
*
* 	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* 	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* 	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* 	FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
*	COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* 	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
* 	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
* 	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* 	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* 	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
* 	ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
* 	POSSIBILITY OF SUCH DAMAGE.
*
*	(c) Copyright 2019 Iowa State University.
*	All rights reserved.
*	
*	Author: Murad Qasaimeh <qasaimeh@iastate.edu>
*	Date:   2018/03/27
*****************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "OpenCVUtils.h" 
#include "opencv2/core/cuda.hpp"   
#include "ivx.hpp" 
#include <HRTimer.h> 
 

int main(int argc, char** argv)
{
	HRTimer timer;

	try {	
		const cv::String keys =
			"{help h usage ? |                 | print this message                   }"
			"{@image         |                 | input image 	                      }"	 
			"{goldenFile gf  |                 | golden output image (from SW)        }"
			"{outFile of     |                 | output image (from PL)               }"
			"{display d      |                 | diplay result with imshow            }"
			"{iterations n   | 1               | number of iterations to measure time }"
			;

		cv::CommandLineParser parser(argc, argv, keys.c_str());
		if (parser.has("help") || argc < 2)
		{
			parser.printMessage();
			std::cout << "\nhit enter to quit...";
			std::cin.get();
			return 0;
		}

		cv::String filenameIn = parser.get<cv::String>(0);	 

		cv::String filenameSW; bool writeSWResult = false;
		if (parser.has("goldenFile")) {
			filenameSW = parser.get<cv::String>("goldenFile");
			writeSWResult = true;
		}

		cv::String filenamePL; bool writePLResult = false;
		if (parser.has("outFile")) {
			filenamePL = parser.get<cv::String>("outFile");
			writePLResult = true;
		}

		bool imShowOn = parser.has("display");
		unsigned int numberOfIterations;
		if (parser.has("iterations"))
			numberOfIterations = parser.get<unsigned int>("iterations");

		if (!parser.check())
		{
			parser.printErrors();
			return(-1);
		}

		// Initialize
		cv::Mat srcIn;
		initializeSingleGrayImageTest(filenameIn, srcIn); 

		int width  = srcIn.size().width;
		int height = srcIn.size().height;
		double alpha =1.0;
  		cv::Mat dstSW= cv::Mat::zeros(height, width,CV_32F); 
		 
		std::cout << "running on CPU" << std::endl;
		 
		//warming up       
		cv::accumulateWeighted(srcIn, dstSW, alpha); 
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++) 
		{
			cv::accumulateWeighted(srcIn, dstSW, alpha);
		}
		timer.StopTimer(); 
		std::cout << "Elapsed time over " << numberOfIterations << " SW call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;
 
		
		/*-------------------------- CUDA ---------------------------------------------*/
		bool ranCuda = false;
		cv::Mat dstCudaOnHost;
		
#ifdef WITH_CUDA 
		
		std::cout << "CUDA gpu device count: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
		if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
			std::cout << "no Cuda devices available" << std::endl;
		}
		else
		{	
			cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
			// src Cuda
			srcIn.convertTo(srcIn, CV_32F);
			cv::cuda::GpuMat srcCuda(height, width, CV_32F);
			srcCuda.upload(srcIn);	 

			// dst Cuda
			cv::cuda::GpuMat dstCuda(height, width, CV_32F);
			dstCuda.setTo(cv::Scalar::all(0)); 
			
			cv::cuda::GpuMat srcCuda1(height, width, CV_32F);
			cv::cuda::GpuMat srcCuda2(height, width, CV_32F);

	
			std::cout << "running CUDA" << std::endl;	 	
			//warming up
			cv::cuda::multiply(srcCuda,cv::Scalar::all(alpha),srcCuda1);
			cv::cuda::multiply(dstCuda,cv::Scalar::all((1-alpha)),dstCuda);
    		cv::cuda::add(srcCuda1, dstCuda, dstCuda); 
			timer.StartTimer();
			for (int i = 0; i < numberOfIterations; i++) 
			{    			
				cv::cuda::multiply(srcCuda,cv::Scalar::all(alpha),srcCuda1);
				cv::cuda::multiply(dstCuda,cv::Scalar::all((1-alpha)),dstCuda);
    			cv::cuda::add(srcCuda1, dstCuda, dstCuda);
			}
			timer.StopTimer(); 
			dstCuda.download(dstCudaOnHost); 
			dstCudaOnHost.convertTo(dstCudaOnHost, CV_32F);
			std::cout << "Elapsed time over " << numberOfIterations << " CUDA call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;
			ranCuda = true;
		}
#endif

		/*-------------------------- OPENVX ---------------------------------------------*/
		bool ranOpenvx = false;
		cv::Mat dstOpenvx; 

#ifdef WITH_OPENVX 

		//create context
		ivx::Context context = ivx::Context::create();

		//copy cv::Mat to vx_image   
        ivx::Image ivxImage = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        ivxImage.copyFrom(0, srcIn); 

		//dst image
 		ivx::Image ivxResult = ivx::Image::create(context, width, height, VX_DF_IMAGE_U8);
        //ivxResult.copyFrom(0, srcIn);        
 		ivxResult.copyFrom(0, cv::Mat::zeros(height,width,CV_16S)); 

		//create graph
 		ivx::Graph graph = ivx::Graph::create(context); 
		ivx::Node::create(graph,VX_KERNEL_ACCUMULATE_WEIGHTED, ivxImage,ivx::Scalar::create<VX_TYPE_FLOAT32>(context, alpha), ivxResult);  
		 
		graph.verify();  

		std::cout << "running OpenVX" << std::endl;			
		
		//warming up
 		graph.process();  
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++){
 			graph.process(); 
		}
		timer.StopTimer(); 
        //copy vx_image to cv::Mat
        ivxResult.copyTo(0, dstOpenvx); 
		std::cout << "Elapsed time over " << numberOfIterations << " OpenVX call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;

		ranOpenvx = true;
        dstOpenvx.convertTo(dstOpenvx, CV_32F);
#endif

		/*------------- end of measurements ----------------------*/
		
		// compare results
		int numberOfDifferences = 0;
		double errorPerPixel = 0; 

		if (ranCuda)
		{
			std::cout << "comparing CUDA versus CPU" << std::endl;
			imageCompare(dstCudaOnHost, dstSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		if (ranOpenvx)
		{
			std::cout << "comparing Openvx versus CPU" << std::endl;
			imageCompare(dstOpenvx, dstSW, numberOfDifferences, errorPerPixel, true, false);
			std::cout << "number of differences: " << numberOfDifferences << " average error per pixel: " << errorPerPixel << std::endl;
		}
		
		//show results
		if (imShowOn) {
			cv::imshow("Input image", srcIn);
			cv::imshow("Processed (SW)", dstSW); 
			if (ranCuda)
				cv::imshow("Processed (CUDA)", dstCudaOnHost);
			if (ranOpenvx)
				cv::imshow("Processed (Openvx)", dstOpenvx);
			cv::waitKey(0);
		}
	}
	catch (std::exception &e)
	{
		const char* errorMessage = e.what();
		std::cerr << "Exception caught: " << errorMessage << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}
	catch (const char *e) {
		std::cerr << "Exception caught: " << e << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}

	std::cout << "\nhit enter to quit...";
	std::cin.get();
	return 0;
}
 
