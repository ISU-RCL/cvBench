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
		cv::Mat srcIn, dstSW;
		initializeSingleGrayImageTest(filenameIn, srcIn);

		int width = srcIn.size().width;
		int height = srcIn.size().height;
		
		//canny edge detector parameters  
		double lowThreshold = 40;
		double maxThreshold = 250;
		int apperture_size  = 3;
		bool L2gradient  = false;

		std::cout << "running on CPU" << std::endl;
		//warming up
		cv::Canny(srcIn, dstSW, lowThreshold, maxThreshold, apperture_size, L2gradient); 			
		timer.StartTimer();
		for (int i = 0; i < numberOfIterations; i++) 
		{
			cv::Canny(srcIn, dstSW, lowThreshold, maxThreshold, apperture_size, L2gradient);
		}
		timer.StopTimer(); 		
		std::cout << "Elapsed time over " << numberOfIterations << "SW call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;
 
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
	
			cv::cuda::GpuMat srcCuda, dstCuda;
			srcCuda.upload(srcIn); 

			std::cout << "running CUDA" << std::endl;	 
 			cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(lowThreshold, maxThreshold, apperture_size, L2gradient);

			//warming up 
			canny->detect(srcCuda, dstCuda); 				
			timer.StartTimer();
			for (int i = 0; i < numberOfIterations; i++) 
			{
				canny->detect(srcCuda, dstCuda);			
			}
			timer.StopTimer(); 			
			dstCuda.download(dstCudaOnHost);
			std::cout << "Elapsed time over " << numberOfIterations << "CUDA call(s): " << timer.GetElapsedUs() << " us or " << (float)timer.GetElapsedUs() / (float)numberOfIterations << "us per frame" << std::endl;		

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

  		ivx::Threshold threshCanny = ivx::Threshold::createRange(context, VX_TYPE_UINT8, (vx_uint32)lowThreshold, (vx_uint32) maxThreshold);
		//create graph
 		ivx::Graph graph = ivx::Graph::create(context); 
		 
		ivx::Node::create(graph, VX_KERNEL_CANNY_EDGE_DETECTOR, ivxImage, threshCanny, ivx::Scalar::create<VX_TYPE_INT32>(context, 3),
                 ivx::Scalar::create<VX_TYPE_ENUM>(context, VX_NORM_L2), ivxResult);
		 
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
			cv::namedWindow("Input image", cv::WINDOW_NORMAL);
			cv::imshow("Input image", srcIn);
			cv::namedWindow("Processed (SW)", cv::WINDOW_NORMAL);
			cv::imshow("Processed (SW)", dstSW); 
			if (ranCuda)
			{				
				cv::namedWindow("Processed (CUDA)", cv::WINDOW_NORMAL);
				cv::imshow("Processed (CUDA)", dstCudaOnHost);
			}
			if (ranOpenvx)			
			{
				cv::namedWindow("Processed (Openvx)", cv::WINDOW_NORMAL);
				cv::imshow("Processed (Openvx)", dstOpenvx);
			}
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
 
