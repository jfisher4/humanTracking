#pragma once
#include "Tools.h"
#include "OpenCV.h"
namespace Simulation{
	class Simulator {
	public:
		std::string name;
		double scale;
		cv::Size ROI_RESIZE_DIM;
		cv::Size WINDOW_RESIZE_DIM;
		cv::Size win_size;
		cv::Size block_size;
		cv::Size block_stride;
		cv::Size cell_size;
		cv::Size win_stride;
		cv::Size padding;
		std::string videoName;
		int frameNumber;
		cv::Mat frame, bgMaskMOG2, element;
		cv::VideoCapture cap;
		cv::Ptr<cv::cuda::HOG> gpu_hog;
		cv::Mat detector;
		cv::cuda::GpuMat gpu_img;
		cv::Mat img_aux, img, img_to_show;
		cv::Ptr <cv::BackgroundSubtractor> mog2;
		cv::Ptr <cv::BackgroundSubtractorMOG2> fgbg;
		cv::cuda::GpuMat d_fgmask;
		cv::Mat fgmask;
		cv::TermCriteria termCrit;
		cv::Mat ROI;
		cv::HOGDescriptor cpu_hog;
		cv::Mat homography;
		cv::Mat cameraPosition;
		Tools::People people;
		//homography with pickle files
		//rotationMatrix with pickle files
		//cameraPosition with pickle files and Tools
		Simulator();
		Simulator(std::string videoname);
		int retrieveCPU(std::ofstream&);
		int retrieveGPU(std::ofstream&);
	};
}
