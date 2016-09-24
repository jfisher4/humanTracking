#pragma once
#include "ADT.h"
namespace Tools{
	
	class Person{
	public:
		Person(int i, int fx, int fy, int fw, int fh, int v);
		int id;
		int x;
		int y;
		int w;
		int h;
		int visible;
		std::vector<ADT::cartesianPointT> location;
		float velocity;
		float height;
	};
	class People{
	public:
		std::vector<Person> listOfPeople;
		std::vector<Person> lostListOfPeople;
		int index;
		People();
		void update(cv::Rect r, cv::Mat img,cv::Mat fgmask,cv::Mat hsv, int frameNumber, cv::Mat homography, cv::Mat cameraPosition);
		void update(cv::Mat img,cv::Mat fgmask,cv::Mat hsv, int frameNumber, cv::Mat homography, cv::Mat cameraPosition);
		void refresh();
	};
	class KalmanFilter{
	private:
		double kalmanGain;
		double covariance;
		double measurementNoiseModel;
		double prediction;
		double covarianceGain;
		double lastSensorValue;
	public:
		KalmanFilter();
		KalmanFilter(double p);
		void updatePrediction(double sensorValue);
		void updateCovariance();
		void updateKalmanGain();
		void step(double sensorValue);
		double getPrediction();
	};
	void displayConnectedComponents(cv::Mat bgMaskMOG2, cv::Mat labels, int nLabels);
	bool pointInBox(cv::Point p, cv::Rect r);
	bool pointInBox(int x, int y, cv::Rect r);
	bool pointInBox(cv::Point p, ADT::component c);
	bool pointInBox(int x, int y, ADT::component c);
	cv::Mat getCameraPositionFromFile(std::string file);
	cv::Mat getHomographyFromFile(std::string file);
	
	ADT::cartesianPoint worldToPixel(ADT::worldPoint p, cv::Mat h);
	ADT::worldPoint pixelToWorld(ADT::cartesianPoint p, cv::Mat h);
	void generateSystemInfo(bool createLog,bool usingGPU);
	bool isOverlap(cv::Rect r1, cv::Rect r2,float threshold);
	bool isOverlap(cv::Rect r1, cv::Rect r2, float threshold, float &overlapPercentage);
	float getObjectHeight(ADT::worldPoint bottomPoint, ADT::worldPoint topPoint, cv::Mat cameraPoisition);
	float getObjectHeight(ADT::cartesianPoint bottomPoint, ADT::cartesianPoint topPoint, cv::Mat cameraPoisition,cv::Mat homography);
}
