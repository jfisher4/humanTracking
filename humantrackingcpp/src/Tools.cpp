#include "Tools.h"

Tools::People::People(){
	index = 0;
}
void Tools::People::refresh(){
	for (int i = 0; i < listOfPeople.size(); i++){
		listOfPeople.at(i).visible++;
	}
}
void Tools::People::update(cv::Rect r, cv::Mat img, cv::Mat fgmask, cv::Mat hsv,int frameNumber, cv::Mat homography, cv::Mat cameraPosition){
	//Case where no people in list
	int ch[] = { 0, 0 };
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	cv::Mat hue,hist;
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	cv::Mat roi(hue, r), maskroi(fgmask, r);
	calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);

	Tools::Person tmpPerson = Tools::Person(index,r.x,r.y,r.width,r.height,0);
	if (listOfPeople.size()==0){
		listOfPeople.push_back(tmpPerson);
		index++;
	}
	//Case where people in list
	else{
		float overlapPercentage = 0.0;
		bool match = false;
		int replacementIndex;
		for (int i = 0;i<listOfPeople.size(); i++){
			float tmpOverlapPercentage = 0.0;
			if (Tools::isOverlap(r, cv::Rect(listOfPeople.at(i).x, listOfPeople.at(i).y, listOfPeople.at(i).w, listOfPeople.at(i).h), 50,tmpOverlapPercentage)){
				if (tmpOverlapPercentage > overlapPercentage){
					match = true;
					overlapPercentage = tmpOverlapPercentage;
					replacementIndex = i;
				}
			}
			
		}
		if (match){
			listOfPeople.at(replacementIndex).x = r.x;
			listOfPeople.at(replacementIndex).y = r.y;
			listOfPeople.at(replacementIndex).w = r.width;
			listOfPeople.at(replacementIndex).h = r.height;
			listOfPeople.at(replacementIndex).visible = 0;
		}
		else{
			listOfPeople.push_back(tmpPerson);
			index++;
		}
	}
}

void Tools::People::update(cv::Mat img,cv::Mat fgmask ,cv::Mat hsv, int frameNumber, cv::Mat homography, cv::Mat cameraPosition){
	int ch[] = { 0, 0 };
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float* phranges = hranges;
	cv::Mat hue, hist;
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	
	for (int i = 0; i < listOfPeople.size(); i++){
		if (listOfPeople.at(i).visible>0 && listOfPeople.at(i).visible<30){
			cv::Mat roi(hue, cv::Rect(listOfPeople.at(i).x, listOfPeople.at(i).y, listOfPeople.at(i).w, listOfPeople.at(i).h)), maskroi(fgmask, cv::Rect(listOfPeople.at(i).x, listOfPeople.at(i).y, listOfPeople.at(i).w, listOfPeople.at(i).h));
			calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
		}
		if (listOfPeople.at(i).visible >= 30){
			listOfPeople.erase(listOfPeople.begin()+ i);
			i--;
		}
	}
}

Tools::Person::Person(int i, int fx, int fy, int fw, int fh, int v){
	id = i;
	x = fx;
	y = fy;
	w = fw;
	h = fh;
	visible = v;
	velocity = 0;
	height = 0;

}

void Tools::displayConnectedComponents(cv::Mat bgMaskMOG2, cv::Mat labels, int nLabels){
	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);//background
	for (int label = 1; label < nLabels; ++label){
		colors[label] = cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}
	cv::Mat dst(bgMaskMOG2.size(), CV_8UC3);
	for (int r = 0; r < dst.rows; ++r){
		for (int c = 0; c < dst.cols; ++c){
			int label = labels.at<int>(r, c);
			cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}
	cv::imshow("CC", dst);
}

bool Tools::pointInBox(cv::Point p, ADT::component c){
	if ((p.x >= c.x) && (p.x <= c.x+c.w) && (p.y >= c.y) && (p.y <= c.y+c.h)){
		return true;
	}
	else{
		return false;
	}
}
bool Tools::pointInBox(cv::Point p, cv::Rect r){
	if ((p.x >= r.x) && (p.x <= r.x + r.width) && (p.y >= r.y) && (p.y <= r.y + r.height)){
		return true;
	}
	else{
		return false;
	}
}
bool Tools::pointInBox(int x, int y, cv::Rect r){
	if ((x >= r.x) && (x <= r.x + r.width) && (y >= r.y) && (y <= r.y + r.height)){
		return true;
	}
	else{
		return false;
	}
}
bool Tools::pointInBox(int x, int y, ADT::component c){
	if ((x >= c.x) && (x <= c.x+c.w) && (y >= c.y) && (y <= c.y+c.h)){
		return true;
	}
	else{
		return false;
	}
}
cv::Mat Tools::getCameraPositionFromFile(std::string file){
	std::string dir = file.substr(0,10) + "_CP.txt";
	std::ifstream inputFile(dir);//"sajdksadhksja.txt"
	if (!inputFile.is_open()) {
		std::cout << "input file not open!";
	}
	std::string line;
	std::getline(inputFile, line);
	int rows = std::stoi(line);
	std::getline(inputFile, line);
	int cols = stoi(line);
	int count = rows * cols;
	std::vector<double> camPos;
	cv::Mat cameraPosition(rows, cols, CV_64F);
	long double tmp;
	for (int i = 0; i < count; i++) {
		inputFile >> tmp;
		camPos.push_back(tmp);
	}
	for (int i = 0; i < count; i++) {
		cameraPosition.at<double>(i) = camPos[i];
	}
	inputFile.close();
	return cameraPosition;
}
cv::Mat Tools::getHomographyFromFile(std::string file){
	std::string dir = file.substr(0,10) + "_H.txt";
	std::ifstream inputFile(dir);
	if (!inputFile.is_open()) {
		std::cout << "input file not open!";
	}
	std::string line;
	std::getline(inputFile, line);
	int rows = std::stoi(line);
	std::getline(inputFile, line);
	int cols = stoi(line);
	int count = rows * cols;
	std::vector<double> hom;
	cv::Mat homography(rows, cols, CV_64F);
	long double tmp;
	for (int i = 0; i < count; i++) {
		inputFile >> tmp;
		hom.push_back(tmp);
	}
	for (int i = 0; i < count; i++) {
		homography.at<double>(i) = hom[i];
	}
	inputFile.close();
	return homography;
}

ADT::cartesianPoint Tools::worldToPixel(ADT::worldPoint p,cv::Mat homography){
	ADT::cartesianPoint q;
	cv::Mat worldVector(3, 1, CV_64F);
	worldVector.at<double>(0) = p.latitude;
	worldVector.at<double>(1) = p.longitude;
	worldVector.at<double>(2) = 1;
	cv::Mat pixelVector = homography.inv() * worldVector;
	q.xCoordinate = int(pixelVector.at<double>(0) / pixelVector.at<double>(2));
	q.yCoordinate = int(pixelVector.at<double>(1) / pixelVector.at<double>(2));

	return q;
}

ADT::worldPoint Tools::pixelToWorld(ADT::cartesianPoint p, cv::Mat homography){
	ADT::worldPoint q;
	cv::Mat pixelVector(3,1,CV_64F);
	pixelVector.at<double>(0) = double(p.xCoordinate);
	pixelVector.at<double>(1) = double(p.yCoordinate);
	pixelVector.at<double>(2) = 1;
	cv::Mat worldVector = homography * pixelVector;
	q.latitude = worldVector.at<double>(0) / worldVector.at<double>(2);
	q.longitude = worldVector.at<double>(1) / worldVector.at<double>(2);

	return q;
}
void Tools::generateSystemInfo(bool createLog, bool usingGPU){
	if (createLog){
		return;
	}
	else{
		return;
	}
}
bool Tools::isOverlap(cv::Rect r1, cv::Rect r2,float threshold){
	float percentage = 0.0;
	if (threshold>=0.0 && threshold<=100.0){
		percentage = threshold/100.0;
	}
	int x_overlap = std::max(0, std::min(r1.x+r1.width  , r2.x+r2.width  ) - std::max(r1.x , r2.x ));
	int y_overlap = std::max(0, std::min(r1.y+r1.height , r2.y+r2.height ) - std::max(r1.y , r2.y ));
	int overlapArea = x_overlap * y_overlap;
	if (float(r1.width*r1.height)*percentage>float(overlapArea) && float(r2.width*r2.height)*percentage>float(overlapArea)){
		return false;
	}
	else{
		return true;
	}
}
bool Tools::isOverlap(cv::Rect r1, cv::Rect r2, float threshold, float &overlapPercentage){
	float percentage = 0.0;
	if (threshold >= 0.0 && threshold <= 100.0){
		percentage = threshold / 100.0;
	}
	int x_overlap = std::max(0, std::min(r1.x + r1.width, r2.x + r2.width) - std::max(r1.x, r2.x));
	int y_overlap = std::max(0, std::min(r1.y + r1.height, r2.y + r2.height) - std::max(r1.y, r2.y));
	int overlapArea = x_overlap * y_overlap;
	if (float(r1.width*r1.height)*percentage>float(overlapArea) && float(r2.width*r2.height)*percentage>float(overlapArea)){
		overlapPercentage = float(overlapArea)/float(std::min(r1.width*r1.height,r2.width*r2.height));
		return false;
	}
	else{
		overlapPercentage = float(overlapArea) / float(std::min(r1.width*r1.height, r2.width*r2.height));
		return true;
	}
}
float Tools::getObjectHeight(ADT::cartesianPoint bottomPoint, ADT::cartesianPoint topPoint, cv::Mat cameraPosition,cv::Mat homography){
	ADT::worldPoint bPoint = Tools::pixelToWorld(bottomPoint,homography);
	ADT::worldPoint tPoint = Tools::pixelToWorld(topPoint,homography);
	
	double cameraHeight = cameraPosition.at<double>(2);

	cv::Mat bottomVector(3, 1, CV_64F);
	bottomVector.at<double>(0) = bPoint.latitude;
	bottomVector.at<double>(1) = bPoint.longitude;
	bottomVector.at<double>(2) = 0;

	cv::Mat topVector(3, 1, CV_64F);
	topVector.at<double>(0) = tPoint.latitude;
	topVector.at<double>(1) = tPoint.longitude;
	topVector.at<double>(2) = 0;

	double radAlpha = cameraHeight / cv::norm(cameraPosition - topVector);
	double tanAlpha = std::tan(radAlpha);
	double objectHeight = cv::norm(bottomVector - topVector) * tanAlpha;
	return float(objectHeight);
}

float Tools::getObjectHeight(ADT::worldPoint bottomPoint, ADT::worldPoint topPoint, cv::Mat cameraPosition){
	double cameraHeight = cameraPosition.at<double>(2);
	
	cv::Mat bottomVector(3, 1, CV_64F);
	bottomVector.at<double>(0) = bottomPoint.latitude;
	bottomVector.at<double>(1) = bottomPoint.longitude;
	bottomVector.at<double>(2) = 0;
	
	cv::Mat topVector(3, 1, CV_64F);
	topVector.at<double>(0) = topPoint.latitude;
	topVector.at<double>(1) = topPoint.longitude;
	topVector.at<double>(2) = 0;

	double radAlpha = cameraHeight / cv::norm(cameraPosition - topVector);
	double tanAlpha = std::tan(radAlpha);
	double objectHeight = cv::norm(bottomVector - topVector) * tanAlpha;
	return float(objectHeight);
}