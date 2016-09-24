#include "Simulation.h"

Simulation::Simulator::Simulator(std::string videoName) {
	size_t idx = videoName.find_last_of(".");
	name = videoName.substr(0, idx);
	scale = 1.05;
	ROI_RESIZE_DIM = cv::Size(64, 128);
	WINDOW_RESIZE_DIM = cv::Size(768, 432);
	win_size = cv::Size(64, 128);
	block_size = cv::Size(16, 16);
	block_stride = cv::Size(8, 8);
	cell_size = cv::Size(8, 8);
	win_stride = cv::Size(8, 8);
	padding = cv::Size(32, 32);
	gpu_hog = cv::cuda::HOG::create(win_size);
	detector = gpu_hog->getDefaultPeopleDetector();
	fgbg = cv::createBackgroundSubtractorMOG2(200, 16, false);
	cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
	cpu_hog = cv::HOGDescriptor(win_size, block_size, block_stride, cell_size, 9);
	cpu_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size);
	cv::Mat detector = gpu_hog->getDefaultPeopleDetector();
	//Tweak var threshold
	mog2 = cv::cuda::createBackgroundSubtractorMOG2(0, 50, true);
	element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2 * 4 + 1, 2 * 4 + 1), cv::Point(4, 4));
	homography = Tools::getHomographyFromFile(name).clone();
	cameraPosition = Tools::getCameraPositionFromFile(name).clone();
	people = Tools::People();
	cap = cv::VideoCapture(videoName);
	cap.read(frame);
	if (!cap.isOpened()) {
		std::cout << "The video file did not open successfully!\n";
		std::cout << "The program will now exit!\n";
		exit(0);
	}
	fgbg->apply(frame, bgMaskMOG2);
	frameNumber++;
	videoName = videoName;
}

int Simulation::Simulator::retrieveCPU(std::ofstream& outputFile) {

	//Start timer for FPS counter
	auto before = std::chrono::high_resolution_clock::now();

	//Check to see if the video feed is opened
	if (!cap.read(frame)){
		std::cout << "End of Video File" << std::endl;
		cap.release();
		cv::destroyAllWindows();
		return 1;
	}

	//Main phase of code
	else{
		cv::Mat imgDisplay, labels, stats, centroids,hsv;
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		imgDisplay = frame.clone();
		fgbg->apply(frame, bgMaskMOG2);
		cv::erode(bgMaskMOG2, bgMaskMOG2, element);
		cv::dilate(bgMaskMOG2, bgMaskMOG2, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
		int nLabels = connectedComponentsWithStats(bgMaskMOG2, labels, stats, centroids, 4, CV_32S);
		std::vector<ADT::component> roi;
		for (int i = 0; i < nLabels; i++){
			if (stats.at<int>(i, cv::CC_STAT_AREA) > 200){
				ADT::component tmp;
				tmp.a = stats.at<int>(i, cv::CC_STAT_AREA);
				tmp.x = stats.at<int>(i, cv::CC_STAT_LEFT) - 75;
				if (tmp.x < 0){ tmp.x = 0; }
				tmp.y = stats.at<int>(i, cv::CC_STAT_TOP) - 100;
				if (tmp.y < 0){ tmp.y = 0; }
				tmp.w = stats.at<int>(i, cv::CC_STAT_WIDTH) + 150;
				if (tmp.w + tmp.x > frame.cols){ tmp.w = frame.cols - tmp.x - 1; }
				tmp.h = stats.at<int>(i, cv::CC_STAT_HEIGHT) + 200;
				if (tmp.h + tmp.y > frame.rows){ tmp.h = frame.rows - tmp.y - 1; }
				roi.push_back(tmp);
			}
		}

		//Display all connected component regions in frame
		for (int i = 0; i < roi.size(); i++) {
			cv::rectangle(imgDisplay, cv::Point(roi[i].x, roi[i].y), cv::Point(roi[i].x + roi[i].w, roi[i].y + roi[i].h), cv::Scalar(255, 0, 0), 3);
		}

		roi.erase(roi.begin());
		std::vector<cv::Rect> newlist;
		for (int i = 0; i < roi.size(); i++){
			if (i == 0){
				cv::Rect r(roi[0].x, roi[0].y, roi[0].w, roi[0].h);
				newlist.push_back(r);
			}
			else{
				int tlx = roi[i].x;
				int tly = roi[i].y;
				int brx = roi[i].x + roi[i].w;
				int bry = roi[i].y + roi[i].h;
				int j = 0;
				while (j < newlist.size()){
					if (Tools::pointInBox(roi[i].x, roi[i].y, newlist[j]) || Tools::pointInBox(roi[i].x + roi[i].w, roi[i].y, newlist[j]) || Tools::pointInBox(roi[i].x + roi[i].w, roi[i].y + roi[i].h, newlist[j]) || Tools::pointInBox(roi[i].x, roi[i].y + roi[i].h, newlist[j])){
						tlx = std::min(roi[i].x, newlist[j].x);
						tly = std::min(roi[i].y, newlist[j].y);
						brx = std::max(roi[i].x + roi[i].w, newlist[j].x + newlist[j].width);
						bry = std::max(roi[i].y + roi[i].h, newlist[j].y + newlist[j].height);
						newlist.erase(newlist.begin() + j);
					}
					else {
						j++;
					}
				}
				cv::Rect new_r(tlx, tly, brx - tlx, bry - tly);
				newlist.push_back(new_r);
			}
		}

		//Display non-maxima suppression rectangles
		for (int i = 0; i < newlist.size(); i++) {
			cv::Rect r = newlist[i];
			cv::rectangle(imgDisplay, r.tl(), r.br(), cv::Scalar(0, 0, 255), 3);
		}
		std::vector<cv::Rect> found;

		//Case for no regions of interests
		if (newlist.empty()){
			//cpu_hog.detectMultiScale(frame, found, 0.0, win_stride, padding, scale);
		}

		//Case for detected region of interests
		else{
			for (int i = 0; i < newlist.size(); i++){
				std::vector<cv::Rect> tmpFound;
				cpu_hog.detectMultiScale(frame(cv::Rect(newlist[i].x, newlist[i].y, newlist[i].width, newlist[i].height)), tmpFound, 0.0, win_stride, padding, scale);
				if (tmpFound.size() != 0){
					for (int j = 0; j < tmpFound.size(); j++){
						tmpFound[j].x = tmpFound[j].x + newlist[i].x;
						tmpFound[j].y = tmpFound[j].y + newlist[i].y;
					}
					found.insert(found.end(), tmpFound.begin(), tmpFound.end());
				}
			}

		}

		//Displays hog rectangles in frame
		for (int i = 0; i < found.size(); i++) {
			cv::Rect r = found[i];
			cv::rectangle(imgDisplay, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}

		//ADD PEOPLE CODE HERE - NOTE LOOP IS UNNECCESSARY BUT COPIED FOR DEBUGGING PURPOSES
		/**/for (size_t i = 0; i < found.size(); i++){
			cv::Rect r = found[i];
			people.update(r, img, fgmask, hsv, frameNumber, homography, cameraPosition);
		}
		people.update(img, fgmask, hsv, frameNumber, homography, cameraPosition);
		people.refresh();
		/**/
		frameNumber++;

		//Calculate FPS and display in video
		auto after = std::chrono::high_resolution_clock::now();
		auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
		float fps = 1.0 / (float(timeElapsed)*0.001);
		outputFile << std::to_string(frameNumber) << "," << std::to_string(fps) << std::endl;
		int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
		double fontScale = 1.0;
		int thickness = 2;
		std::string frameText = "Frame #: " + std::to_string(frameNumber);
		std::string fpsText = "FPS: " + std::to_string(fps);
		cv::putText(imgDisplay, frameText, cv::Point(imgDisplay.cols / 50, imgDisplay.rows / 20), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
		cv::putText(imgDisplay, fpsText, cv::Point(imgDisplay.cols / 50, imgDisplay.rows / 20 + 40), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);

		//Show the video
		imshow("HOG", imgDisplay);

		//Check to end video by user input
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Ending video..." << std::endl;
			cap.release();
			cv::destroyAllWindows();
			return 1;
		}
		return 0;
	}
}

int Simulation::Simulator::retrieveGPU(std::ofstream& outputFile) {

	//Start timer for FPS counter
	auto before = std::chrono::high_resolution_clock::now();

	//Check to see if video feed is opened
	if (!cap.read(frame)){
		std::cout << "End of Video File" << std::endl;
		cap.release();
		cv::destroyAllWindows();
		return 1;
	}

	//Main phase of code
	else{
		cv::Mat labels, stats, centroids, hsv;
		gpu_hog->setSVMDetector(detector);
		cv::cvtColor(frame, img, cv::COLOR_BGR2BGRA);
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		img_to_show = img;
		cv::cuda::GpuMat d_frame(frame);
		d_frame.upload(frame);
		mog2->apply(d_frame, d_fgmask);
		d_fgmask.download(fgmask);
		int nLabels = connectedComponentsWithStats(fgmask, labels, stats, centroids, 4, CV_32S);
		std::vector<ADT::component> roi;
		for (int i = 0; i < nLabels; i++){
			if (stats.at<int>(i, cv::CC_STAT_AREA) > 200 && stats.at<int>(i, cv::CC_STAT_TOP) < 1000){
				ADT::component tmp;
				tmp.a = stats.at<int>(i, cv::CC_STAT_AREA);
				tmp.x = stats.at<int>(i, cv::CC_STAT_LEFT) - 75;
				if (tmp.x < 0){ tmp.w = tmp.w - tmp.x; tmp.x = 0; }
				tmp.y = stats.at<int>(i, cv::CC_STAT_TOP) - 100;
				if (tmp.y < 0){ tmp.h = tmp.h - tmp.y; tmp.y = 0; }
				tmp.w = stats.at<int>(i, cv::CC_STAT_WIDTH) + 150;
				if (tmp.w + tmp.x > frame.cols){ tmp.w = frame.cols - tmp.x - 2; }
				tmp.h = stats.at<int>(i, cv::CC_STAT_HEIGHT) + 200;
				if (tmp.h + tmp.y > frame.rows){ tmp.h = frame.rows - tmp.y - 2; }
				roi.push_back(tmp);
			}
		}

		//Display connected component regions in frame
		for (int i = 0; i < roi.size(); i++) {
			cv::rectangle(img_to_show, cv::Point(roi[i].x, roi[i].y), cv::Point(roi[i].x + roi[i].w, roi[i].y + roi[i].h), cv::Scalar(255, 0, 0), 3);
		}

		roi.erase(roi.begin());
		std::vector<cv::Rect> newlist;
		for (int i = 0; i < roi.size(); i++){
			if (i == 0){
				cv::Rect r(roi[0].x, roi[0].y, roi[0].w, roi[0].h);
				newlist.push_back(r);
			}
			else{
				int tlx = roi[i].x;
				int tly = roi[i].y;
				int brx = roi[i].x + roi[i].w;
				int bry = roi[i].y + roi[i].h;
				int j = 0;
				while (j < newlist.size()){
					if (Tools::pointInBox(roi[i].x, roi[i].y, newlist[j]) || Tools::pointInBox(roi[i].x + roi[i].w, roi[i].y, newlist[j]) || Tools::pointInBox(roi[i].x + roi[i].w, roi[i].y + roi[i].h, newlist[j]) || Tools::pointInBox(roi[i].x, roi[i].y + roi[i].h, newlist[j])){
						tlx = std::min(roi[i].x, newlist[j].x);
						tly = std::min(roi[i].y, newlist[j].y);
						brx = std::max(roi[i].x + roi[i].w, newlist[j].x + newlist[j].width);
						bry = std::max(roi[i].y + roi[i].h, newlist[j].y + newlist[j].height);
						newlist.erase(newlist.begin() + j);
					}
					else {
						j++;
					}
				}
				cv::Rect new_r(tlx, tly, brx - tlx, bry - tly);
				newlist.push_back(new_r);
			}
		}

		//Display non-maxima suppression rectangles
		for (int i = 0; i < newlist.size(); i++) {
			cv::Rect r = newlist[i];
			cv::rectangle(img_to_show, r.tl(), r.br(), cv::Scalar(0, 0, 255), 3);
		}

		gpu_hog->setNumLevels(10);
		gpu_hog->setHitThreshold(0.0);
		gpu_hog->setWinStride(win_stride);
		gpu_hog->setScaleFactor(1.07);
		gpu_hog->setGroupThreshold(1);

		std::vector<cv::Rect> found;

		//Case for no regions of interests
		if (newlist.empty()){
			gpu_img.upload(img);
			//gpu_hog->detectMultiScale(gpu_img, found);
		}

		//Case for detected region of interests
		else{
			for (int i = 0; i < newlist.size(); i++){
				std::vector<cv::Rect> tmpFound;
				gpu_img.upload(img(cv::Rect(newlist[i].x, newlist[i].y, newlist[i].width, newlist[i].height)));
				gpu_hog->detectMultiScale(gpu_img, tmpFound);
				if (tmpFound.size() != 0){
					for (int j = 0; j < tmpFound.size(); j++){
						tmpFound[j].x = tmpFound[j].x + newlist[i].x;
						tmpFound[j].y = tmpFound[j].y + newlist[i].y;
					}
					found.insert(found.end(), tmpFound.begin(), tmpFound.end());
				}
			}

		}

		//Display hog detections in frame
		for (size_t i = 0; i < found.size(); i++){
			cv::Rect r = found[i];
			cv::rectangle(img_to_show, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}

		//ADD PEOPLE CODE HERE - NOTE LOOP IS UNNECCESSARY BUT COPIED FOR DEBUGGING PURPOSES
		/*for (size_t i = 0; i < found.size(); i++){
			cv::Rect r = found[i];
			people.update(r,img,fgmask,hsv,frameNumber,homography,cameraPosition);
		}
		people.update(img, fgmask, hsv, frameNumber, homography, cameraPosition);
		people.refresh();
		
		for (int i = 0; i < people.listOfPeople.size();i++){
			cv::rectangle(img_to_show, cv::Point(people.listOfPeople.at(i).x, people.listOfPeople.at(i).y), cv::Point(people.listOfPeople.at(i).x + people.listOfPeople.at(i).w, people.listOfPeople.at(i).y + people.listOfPeople.at(i).h), cv::Scalar(255, 255, 255), 3);
			cv::putText(img_to_show, std::to_string(people.listOfPeople.at(i).id), cv::Point(people.listOfPeople.at(i).x, people.listOfPeople.at(i).y), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255,255,255), 2, 8);
		}
		*/
		frameNumber++;

		//Calculates FPS and display information on video
		auto after = std::chrono::high_resolution_clock::now();
		auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count();
		float fps = 1.0 / (float(timeElapsed)*0.001);
		outputFile << std::to_string(frameNumber) << "," << std::to_string(fps) << std::endl;
		int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
		double fontScale = 1.0;
		int thickness = 2;
		std::string frameText = "Frame #: " + std::to_string(frameNumber);
		std::string fpsText = "FPS: " + std::to_string(fps);
		cv::putText(img_to_show, frameText, cv::Point(img.cols / 50, img.rows / 20), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
		cv::putText(img_to_show, fpsText, cv::Point(img.cols / 50, img.rows / 20 + 40), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);

		//Display the video 
		imshow("HOG_GPU", img_to_show);

		//End video on user input
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Ending video..." << std::endl;
			cap.release();
			cv::destroyAllWindows();
			return 1;
		}
		return 0;
	}
}