#include "Controller.h"
#include <unistd.h>
void Controller::testSingleView(std::string directory, std::string videoname, bool useGPU) {
	//Change to working directory
	if (chdir(directory.c_str()) == -1){
		std::cout << "Error changing to directory. Ending program..." << std::endl;
	}
	else{
		Simulation::Simulator simulationA(videoname);
		size_t idx2 = videoname.find_last_of(".");
		std::string v = videoname.substr(0, idx2);
		int status = 0;
		std::ofstream oFile;
		if (useGPU){
			std::ofstream oFile("data/" + v + "_GPU_FPS.csv");
			while (true) {
				status = simulationA.retrieveGPU(oFile);
				if (status == 1){ break; }
			}
		}
		else{
			std::ofstream oFile("data/" + v + "_CPU_FPS.csv");
			while (true) {
				status = simulationA.retrieveCPU(oFile);
				if (status == 1){ break; }
			}
		}
		oFile.close();
	}
}

//TODO: UPDATE TO BE SIMILAR TO SINGLEVIEW
void Controller::testMultiView(std::string directory, std::string videoname1, std::string videoname2,bool useGPU){
	if (chdir(directory.c_str()) == -1){
		std::cout << "Error changing to directory. Ending program..." << std::endl;
	}
	else{
		Simulation::Simulator simulationA(videoname1);
		size_t idx1 = videoname1.find_last_of(".");
		std::string v1 = videoname1.substr(0, idx1);
		Simulation::Simulator simulationB(videoname2);
		size_t idx2 = videoname2.find_last_of(".");
		std::string v2 = videoname2.substr(0, idx2);
		int statusA = 0;
		int statusB = 0;
		while (true) {
			//statusA = simulationA.retrieveCPU(nullptr);
			//statusB = simulationB.retrieveCPU(nullptr);
			if (statusA == 1 || statusB == 1){ break; }
		}
	}
}

int main (){
	
	std::string directory1 = "/home/fish/humanTracking/c/videos";

	std::string videoname1 = "01072016A5_B1.mp4";
	std::string videoname2 = "01072016A5_B2.mp4";
	std::string videoname3 = "01072016A5_C1.mp4";
	std::string videoname4 = "01072016A5_C2.mp4";
	std::string videoname5 = "01072016A5_D1.mp4";
	std::string videoname6 = "01072016A5_D2.mp4";
	std::string videoname7 = "01072016A5_E1.mp4";
	std::string videoname8 = "01072016A5_E2.mp4";
	std::string videoname9 = "01072016A5_F1.mp4";
	std::string videoname10 = "01072016A5_F2.mp4";
	std::string videoname11 = "01072016A5_F3.mp4";
	std::string videoname12 = "01072016A5_F4.mp4";
	std::string videoname13 = "01072016A5_F5.mp4";
	std::string videoname14 = "01072016A5_F6.mp4";
	std::string videoname15 = "01072016A5_J1.mp4";
	std::string videoname16 = "01072016A5_J2.mp4";
	std::string videoname17= "01072016A5_M1.mp4";
	Controller::testSingleView(directory1, videoname1, true);
	return 0;
}
