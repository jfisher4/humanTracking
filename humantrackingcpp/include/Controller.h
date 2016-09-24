#pragma once
#include "Simulation.h"
namespace Controller{
	class PeopleController{
	private:
	public:
		PeopleController();
		void update();
};
	void testSingleView(std::string directory,std::string videoname,bool useGPU);
	void testMultiView(std::string directory,std::string videoname1, std::string videoname2, bool useGPU);
}