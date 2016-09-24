#pragma once
#include "OpenCV.h"
namespace ADT{
	struct component{
		int a;
		int x;
		int y;
		int w;
		int h;
	};
	struct worldPoint{
		double latitude;
		double longitude;
	};
	struct worldPointT{
		int frameNumber;
		double latitude;
		double longitude;

	};
	struct cartesianPoint{
		int xCoordinate;
		int yCoordinate;
	};
	struct cartesianPointT{
		int frameNumber;
		int xCoordinate;
		int yCoordinate;

	};
	struct cartesianVectorT{
		int frameNumber;
		cartesianPoint previousPoint;
		cartesianPoint nextPoint;
	};
	struct worldVectorT{
		int frameNumber;
		worldPoint previousPoint;
		worldPoint nextPoint;
	};
}