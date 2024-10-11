/**
 * SLFNMat.hpp
 *
 * Created on: 4/18/2013
 *     Author: ZHU Qiuxi
 */

#ifndef SLFN_MAT_H
#define SLFN_MAT_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "SLFN.h"
#include "ActivationFunction.h"

namespace model{

class SLFNMat:public SLFN{
private:
	cv::Mat wIn;
	cv::Mat b;
	cv::Mat wOut;
	ActivationFunction* g;

public:
	static const int MODEL_MAT_TYPE=CV_64FC1;

public:
	SLFNMat();
	~SLFNMat();
	
/*These member functions will throw -1 for NullPointerException (in Java),
                                and -2 for IllegaL_ARGumentException*/
//                  they will throw -3 for unexpected internal miscv::Matches

	cv::Mat& getWIn();
	cv::Mat& getB();
	cv::Mat& getWOut();
	ActivationFunction* getG();
	int getN();
	int getNH();
	int getM();
	
	void setWIn(const cv::Mat& arg);
	void setB(const cv::Mat& arg);
	void setWOut(const cv::Mat& arg);
	void setG(ActivationFunction* arg);
	
	virtual cv::Mat f(const cv::Mat& arg);
};

}

#endif //SLFN_MAT_H
