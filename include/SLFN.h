/**
 * SLFN.h
 *
 * Created on: 3/28/2013
 *     Author: ZHU Qiuxi
 */

#ifndef SLFN_H
#define SLFN_H

#include "opencv2/opencv.hpp"
#include "ActivationFunction.h"

namespace model{

class SLFN{
public:
	virtual cv::Mat f(const cv::Mat& arg)=0;
	
	virtual cv::Mat& getWIn()=0;
	virtual cv::Mat& getB()=0;
	virtual cv::Mat& getWOut()=0;
	virtual ActivationFunction* getG()=0;
	virtual int getN()=0;
	virtual int getNH()=0;
	virtual int getM()=0;
	
	virtual void setWIn(const cv::Mat& arg)=0;
	virtual void setB(const cv::Mat& arg)=0;
	virtual void setWOut(const cv::Mat& arg)=0;
	virtual void setG(ActivationFunction* arg)=0;

	virtual ~SLFN();
};

}

#endif //SLFN_H
