/**
 * SLFNELM.hpp
 *
 * Created on: 3/29/2013
 *     Author: ZHU Qiuxi
 */

#ifndef SLFN_ELM_HPP
#define SLFN_ELM_HPP

#include "opencv2/opencv.hpp"
#include "SLFNTrainer.hpp"

namespace model{

class SLFNELM:public SLFNTrainer{
private:
	SLFNELM();
	~SLFNELM();

public:
	static const int MODEL_MAT_TYPE=CV_64FC1;

public:
	static SLFNELM* getInstance();

	void run(SLFN* net,const cv::Mat& tIn,const cv::Mat& tOut);
	static void init(SLFN* net,int netN,int netNH);

private:
	static cv::Mat getH(SLFN* net,const cv::Mat& tIn);
};

}

#endif //SLFN_ELM_HPP
