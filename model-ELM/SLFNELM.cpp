/**
 * SLFNELM.cpp
 *
 * Created on: 3/29/2013
 *     Author: ZHU Qiuxi
 */

#include <iostream>
#include <ctime>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "SLFNELM.hpp"
#include "ActivationFunction.h"

using namespace cv;
using namespace model;

SLFNELM::SLFNELM(){;}
SLFNELM::~SLFNELM(){;}

SLFNELM* SLFNELM::getInstance(){
	static SLFNELM* instance=new SLFNELM();
	
	return instance;
}

void SLFNELM::run(SLFN* net,const Mat& tIn,const Mat& tOut){
	if(net==NULL) throw -1;
	if(tIn.empty() || tOut.empty()) throw -1;
	if(tIn.dims!=2 || tIn.rows<1 || tIn.cols<1 || tIn.type()!=MODEL_MAT_TYPE)
		throw -2;
	if(tOut.dims!=2 || tOut.rows<1 || tOut.cols<1 || tOut.type()!=MODEL_MAT_TYPE)
			throw -2;
	if(tIn.rows!=tOut.rows) throw -2;
	
	int netN=net->getN();
//	int netNH=net->getNH(); //unused
//	int netM=net->getM(); //M is unknown

	try{
		net->getWOut();
		throw -2; //wOut is already set
	} catch(int e){
		if(e!=-1) throw -2;//wOut is absent, which is expected
	}
	if(tIn.cols!=netN) throw -2;

	Mat tH=getH(net,tIn);
	Mat tInvH=tH.inv(DECOMP_SVD);
	Mat netWOut=tInvH*tOut;
	
	net->setWOut(netWOut);
}

static double rand_double(double value1,double value2);
static double rand_double();

void SLFNELM::init(SLFN* net,int netN,int netNH){
	if(net==NULL) throw -1;
	try{
		net->getWIn();
		throw -2;
	} catch(int e){
		if(e!=-1) throw -2;//defined as NullPointerException in Java
	}
	try{
		net->getB();
		throw -2;
	} catch(int e){
		if(e!=-1) throw -2;//defined as NullPointerException in Java
	}
	if(netN<1) throw -2;
	if(netNH<1) throw -2;

	Mat netWIn(netNH,netN,MODEL_MAT_TYPE);
	Mat netB(netNH,1,MODEL_MAT_TYPE);

	srand((int)time(NULL));
	for(int i=0;i<netNH*netN;++i)
		netWIn.at<double>(i)=rand_double();
	for(int i=0;i<netNH;++i)
		netB.at<double>(i)=rand_double();
	net->setWIn(netWIn);
	net->setB(netB);
}

Mat SLFNELM::getH(SLFN* net,const Mat& tIn){
	Mat netWIn=net->getWIn();
	Mat netB=net->getB();
	int netN=net->getN();
	int netNH=net->getNH();
	
	if(netN<1 || netNH<1) throw -2;
	
	ActivationFunction* netG=net->getG();
	int tN=tIn.rows;
	Mat tH(tN,netNH,MODEL_MAT_TYPE);
	
	for(int i=0;i<tN;++i)
		for(int j=0;j<netNH;++j)
			tH.at<double>(i,j)=netG->f(netWIn.row(j).dot(tIn.row(i))+netB.at<double>(j));
	return tH;
}

static double rand_double(double value1,double value2){
	return value1+(value2-value1)*rand()/(RAND_MAX+1.);
}
static double rand_double(){
	return rand_double(0,1);
}
