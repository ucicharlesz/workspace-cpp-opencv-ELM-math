/**
 * SLFNMat.cpp
 *
 * Created on: 4/18/2013
 *     Author: ZHU Qiuxi
 */

#include <iostream>
#include <iomanip>
#include <limits>
#include "opencv2/opencv.hpp"
#include "SLFNMat.hpp"
#include "ActivationFunction.h"

using namespace std;
using namespace cv;
using namespace model;

SLFNMat::SLFNMat()
	:wIn(),b(),wOut(),g(NULL){
	;
}

SLFNMat::~SLFNMat(){;}

/*These member functions will throw -1 for NullPointerException (in Java),
                                and -2 for IllegaL_ARGumentException*/
//                  they will throw -3 for unexpected internal mismatches

Mat& SLFNMat::getWIn(){
	if(wIn.empty()) throw -1;
	return wIn;
}

Mat& SLFNMat::getB(){
	if(b.empty()) throw -1;
	return b;
}

Mat& SLFNMat::getWOut(){
	if(wOut.empty()) throw -1;
	return wOut;
}

ActivationFunction* SLFNMat::getG(){
	if(g==NULL) throw -1;
	return g;
}

int SLFNMat::getN(){
	return getWIn().cols;
}

int SLFNMat::getNH(){
	return getB().rows;
}

int SLFNMat::getM(){
	return getWOut().cols;
}

void SLFNMat::setWIn(const Mat& arg){
	if(arg.empty()) throw -1;
	if(arg.dims!=2 || arg.rows<1 || arg.cols<1 || arg.type()!=MODEL_MAT_TYPE)
		throw -2;
	if(!b.empty()) if(arg.rows!=b.rows) throw -2;
	if(!wOut.empty()) if(arg.rows!=wOut.rows) throw -2;
	wIn=arg.clone();
}

void SLFNMat::setB(const Mat& arg){
	if(arg.empty()) throw -1;
	if(arg.dims!=2 || arg.rows<1 || arg.cols!=1 || arg.type()!=MODEL_MAT_TYPE)
		throw -2;
	if(!wIn.empty()) if(arg.rows!=wIn.rows) throw -2;
	if(!wOut.empty()) if(arg.rows!=wOut.rows) throw -2;
	b=arg.clone();
}

void SLFNMat::setWOut(const Mat& arg){
	if(arg.empty()) throw -1;
	if(arg.dims!=2 || arg.rows<1 || arg.cols<1 || arg.type()!=MODEL_MAT_TYPE)
		throw -2;
	if(!wIn.empty()) if(arg.rows!=wIn.rows) throw -2;
	if(!b.empty()) if(arg.rows!=b.rows) throw -2;
	wOut=arg.clone();
}

void SLFNMat::setG(ActivationFunction* arg){
	if(arg==NULL) throw -1;
	g=arg;
}

Mat SLFNMat::f(const Mat& in){
	if(wIn.empty() || b.empty() || wOut.empty() || g==NULL) throw -1;
	if(in.empty()) throw -1;
	if(in.dims!=2 || in.rows<1 || in.cols!=1 || in.type()!=MODEL_MAT_TYPE)
		throw -2;
	
	Mat r=Mat::zeros(getM(),1,MODEL_MAT_TYPE);
	
	for(int i=0;i<getNH();++i)
		r+=wOut.row(i).t()*g->f(in.t().dot(wIn.row(i))+b.at<double>(i));
	return r;
}
