#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "ActivationFunction.h"
#include "HyperbolicTangentFunction.h"
#include "SLFN.h"
#include "SLFNMat.hpp"

using namespace std;
using namespace cv;
using namespace model;

void out_exception(int arg);
void printMat(const Mat& arg);

int main(int argc,char* argv[]){
	SLFN* net=new SLFNMat();
	const int N=1;
	const int NH=3;
	const int M=1;
	double _wIn[NH*N]={-0.5,0.8,0.2};
	double _b[NH]={0.4,-0.7,0.3};
	double _wOut[NH*M]={0.6,-0.7,0.2};
	ActivationFunction* g=HyperbolicTangentFunction::getInstance();
	Mat wIn(NH,N,SLFNMat::MODEL_MAT_TYPE,_wIn);
	Mat b(NH,1,SLFNMat::MODEL_MAT_TYPE,_b);
	Mat wOut(NH,M,SLFNMat::MODEL_MAT_TYPE,_wOut);
	
/*	cout<<"Input weights:"<<endl;
	printMat(wIn);
	cout<<endl;
	cout<<"Hidden layer biases:"<<endl;
	printMat(b);
	cout<<endl;
	cout<<"Output weights:"<<endl;
	printMat(wOut);
	cout<<endl;*/

	try{
		net->setWIn(wIn);
		net->setB(b);
		net->setWOut(wOut);
		net->setG(g);
	} catch(int e){
		out_exception(e);
		return e;
	}
	
/*	cout<<"Input length: "<<net->getN()<<endl;
	cout<<"Hidden nodes: "<<net->getNH()<<endl;
	cout<<"Output length: "<<net->getM()<<endl;
	cout<<endl;

	cout<<"Input weights:"<<endl;
	printMat(net->getWIn());
	cout<<endl;
	cout<<"Hidden layer biases:"<<endl;
	printMat(net->getB());
	cout<<endl;
	cout<<"Output weights:"<<endl;
	printMat(net->getWOut());
	cout<<endl;*/

	const int L_IN=1;
	double _in[L_IN]={0.25};
	Mat in(L_IN,1,SLFNMat::MODEL_MAT_TYPE,_in);
	Mat out=net->f(in);
	
	delete net;
	cout<<"SLFN output:"<<endl;
	printMat(out);
	return 0;
}

void out_exception(int arg){
	cout<<"Exception caught: ";
	switch(arg){
	case -1: cout<<"null pointer"; break;
	case -2: cout<<"illegal argument"; break;
	case -3: cout<<"unexpected internal mismatch"; break;
	default: cout<<"unknown";
	}
	cout<<endl;
}

void printMat(const Mat& arg){
	if(arg.empty()) return;
	if(arg.rows<1 || arg.cols<1 || arg.type()!=SLFNMat::MODEL_MAT_TYPE) return;
	for(int i=0;i<arg.rows;++i){
		cout<<"\t";
		for(int j=0;j<arg.cols;++j)
			cout<<fixed<<setprecision(4)<<arg.at<double>(i,j)<<" ";
		cout<<endl;
	}
}
