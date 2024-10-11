#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <limits>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "SLFN.h"
#include "SLFNMat.hpp"
#include "SigmoidFunction.h"
#include "SLFNTrainer.hpp"
#include "SLFNELM.hpp"

using namespace std;
using namespace cv;
using namespace model;

void out_exception(int arg);
void printMat(const Mat& arg);

int main(int argc,char *argv[]){
	if(argc<4) return -1;
	if(argc>4) return -2;
	
	int ch=std::atoi(argv[3]);
	ifstream fin(argv[1]);
	vector<string> lines;
	
	if(!fin.is_open()) return -4; //failure opening file
	while(1){
		string line;
		
		if(!getline(fin,line)) break;
		lines.push_back(line);
	}
	fin.close();
	
	int size=lines.size();
	double* dxa=new double[size]; //ATTENTION: new
	double* dya=new double[size]; //ATTENTION: new
	
	for(int i=0;i<size;++i){
		const char* line=lines.at(i).c_str();
		
		sscanf(line,"%lf,%lf",&dxa[i],&dya[i]);
	}
	
	SLFN* net=new SLFNMat();
	
	net->setG(SigmoidFunction::getInstance());
	SLFNELM::init(net,1,ch);
	
	Mat tIn(size,1,SLFNELM::MODEL_MAT_TYPE);
	Mat tOut(size,1,SLFNELM::MODEL_MAT_TYPE);
	
	for(int i=0;i<size;++i){
		tIn.at<double>(i)=dxa[i];
		tOut.at<double>(i)=dya[i];
	}
	
	SLFNTrainer* trainer=SLFNELM::getInstance();
	
	try{
		trainer->run(net,tIn,tOut);
	} catch(int e){
		out_exception(e);
		return e;
	}
	
	//clean up
	delete[] dxa;
	delete[] dya;
	
	ofstream fout(argv[2]);
	
	if(!fout.is_open()) return -4; //failure opening file
	fout.precision(numeric_limits<double>::digits10);
	for(int txint=-2500;txint<2500;txint+=1){
		double tx=txint/250.;
		Mat txa(1,1,SLFNELM::MODEL_MAT_TYPE,&tx);
		Mat tya=net->f(txa);
		
		fout<<tx<<","<<tya.at<double>(0)<<endl;
	}
	fout.close();
	delete net;
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
	cout<<" ("<<arg<<")"<<endl;
}

void printMat(const Mat& arg){
	if(arg.empty()) return;
	if(arg.rows<1 || arg.cols<1 || arg.type()!=SLFNELM::MODEL_MAT_TYPE) return;
	for(int i=0;i<arg.rows;++i){
		cout<<"\t";
		for(int j=0;j<arg.cols;++j)
			cout<<fixed<<setprecision(4)<<arg.at<double>(i,j)<<" ";
		cout<<endl;
	}
}
