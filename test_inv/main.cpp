#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/opencv.hpp"
#include "SLFNMat.hpp"

using namespace std;
using namespace cv;
using namespace model;

void printMat(const Mat& arg);
void runTestOnce(const Mat& arg,string s);

int main(int argc,char *argv[]){
	const int L_A_1=4;
	const int L_A_2=4;
	const int L_B_1=4;
	const int L_B_2=4;
	const int L_X_1=3;
	const int L_X_2=5;
	double _a[L_A_1*L_A_2]={1.,0.,0.,0.,
	                        0.,1.,0.,0.,
	                        0.,0.,1.,0.,
	                        0.,0.,0.,1.};
	double _b[L_B_1*L_B_2]={1.,0.,0.,0.,
	                        0.,2.,0.,0.,
	                        0.,0.,3.,0.,
	                        0.,0.,0.,4.};
	double _x[L_X_1*L_X_2]={1.,1.,0.,1.,0.,
	                        0.,1.,1.,1.,1.,
	                        1.,0.,1.,1.,0.};
	Mat a(L_A_1,L_A_2,SLFNMat::MODEL_MAT_TYPE,_a);
	Mat b(L_B_1,L_B_2,SLFNMat::MODEL_MAT_TYPE,_b);
	Mat x(L_X_1,L_X_2,SLFNMat::MODEL_MAT_TYPE,_x);
	Mat y=x.t();
	
	runTestOnce(a,"A");
	runTestOnce(b,"B");
	runTestOnce(x,"X");
	runTestOnce(y,"Y");
	return 0;
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

void runTestOnce(const Mat& arg,string s){
	Mat inv=arg.inv(DECOMP_SVD);
	
	cout<<"Matrix "<<s<<":"<<endl;
	printMat(arg);
	cout<<endl;
	cout<<"Matrix "<<s<<", "<<s<<"^+:"<<endl;
	printMat(inv);
	cout<<endl;
}
