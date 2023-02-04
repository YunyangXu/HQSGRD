#include<iostream>
#include<vector>
#include <thread>
#include<opencv2\opencv.hpp>  
#include "priorityQueue.h"
using namespace cv;
using namespace std;


typedef struct pixelsLocal {
	int x=-1;
	int y=-1;
	int z=-1;
	int initial_x=-1;
	int initial_y=-1;
}*pixels_Local;
typedef struct pixelsPoinT {
	int R;
	int G;
	int B;
	pixelsLocal closestLocal[8];
	int D[9] = { -1,-1,-1,-1,-1,-1,-1,-1,-1 };
}*pixels_PoinT;
typedef struct TempShortArray {
	int distance;
	int x = 0;
	int y = 0;
	int z = 0;
	int Min_Sign = -1;
	int NO = -1;
}*eight_neighborhood_pixels;

class HQSGTRD
{
public:


	typedef struct  differentialQuotient {
		int x=-1;
		int y=-1;
		float dQNum=-1;
	}*differential_Quotient;
	typedef struct pixelsPointNew {
		int L;
		int a;
		int b;

		int R;
		int G;
		int B;

		double NormL = 0;
		double NormA = 0;
		double NormB = 0;
		double NormX = 0;
		double NormY = 0;
		double NormG = 0;

		double g = 0.0;
		double u = 0.0;

		int ClassSign = -1;
		int flag;
		int Lately_num = 0;
		pixels_Local closestLocal = new  pixelsLocal[8];//与该像素距离最近的两个点


		int SuperpixelLabel = -1;
		int Superpixel_Boundary = -1;
		int WalkingDirection = -1;
		int W_sign = -1;

		int BSign = -1;
		int x;
		int y;

		int end = -1;

		int CurrentSuperpixelBoundary = -1;

		int Fsign = -1;
		int GapSign = -1;

		int D[9] = { -1,-1,-1,-1,-1,-1,-1,-1,-1 };

	};
	typedef struct SeedPoint_T {
		int x=-1;
		int y=-1;

		int Modified_x=-1;
		int Modified_y=-1;

		int Sign=-1;
		int UpperVertex_x=-1;
		int UpperVertex_y=-1;
		int LowerVertex_x=-1;
		int LowerVertex_y=-1;

		int LeftVerticesOne_x=-1;
		int LeftVerticesOne_y=-1;

		int LeftVerticesTwo_x=-1;
		int LeftVerticesTwo_y=-1;

		int RightVerticesOne_x=-1;
		int RightVerticesOne_y=-1;

		int RightVerticesTwo_x=-1;
		int RightVerticesTwo_y=-1;

		int DELETE = 0;

	}*Seed_Point_T;

	typedef struct SeedPN {
		int SeedSign = -1;
	}*Seed_PN;

	typedef struct centroidOfCluster
	{
		int  Csign = 0;
		double cL = 0;
		double ca = 0;
		double cb = 0;
		double cg = 0;
		double cx = 0;
		double cy = 0;
	}*centroid_Cluster;

	typedef struct MarkMSW
	{
		double sumL = 0;
		double suma = 0;
		double sumb = 0;
		double sumg = 0;
		double sumu = 0;
		double sums = 0;

		double SumNum = 0;
		double maxL = 0;
		double maxa = 0;
		double maxb = 0;
		double maxg = 0;
		double maxu = 0;
		double maxs = 0;

		double wL = 0;
		double wa = 0;
		double wb = 0;
		double wg = 0;
		double wu = 0;
		double ws = 0;

	}*Mark_SW;

	typedef struct MarkMparameter
	{
		int  Csign = 0;
		double dL = 0;
		double da = 0;
		double db = 0;
		double dg = 0;
		double du = 0;
		double ds = 0;
	}*Mark_parameter;

	typedef struct MarkMatrix
	{
		int  Csign = 0;
		int Lable = -1;
		int Fsign = -1;
		int TRSign = -1;
	}*Mark_Matri;

	const int N8[8][2] =
	{ 
		{0,-1},
		{-1,0},
		{0,1},
		{1,0},

		{-1,-1},
		{-1,1},
		{1,1},
		{1,-1}
	};

	int DrawcolourRed[256];
	int DrawcolourBlue[256];
	int DrawcolourGreen[256];
	string FileName = "";

	int MaxL = 0, MaxA = 0, MaxB = 0;
	int MinL = 0, MinA = 0, MinB = 0;
	int PictureI = 0;
	int PictureJ = 0;

	int kNum = 0;

	double  maxGNum = 0, minGNum = 0;
	int classificationLable = 0;
	
	int SuperPixel_size = 0;

	vector<vector<pixelsPointNew>>PixelMatrix;
	
	vector<vector<pixelsPoinT>>PixelMatrix_temp;

	vector<vector<MarkMparameter>>Pixel_Information;

	vector<vector<MarkMatrix>>Superpixel_Label;

	Seed_Point_T  Seed_Point;
	
	vector<vector<SeedPN>>Seed_label;


	HQSGTRD();
	~HQSGTRD();

	void Superpixel_Segmentation(string image_file_name, string edge_file_name, int Superpixel_Num, string file_name, string Output_folder, vector<double>& run_time, int iteration_num, int Save_Image, int Save_Label_file);

	void Calculate_Contour_Information(Mat LabImage, vector<vector<int>>& t_value);

	void RearC(Mat image, int start, vector<vector<pixelsPoinT>>& PixelMatrixT, vector<vector<int>> t_value);

	void middleC(Mat image, int start, vector<vector<pixelsPoinT>>& PixelMatrixT, vector<vector<int>> t_value);

	void Clustering(Mat LabImage, vector<vector<int>> t_value);

	void Generate_Classification(vector<vector<pixelsPoinT>>& PixelMatrixT);

	void changsignPMN(int x, int y, int PreSign, int Aftersign);

	void Cluster_Image();

	int Superpixel_Generation(Mat Image, int Superpixel_Num, Mat& Finally_labelMat, int iteration_num);

	void Flat_Regions_Generation(Mat img, int Superpixel_Num, int cstrips, double SuperPixel_size, double grid_interval_S);

	void ProduceSuperpixelLable(int x, int y, int m, int SeedX, int SeedY, int& SuperPixelsNum);

	int Seed_Point_Generation(int k, Mat img);

	void DDALine(int x0, int y0, int x1, int y1, Mat img, int sign);

	int Non_Flat_Regions_Generation(string path, int superpixel_num_K, int grid_interval_S, Mat img, int iteration_num, Mat& Finally_labelMat);

	void Label2Boundary(int* label, Mat img, Mat img_boundary);

	void Iterative_Generation(cv::Mat& labelMat, int numseeds, int grid_interval_S, int itr_times, cv::Mat img);

	void Weights_Adjustment(cv::Mat labelMat_sign, cv::Mat Label, double& WL, double& Wa, double& Wb, double& Wg, double& Ws, centroid_Cluster coC, int numseeds);

	void Update_Seed_Point(cv::Mat Label, cv::Mat img_lab, int num_region, centroid_Cluster coC, int* spixel_size);

	void Superpixel_Combination(cv::Mat labelMat);

	void number_Statistical_SuperpixelsLable(int CNum, Nodem NumberSuperPixle);

	void CombineSuperpixelsPriority(int countNum, int& NumberSuperpixels, int length, Nodem NumberSuperPixle);

	void CombineSuperpixelsPriorityS(int countNum, int& NumberSuperpixels, int length, Nodem NumberSuperPixle);


	void FindNeighborhood(int x, int y, int lable, vector<int>& nborhood, int RM);

	void mergeSuperpixelsLable(int x, int y, int PreSign, int Aftersign);

	void Save_result(Mat Finally_labelMat, string file_name, string Output_folder, int final_label, int Save_Image, int Save_Label_file);
	//void Clustering(Mat LabImage, vector<vector<int>>& t_value);
private:

	

};

