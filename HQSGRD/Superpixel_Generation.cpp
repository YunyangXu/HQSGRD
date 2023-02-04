#include"Superpixel_Generation.h"
#include<opencv2\opencv.hpp>  
#include<time.h>
#include <fstream>
#include <string> 
using namespace cv;

HQSGTRD::HQSGTRD()
{
	for (int i = 0; i < 255; i++)
	{
		DrawcolourRed[i] = -1;
		DrawcolourBlue[i] = -1;
		DrawcolourGreen[i] = -1;
	}
	int i = 0;
	int Cnum = 0;
	int temc = 0;

	while (i < 255)
	{
		Cnum = rand() % 255;
		for (int j = 0; j <= i; j++)
		{
			if (DrawcolourRed[j] == Cnum) {
				temc++;
			}
		}
		if (temc == 0) {
			DrawcolourRed[i++] = Cnum;
			//cout << Cnum<<" ";
		}
		temc = 0;
	}

	i = 0;
	Cnum = 0;
	temc = 0;

	while (i < 255)
	{
		Cnum = rand() % 255;

		for (int j = 0; j <= i; j++)
		{
			if (DrawcolourBlue[j] == Cnum) {
				temc++;
			}
		}
		if (temc == 0) {
			DrawcolourBlue[i++] = Cnum;
			//cout << Cnum<<" ";
		}
		temc = 0;
	}
	i = 0;
	Cnum = 0;
	temc = 0;

	while (i < 255)
	{
		Cnum = rand() % 255;
		for (int j = 0; j <= i; j++)
		{
			if (DrawcolourGreen[j] == Cnum) {
				temc++;
			}
		}
		if (temc == 0) {
			DrawcolourGreen[i++] = Cnum;
			//cout << Cnum<<" ";
		}
		temc = 0;
	}

}

HQSGTRD::~HQSGTRD()
{
}
void HQSGTRD::Superpixel_Segmentation(string image_file_name, string edge_file_name, int Superpixel_Num, string file_name, string Output_folder, vector<double>& run_time, int iteration_num, int Save_Image, int Save_Label_file)
{
	Mat img = imread(image_file_name);
	Mat imgEge = imread(edge_file_name, 0);
	Mat Enhance_boundaries;// = imread(edge_file_name, 0);
	imgEge.copyTo(Enhance_boundaries);

	Mat LabImage;
	Mat LabImage1;

	clock_t startTime, endTime;

	cvtColor(img, LabImage1, COLOR_BGR2Lab);

	vector<vector<int>> t_value;
	SeedPN Seedpn;

	MarkMparameter SWtemp;
	MarkMatrix MM;
	PixelMatrix.clear();
	PixelMatrix_temp.clear();
	Seed_label.clear();
	Pixel_Information.clear();

	startTime = clock();
	GaussianBlur(LabImage1, LabImage, Size(3, 3), 0, 0, BORDER_DEFAULT);

	for (int l = 0; l < LabImage.rows; l++)
	{
		vector<int> t_value_temp;
		vector<SeedPN> Seed;
		vector<MarkMparameter > swIn;
		vector<MarkMatrix >MaM;
		for (int l = 0; l < LabImage.cols; l++)
		{
			t_value_temp.push_back(50);
			Seedpn.SeedSign = -1;
			Seed.push_back(Seedpn);
			swIn.push_back(SWtemp);
			MaM.push_back(MM);
		}
		t_value.push_back(t_value_temp);
		Seed_label.push_back(Seed);
		Pixel_Information.push_back(swIn);
		Superpixel_Label.push_back(MaM);
	}

	FileName = image_file_name;

	MaxL = 0, MaxA = 0, MaxB = 0;
	MinL = 1000, MinA = 1000, MinB = 1000;

	vector<pixelsPointNew>PixelM;
	vector<pixelsPoinT>PixelM1;

	pixelsPointNew pixel;
	pixelsPoinT pixel1;
	for (int i = 0; i < LabImage.rows; i++) {

		PixelM.clear();
		PixelM1.clear();
		for (int j = 0; j < LabImage.cols; j++) {

			pixel.L = LabImage.at<cv::Vec3b>(i, j)[0];
			pixel.a = LabImage.at<cv::Vec3b>(i, j)[1];
			pixel.b = LabImage.at<cv::Vec3b>(i, j)[2];

			if (pixel.L > MaxL) {
				MaxL = pixel.L;
			}

			if (pixel.a > MaxA) {
				MaxA = pixel.a;
			}
			if (pixel.b > MaxB) {
				MaxB = pixel.b;
			}
			if (pixel.L < MinL) {
				MinL = pixel.L;
			}
			if (pixel.a < MinA) {
				MinA = pixel.a;
			}
			if (pixel.b < MinB) {
				MinB = pixel.b;
			}
			pixel1.R = pixel.L;
			pixel1.G = pixel.a;
			pixel1.B = pixel.b;

			PixelM.push_back(pixel);
			PixelM1.push_back(pixel1);
		}
		PixelMatrix.push_back(PixelM);
		PixelMatrix_temp.push_back(PixelM1);
	}


	for (int i = 0; i < imgEge.rows; i++)
	{
		for (int j = 0; j < imgEge.cols; j++) {

			if ((int)imgEge.at<uchar>(i, j) <= 5) {
				imgEge.at<uchar>(i, j) = 0;
				Enhance_boundaries.at<uchar>(i, j) = 0;
			}
		}
	}

	for (int i = 2; i < imgEge.rows - 2; i++)
	{
		for (int j = 2; j < imgEge.cols - 2; j++) {

			if ((int)imgEge.at<uchar>(i, j) > 2) {//&& we[i][j]<7) {

				for (int n = i - 2; n <= i + 1; n++)
				{
					for (int m = j - 2; m <= j + 1; m++)
					{
						if ((int)imgEge.at<uchar>(n, m) > 2) {
							continue;
						}
						Enhance_boundaries.at<uchar>(n, m) = imgEge.at<uchar>(i, j);// % 255;
					}
				}
			}
			else
			{
				Enhance_boundaries.at<uchar>(i, j) = imgEge.at<uchar>(i, j);// % 255;
			}

		}
	}

	Calculate_Contour_Information(Enhance_boundaries, t_value);

	Clustering(LabImage, t_value);

	Mat Finally_labelMat;

	int final_label = Superpixel_Generation(imgEge, Superpixel_Num, Finally_labelMat, iteration_num);

	endTime = clock();
	run_time.push_back((double)(endTime - startTime) / CLOCKS_PER_SEC);

	cout << Superpixel_Num << "\n";

	Save_result(Finally_labelMat, file_name, Output_folder, final_label, Save_Image, Save_Label_file);

}

void HQSGTRD::Calculate_Contour_Information(Mat LabImage, vector<vector<int>>& t_value)
{

	vector<vector<int>> Edge_enhanced_L_channel;
	for (int l = 0; l < LabImage.rows; l++)
	{
		vector<int> temp;
		for (int l = 0; l < LabImage.cols; l++)
		{
			temp.push_back(0);
		}
		Edge_enhanced_L_channel.push_back(temp);
	}

	double v = 0;
	double g = 0;
	double v2 = 0;
	double g2 = 0;
	double v1 = 0;
	double g1 = 0;

	for (int i = 0; i < LabImage.rows; i++)
	{
		vector<int> W_temp;
		for (int j = 0; j < LabImage.cols; j++) {

			Edge_enhanced_L_channel[i][j] = PixelMatrix[i][j].L + (int)LabImage.at<uchar>(i, j);
			
		}
	}

	maxGNum = 0.0;
	minGNum = 1000;

	for (int i = 1; i < LabImage.rows - 1; i++)
	{
		for (int j = 1; j < LabImage.cols - 1; j++) {

			v2 = (Edge_enhanced_L_channel[i - 1][j] + Edge_enhanced_L_channel[i - 1][j - 1] + Edge_enhanced_L_channel[i - 1][j + 1]) - (Edge_enhanced_L_channel[i + 1][j] + Edge_enhanced_L_channel[i + 1][j - 1] + Edge_enhanced_L_channel[i + 1][j + 1]);

			g2 = (Edge_enhanced_L_channel[i - 1][j - 1] + Edge_enhanced_L_channel[i][j - 1] + Edge_enhanced_L_channel[i + 1][j - 1]) - (Edge_enhanced_L_channel[i + 1][j + 1] + Edge_enhanced_L_channel[i][j + 1] + Edge_enhanced_L_channel[i - 1][j + 1]);

			v = Edge_enhanced_L_channel[i - 1][j] - Edge_enhanced_L_channel[i + 1][j];

			g = Edge_enhanced_L_channel[i][j - 1] - Edge_enhanced_L_channel[i][j + 1];


			t_value[i][j] = sqrt(v * v + g * g);
			PixelMatrix[i][j].g = sqrt(v2 * v2 + g2 * g2);

			if (t_value[i][j] < 5) {
				t_value[i][j] = 0;
			}
			if (PixelMatrix[i][j].g < 5) {
				PixelMatrix[i][j].g = 0.0;
			}

			if (PixelMatrix[i][j].g > maxGNum) {
				maxGNum = PixelMatrix[i][j].g;
			}

			if (PixelMatrix[i][j].g < minGNum) {
				minGNum = PixelMatrix[i][j].g;
			}

		}
	}

	return;
	int z = 0;
	int colourRed;
	int colourBlue;
	int colourGreen;

	Mat img56 = imread(FileName, 1);
//	Mat img56(LabImage.rows, LabImage.cols, CV_8U);
	for (int i = 1; i < LabImage.rows - 1; i++)
	{
		for (int j = 1; j < LabImage.cols - 1; j++) {


			/*if (t_value[i][j] == 0) {
				img56.at<uchar>(i, j) = t_value[i][j];
			
			}
			else {
				img56.at<uchar>(i, j) = t_value[i][j]+50;

			}*/

			if (t_value[i][j] == 0) {

				img56.at<Vec3b>(i, j)[0] = t_value[i][j];// % 255;
				img56.at<Vec3b>(i, j)[1] = t_value[i][j];// % 255;
				img56.at<Vec3b>(i, j)[2] = t_value[i][j];// % 255;
			}
			else {
				img56.at<Vec3b>(i, j)[0] = t_value[i][j] + 50;// % 255;
				img56.at<Vec3b>(i, j)[1] = t_value[i][j] + 50;// % 255;
				img56.at<Vec3b>(i, j)[2] = t_value[i][j] + 50;// % 255;
			}

			/*	if (( (double)Tweight[i][j] /( maxPicturerNum/3)) >0.05)
				{

					//cout << (double)Tweight[i][j] / max << "\n";
					img56.at<Vec3b>(i, j)[0] = Tweight[i][j] + 40;// % 255;
					img56.at<Vec3b>(i, j)[1] = Tweight[i][j] + 40;// % 255;
					img56.at<Vec3b>(i, j)[2] = Tweight[i][j] + 40;// % 255;
				}
				else
				{
					img56.at<Vec3b>(i, j)[0] = 0;// % 255;
					img56.at<Vec3b>(i, j)[1] = 0;// % 255;
					img56.at<Vec3b>(i, j)[2] = 0;// % 255;
				}*/

				/*if (PixelMatrixNew[i][j].g > 0) {

					img56.at<Vec3b>(i, j)[0] = (int)PixelMatrixNew[i][j].g + 40;// % 255;
					img56.at<Vec3b>(i, j)[1] = (int)PixelMatrixNew[i][j].g + 40;// % 255;
					img56.at<Vec3b>(i, j)[2] = (int)PixelMatrixNew[i][j].g + 40;// % 255;
				}
				else
				{
					img56.at<Vec3b>(i, j)[0] = (int)PixelMatrixNew[i][j].g;// % 255;
					img56.at<Vec3b>(i, j)[1] = (int)PixelMatrixNew[i][j].g;// % 255;
					img56.at<Vec3b>(i, j)[2] = (int)PixelMatrixNew[i][j].g;// % 255;
				}
				*/
		}
	}

	//imshow("img", img);

	string firstname(FileName.substr(FileName.find_last_of("\\\\") + 1, FileName.length() - FileName.find_last_of("\\\\")));

	string fileName(firstname.substr(0, firstname.find_last_of(".")));

	fileName = "E:\\5B\\" + fileName + ".bmp";

	//fileName = "E:\\BSD5008\\" + fileName + ".jpg";
	cv::imwrite(fileName, img56);   //  将image图像保存为my.jpg
}

void forwardC(Mat image, int end, vector<vector<pixelsPoinT>>&PixelMatrixT, vector<vector<int>> t_value) {
	int B[9];
	int CCCnum = 0;
	for (int i = 0; i < 9; i++)
	{
		B[i] = 9 - i;
	}

	for (int i = 1; i <= end; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			int iNum = i;
			int jNum = j;
			CCCnum++;
			int u = 1;

			eight_neighborhood_pixels array = new TempShortArray[8];
			int z = 0;
			int Length = 0;
			int S[8];

			for (int I = i - 1; I <= i + 1; I++)
			{
				for (int J = j - 1; J <= j + 1; J++)
				{
					if (I == iNum && J == jNum) {
						continue;
					}
					array[Length].x = I;
					array[Length].y = J;

					//cout << A[n][m] << " ";
					if (PixelMatrixT[i][j].D[u] == -1)
					{
						PixelMatrixT[i][j].D[u] = round((double)sqrt(
							(PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) * (PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) +
							(PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) * (PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) +
							(PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B) * (PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B)
						));
						PixelMatrixT[I][J].D[B[u]] = PixelMatrixT[i][j].D[u];
						array[Length].distance = PixelMatrixT[i][j].D[u];
					}
					else
					{
						array[Length].distance = PixelMatrixT[i][j].D[u];
						//cout << C[i][j].D[u] << " ";
					}
					array[Length].z = u - 1;
					u++;

					array[Length].distance += t_value[I][J];// *Tweight[I][J];

					array[Length].Min_Sign = -1;
					Length++;

				}
			}

			int min1 = 99999;
			int min2 = 99999;
			for (int i = 0; i < Length; i++) {

				if (array[i].distance < min1) {
					min1 = array[i].distance;
				}

			}
			for (int i = 0; i < Length; i++) {

				if (array[i].distance == min1) {
					continue;
				}
				else
				{
					if (array[i].distance < min2) {
						min2 = array[i].distance;
					}
				}
			}
			for (int numi = 0; numi < Length; numi++)
			{
				if (array[numi].distance == min2 || array[numi].distance == min1) {
					PixelMatrixT[iNum][jNum].closestLocal[numi].z = 1;
				}
			}
			delete[]array;
		}
	}

}

void HQSGTRD::RearC(Mat image, int start, vector<vector<pixelsPoinT>> &PixelMatrixT, vector<vector<int>> t_value) {
	int B[9];
	int CCCnum = 0;
	for (int i = 0; i < 9; i++)
	{
		B[i] = 9 - i;
	}

	for (int i = start + 3; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			int iNum = i;
			int jNum = j;
			//CCCnum++;
			int u = 1;
			eight_neighborhood_pixels array = new TempShortArray[8];
			int z = 0;
			int Length = 0;
			int S[8];

			for (int I = i - 1; I <= i + 1; I++)
			{
				for (int J = j - 1; J <= j + 1; J++)
				{
					if (I == iNum && J == jNum) {
						continue;
					}
					array[Length].x = I;
					array[Length].y = J;

					if (PixelMatrixT[i][j].D[u] == -1)
					{
						PixelMatrixT[i][j].D[u] = round((double)sqrt(
							(PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) * (PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) +
							(PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) * (PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) +
							(PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B) * (PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B)
						));

						PixelMatrixT[I][J].D[B[u]] = PixelMatrixT[i][j].D[u];
						array[Length].distance = PixelMatrixT[i][j].D[u];
					}
					else
					{
						array[Length].distance = PixelMatrixT[i][j].D[u];
					}
					array[Length].z = u - 1;
					u++;

					array[Length].distance += t_value[I][J];// *Tweight[I][J];
					array[Length].Min_Sign = -1;
					Length++;

				}
			}

			int min1 = 99999;
			int min2 = 99999;
			for (int i = 0; i < Length; i++) {

				if (array[i].distance < min1) {
					min1 = array[i].distance;
				}

			}
			for (int i = 0; i < Length; i++) {

				if (array[i].distance == min1) {
					continue;
				}
				else
				{
					if (array[i].distance < min2) {
						min2 = array[i].distance;
					}
				}
			}

			for (int numi = 0; numi < Length; numi++)
			{
				if (array[numi].distance == min2 || array[numi].distance == min1) {
					PixelMatrixT[iNum][jNum].closestLocal[numi].z = 1;
				}
			}
			delete[]array;
		}
	}

}

void HQSGTRD::middleC(Mat image, int start, vector<vector<pixelsPoinT>> &PixelMatrixT, vector<vector<int>> t_value) {
	int B[9];
	int CCCnum = 0;
	for (int i = 0; i < 9; i++)
	{
		B[i] = 9 - i;
	}

	for (int i = start + 1; i <= start + 2; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			int iNum = i;
			int jNum = j;

			int u = 1;
			eight_neighborhood_pixels array = new TempShortArray[8];
			int z = 0;
			int Length = 0;
			int S[8];

			for (int I = i - 1; I <= i + 1; I++)
			{
				for (int J = j - 1; J <= j + 1; J++)
				{
					if (I == iNum && J == jNum) {
						continue;
					}
					array[Length].x = I;
					array[Length].y = J;

					if (PixelMatrixT[i][j].D[u] == -1)
					{
						PixelMatrixT[i][j].D[u] = round((double)sqrt(
							(PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) * (PixelMatrixT[iNum][jNum].R - PixelMatrixT[I][J].R) +
							(PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) * (PixelMatrixT[iNum][jNum].G - PixelMatrixT[I][J].G) +
							(PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B) * (PixelMatrixT[iNum][jNum].B - PixelMatrixT[I][J].B)
						));
						PixelMatrixT[I][J].D[B[u]] = PixelMatrixT[i][j].D[u];
						array[Length].distance = PixelMatrixT[i][j].D[u];
					}
					else
					{
						array[Length].distance = PixelMatrixT[i][j].D[u];
					}
					array[Length].z = u - 1;
					u++;

					array[Length].distance += t_value[I][J];// *Tweight[I][J];

					array[Length].Min_Sign = -1;
					Length++;

				}
			}

			int min1 = 99999;
			int min2 = 99999;
			for (int i = 0; i < Length; i++) {

				if (array[i].distance < min1) {
					min1 = array[i].distance;
				}

			}
			for (int i = 0; i < Length; i++) {

				if (array[i].distance == min1) {
					continue;
				}
				else
				{
					if (array[i].distance < min2) {
						min2 = array[i].distance;
					}
				}
			}

			for (int numi = 0; numi < Length; numi++)
			{
				if (array[numi].distance == min2 || array[numi].distance == min1) {
					PixelMatrixT[iNum][jNum].closestLocal[numi].z = 1;
				}
			}
			delete[]array;

		}
	}

}

void HQSGTRD::Clustering(Mat LabImage, vector<vector<int>> t_value) {


	PictureI = LabImage.rows - 1;
	PictureJ = LabImage.cols - 1;

	int end1 = LabImage.rows / 2;
	thread t(forwardC, LabImage, end1, ref(PixelMatrix_temp), t_value);
	RearC(LabImage, end1, PixelMatrix_temp, t_value);
	middleC(LabImage, end1, PixelMatrix_temp, t_value);
	t.join();
	
	//int hj = 0;
	Generate_Classification(PixelMatrix_temp);
	//Cluster_Image();
	
}

void HQSGTRD::Generate_Classification(vector<vector<pixelsPoinT>>&PixelMatrixT) {

	classificationLable = 0;

	int Sx = 0;
	int Sy = 0;

	int I_Capped2 = 0;
	int I_LowerLimit2 = 0;

	int J_Capped2 = 0;
	int J_LowerLimit2 = 0;

	int I_Capped = 0;
	int I_LowerLimit = PictureI;

	int J_Capped = 0;
	int J_LowerLimit = PictureJ;

	int iNum = 0;
	int jNum = 0;

	for (int i = 1; i <= PictureI - 1; i++)
	{
		for (int j = 1; j <= PictureJ - 1; j++)
		{
			iNum = i;
			jNum = j;
			int au = 8;
			int bu = -1;

			for (int I = iNum - 1; I <= iNum + 1; I++)
			{
				for (int J = jNum - 1; J <= jNum + 1; J++)
				{
					if (iNum == I && jNum == J)
					{
						continue;
					}
					au--;
					bu++;

					//cout << PixelMatrixT[iNum][jNum].closestLocal[bu].z << " ";
					if (PixelMatrixT[iNum][jNum].closestLocal[bu].z == -1) {

						continue;
					}
					if (PixelMatrixT[I][J].closestLocal[au].z == -1) {
						continue;
					}


					if (PixelMatrix[I][J].ClassSign != -1) {

						if (PixelMatrix[iNum][jNum].ClassSign != -1)
						{

							if (PixelMatrix[iNum][jNum].ClassSign != PixelMatrix[I][J].ClassSign)
							{

								if (PixelMatrix[iNum][jNum].ClassSign > PixelMatrix[I][J].ClassSign)
								{
									if (classificationLable == PixelMatrix[iNum][jNum].ClassSign) {
										classificationLable--;
									}
									int sign = PixelMatrix[iNum][jNum].ClassSign;

									PixelMatrix[iNum][jNum].ClassSign = PixelMatrix[I][J].ClassSign;

									changsignPMN(iNum, jNum, sign, PixelMatrix[I][J].ClassSign);

								}

								if (PixelMatrix[iNum][jNum].ClassSign < PixelMatrix[I][J].ClassSign)
								{
									if (classificationLable == PixelMatrix[I][J].ClassSign) {
										classificationLable--;
									}

									int sign = PixelMatrix[I][J].ClassSign;
									PixelMatrix[I][J].ClassSign = PixelMatrix[iNum][jNum].ClassSign;

									changsignPMN(I, J, sign, PixelMatrix[iNum][jNum].ClassSign);

								}
							}

						}
						if (PixelMatrix[iNum][jNum].ClassSign == -1) {
							PixelMatrix[iNum][jNum].ClassSign = PixelMatrix[I][J].ClassSign;
						}
					}
					if (PixelMatrix[I][J].ClassSign == -1) {

						if (PixelMatrix[iNum][jNum].ClassSign == -1)
						{
							PixelMatrix[iNum][jNum].ClassSign = classificationLable++;
						}

						PixelMatrix[I][J].ClassSign = PixelMatrix[iNum][jNum].ClassSign;
						continue;
					}
				}
			}
		}
	}
}

void HQSGTRD::changsignPMN(int x, int y, int PreSign, int Aftersign)
{

	int I_Capped2 = 0;
	int I_LowerLimit2 = 0;

	int J_Capped2 = 0;
	int J_LowerLimit2 = 0;

	int I_Capped = 0;
	int I_LowerLimit = PictureI;

	int J_Capped = 0;
	int J_LowerLimit = PictureJ;

	int iNum = x;
	int jNum = y;

	if (iNum - I_Capped > 0) {
		I_Capped2 = iNum - 1;
	}
	else
	{
		I_Capped2 = I_Capped;
	}

	if (iNum - I_LowerLimit != 0)
	{
		I_LowerLimit2 = iNum + 1;
	}
	else
	{
		I_LowerLimit2 = I_LowerLimit;
	}

	if (jNum - J_Capped > 0)
	{
		J_Capped2 = jNum - 1;
	}
	else
	{
		J_Capped2 = J_Capped;
	}

	if (jNum - J_LowerLimit != 0)
	{
		J_LowerLimit2 = jNum + 1;
	}
	else
	{
		J_LowerLimit2 = J_LowerLimit;
	}

	for (int i = I_Capped2; i <= I_LowerLimit2; i++)
	{
		for (int j = J_Capped2; j <= J_LowerLimit2; j++)
		{
			if (iNum == i && jNum == j)
			{
				continue;
			}

			if (PreSign == PixelMatrix[i][j].ClassSign) {
				//	cout << "Change "<< PixelM[i][j].sign<<"  to"<< Aftersign<< "    ";
				PixelMatrix[i][j].ClassSign = Aftersign;

				changsignPMN(i, j, PreSign, Aftersign);
			}
		}

	}
	return;
}

 void HQSGTRD::Cluster_Image()
{
	Mat img = imread(FileName, 1);
	int z = 0;
	int colourRed;
	int colourBlue;
	int colourGreen;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (PixelMatrix[i][j].ClassSign == -1)//当前的值不是像素
			{
				continue;
			}
			//cout << PixelMatrixNew[i][j].ClassSign<<" ";

			colourRed = DrawcolourRed[PixelMatrix[i][j].ClassSign % 255];
			colourBlue = DrawcolourBlue[PixelMatrix[i][j].ClassSign % 255];
			colourGreen = DrawcolourGreen[PixelMatrix[i][j].ClassSign % 255];


			img.at<Vec3b>(i, j)[0] = colourBlue;
			img.at<Vec3b>(i, j)[1] = colourGreen;
			img.at<Vec3b>(i, j)[2] = colourRed;

			/*	img.at<Vec3b>(i, j)[0] = PixelMatrixNew[i][j].zB;
				img.at<Vec3b>(i, j)[1] = PixelMatrixNew[i][j].zG;
				img.at<Vec3b>(i, j)[2] = PixelMatrixNew[i][j].zR;*/
		}
	}


	string firstname(FileName.substr(FileName.find_last_of("\\\\") + 1, FileName.length() - FileName.find_last_of("\\\\")));

	string fileName(firstname.substr(0, firstname.find_last_of(".")));

	fileName = "E:\\5B\\" + fileName + ".bmp";

	cv::imwrite(fileName, img);

}

 int  HQSGTRD::Superpixel_Generation(Mat Image, int Superpixel_Num, Mat &Finally_labelMat, int iteration_num)
 {
	 int K = Superpixel_Num;
	 kNum = Superpixel_Num;

	 int Superpixel_Num_Actually = Seed_Point_Generation(Superpixel_Num, Image);

	 int grid_interval_S;
	 int N = Image.rows * Image.cols;
	 double spixel_size = 1.0 * N / K; 

	 SuperPixel_size = spixel_size;

	 grid_interval_S = (int)(sqrt(2 * sqrt(3.0) / 9 * spixel_size) + 0.5); 

	 int r_interval = (int)(1.5 * grid_interval_S + 0.5);
	 int c_interval = (int)(sqrt(3.0) * grid_interval_S + 0.5);
	 int rstrips = Image.rows / r_interval + 1; 
	 int cstrips = Image.cols / c_interval + 1;

	 Flat_Regions_Generation(Image, Superpixel_Num_Actually, cstrips, spixel_size, grid_interval_S);
	 int final_label= Non_Flat_Regions_Generation(FileName, Superpixel_Num_Actually, grid_interval_S, Image, iteration_num, Finally_labelMat);

	 return final_label;
 }


 inline void HQSGTRD::Flat_Regions_Generation(Mat img, int Superpixel_Num, int cstrips, double spixel_size,double grid_interval_S) {

	 int SuperPixelsNum = 0;

	 for (int s = 0; s < Superpixel_Num; s++)
	 {
		 SuperPixelsNum = 0;
		 ProduceSuperpixelLable(Seed_Point[s].Modified_x, Seed_Point[s].Modified_y, s, Seed_Point[s].Modified_x, Seed_Point[s].Modified_y, SuperPixelsNum);

		 if (SuperPixelsNum <= 0.80 * spixel_size)
		 {

			 int I_Capped2 = 0;
			 int I_LowerLimit2 = 0;

			 int J_Capped2 = 0;
			 int J_LowerLimit2 = 0;

			 int bblength = 2 * grid_interval_S;


			 int I_Capped = bblength;

			 int I_LowerLimit = PictureI;

			 int J_Capped = bblength;
			 int J_LowerLimit = PictureJ;

			 int iNum = Seed_Point[s].Modified_x;
			 int jNum = Seed_Point[s].Modified_y;


			 if (iNum - I_Capped > 0) {

				 I_Capped2 = iNum - I_Capped;
			 }
			 else
			 {
				 I_Capped2 = 0;
			 }


			 if (iNum + bblength - I_LowerLimit < 0)
			 {
				 I_LowerLimit2 = iNum + bblength;
			 }
			 else
			 {
				 I_LowerLimit2 = I_LowerLimit;
			 }


			 if (jNum - J_Capped > 0)
			 {
				 J_Capped2 = jNum - bblength;
			 }
			 else
			 {
				 J_Capped2 = 0;
			 }

			 if (jNum + bblength - J_LowerLimit < 0)
			 {
				 J_LowerLimit2 = jNum + bblength;
			 }
			 else
			 {
				 J_LowerLimit2 = J_LowerLimit;
			 }

			 for (int i = I_Capped2; i <= I_LowerLimit2; i++)
			 {
				 for (int j = J_Capped2; j <= J_LowerLimit2; j++)
				 {

					 if (iNum == i && jNum == j)
					 {
						 continue;
					 }
					 if (PixelMatrix[i][j].SuperpixelLabel == s) {

						 PixelMatrix[i][j].SuperpixelLabel = -1;
					 }

				 }

			 }

		 }
	
	 }

 }
 inline void HQSGTRD::ProduceSuperpixelLable(int x, int y, int m, int SeedX, int SeedY, int& SuperPixelsNum)
 {

	 if (PixelMatrix[x][y].Superpixel_Boundary == 1) {
		 return;
	 }
	 if (PixelMatrix[x][y].ClassSign == -1) {
		 return;
	 }
	 if (PixelMatrix[SeedX][SeedY].ClassSign != PixelMatrix[x][y].ClassSign) {
		 return;
	 }

	 if (PixelMatrix[x][y].SuperpixelLabel != m && PixelMatrix[x][y].SuperpixelLabel != -1) {
		 return;
	 }


	 int I_Capped2 = 0;
	 int I_LowerLimit2 = 0;

	 int J_Capped2 = 0;
	 int J_LowerLimit2 = 0;

	 int I_Capped = 0;
	 int I_LowerLimit = PictureI;

	 int J_Capped = 0;
	 int J_LowerLimit = PictureJ;

	 int iNum = x;
	 int jNum = y;


	 if (iNum - I_Capped > 0) {
		 I_Capped2 = iNum - 1;
	 }
	 else
	 {
		 I_Capped2 = I_Capped;
	 }


	 if (iNum - I_LowerLimit != 0)
	 {
		 I_LowerLimit2 = iNum + 1;
	 }
	 else
	 {
		 I_LowerLimit2 = I_LowerLimit;
	 }

	 if (jNum - J_Capped > 0)
	 {
		 J_Capped2 = jNum - 1;
	 }
	 else
	 {
		 J_Capped2 = J_Capped;
	 }

	 if (jNum - J_LowerLimit != 0)
	 {
		 J_LowerLimit2 = jNum + 1;
	 }
	 else
	 {
		 J_LowerLimit2 = J_LowerLimit;
	 }

	 for (int i = I_Capped2; i <= I_LowerLimit2; i++)
	 {
		 for (int j = J_Capped2; j <= J_LowerLimit2; j++)
		 {

			 if (iNum - 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum == i && jNum == j)
			 {
				 continue;
			 }

			 if (iNum - 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }

			 if (PixelMatrix[i][j].SuperpixelLabel == m) {
				 continue;
			 }

			 if (PixelMatrix[i][j].SuperpixelLabel == -1) {

				 PixelMatrix[i][j].SuperpixelLabel = m;
			 }

			 //cout << i << "  " <<j << "\n";

			 SuperPixelsNum++;
			 ProduceSuperpixelLable(i, j, m, SeedX, SeedY, SuperPixelsNum);

		 }

	 }
	 return;
 }
 int  HQSGTRD::Seed_Point_Generation(int k, Mat img)
 {
	 int K = k;
	 int grid_interval_S;
	 int a;
	 int TempX = 0;
	 int TempY = 0;

	 int HBSign = 0;


	 int N = img.rows * img.cols;

	 double spixel_size = 1.0 * N / K; 

	 grid_interval_S = (int)(sqrt(2 * sqrt(3.0) / 9 * spixel_size) + 0.5); 

	 int r_interval = (int)(1.5 * grid_interval_S + 0.5);
	 int c_interval = (int)(sqrt(3.0) * grid_interval_S + 0.5);

	 int rstrips = img.rows / r_interval + 1; 
	 int cstrips = img.cols / c_interval + 1;
	 int r_off = (img.rows - r_interval * (rstrips - 1)) / 2; 
	 int c_off = (img.cols - c_interval * (cstrips - 1)) / 2;

	 int numseeds = rstrips * cstrips; 
	 a = numseeds;


	 Seed_Point = new SeedPoint_T[a];

	 int dQNum = 0;
	 int idx = 0;

	 for (int i = 0; i < rstrips; i++)
	 {
		 int seed_r = r_off + i * r_interval;
		 int c_move = 0;
		 if (i % 2 != 0)
			 c_move = c_interval / 2;

		 for (int j = 0; j < cstrips; j++)
			 //int j = cstrips-1;
		 {
			 int seed_c = c_move + c_off + j * c_interval;

			 if (seed_r >= img.rows)
				 seed_r = img.rows - 1;
			 if (seed_c >= img.cols)
				 seed_c = img.cols - 1;

			 int x = seed_r;
			 int y = seed_c;

			 //cout << x << " " << y << "\n";

			 if (i == 0 && j == cstrips - 1) {
				 if ((img.cols - y) < c_interval / 2)
					 HBSign = 1;
			 }

			 HBSign = 1;

			 if (HBSign == 1 && i % 2 == 1 && j == cstrips - 1) {
				 continue;
			 }

			 int UpperVertex_x = x - grid_interval_S;
			 int UpperVertex_y = y;

			 int LowerVertex_x = x + grid_interval_S;
			 int LowerVertex_y = y;


			 int LeftVerticesOne_x = x - grid_interval_S / 2;
			 int LeftVerticesOne_y = y - c_interval / 2;

			 int LeftVerticesTwo_x = x + grid_interval_S / 2;
			 int LeftVerticesTwo_y = y - c_interval / 2;


			 int RightVerticesOne_x = x - grid_interval_S / 2;
			 int RightVerticesOne_y = y + c_interval / 2;


			 int RightVerticesTwo_x = x + grid_interval_S / 2;
			 int RightVerticesTwo_y = y + c_interval / 2;

			 Seed_Point[idx].x = x;
			 Seed_Point[idx].y = y;

			 if (x - grid_interval_S <= 0) {

				 Seed_Point[idx].UpperVertex_x = 0;
			 }
			 else
			 {
				 Seed_Point[idx].UpperVertex_x = x - grid_interval_S;
			 }

			 Seed_Point[idx].UpperVertex_y = y;


			 if (x + grid_interval_S >= img.rows)
			 {
				 Seed_Point[idx].LowerVertex_x = img.rows - 1;
			 }
			 else
			 {
				 Seed_Point[idx].LowerVertex_x = x + grid_interval_S;
			 }

			 Seed_Point[idx].LowerVertex_y = y;

			 if (x - grid_interval_S / 2 <= 0)
			 {
				 Seed_Point[idx].LeftVerticesOne_x = 0;
			 }
			 else {
				 Seed_Point[idx].LeftVerticesOne_x = x - grid_interval_S / 2;
			 }
			 if (y - c_interval / 2 <= 0)
			 {
				 Seed_Point[idx].LeftVerticesOne_y = 0;
			 }
			 else {
				 Seed_Point[idx].LeftVerticesOne_y = y - c_interval / 2;
			 }

			 if (x + grid_interval_S / 2 >= img.rows)
			 {
				 Seed_Point[idx].LeftVerticesTwo_x = img.rows - 1;
			 }
			 else {
				 Seed_Point[idx].LeftVerticesTwo_x = x + grid_interval_S / 2;
			 }

			 if (y - c_interval / 2 <= 0)
			 {
				 Seed_Point[idx].LeftVerticesTwo_y = 0;
			 }
			 else {
				 Seed_Point[idx].LeftVerticesTwo_y = y - c_interval / 2;
			 }


			 if (x - grid_interval_S / 2 <= 0)
			 {
				 Seed_Point[idx].RightVerticesOne_x = 0;
			 }
			 else {
				 Seed_Point[idx].RightVerticesOne_x = x - grid_interval_S / 2;
			 }

			 if (y + c_interval / 2 >= img.cols)
			 {
				 Seed_Point[idx].RightVerticesOne_y = img.cols - 1;
			 }
			 else {
				 Seed_Point[idx].RightVerticesOne_y = y + c_interval / 2;
			 }

			 if (x + grid_interval_S / 2 >= img.rows)
			 {
				 Seed_Point[idx].RightVerticesTwo_x = img.rows - 1;
			 }
			 else {
				 Seed_Point[idx].RightVerticesTwo_x = x + grid_interval_S / 2;
			 }


			 if (y + c_interval / 2 >= img.cols)
			 {
				 Seed_Point[idx].RightVerticesTwo_y = img.cols - 1;
			 }
			 else {
				 Seed_Point[idx].RightVerticesTwo_y = y + c_interval / 2;
			 }

			 if (j == 0 && i == 0) {
				 Seed_Point[idx].LowerVertex_y = 0;
				 if (Seed_Point[idx].UpperVertex_x < Seed_Point[idx].RightVerticesOne_x) {
					 Seed_Point[idx].RightVerticesOne_x = Seed_Point[idx].UpperVertex_x;
				 }
			 }

			 if (j != 0 && i == 0 && j != cstrips - 1) {

				 if (Seed_Point[idx].UpperVertex_x < Seed_Point[idx].RightVerticesOne_x) {
					 Seed_Point[idx].RightVerticesOne_x = Seed_Point[idx].UpperVertex_x;
				 }

				 if (Seed_Point[idx].UpperVertex_x < Seed_Point[idx].LeftVerticesOne_x) {
					 Seed_Point[idx].LeftVerticesOne_x = Seed_Point[idx].UpperVertex_x;
				 }
			 }

			 if (j != 0 && i == rstrips - 1 && j != cstrips - 1) {

				 if (Seed_Point[idx].LowerVertex_x > Seed_Point[idx].RightVerticesTwo_x) {
					 Seed_Point[idx].RightVerticesTwo_x = Seed_Point[idx].LowerVertex_x;
				 }

				 if (Seed_Point[idx].LowerVertex_x > Seed_Point[idx].LeftVerticesTwo_x) {
					 Seed_Point[idx].LeftVerticesTwo_x = Seed_Point[idx].LowerVertex_x;
				 }
			 }


			 if (j == 0 && i % 2 == 1 && i != 0) {
				 Seed_Point[idx].LeftVerticesOne_y = 0;
				 Seed_Point[idx].LeftVerticesTwo_y = 0;
			 }
			 if (j == 0 && i % 2 == 0 && i != 0) {
				 Seed_Point[idx].UpperVertex_y = 0;
				 Seed_Point[idx].LowerVertex_y = 0;
			 }


			 if (j == cstrips - 1 && HBSign == 1 && i == 0) {
				 Seed_Point[idx].LowerVertex_y = img.cols - 1;
				 if (Seed_Point[idx].UpperVertex_x < Seed_Point[idx].LeftVerticesOne_x) {
					 Seed_Point[idx].LeftVerticesOne_x = Seed_Point[idx].UpperVertex_x;
				 }
			 }

			 if (j == cstrips - 2 && HBSign == 1 && i % 2 == 1) {

				 Seed_Point[idx].RightVerticesOne_y = img.cols - 1;
				 Seed_Point[idx].RightVerticesTwo_y = img.cols - 1;

			 }

			 if (j == cstrips - 1 && HBSign == 1 && i % 2 == 0) {
				 Seed_Point[idx].UpperVertex_y = img.cols - 1;
				 Seed_Point[idx].LowerVertex_y = img.cols - 1;
			 }


			 if (j == 0 && i == rstrips - 1) {

				 if (Seed_Point[idx].LowerVertex_x > Seed_Point[idx].RightVerticesTwo_x) {
					 Seed_Point[idx].RightVerticesTwo_x = Seed_Point[idx].LowerVertex_x;
				 }
			 }

			 if (i == rstrips - 1 && j == cstrips - 1) {
				 if (Seed_Point[idx].LowerVertex_x > Seed_Point[idx].LeftVerticesTwo_x) {
					 Seed_Point[idx].LeftVerticesTwo_x = Seed_Point[idx].LowerVertex_x;
				 }
			 }
			 idx++;

		 }

	 }

	 for (int i = 0; i < idx; i++)
	 {

		 DDALine(Seed_Point[i].UpperVertex_x, Seed_Point[i].UpperVertex_y,Seed_Point[i].LeftVerticesOne_x, Seed_Point[i].LeftVerticesOne_y, img, 1);

		 DDALine(Seed_Point[i].UpperVertex_x, Seed_Point[i].UpperVertex_y,Seed_Point[i].RightVerticesOne_x, Seed_Point[i].RightVerticesOne_y, img, 1);

		 DDALine(Seed_Point[i].LeftVerticesOne_x, Seed_Point[i].LeftVerticesOne_y,Seed_Point[i].LeftVerticesTwo_x, Seed_Point[i].LeftVerticesTwo_y, img, 1);

		 DDALine(Seed_Point[i].LeftVerticesTwo_x, Seed_Point[i].LeftVerticesTwo_y,Seed_Point[i].LowerVertex_x, Seed_Point[i].LowerVertex_y, img, 1);

		 DDALine(Seed_Point[i].RightVerticesOne_x, Seed_Point[i].RightVerticesOne_y,Seed_Point[i].RightVerticesTwo_x, Seed_Point[i].RightVerticesTwo_y, img, 1);

		 DDALine(Seed_Point[i].RightVerticesTwo_x, Seed_Point[i].RightVerticesTwo_y,Seed_Point[i].LowerVertex_x, Seed_Point[i].LowerVertex_y, img, 1);

		 //cout << i << " " << SEED[i].x << " " << SEED[i].y << "\n";
	 }
	 /*****************************************************************/

	 int Csign = 0;
	 for (int n = 0; n < idx; n++)
		 //	int n = 0;
	 {
		 dQNum = 0;
		 differential_Quotient dQ = new  differentialQuotient[spixel_size*2];
		 for (int i = Seed_Point[n].LeftVerticesOne_x ; i <= Seed_Point[n].LeftVerticesTwo_x ; i++)
		 {
			 for (int j = Seed_Point[n].LeftVerticesOne_y ; j <= Seed_Point[n].RightVerticesOne_y ; j++)
			 {
				 Csign = 0;

				 if ((i - 1 < 0) || (i + 1 > PictureI)) {
					 continue;
				 }

				 if ((j - 1 < 0) || (j + 1 > PictureJ)) {
					 continue;
				 }

				 for (int n = i - 1; n <= i + 1; n++)
				 {
					 for (int m = j - 1; m <= j + 1; m++)
					 {
						 if (PixelMatrix[n][m].ClassSign !=
							 PixelMatrix[i][j].ClassSign) {
							 Csign++;
						 }

					 }
				 }
				 if (Csign != 0)
				 {
					 continue;
				 }
				 dQ[dQNum].x = i;
				 dQ[dQNum].y = j;
				 dQ[dQNum].dQNum = (float)sqrt((i - Seed_Point[n].x) * (i - Seed_Point[n].x) + (j - Seed_Point[n].y) * (j - Seed_Point[n].y));
				 dQNum++;
			 }

		 }

		 float minNum;
		 int   minNumL = 0;
		 for (int i = 1; i < dQNum; i++)
		 {
			 if (dQ[minNumL].dQNum > dQ[i].dQNum) {
				 minNumL = i;
			 }

		 }

		 if (dQNum != 0) {

			 Seed_Point[n].Modified_x = dQ[minNumL].x;
			 Seed_Point[n].Modified_y = dQ[minNumL].y;
			 Seed_label[dQ[minNumL].x][dQ[minNumL].y].SeedSign = n;

		 }
		 else
		 {
			 dQNum = 0;
			 for (int i = Seed_Point[n].LeftVerticesOne_x + 0; i <= Seed_Point[n].LeftVerticesTwo_x - 0; i++)
			 {
				 for (int j = Seed_Point[n].LeftVerticesOne_y + 0; j <= Seed_Point[n].RightVerticesOne_y - 0; j++)			 
				 {
					 if ((i - 1 < 0) || (i + 1 > PictureI)) {
						 continue;
					 }

					 if ((j - 1 < 0) || (j + 1 > PictureJ)) {
						 continue;
					 }

					 float A1, A1L, A1A, A1B,
						 A2, A2L, A2A, A2B,
						 A3, A3L, A3A, A3B,
						 A4, A4L, A4A, A4B;

					 float  L = 0.0, A = 0.0, B = 0.0;

					 A1L = (((float)PixelMatrix[i - 1][j - 1].L - (float)PixelMatrix[i][j].L) -
						 ((float)PixelMatrix[i][j].L - (float)PixelMatrix[i + 1][j + 1].L)) / 2.0;

					 A1A = (((float)PixelMatrix[i - 1][j - 1].a - (float)PixelMatrix[i][j].a) -
						 ((float)PixelMatrix[i][j].a - (float)PixelMatrix[i + 1][j + 1].a)) / 2.0;

					 A1B = (((float)PixelMatrix[i - 1][j - 1].b - (float)PixelMatrix[i][j].b) -
						 ((float)PixelMatrix[i][j].b - (float)PixelMatrix[i + 1][j + 1].b)) / 2.0;


					 A1 = (abs(A1L) + abs(A1A) + abs(A1B)) / 3.0;



					 A2L = (((float)PixelMatrix[i - 1][j].L - (float)PixelMatrix[i][j].L) -
						 ((float)PixelMatrix[i][j].L - (float)PixelMatrix[i + 1][j].L)) / 2.0;
					 A2A = (((float)PixelMatrix[i - 1][j].a - (float)PixelMatrix[i][j].a) -
						 ((float)PixelMatrix[i][j].a - (float)PixelMatrix[i + 1][j].a)) / 2.0;
					 A2B = (((float)PixelMatrix[i - 1][j].b - (float)PixelMatrix[i][j].b) -
						 ((float)PixelMatrix[i][j].b - (float)PixelMatrix[i + 1][j].b)) / 2.0;

					 A2 = (abs(A2L) + abs(A2A) + abs(A2B)) / 3.0;



					 A3L = (((float)PixelMatrix[i - 1][j + 1].L - (float)PixelMatrix[i][j].L) -
						 ((float)PixelMatrix[i][j].L - (float)PixelMatrix[i + 1][j - 1].L)) / 2.0;

					 A3A = (((float)PixelMatrix[i - 1][j + 1].a - (float)PixelMatrix[i][j].a) -
						 ((float)PixelMatrix[i][j].a - (float)PixelMatrix[i + 1][j - 1].a)) / 2.0;

					 A3B = (((float)PixelMatrix[i - 1][j + 1].b - (float)PixelMatrix[i][j].b) -
						 ((float)PixelMatrix[i][j].b - (float)PixelMatrix[i + 1][j - 1].b)) / 2.0;

					 A3 = (abs(A3L) + abs(A3A) + abs(A3B)) / 3.0;


					 A4L = (((float)PixelMatrix[i][j - 1].L - (float)PixelMatrix[i][j].L) -
						 ((float)PixelMatrix[i][j].L - (float)PixelMatrix[i][j + 1].L)) / 2.0;

					 A4A = (((float)PixelMatrix[i][j - 1].a - (float)PixelMatrix[i][j].a) -
						 ((float)PixelMatrix[i][j].a - (float)PixelMatrix[i][j + 1].a)) / 2.0;

					 A4B = (((float)PixelMatrix[i][j - 1].b - (float)PixelMatrix[i][j].b) -
						 ((float)PixelMatrix[i][j].b - (float)PixelMatrix[i][j + 1].b)) / 2.0;

					 A4 = (abs(A4L) + abs(A4A) + abs(A4B)) / 3.0;


					 L = (abs(A1L) + abs(A2L) + abs(A3L) + abs(A4L)) / 4.0;
					 A = (abs(A1A) + abs(A2A) + abs(A3A) + abs(A4A)) / 4.0;
					 B = (abs(A1B) + abs(A2B) + abs(A3B) + abs(A4B)) / 4.0;

					 dQ[dQNum].x = i;
					 dQ[dQNum].y = j;

			
					 dQ[dQNum].dQNum = (L + A + B) +(float)sqrt((i - Seed_Point[n].x) * (i - Seed_Point[n].x) + (j - Seed_Point[n].y) * (j - Seed_Point[n].y));
					
					 dQNum++;
				 }
			 }

			 minNumL = 0;
			 for (int i = 1; i < dQNum; i++)
			 {
				 if (dQ[minNumL].dQNum > dQ[i].dQNum) {
					 minNumL = i;
				 }

			 }

			 Seed_Point[n].Modified_x = dQ[minNumL].x;
			 Seed_Point[n].Modified_y = dQ[minNumL].y;
			 Seed_label[dQ[minNumL].x][dQ[minNumL].y].SeedSign = n;

		 }
		 delete[]dQ;
	 }
	 //cout << "\n";
 /************************************************************************************/

 /*画图六边形*/

	 /*for (int i = 0; i < idx; i++)
	 {

		 /* img.at<Vec3b>(SEED[i].x, SEED[i].y)[0] = 0;
		 img.at<Vec3b>(SEED[i].x, SEED[i].y)[1] = 0;
		 img.at<Vec3b>(SEED[i].x, SEED[i].y)[2] = 255;*/


		 /*	img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[0] = DrawcolourRed[i % 255];
		 img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[1] = DrawcolourBlue[i % 255];
		 img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[2] = DrawcolourGreen[i % 255];
		 */


		 /*img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[0] = 0;
		 img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[1] = 0;
		 img.at<Vec3b>(SEED[i].Modified_x, SEED[i].Modified_y)[2] = 0;*/

		 /*	cv::Point start;
			 cv::Point end;


			 //if (SEED[i].UpperVertex_x != 0 && SEED[i].LeftVerticesOne_y != 0)

			 {
				 cv::Point start = cv::Point(SEED[i].UpperVertex_y, SEED[i].UpperVertex_x); //直线起点
				 cv::Point end = cv::Point(SEED[i].LeftVerticesOne_y, SEED[i].LeftVerticesOne_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }

			 //	if (SEED[i].UpperVertex_x != 0 && SEED[i].RightVerticesOne_y != img.cols - 1)
			 {
				 start = cv::Point(SEED[i].UpperVertex_y, SEED[i].UpperVertex_x); //直线起点
				 end = cv::Point(SEED[i].RightVerticesOne_y, SEED[i].RightVerticesOne_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }


			 //if (SEED[i].LeftVerticesOne_y != 0 && SEED[i].LeftVerticesTwo_y != 0)
			 {
				 start = cv::Point(SEED[i].LeftVerticesOne_y, SEED[i].LeftVerticesOne_x); //直线起点
				 end = cv::Point(SEED[i].LeftVerticesTwo_y, SEED[i].LeftVerticesTwo_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }


			 //if (SEED[i].LowerVertex_x != img.rows - 1 && SEED[i].LeftVerticesTwo_y != 0)
			 {

				 start = cv::Point(SEED[i].LeftVerticesTwo_y, SEED[i].LeftVerticesTwo_x); //直线起点
				 end = cv::Point(SEED[i].LowerVertex_y, SEED[i].LowerVertex_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }

			 //if (SEED[i].RightVerticesOne_y != img.cols - 1 && SEED[i].RightVerticesTwo_y != img.cols - 1)
			 {
				 start = cv::Point(SEED[i].RightVerticesOne_y, SEED[i].RightVerticesOne_x); //直线起点
				 end = cv::Point(SEED[i].RightVerticesTwo_y, SEED[i].RightVerticesTwo_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }

			 //	if (SEED[i].LowerVertex_x != img.rows - 1 && SEED[i].RightVerticesTwo_y != img.cols - 1)
			 {
				 start = cv::Point(SEED[i].RightVerticesTwo_y, SEED[i].RightVerticesTwo_x); //直线起点
				 end = cv::Point(SEED[i].LowerVertex_y, SEED[i].LowerVertex_x);   //直线终点
				 cv::line(img, start, end, cv::Scalar(0, 0, 0));
			 }


		 }
		 cv::imwrite("E:\\liubix44.bmp", img);*/
		 /***************************************************************************************/

	 int AA = 0;
	 int DD = 0;

	 for (int i = 0; i < rstrips; i++)
		 //int i = 0;
	 {
		 AA = 0;
		 DD = 0;

		 for (int n = i; n >= 0; n--)
		 {
			 if (n % 2 == 0) {
				 AA = AA + cstrips - 1;
			 }
			 if (n % 2 == 1)
			 {
				 AA = AA + cstrips - 2;
			 }
		 }
		 AA = AA + i;

		 if (i % 2 == 0) {
			 DD = AA - (cstrips - 1);
		 }
		 else
		 {
			 DD = AA - (cstrips - 2);
		 }

		 if (i % 2 == 1) {
			 for (int m = DD; m <= AA; m++) {


				 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - cstrips].Modified_x, Seed_Point[m - cstrips].Modified_y, img, 2);

				 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - cstrips + 1].Modified_x, Seed_Point[m - cstrips + 1].Modified_y, img, 2);

				 if (m != DD)
				 {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - 1].Modified_x, Seed_Point[m - 1].Modified_y, img, 2);
				 }

				 if (m != AA)
				 {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + 1].Modified_x, Seed_Point[m + 1].Modified_y, img, 2);
				 }

				 if (i != rstrips - 1)
				 {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + cstrips - 1].Modified_x, Seed_Point[m + cstrips - 1].Modified_y, img, 2);
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + cstrips].Modified_x, Seed_Point[m + cstrips].Modified_y, img, 2);
				 }

			 }
		 }
		 if (i % 2 == 0 && i != rstrips - 1) {
			 for (int m = DD; m <= AA; m++) {



				 if (i != 0) {
					 if (m != DD)
					 {
						 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - cstrips].Modified_x, Seed_Point[m - cstrips].Modified_y, img, 2);
					 }
					 if (m != AA) {
						 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - cstrips + 1].Modified_x, Seed_Point[m - cstrips + 1].Modified_y, img, 2);
					 }

				 }

				 if (m != DD) {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m - 1].Modified_x, Seed_Point[m - 1].Modified_y, img, 2);
				 }

				 if (m != AA)
				 {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + 1].Modified_x, Seed_Point[m + 1].Modified_y, img, 2);
				 }

				 if (m != DD) {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + cstrips - 1].Modified_x, Seed_Point[m + cstrips - 1].Modified_y, img, 2);
				 }
				 if (m != AA) {
					 DDALine(Seed_Point[m].Modified_x, Seed_Point[m].Modified_y, Seed_Point[m + cstrips].Modified_x, Seed_Point[m + cstrips].Modified_y, img, 2);
				 }


			 }
		 }
		 /*	if (i % 2 == 1) {
				 for (int m = DD; m <= AA; m++) {



					 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - cstrips].Modified_x, SEED[m - cstrips].Modified_y, img);

					 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - cstrips + 1].Modified_x, SEED[m - cstrips + 1].Modified_y, img);

					 if (m != DD)
					 {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - 1].Modified_x, SEED[m - 1].Modified_y, img);
					 }

					 if (m != AA)
					 {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + 1].Modified_x, SEED[m + 1].Modified_y, img);
					 }

					 if (i != rstrips - 1)
					 {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + cstrips - 1].Modified_x, SEED[m + cstrips - 1].Modified_y, img);
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + cstrips].Modified_x, SEED[m + cstrips].Modified_y, img);
					 }

				 }
			 }
			 if (i % 2 == 0 && i != rstrips-1) {
				 for (int m = DD; m <= AA; m++) {

					 if (i != 0) {
						 if (m != DD)
						 {
							 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - cstrips].Modified_x, SEED[m - cstrips].Modified_y, img);
						 }
						 if (m != AA) {
							 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - cstrips + 1].Modified_x, SEED[m - cstrips + 1].Modified_y, img);
						 }

					 }

					 if (m != DD) {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m - 1].Modified_x, SEED[m - 1].Modified_y, img);
					 }

					 if (m != AA)
					 {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + 1].Modified_x, SEED[m + 1].Modified_y, img);
					 }

					 if (m != DD) {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + cstrips - 1].Modified_x, SEED[m + cstrips - 1].Modified_y, img);
					 }
					 if (m != AA) {
						 DDALine(SEED[m].Modified_x, SEED[m].Modified_y, SEED[m + cstrips].Modified_x, SEED[m + cstrips].Modified_y, img);
					 }

				 }
			 }*/
	 }

	 return idx;
 }

 inline void HQSGTRD::DDALine(int x0, int y0, int x1, int y1, Mat img, int sign)
 {

	 if (x1 < x0)
	 {
		 int temp;
		 temp = x1, x1 = x0, x0 = temp;
		 temp = y1, y1 = y0, y0 = temp;
	 }
	 if (y0 == y1)
	 {
		 for (int i = x0; i <= x1; i++)
		 {
			 if (sign == 1) {

				 PixelMatrix[i][y0].Superpixel_Boundary = 1;

			 }//
			 if (sign == 2) {
				 PixelMatrix[i][y0].WalkingDirection = 2;
			 }//

		 }
		 return;
	 }

	 if (x0 == x1)
	 {
		 if (y0 > y1) {

			 swap(y0, y1);
		 }
		 for (int i = y0; i <= y1; i++)
		 {
			 if (sign == 1) {
				 PixelMatrix[x0][i].Superpixel_Boundary = 1;
			 }//
			 if (sign == 2) {
				 PixelMatrix[x0][i].WalkingDirection = 2;
			 }//
		 }

		 return;
	 }


	 int dx = x1 - x0, dy = y1 - y0;
	 float steps_x, steps_y, steps;
	 if (abs(dx) > abs(dy))    //|K|<1
	 {
		 steps = abs(dx);
		 steps_x = 1;
		 steps_y = (float)dy / (float)steps;
	 }
	 else                     //|K|>=1
	 {
		 steps = abs(dy);
		 if (dy < 0)         //K<-1
		 {
			 steps_y = -1;
		 }
		 else
		 {
			 steps_y = 1;
		 }
		 steps_x = (float)dx / (float)steps;
	 }

	 float x = x0, y = y0;

	 if (sign == 1) {

		 PixelMatrix[int(x)][int(y)].Superpixel_Boundary = 1;

	 }//
	 if (sign == 2) {
		 PixelMatrix[int(x)][int(y)].WalkingDirection = 2;
	 }//
	 for (int i = 1; i <= steps; i++)
	 {
		 x += steps_x;
		 y += steps_y;

		 if (sign == 1) {
			 PixelMatrix[(int)(x + 0.5)][(int)(y + 0.5)].Superpixel_Boundary = 1;
		 }
		 if (sign == 2) {
			 PixelMatrix[(int)(x + 0.5)][(int)(y + 0.5)].WalkingDirection = 2;
		 }
	 }

 }

 int  HQSGTRD::Non_Flat_Regions_Generation(string path, int superpixel_num_K, int grid_interval_S, Mat img, int iteration_num, Mat& Finally_labelMat)
 {

	 std::vector<cv::Vec3d> seed_color;
	 std::vector<cv::Vec2i> seed_pos; 

	 cv::Mat labelMat;
	 std::vector<cv::Vec3d> spixel_color;
	 std::vector<cv::Vec2d> spixel_pos;

	 std::vector<int> spixel_size; 

	 labelMat.create(PictureI + 1, PictureJ + 1, CV_32S);

	 MaxL = MaxL - MinL;
	 MaxA = MaxA - MinA;
	 MaxB = MaxB - MinB;
	 maxGNum = maxGNum - minGNum;

	 for (int i = 0; i < img.rows; i++) {
		 for (int j = 0; j < img.cols; j++) {

			 labelMat.at<int>(i, j) = PixelMatrix[i][j].SuperpixelLabel;

			 if (labelMat.at<int>(i, j) == -1) {

				 PixelMatrix[i][j].NormL = double(PixelMatrix[i][j].L - MinL) / double(MaxL);

				 PixelMatrix[i][j].NormA = double(PixelMatrix[i][j].a - MinA) / double(MaxA);

				 PixelMatrix[i][j].NormB = double(PixelMatrix[i][j].b - MinB) / double(MaxB);

				 PixelMatrix[i][j].NormX = double(i) / double(img.rows);

				 PixelMatrix[i][j].NormY = double(j) / double(img.cols);

				 PixelMatrix[i][j].NormG = double(PixelMatrix[i][j].g - minGNum) / double(maxGNum);

			 }
		 }
	 }

	 Iterative_Generation(labelMat, superpixel_num_K, grid_interval_S, iteration_num, img);

	 for (int i = 0; i < img.rows; i++)
	 {
		 for (int j = 0; j < img.cols; j++)
		 {
			 if ((int)img.at<uchar>(i, j) > 5)
			 {
				 labelMat.at<int>(i, j) = -4;		
			 }
		 }
	 }

	 Superpixel_Combination(labelMat);

	

	 //return  0;
	 int final_label = 0;
	
	 for (int i = 0; i <= PictureI; i++) {
		 for (int j = 0; j <= PictureJ; j++) {
			 labelMat.at<int>(i, j) = Superpixel_Label[i][j].Lable;
		 }
	 }

	 int u = 0;
	 final_label = 0;
	 Mat old_labelMat;
	 
	 labelMat.copyTo(old_labelMat);
	 labelMat.setTo(-1);//labelMat元素初始化为-1	

	 vector<cv::Vec2i> vec3(img.rows * img.cols);//记录第③步中每个超像素的像素
	 vector<cv::Vec2i> first_pixel(img.rows * img.cols);//记录每个超像素第一个像素

	 spixel_color.assign(labelMat.rows * labelMat.cols, cv::Vec3d(0, 0, 0)); 
	 spixel_pos.assign(labelMat.rows * labelMat.cols, cv::Vec2d(0, 0));
	 spixel_size.assign(labelMat.rows * labelMat.cols, 0);
	 first_pixel.assign(img.rows * img.cols, cv::Vec2i(0, 0));

	 for (int i = 0; i < img.rows; i++)
	 {
		 for (int j = 0; j < img.cols; j++)
		 {
			 if (labelMat.at<int>(i, j) < 0)
			 {
				 labelMat.at<int>(i, j) = final_label;

				 spixel_size[final_label] += 1;
				 first_pixel[final_label] = cv::Vec2i(i, j);

				 vec3[0] = cv::Vec2i(i, j);
				 int cur_label = old_labelMat.at<int>(i, j);

				 int count = 1;
				 for (int c = 0; c < count; c++)
				 {
					 for (int n = 0; n < 4; n++)
					 {
						 cv::Vec2i np = vec3[c] + cv::Vec2i(N8[n][0], N8[n][1]);
						 if (np[0] >= 0 && np[0] < img.rows && np[1] >= 0 && np[1] < img.cols)
						 {
							 if (labelMat.at<int>(np.val) < 0 && cur_label == old_labelMat.at<int>(np.val))
							 {
								 vec3[count] = np;

								 labelMat.at<int>(np.val) = final_label;

								 count++;
							 }
						 }
					 }
				 }
				 final_label++;
			 }
		 }
	 }

	 for (int i = 0; i < img.rows; i++)
	 {
		 for (int j = 0; j < img.cols; j++)
		 {
			 labelMat.at<int>(i, j) += 1;
		 }
	 }
	 //return 0;
	 labelMat.copyTo(Finally_labelMat);
	delete(Seed_Point);
	return final_label;

 }

 inline void HQSGTRD::Label2Boundary(int* label, Mat img, Mat img_boundary)
 {
	 int rows = img.rows;
	 int cols = img.cols;
	 img.copyTo(img_boundary);

	 Mat istaken = Mat::zeros(img.size(), CV_8U);

	 for (int i = 0; i < rows; i++)
		 for (int j = 0; j < cols; j++)
		 {
			 int np(0);
			 int fij = label[i * cols + j];
			 //cout << fij << endl;
			 bool flag = false;
			 int l = max(0, j - 1);
			 int r = min(cols - 1, j + 1);
			 int u = max(0, i - 1);
			 int b = min(rows - 1, i + 1);
			 for (int ii = u; ii <= b; ii++)
				 for (int jj = l; jj <= r; jj++)
				 {
					 int fn = label[ii * cols + jj];
					 if (0 == istaken.at<uchar>(ii, jj))
					 {
						 if (fij != fn) np++;
					 }
				 }
			 if (np > 1)
			 {
				 istaken.at<uchar>(i, j) = 255;
				 img_boundary.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			 }
		 }
 }

 inline void HQSGTRD::Iterative_Generation(cv::Mat& labelMat, int numseeds, int grid_interval_S, int itr_times, cv::Mat img)
 {
	 double WL = 1.0 / 5.0;
	 double Wa = 1.0 / 5.0;
	 double Wb = 1.0 / 5.0;
	 double Wg = 1.0 / 5.0;
	 double Ws = 1.0 / 5.0;

	 centroid_Cluster  coC = new centroidOfCluster[numseeds];

	 int* spixel_size;
	 int* spixel_size_Old;
	 int* SuperPixelCircle_radius;

	 numseeds = numseeds - 1;

	 for (int k = 0; k < numseeds; k++)
	 {
		 coC[k].cx = Seed_Point[k].Modified_x;
		 coC[k].cy = Seed_Point[k].Modified_y;
		 PixelMatrix[Seed_Point[k].Modified_x][Seed_Point[k].Modified_y].SuperpixelLabel = k;
		 coC[k].cb = PixelMatrix[Seed_Point[k].Modified_x][Seed_Point[k].Modified_y].b;//种子点颜色值
		 coC[k].ca = PixelMatrix[Seed_Point[k].Modified_x][Seed_Point[k].Modified_y].a;//种子点颜色值
		 coC[k].cL = PixelMatrix[Seed_Point[k].Modified_x][Seed_Point[k].Modified_y].L;//种子点颜色值
		 coC[k].cg = PixelMatrix[Seed_Point[k].Modified_x][Seed_Point[k].Modified_y].g;
	 }

	 cv::Mat distvec(PictureI + 1, PictureJ + 1, CV_64F, cv::Scalar(DBL_MAX));//distvec为rows*cols的64为float类型的矩阵

	 int Circle_radius = (int)(2.5 * grid_interval_S + 0.5);//搜索范围的半径

	 int Circle_diameter = 2 * Circle_radius + 1;//搜索范围的直径

	 cv::Mat SolidCircle(Circle_diameter, Circle_diameter, CV_8UC1, cv::Scalar(0));//直径*直径大小的矩阵，初始为0，8位无符号整型

	 cv::circle(SolidCircle, cv::Point2i(Circle_radius, Circle_radius), Circle_radius, cv::Scalar(1), -1);//画圆，圆心（半径，半径），半径为半径

	 cv::Mat labelMat_old;
	 labelMat.copyTo(labelMat_old);

	 cv::Mat labelMat_sign;
	 labelMat.copyTo(labelMat_sign);

	 spixel_size = new int[numseeds + 5];

	 spixel_size_Old = new int[numseeds + 5];

	 SuperPixelCircle_radius = new int[numseeds + 5];

	 Mark_SW  SWi = new MarkMSW[numseeds + 5];

	 double centL = 0;
	 double centA = 0;
	 double centB = 0;
	 double centX = 0;
	 double centY = 0;
	 double centG = 0;

	 for (int itr = 0; itr < itr_times; itr++)
	 {	
		 distvec.setTo(cv::Scalar(DBL_MAX));
		 if (itr >= 0) {
			 for (int k = 0; k < numseeds; k++)
			 {
				 centL = (coC[k].cL - MinL) / double(MaxL);
				 centA = (coC[k].ca - MinA) / double(MaxA);
				 centB = (coC[k].cb - MinB) / double(MaxB);
				 centX = (coC[k].cx) / double(labelMat.rows);
				 centY = (coC[k].cy) / double(labelMat.cols);
				 centG = (coC[k].cg - minGNum) / double(maxGNum);

				 for (int i = 0; i < Circle_diameter; i++)
				 {
					 for (int j = 0; j < Circle_diameter; j++)
					 {
						 int r = (int)coC[k].cx - Circle_radius + i;
						 int c = (int)coC[k].cy - Circle_radius + j;

						 if (SolidCircle.at<uchar>(i, j) == 1 &&
							 r > -1 && r <= PictureI &&
							 c > -1 && c <= PictureJ)
						 {
							 if (labelMat_sign.at<int>(r, c) != -1) {
								 continue;
							 }
							 double dL = (centL - PixelMatrix[r][c].NormL) * (centL - PixelMatrix[r][c].NormL);

							 double da = (centA - PixelMatrix[r][c].NormA) * (centA - PixelMatrix[r][c].NormA);

							 double db = (centB - PixelMatrix[r][c].NormB) * (centB - PixelMatrix[r][c].NormB);

							 double ds = (centX - PixelMatrix[r][c].NormX) * (centX - PixelMatrix[r][c].NormX)
								 + (centY - PixelMatrix[r][c].NormY) * (centY - PixelMatrix[r][c].NormY);

							 double dg1 = centG - PixelMatrix[r][c].NormG;

							 double dg = dg1 * dg1;


							 double dist = WL * dL +
								 Wb * db +
								 Wa * da +
								 Wg * dg +
								 Ws * ds;

							 if (dist < distvec.at<double>(r, c))
							 {

								 distvec.at<double>(r, c) = dist;
								 labelMat.at<int>(r, c) = k;
								 Pixel_Information[r][c].Csign = 1;
								 Pixel_Information[r][c].dL = dL;
								 Pixel_Information[r][c].da = da;
								 Pixel_Information[r][c].db = db;
								 Pixel_Information[r][c].dg = dg;
								 Pixel_Information[r][c].ds = ds;
							 }
						 }
					 }
				 }
			 }
			 for (int m = 0; m < numseeds; m++) {
				 spixel_size[m] = 0;				
			 }

			 Weights_Adjustment(labelMat_sign, labelMat, WL, Wa, Wb, Wg, Ws, coC, numseeds);

			 Update_Seed_Point(labelMat, labelMat_sign, numseeds, coC, spixel_size);

		 }
	 }

	 /*if (Wg > 0.1) {
		 cout << WL << " " << Wa << " " << Wb << " " << Ws << " " << Wg << "\n";
	 }*/
	 //cout << WL << " " << Wa << " " << Wb << " " << Ws << " " << Wg << "\n";

	 delete[]spixel_size_Old;
	 delete[]coC;
	 delete[]spixel_size;
	 delete[]SuperPixelCircle_radius;

 }

 inline void HQSGTRD::Weights_Adjustment(cv::Mat labelMat_sign, cv::Mat Label, double& WL, double& Wa, double& Wb, double& Wg, double& Ws, centroid_Cluster  coC, int numseeds) {

	 double  SWL = 0.0;
	 double  SWa = 0.0;
	 double  SWb = 0.0;
	 double  SWg = 0.0;
	 double  SWs = 0.0;

	 double  nSWL = 0.0;
	 double  nSWa = 0.0;
	 double  nSWb = 0.0;
	 double  nSWg = 0.0;
	 double  nSWs = 0.0;

	 double pixelNum = 0;

	 double pixelNumG = 0;

	 for (int i = 0; i <= PictureI; i++)
	 {
		 for (int j = 0; j <= PictureJ; j++)
		 {

			 if (labelMat_sign.at<int>(i, j) != -1) {
				 continue;
			 }

			 int k = Label.at<int>(i, j);

			 if (k >= 0)
			 {
				 pixelNum++;
		
				 SWL += Pixel_Information[i][j].dL;

				 SWa += Pixel_Information[i][j].da;
			
				 SWb += Pixel_Information[i][j].db;
			
				 SWs += Pixel_Information[i][j].ds;
				
				 SWg += Pixel_Information[i][j].dg;
			 }

		 }
	 }

	 nSWL = SWL / numseeds + 0.001;
	 nSWa = SWa / numseeds + 0.001;
	 nSWb = SWb / numseeds + 0.001;
	 nSWg = SWg / numseeds + 0.001;
	 nSWs = SWs / numseeds + 0.001;

	 double gama = 1.3 / 1.0;
	
	 WL = pow(nSWL, -gama) / (pow(nSWL, -gama) + pow(nSWa, -gama) + pow(nSWb, -gama) + pow(nSWg, -gama) + pow(nSWs, -gama));
	 Wa = pow(nSWa, -gama) / (pow(nSWL, -gama) + pow(nSWa, -gama) + pow(nSWb, -gama) + pow(nSWg, -gama) + pow(nSWs, -gama));
	 Wb = pow(nSWb, -gama) / (pow(nSWL, -gama) + pow(nSWa, -gama) + pow(nSWb, -gama) + pow(nSWg, -gama) + pow(nSWs, -gama));
	 Wg = pow(nSWg, -gama) / (pow(nSWL, -gama) + pow(nSWa, -gama) + pow(nSWb, -gama) + pow(nSWg, -gama) + pow(nSWs, -gama));
	 Ws = pow(nSWs, -gama) / (pow(nSWL, -gama) + pow(nSWa, -gama) + pow(nSWb, -gama) + pow(nSWg, -gama) + pow(nSWs, -gama));

 }
 inline void HQSGTRD::Update_Seed_Point(cv::Mat Label, cv::Mat img_lab, int num_region, centroid_Cluster  coC, int* spixel_size)
 {

	 for (int k = 0; k < num_region; k++)
	 {
		 coC[k].cL = 0.0;
		 coC[k].ca = 0.0;
		 coC[k].cb = 0.0;
		 coC[k].cg = 0.0;
		 coC[k].cx = 0.0;
		 coC[k].cy = 0.0;

	 }

	 for (int i = 0; i < img_lab.rows; i++)
	 {
		 for (int j = 0; j < img_lab.cols; j++)
		 {

			 int k = Label.at<int>(i, j);

			 if (k >= 0)//&& img_lab.at<int>(i, j)== -1)
			 {

				 coC[k].cb += PixelMatrix[i][j].b;//种子点颜色值
				 coC[k].ca += PixelMatrix[i][j].a;//种子点颜色值
				 coC[k].cL += PixelMatrix[i][j].L;//种子点颜色值

				 coC[k].cg += PixelMatrix[i][j].g;

				 coC[k].cx += i;
				 coC[k].cy += j;

				 spixel_size[k]++;

			 }
		 }
	 }

	 for (int k = 0; k < num_region; k++)
	 {

		 coC[k].cb = coC[k].cb / double(spixel_size[k]);
		 coC[k].ca = coC[k].ca / double(spixel_size[k]);
		 coC[k].cL = coC[k].cL / double(spixel_size[k]);

		 coC[k].cg = coC[k].cg / double(spixel_size[k]);

		 coC[k].cx = coC[k].cx / double(spixel_size[k]);
		 coC[k].cy = coC[k].cy / double(spixel_size[k]);

	 }

 }

void  HQSGTRD::Superpixel_Combination(cv::Mat labelMat) {
	 int CurrentSeedNum = 0;

	 cv::Mat old_labelMat;
	 labelMat.copyTo(old_labelMat);
	 labelMat.setTo(-4);

	 vector<cv::Vec2i> vec(labelMat.rows * labelMat.cols); 
	 vector<cv::Vec2i> first_pixel(labelMat.rows * labelMat.cols);

	 int new_label = 0;

	 for (int i = 0; i < labelMat.rows; i++)
	 {
		 for (int j = 0; j < labelMat.cols; j++)
		 {
			 if (old_labelMat.at<int>(i, j) == -4) {
				 continue;
			 }
			 if (labelMat.at<int>(i, j) < 0)
			 {
				 labelMat.at<int>(i, j) = new_label;

				 first_pixel[new_label] = cv::Vec2i(i, j);

				 vec[0] = cv::Vec2i(i, j);
				 int cur_label = old_labelMat.at<int>(i, j);

				 int count = 1;
				 for (int c = 0; c < count; c++)
				 {
					 for (int n = 0; n < 4; n++)
					 {
						 cv::Vec2i np = vec[c] + cv::Vec2i(N8[n][0], N8[n][1]);
						 if (np[0] >= 0 && np[0] < labelMat.rows && np[1] >= 0 && np[1] < labelMat.cols)
						 {
							 if (labelMat.at<int>(np.val) < 0 && cur_label == old_labelMat.at<int>(np.val))
							 {
								 vec[count] = np;
								 labelMat.at<int>(np.val) = new_label;
								 count++;
							 }
						 }
					 }
				 }
				 new_label++;
			 }
		 }
	 }

	 for (int i = 0; i <= PictureI; i++) {
		 for (int j = 0; j <= PictureJ; j++) {
			 Superpixel_Label[i][j].Lable = labelMat.at<int>(i, j);
		 }
	 }

	 CurrentSeedNum = new_label;

	 int CurrentNumberSuperpixels = CurrentSeedNum - 1;

	 Nodem statisticsSuperPixle;

	 statisticsSuperPixle = new NodeM[CurrentSeedNum];

	 priorityQueue NN;

	 number_Statistical_SuperpixelsLable(CurrentSeedNum, statisticsSuperPixle);


	 int NumC = 1;
	 int NumCondition = 0.3 * (SuperPixel_size);
	 int colorZeng = 5;
	 int colorThreshold = 50;

	 CombineSuperpixelsPriority(NumCondition, CurrentNumberSuperpixels, CurrentSeedNum, statisticsSuperPixle);


	 int num = -5;
	 for (int i = 0; i <= PictureI; i++) {
		 for (int j = 0; j <= PictureJ; j++) {

			 if (Superpixel_Label[i][j].Lable == -4) {
				 Superpixel_Label[i][j].Lable = num--;
			 }
			 labelMat.at<int>(i, j) = Superpixel_Label[i][j].Lable;
		 }
	 }


	 cv::Mat old_labelMat1;
	 labelMat.copyTo(old_labelMat1);
	 labelMat.setTo(-6);

	 vector<cv::Vec2i> vec1(labelMat.rows * labelMat.cols); 
	 vector<cv::Vec2i> first_pixel1(labelMat.rows * labelMat.cols);

	 new_label = 0;

	 for (int i = 0; i < labelMat.rows; i++)
	 {
		 for (int j = 0; j < labelMat.cols; j++)
		 {

			 if (labelMat.at<int>(i, j) < 0)
			 {
				 labelMat.at<int>(i, j) = new_label;

				 first_pixel1[new_label] = cv::Vec2i(i, j);

				 vec1[0] = cv::Vec2i(i, j);
				 int cur_label = old_labelMat.at<int>(i, j);

				 int count = 1;
				 for (int c = 0; c < count; c++)
				 {
					 for (int n = 0; n < 4; n++)
					 {
						 cv::Vec2i np = vec1[c] + cv::Vec2i(N8[n][0], N8[n][1]);
						 if (np[0] >= 0 && np[0] < labelMat.rows && np[1] >= 0 && np[1] < labelMat.cols)
						 {
							 if (labelMat.at<int>(np.val) < 0 && cur_label == old_labelMat.at<int>(np.val))
							 {
								 vec1[count] = np;

								 labelMat.at<int>(np.val) = new_label;

								 count++;
							 }
						 }
					 }
				 }
				 new_label++;
			 }
		 }
	 }

	 for (int i = 0; i <= PictureI; i++) {
		 for (int j = 0; j <= PictureJ; j++) {

			 Superpixel_Label[i][j].Lable = labelMat.at<int>(i, j);

		 }
	 }


	 CurrentSeedNum = new_label;

	 CurrentNumberSuperpixels = CurrentSeedNum - 1;

	 Nodem statisticsSuperPixle1;

	 statisticsSuperPixle1 = new NodeM[CurrentSeedNum];

	 number_Statistical_SuperpixelsLable(CurrentSeedNum, statisticsSuperPixle1);

	 CombineSuperpixelsPriorityS(NumCondition, CurrentNumberSuperpixels, CurrentSeedNum, statisticsSuperPixle1);

 }

 inline void HQSGTRD::number_Statistical_SuperpixelsLable(int CNum, Nodem NumberSuperPixle)
 {
	 int arr[] = { -1 };
	 int size = 1;


	 for (int i = 0; i < CNum; i++)
	 {
		 NumberSuperPixle[i].totalNum = 0;
		 NumberSuperPixle[i].sign = -1;
		 NumberSuperPixle[i].old = 0;

		 NumberSuperPixle[i].AX = 0;
		 NumberSuperPixle[i].AY = 0;

		 NumberSuperPixle[i].TotalL = 0;
		 NumberSuperPixle[i].TotalA = 0;
		 NumberSuperPixle[i].TotalB = 0;

		 NumberSuperPixle[i].StandardDeviationR = 0;
		 NumberSuperPixle[i].StandardDeviationG = 0;
		 NumberSuperPixle[i].StandardDeviationB = 0;

		 NumberSuperPixle[i].initial_x = -1;//初始位置的X
		 NumberSuperPixle[i].initial_y = -1;//初始位置的Y
	 }

	 //return;
	 int Tempi = 0;
	 int Tempj = 0;
	 for (int i = 0; i <= PictureI; i++)
	 {
		 for (int j = 0; j <= PictureJ; j++)
		 {
			 if (Superpixel_Label[i][j].Lable == -4) {
				 continue;
			 }

			 NumberSuperPixle[Superpixel_Label[i][j].Lable].totalNum++;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].sign = Superpixel_Label[i][j].Lable;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].initial_x = i;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].initial_y = j;

			 NumberSuperPixle[Superpixel_Label[i][j].Lable].AX += i;
			 //Tempi += i;
			 //	cout << "CESHI  " << i;
			 //cout << NumberSuperPixle[PixelM[i][j].sign].AX << "  ";
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].AY += j;

			 //NumberSuperPixle[MarkM[i][j].Lable].TotalGray = NumberSuperPixle[MarkM[i][j].Lable].TotalGray + PixelMatrixNew[i][j].z;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalL = NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalL + PixelMatrix[i][j].L;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalA = NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalA + PixelMatrix[i][j].a;
			 NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalB = NumberSuperPixle[Superpixel_Label[i][j].Lable].TotalB + PixelMatrix[i][j].b;
		 }
	 }
 }

 inline void HQSGTRD::CombineSuperpixelsPriority(int countNum, int& NumberSuperpixels, int length, Nodem NumberSuperPixle) {

	 priorityQueue pQ;

	 for (int i = 0; i < length; i++)
	 {
		 NodeM tem;
		 tem.totalNum = NumberSuperPixle[i].totalNum;
		 tem.AX = NumberSuperPixle[i].AX;
		 tem.AY = NumberSuperPixle[i].AY;

		 tem.TotalL = NumberSuperPixle[i].TotalL;
		 tem.TotalA = NumberSuperPixle[i].TotalA;
		 tem.TotalB = NumberSuperPixle[i].TotalB;

		 tem.old = NumberSuperPixle[i].old;

		 tem.initial_x = NumberSuperPixle[i].initial_x;
		 tem.initial_y = NumberSuperPixle[i].initial_y;
		 tem.sign = NumberSuperPixle[i].sign;
		 pQ.enQueue(tem);
		 //	cout << num << " ";
	 }

	 int RM = 0;
	 double TempC = 0;

	 int numC = 0;

	 while (true)
	 {
		 NodeM temp = pQ.deQueue();

		 if (temp.totalNum <= 0) {
			 continue;
		 }

		 //cout << NN.getSize() << " ";
		 //cout << temp.totalNum << " "<< temp.sign<<" |||||||| "<<"\n";
		 if (NumberSuperpixels < kNum + 80) {

			 return;
		 }

		 int i = temp.initial_x;

		 int j = temp.initial_y;

		 //cout << NumberSuperPixle[MarkM[i][j].Lable].totalNum <<"   "<< MarkM[i][j].Lable << "\n";

		 if (temp.old != NumberSuperPixle[Superpixel_Label[i][j].Lable].old) {
			 continue;
		 }

		 //cout << NN.getSize() << " ";

		 vector<int> nborhood;
		 RM++;

		 FindNeighborhood(i, j, Superpixel_Label[i][j].Lable, nborhood, RM);

		 float aversgeL = 0;
		 float aversgeA = 0;
		 float aversgeB = 0;
		 float ColorAver = 0;
		 float aversgeX = 0;
		 float aversgeY = 0;

		 aversgeL = temp.TotalL / temp.totalNum;
		 aversgeA = temp.TotalA / temp.totalNum;
		 aversgeB = temp.TotalB / temp.totalNum;

		 aversgeL = (aversgeL - MinL) / double(MaxL);
		 aversgeA = (aversgeA - MinA) / double(MaxA);
		 aversgeB = (aversgeB - MinB) / double(MaxB);



		 aversgeX = temp.AX / temp.totalNum;
		 aversgeY = temp.AY / temp.totalNum;

		 aversgeX = aversgeX / PictureI;
		 aversgeY = aversgeY / PictureJ;

		 float minGL = 1000.0;
		 float minAL = 1000.0;
		 int ASign = 0;

		 int ACSign = 0;
		 float AVer;
		 for (int nb = 0; nb < nborhood.size(); nb++) {

			 if (nborhood[nb] == -4) {
				 continue;
			 }

			 if (NumberSuperPixle[nborhood[nb]].totalNum == 0)
			 {
				 continue;
			 }

			 float aversgeLC = NumberSuperPixle[nborhood[nb]].TotalL / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeAC = NumberSuperPixle[nborhood[nb]].TotalA / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeBC = NumberSuperPixle[nborhood[nb]].TotalB / NumberSuperPixle[nborhood[nb]].totalNum;

			 float aversgeXC = NumberSuperPixle[nborhood[nb]].AX / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeYC = NumberSuperPixle[nborhood[nb]].AY / NumberSuperPixle[nborhood[nb]].totalNum;


			 aversgeLC = (aversgeLC - MinL) / double(MaxL);
			 aversgeAC = (aversgeAC - MinA) / double(MaxA);
			 aversgeBC = (aversgeBC - MinB) / double(MaxB);

			 aversgeXC = aversgeXC / PictureI;
			 aversgeYC = aversgeYC / PictureJ;


			 TempC = sqrt(
				 (aversgeL - aversgeLC) * (aversgeL - aversgeLC) +
				 (aversgeA - aversgeAC) * (aversgeA - aversgeAC) +
				 (aversgeB - aversgeBC) * (aversgeB - aversgeBC));

			 float AD = sqrt((aversgeX - aversgeXC) * (aversgeX - aversgeXC) + (aversgeY - aversgeYC) * (aversgeY - aversgeYC));

			 AVer = 1 * TempC + AD;

			 ColorAver = TempC;// / 3;


			 if (AVer < minGL) {
				 minGL = AVer;
				 ASign = nborhood[nb];
			 }

		 }


		 NumberSuperpixels--;

		 NumberSuperPixle[ASign].totalNum += temp.totalNum;

		 NumberSuperPixle[ASign].AX += temp.AX;
		 NumberSuperPixle[ASign].AY += temp.AY;

		 NumberSuperPixle[ASign].TotalL += temp.TotalL;
		 NumberSuperPixle[ASign].TotalA += temp.TotalA;
		 NumberSuperPixle[ASign].TotalB += temp.TotalB;

		 NumberSuperPixle[Superpixel_Label[i][j].Lable].totalNum = 0;


		 int Psign = Superpixel_Label[i][j].Lable;
		 Superpixel_Label[i][j].Lable = ASign;

		 mergeSuperpixelsLable(i, j, Psign, ASign);
		 NumberSuperPixle[ASign].initial_x = i;
		 NumberSuperPixle[ASign].initial_y = j;

		 NumberSuperPixle[ASign].old++;

		 NodeM tem;
		 tem.totalNum = NumberSuperPixle[ASign].totalNum;
		 tem.AX = NumberSuperPixle[ASign].AX;
		 tem.AY = NumberSuperPixle[ASign].AY;

		 tem.TotalL = NumberSuperPixle[ASign].TotalL;
		 tem.TotalA = NumberSuperPixle[ASign].TotalA;
		 tem.TotalB = NumberSuperPixle[ASign].TotalB;

		 tem.old = NumberSuperPixle[ASign].old;

		 tem.initial_x = NumberSuperPixle[ASign].initial_x;
		 tem.initial_y = NumberSuperPixle[ASign].initial_y;

		 tem.sign = ASign;

		 pQ.enQueue(tem);

		 vector<int>().swap(nborhood);
	 }

 }

 inline void HQSGTRD::CombineSuperpixelsPriorityS(int countNum, int& NumberSuperpixels, int length, Nodem NumberSuperPixle) {

	 priorityQueue pQ;

	 for (int i = 0; i < length; i++)
	 {
		 NodeM tem;
		 tem.totalNum = NumberSuperPixle[i].totalNum;
		 tem.AX = NumberSuperPixle[i].AX;
		 tem.AY = NumberSuperPixle[i].AY;

		 tem.TotalL = NumberSuperPixle[i].TotalL;
		 tem.TotalA = NumberSuperPixle[i].TotalA;
		 tem.TotalB = NumberSuperPixle[i].TotalB;

		 tem.old = NumberSuperPixle[i].old;

		 tem.initial_x = NumberSuperPixle[i].initial_x;
		 tem.initial_y = NumberSuperPixle[i].initial_y;
		 tem.sign = NumberSuperPixle[i].sign;
		 pQ.enQueue(tem);
		 //	cout << num << " ";
	 }

	 int RM = 0;
	 double TempC = 0;

	 int numC = 0;

	 while (true)
	 {
		 NodeM temp = pQ.deQueue();

		 if (temp.totalNum <= 0) {
			 continue;
		 }

		 //cout << NN.getSize() << " ";
		 //cout << temp.totalNum << " "<< temp.sign<<" |||||||| "<<"\n";
		 if (NumberSuperpixels < kNum + 15) {

			 return;
		 }

		 int i = temp.initial_x;

		 int j = temp.initial_y;

		 if (temp.old != NumberSuperPixle[Superpixel_Label[i][j].Lable].old) {
			 continue;
		 }

		 //cout << NN.getSize() << " ";

		 vector<int> nborhood;
		 RM++;

		 FindNeighborhood(i, j, Superpixel_Label[i][j].Lable, nborhood, RM);

		 float aversgeL = 0;
		 float aversgeA = 0;
		 float aversgeB = 0;
		 float ColorAver = 0;
		 float aversgeX = 0;
		 float aversgeY = 0;

		 aversgeL = temp.TotalL / temp.totalNum;
		 aversgeA = temp.TotalA / temp.totalNum;
		 aversgeB = temp.TotalB / temp.totalNum;

		 aversgeL = (aversgeL - MinL) / double(MaxL);
		 aversgeA = (aversgeA - MinA) / double(MaxA);
		 aversgeB = (aversgeB - MinB) / double(MaxB);

		 aversgeX = temp.AX / temp.totalNum;
		 aversgeY = temp.AY / temp.totalNum;

		 aversgeX = aversgeX / PictureI;
		 aversgeY = aversgeY / PictureJ;

		 float minGL = 1000.0;
		 float minAL = 1000.0;
		 int ASign = 0;

		 int ACSign = 0;
		 float AVer;
		 for (int nb = 0; nb < nborhood.size(); nb++) {

			 if (nborhood[nb] == -4) {
				 continue;
			 }

			 if (NumberSuperPixle[nborhood[nb]].totalNum == 0)
			 {
				 continue;
			 }

			 float aversgeLC = NumberSuperPixle[nborhood[nb]].TotalL / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeAC = NumberSuperPixle[nborhood[nb]].TotalA / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeBC = NumberSuperPixle[nborhood[nb]].TotalB / NumberSuperPixle[nborhood[nb]].totalNum;

			 float aversgeXC = NumberSuperPixle[nborhood[nb]].AX / NumberSuperPixle[nborhood[nb]].totalNum;
			 float aversgeYC = NumberSuperPixle[nborhood[nb]].AY / NumberSuperPixle[nborhood[nb]].totalNum;

			 aversgeLC = (aversgeLC - MinL) / double(MaxL);
			 aversgeAC = (aversgeAC - MinA) / double(MaxA);
			 aversgeBC = (aversgeBC - MinB) / double(MaxB);

			 aversgeXC = aversgeXC / PictureI;
			 aversgeYC = aversgeYC / PictureJ;

			 TempC = sqrt(
				 (aversgeL - aversgeLC) * (aversgeL - aversgeLC) +
				 (aversgeA - aversgeAC) * (aversgeA - aversgeAC) +
				 (aversgeB - aversgeBC) * (aversgeB - aversgeBC));

			 float AD = sqrt((aversgeX - aversgeXC) * (aversgeX - aversgeXC) + (aversgeY - aversgeYC) * (aversgeY - aversgeYC));

			 AVer = 1 * TempC + AD;

			 ColorAver = TempC;// / 3;


			 if (AVer < minGL) {
				 minGL = AVer;
				 ASign = nborhood[nb];
			 }

			 if (ColorAver < minAL) {
				 minAL = ColorAver;
				 ACSign = nborhood[nb];
			 }
		 }

		 NumberSuperpixels--;

		 NumberSuperPixle[ASign].totalNum += temp.totalNum;

		 NumberSuperPixle[ASign].AX += temp.AX;
		 NumberSuperPixle[ASign].AY += temp.AY;

		 NumberSuperPixle[ASign].TotalL += temp.TotalL;
		 NumberSuperPixle[ASign].TotalA += temp.TotalA;
		 NumberSuperPixle[ASign].TotalB += temp.TotalB;

		 NumberSuperPixle[Superpixel_Label[i][j].Lable].totalNum = 0;


		 int Psign = Superpixel_Label[i][j].Lable;
		 Superpixel_Label[i][j].Lable = ASign;

		 mergeSuperpixelsLable(i, j, Psign, ASign);
		 NumberSuperPixle[ASign].initial_x = i;
		 NumberSuperPixle[ASign].initial_y = j;

		 NumberSuperPixle[ASign].old++;

		 NodeM tem;
		 tem.totalNum = NumberSuperPixle[ASign].totalNum;
		 tem.AX = NumberSuperPixle[ASign].AX;
		 tem.AY = NumberSuperPixle[ASign].AY;

		 tem.TotalL = NumberSuperPixle[ASign].TotalL;
		 tem.TotalA = NumberSuperPixle[ASign].TotalA;
		 tem.TotalB = NumberSuperPixle[ASign].TotalB;

		 tem.old = NumberSuperPixle[ASign].old;

		 tem.initial_x = NumberSuperPixle[ASign].initial_x;
		 tem.initial_y = NumberSuperPixle[ASign].initial_y;

		 tem.sign = ASign;

		 pQ.enQueue(tem);

		 vector<int>().swap(nborhood);
	 }

 }

 inline void HQSGTRD::FindNeighborhood(int x, int y, int lable, vector<int>& nborhood, int RM)
 {

	 if (Superpixel_Label[x][y].Lable != lable) {

		 vector<int>::iterator it = find(nborhood.begin(), nborhood.end(), Superpixel_Label[x][y].Lable);

		 if (it == nborhood.end()) {
			 nborhood.push_back(Superpixel_Label[x][y].Lable);
		 }

		 return;
	 }

	 int I_Capped2 = 0;
	 int I_LowerLimit2 = 0;

	 int J_Capped2 = 0;
	 int J_LowerLimit2 = 0;

	 int I_Capped = 0;
	 int I_LowerLimit = PictureI;

	 int J_Capped = 0;
	 int J_LowerLimit = PictureJ;

	 int iNum = x;
	 int jNum = y;

	 //cout << x << " " << y<<"\n";
	 if (iNum - I_Capped > 0) {
		 I_Capped2 = iNum - 1;
	 }
	 else
	 {
		 I_Capped2 = I_Capped;
	 }


	 if (iNum - I_LowerLimit != 0)
	 {
		 I_LowerLimit2 = iNum + 1;
	 }
	 else
	 {
		 I_LowerLimit2 = I_LowerLimit;
	 }

	 if (jNum - J_Capped > 0)
	 {
		 J_Capped2 = jNum - 1;
	 }
	 else
	 {
		 J_Capped2 = J_Capped;
	 }

	 if (jNum - J_LowerLimit != 0)
	 {
		 J_LowerLimit2 = jNum + 1;
	 }
	 else
	 {
		 J_LowerLimit2 = J_LowerLimit;
	 }


	 for (int i = I_Capped2; i <= I_LowerLimit2; i++)
	 {
		 for (int j = J_Capped2; j <= J_LowerLimit2; j++)
		 {

			 if (iNum - 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum == i && jNum == j)
			 {
				 continue;
			 }

			 if (iNum - 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }

			 if (Superpixel_Label[i][j].Fsign == RM) {
				 continue;
			 }

			 if (Superpixel_Label[i][j].Fsign != RM) {

				 Superpixel_Label[i][j].Fsign = RM;
			 }

			 FindNeighborhood(i, j, lable, nborhood, RM);
		 }

	 }
	 return;
 }
 inline void  HQSGTRD::mergeSuperpixelsLable(int x, int y, int PreLabel, int AfterLabel)
 {

	 if (PreLabel == AfterLabel)return;
	 int I_Capped2 = 0;
	 int I_LowerLimit2 = 0;

	 int J_Capped2 = 0;
	 int J_LowerLimit2 = 0;

	 int I_Capped = 0;
	 int I_LowerLimit = PictureI;

	 int J_Capped = 0;
	 int J_LowerLimit = PictureJ;

	 int iNum = x;
	 int jNum = y;


	 if (iNum - I_Capped > 0) {
		 I_Capped2 = iNum - 1;
	 }
	 else
	 {
		 I_Capped2 = I_Capped;
	 }


	 if (iNum - I_LowerLimit != 0)
	 {
		 I_LowerLimit2 = iNum + 1;
	 }
	 else
	 {
		 I_LowerLimit2 = I_LowerLimit;
	 }

	 if (jNum - J_Capped > 0)
	 {
		 J_Capped2 = jNum - 1;
	 }
	 else
	 {
		 J_Capped2 = J_Capped;
	 }

	 if (jNum - J_LowerLimit != 0)
	 {
		 J_LowerLimit2 = jNum + 1;
	 }
	 else
	 {
		 J_LowerLimit2 = J_LowerLimit;
	 }

	 for (int i = I_Capped2; i <= I_LowerLimit2; i++)
	 {
		 for (int j = J_Capped2; j <= J_LowerLimit2; j++)
		 {
			 if (iNum == i && jNum == j)
			 {
				 continue;
			 }

			 if (iNum - 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum - 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum - 1 == j)
			 {
				 continue;
			 }

			 if (iNum + 1 == i && jNum + 1 == j)
			 {
				 continue;
			 }
			 if (PreLabel == Superpixel_Label[i][j].Lable) {			
				 Superpixel_Label[i][j].Lable = AfterLabel;
				 mergeSuperpixelsLable(i, j, PreLabel, AfterLabel);
			 }
		 }

	 }
	 return;
 }
 void HQSGTRD::Save_result(Mat Finally_labelMat, string file_name, string Output_folder,int final_label,int Save_Image,int Save_Label_file) {

	 string SkNum = std::to_string(kNum);
	

	 if (Save_Image == 1) {
		 Mat img = imread(FileName, 1);
		 Label2Boundary((int*)Finally_labelMat.data, img, img);
		 string fileName = Output_folder + "\\" + file_name + "_" + SkNum + ".png";
		 cv::imwrite(fileName, img);
	 }
	 

	 if (Save_Label_file == 1) {
		
		 string TxtName = file_name + "_" + SkNum + "_HQSGRD.txt";

		 //TxtName = TxtName + "_SASS.txt";

		 TxtName = Output_folder + "\\" + TxtName;
		 ofstream out(TxtName);

		 for (int i = 0; i < Finally_labelMat.rows; i++)
		 {
			 for (int j = 0; j < Finally_labelMat.cols; j++)
			 {
				 out << Finally_labelMat.at<int>(i, j) << "\t";
			 }
			 out << std::endl;
		 }
		 out.close();
	 }

	
 }
