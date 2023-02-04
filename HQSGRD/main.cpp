#include<iostream>
#include <vector>
#include <io.h>
#include "Superpixel_Generation.h"
#include<time.h>
using namespace std;


void getAllFiles(std::string path, std::vector<std::string>& files, std::string fileType)
{
	long long hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + fileType).c_str(), &fileinfo)) != -1) {
		do {
			files.push_back(p.assign(path).append("\\").append(fileinfo.name));
		} while (_findnext(hFile, &fileinfo) == 0); 
		_findclose(hFile);
	}
}

int main() {

	vector<string> image_file_name;
	vector<double> run_time;
	
	string Input_folder = "F:\\BSD\\";

	string Output_folder = "E:\\Result";//The location where the superpixel result is saved.

	getAllFiles(Input_folder, image_file_name, ".jpg");//Enter the folder where the image is located.
	
	string folder_edge = Input_folder;//Enter the folder where the RCF result of the image is located.
	

	
	int iteration_num = 4;

	int Superpixel_Num = 300;//
   
	int Save_Image = 1;
	
	int Save_Label_file = 0;
	
	for (int Superpixel_Num = 100; Superpixel_Num <= 1000; Superpixel_Num += 100)
	{

		
		for (int i = 0; i < image_file_name.size(); i++)
		{
			string temp_file_name = image_file_name[i];
			string tempName(temp_file_name.substr(temp_file_name.find_last_of("\\\\") + 1, temp_file_name.length() - temp_file_name.find_last_of("\\\\")));

			string file_name(tempName.substr(0, tempName.find_last_of(".")));

			string RCF_file_name = folder_edge + file_name + ".png";

			cout << image_file_name[i] << "  " << RCF_file_name << "\n";
			HQSGTRD sp;
			sp.Superpixel_Segmentation(image_file_name[i], RCF_file_name, Superpixel_Num, file_name, Output_folder, run_time, iteration_num,  Save_Image, Save_Label_file);
			cout << "\n";
		}
		double time = 0;
		for (int i = 0; i < run_time.size(); i++)
		{
			time += run_time[i];
		}
		time = time / run_time.size();
		cout << "The running time of the algorithm is:  " << time << "\n";
		cout << "\n";
	}
	return 0;
}