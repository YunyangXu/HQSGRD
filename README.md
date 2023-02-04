# HQSGRD
The official implementation of “High quality superpixel generation through regional decomposition”.<br>
Paper link: https://ieeexplore.ieee.org/document/9927181 <br>
The main requirements are opencv (4.7.0) and Microsoft Visual Studio 2022.


# Get started
1. Download the datasets (Paper_daset.zip) from this [google drive link]().
2. git clone https://github.com/YunyangXu/HQSGRD. 
3. Install Microsoft Visual Studio 2022 and opencv (4.7.0).
4. Open the project file "HQSGRD. sln".
    
# How to test
1. Unzip the downloaded folder Paper_daset.zip. This folder contains the three data sets used in the paper (BSD, FashImages and PASCAL-S). Each folder contains the original image and the results of its RCF method.
2. Set the path of Input_folder and Output_folder in main.cpp.
3. When Save_Image == 1, the visual result of the superpixel is saved. When Save_Label_file == 1, save the split mark file of the super pixel.

# How to use your own data
1. The RCF method is used to obtain the edge detection results of the input image and perform non-maximum suppression. [RCF_code](https://github.com/yun-liu/RCF)
2. Put the result of non-maximum suppression and the original image into Input_folder.  
3. Run the algorithm.

# Citation

@article{xu2022high,
  title={High quality superpixel generation through regional decomposition},
  author={Xu, Yunyang and Gao, Xifeng and Zhang, Caiming and Tan, Jianchao and Li, Xuemei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
