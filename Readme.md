# Readme
- This is the PyTorch implementation of ICRA 2017 paper [DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7989236/)
## Usage
- Download [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) data
	- This shell ```KITTI/downloader.sh``` can be used to download the KITTI images
		- the shell will only keep the left camera color images (image_03 folder) and delete other data
		- the downloaded images will be placed at ```KITTI/images/00/```, ```KITTI/images/01```, ...
		- the images offered by KITTI is already rectified
	- Download the ground truth pose from [KITTI Visual Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
		- direct link is: [odometry ground truth poses (4 MB)](http://www.cvlibs.net/download.php?file=data_odometry_poses.zip)
		- and place the ground truth pose at ```KITTI/pose_GT/```
- Run 'preprocess.py' to 
    - remove unused images based on the readme file in KITTI devkit
    - convert the ground truth poses from KITTI (12 floats [R|t]) into 6 floats (euler angle + translation)
    - and save the transformed ground truth pose into ```.npy``` file
- Pretrained weight of FlowNet ( CNN part ) can be downloaded [here](https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM)
	- the code of CNN layers is modified from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
- Specify the paths and changes hyperparameters in ```params.py```
	- If your computational resource is limited, please be careful with the following arguments:
	- ```batch_size```: choose batch size depends on your GPU memory
	- ```img_w```, ```img_h```: downsample the images to fit to the GPU memory
	- ```pin_mem```: accelerate the data excahnge between GPU and memory, if your RAM is not large enough, please set to False
- Run ```main.py``` to train the model
	- the trained model and optimizer will be saved in ```models/```
	- the records will be saved in ```records/```
- Run ```test.py``` to output predicted pose
	- output to ```result/```
	- file name will be like ``out_00.txt``
- Run ```visualize.py``` to visualize the prediction route path
- Other files:
	- ```model.py```: model is defined here
	- ```data_helper.py```: customized PyTorch dataset and sampler
		- the input images is loaded batch by batch
	


## Result
- Training Sequences
<img src="https://i.imgur.com/P47OOtQ.png" width="200" height="200"> <img src="https://i.imgur.com/yDjWpZ3.png" width="200" height="200">
- Testing Sequence
<img src="https://i.imgur.com/hdTlQkD.png" width="200" height="200"> <img src="https://i.imgur.com/VTAN321.png" width="200" height="200">