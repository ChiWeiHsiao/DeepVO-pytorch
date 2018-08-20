# Readme
- This is the PyTorch implementation of ICRA 2017 paper [DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7989236/)
- Model
    - ![](https://imgur.com/vo0vXgk.png)

## Usage
- Download KITTI data and our pretrained model
	- This shell ```KITTI/downloader.sh``` can be used to download the KITTI images and pretrained model
		- the shell will only keep the left camera color images (image_03 folder) and delete other data
		- the downloaded images will be placed at ```KITTI/images/00/```, ```KITTI/images/01```, ...
		- the images offered by KITTI is already rectified
		- the direct [download link](https://www.polybox.ethz.ch/index.php/s/90OlHg6KWBzG6gR) of pretrained model
	- Download the ground truth pose from [KITTI Visual Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
		- you need to enter your email to request the pose data [here](http://www.cvlibs.net/download.php?file=data_odometry_poses.zip)
		- and place the ground truth pose at ```KITTI/pose_GT/```
- Run 'preprocess.py' to 
    - remove unused images based on the readme file in KITTI devkit
    - convert the ground truth poses from KITTI (12 floats [R|t]) into 6 floats (euler angle + translation)
    - and save the transformed ground truth pose into ```.npy``` file
- Pretrained weight of FlowNet ( CNN part ) can be downloaded [here](https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM)
	- note that this pretrained FlowNet model assumes that RGB value range is [-0.5, 0.5]
	- the code of CNN layers is modified from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
- Specify the paths and changes hyperparameters in ```params.py```
	- If your computational resource is limited, please be careful with the following arguments:
	- ```batch_size```: choose batch size depends on your GPU memory
	- ```img_w```, ```img_h```: downsample the images to fit to the GPU memory
	- ```pin_mem```: accelerate the data excahnge between GPU and memory, if your RAM is not large enough, please set to False
- Run ```main.py``` to train the model
	- the trained model and optimizer will be saved in ```models/```
	- the records will be saved in ```records/```
- The trained weight can be downloaded [here](https://drive.google.com/drive/folders/1Zb6wObjdZ2lvhM07pgvJGhoZgJKFJMOa?usp=sharing)
- Run ```test.py``` to output predicted pose
	- output to ```result/```
	- file name will be like ``out_00.txt``
- Run ```visualize.py``` to visualize the prediction of route
- Other files:
	- ```model.py```: model is defined here
	- ```data_helper.py```: customized PyTorch dataset and sampler
		- the input images is loaded batch by batch
	
	
## Required packages
- pytorch 0.4.0
- torchvision 0.2.1
- numpy
- pandas
- pillow
- matplotlib
- glob


## Result
- Training Sequences
	- <img src="https://i.imgur.com/HQKW42J.png" width="200" height="200"> <img src="https://i.imgur.com/LQj8T8G.png" width="200" height="200"> <img src="https://i.imgur.com/I2Y35Pl.png" width="200" height="200"> <img src="https://i.imgur.com/XgVLtjN.png" width="200" height="200"> <img src="https://i.imgur.com/uuFUbdz.png" width="200" height="200"> 
- Testing Sequence
	- <img src="https://i.imgur.com/eAfHI6N.png" width="200" height="200"> <img src="https://i.imgur.com/LgNSvjB.png" width="200" height="200"><img src="https://i.imgur.com/6OnEjci.png" width="200" height="200"> <img src="https://i.imgur.com/VlL6LH9.png" width="200" height="200">


