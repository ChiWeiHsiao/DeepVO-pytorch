# Readme
- This is an unofficial PyTorch implementation of ICRA 2017 paper [DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks](https://ieeexplore.ieee.org/document/7989236/)
- Model
    - ![](https://imgur.com/vo0vXgk.png)

## Usage
- Download KITTI data and our pretrained model
	- This shell ```KITTI/downloader.sh``` can be used to download the KITTI images and pretrained model
		- the shell will only keep the left camera color images (image_03 folder) and delete other data
		- the downloaded images will be placed at ```KITTI/images/00/```, ```KITTI/images/01```, ...
		- the images offered by KITTI is already rectified
		- the direct [download link](https://drive.google.com/file/d/1l0s3rYWgN8bL0Fyofee8IhN-0knxJF22/view) of pretrained model
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
- Run ```test.py``` to output predicted pose
	- output to ```result/```
	- file name will be like ``out_00.txt``
- Run ```visualize.py``` to visualize the prediction of route
- Other files:
	- ```model.py```: model is defined here
	- ```data_helper.py```: customized PyTorch dataset and sampler
		- the input images is loaded batch by batch
		
## Download trained model
Provided by [alexart13](https://github.com/alexart13).
- [trained model](https://drive.google.com/file/d/1l0s3rYWgN8bL0Fyofee8IhN-0knxJF22/view)
- [optimizer](https://drive.google.com/file/d/1JlVJwEZy4W4EmgtTCNWmM4YAACUHxnr2/view)

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
<table border=1>
<tr>
<td>
<img src="https://user-images.githubusercontent.com/32840403/51480655-96989400-1da2-11e9-9cd7-c0618ed3ff6d.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480656-96989400-1da2-11e9-99d4-b55546e89d54.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480645-95676700-1da2-11e9-842e-ff20a243013d.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480647-95676700-1da2-11e9-95ac-d548d3aec24c.png" width="24%"/>
</td>
</tr>

<tr>
<td>
<img src="https://user-images.githubusercontent.com/32840403/51480651-95fffd80-1da2-11e9-8b37-4cebfa7358e6.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480652-96989400-1da2-11e9-9096-7c93de6109da.png" width="24%"/>
</td>
</tr>
</table>

- Testing Sequence
<table border=1>
<tr>
<td>
<img src="https://user-images.githubusercontent.com/32840403/51480646-95676700-1da2-11e9-9723-ad8e07aa4706.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480648-95fffd80-1da2-11e9-917e-f9e67c1f1500.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480650-95fffd80-1da2-11e9-8c02-96aa37beab94.png" width="24%"/>
<img src="https://user-images.githubusercontent.com/32840403/51480654-96989400-1da2-11e9-8202-ddedae1a9a7a.png" width="24%"/>
</td>
</tr>
</table>

## Acknowledgments
- Thanks [alexart13](https://github.com/alexart13) for providing the trained model and the correct code to process ground truth rotation.

## References
- [paper](https://ieeexplore.ieee.org/document/7989236/)
    - Sen Wang, Ronald Clark, Hongkai Wen, Niki Trigoni
    - ICRA 2017
      ```
      @inproceedings{wang2017deepvo,
	  title={Deepvo: Towards end-to-end visual odometry with deep recurrent convolutional neural networks},
	  author={Wang, Sen and Clark, Ronald and Wen, Hongkai and Trigoni, Niki},
	  booktitle={Robotics and Automation (ICRA), 2017 IEEE International Conference on},
	  pages={2043--2050},
	  year={2017},
	  organization={IEEE}
	  }
      ```
