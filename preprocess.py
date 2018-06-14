import os
import glob
import numpy as np
import time
from helper import R_to_angle
from params import par
from torchvision import transforms

def clean_unused_images():
	seq_frame = {'00': ['000', '004540'],
				'01': ['000', '001100'],
				'02': ['000', '004660'],
				'03': ['000', '000800'],
				'04': ['000', '000270'],
				'05': ['000', '002760'],
				'06': ['000', '001100'],
				'07': ['000', '001100'],
				'08': ['001100', '005170'],
				'09': ['000', '001590'],
				'10': ['000', '001200']
				}
	for dir_id, img_ids in seq_frame.items():
		dir_path = '{}{}/'.format(par.image_dir, dir_id)
		if not os.path.exists(dir_path):
			continue

		print('Cleaning {} directory'.format(dir_id))
		start, end = img_ids
		start, end = int(start), int(end)
		for idx in range(0, start):
			img_name = '{:010d}.png'.format(idx)
			img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)
		for idx in range(end+1, 10000):
			img_name = '{:010d}.png'.format(idx)
			img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)


# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
	info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
	start_t = time.time()
	for video in info.keys():
		fn = '{}{}.txt'.format(par.pose_dir, video)
		print('Transforming {}...'.format(fn))
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]  # list of pose (pose=list of 12 floats)
			poses = np.array(poses)
			np.save(fn.split('.')[0]+'.npy', poses)
			print('Video {}: shape={}'.format(video, poses.shape))
	print('elapsed time = {}'.format(time.time()-start_t))


def calculate_rgb_mean(image_path_list):
	n_images = len(image_path_list)
	print('Numbers of frames in training dataset: {}'.format(n_images))

	mean_np = [0, 0, 0]
	mean_tensor = [0, 0, 0]
	to_tensor = transforms.ToTensor()

	image_sequence = []
	for idx, img_path in enumerate(fpaths):
		print('{} / {}'.format(idx, n_images), end='\r')
		img_as_img = Image.open(img_path)
		img_as_tensor = to_tensor(img_as_img)
		img_as_np = np.array(img_as_img)
		img_as_np = np.rollaxis(img_as_np, 2, 0)
		for c in range(3):
			mean_tensor[c] += torch.mean(img_as_tensor[c])
			mean_np[c] += np.mean(img_as_np[c])
	print('mean_tensor = ', mean_tensor)
	print('mean_np = ', mean_np)
	


if __name__ == '__main__':
	clean_unused_images()
	create_pose_data()
	
	# Calculate RGB means of images in training videos
	train_video = ['00', '02', '08', '09']
	image_path_list = []
	for folder in train_video:
		image_path_list += glob.glob('images/{}/*.png'.format(folder))
	calculate_rgb_mean(image_path_list)

