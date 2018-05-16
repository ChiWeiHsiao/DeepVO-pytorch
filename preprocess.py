import os
import glob
import numpy as np
import time
from helper import R_to_angle

def clean_unused_images():
	seq_frame = {'00': ['000000', '004540'],
				'01': ['000000', '001100'],
				'02': ['000000', '004660'],
				'03': ['000000', '000800'],
				'04': ['000000', '000270'],
				'05': ['000000', '002760'],
				'06': ['000000', '001100'],
				'07': ['000000', '001100'],
				'08': ['001100', '005170'],
				'09': ['000000', '001590'],
				'10': ['000000', '001200']
				}
	for dir_id, img_ids in seq_frame.items():
		dir_path = 'KITTI/images/{}/'.format(dir_id)
		if not os.path.exists(dir_path):
			continue

		print('Cleaning {} directory'.format(dir_id))
		start, end = img_ids
		start, end = int(start), int(end)
		for idx in range(0, start):
			img_name = '{:010d}.png'.format(idx)
			img_path = 'KITTI/images/{}/image_03/data/{}'.format(dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)
		for idx in range(end+1, 10000):
			img_name = '{:010d}.png'.format(idx)
			img_path = 'KITTI/images/{}/image_03/data/{}'.format(dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)

# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def transform_poseGT_to_angles():
	pose_dir = 'KITTI/pose_GT/'
	fnames = glob.glob('{}*.txt'.format(pose_dir))
	for fn in fnames:
		start_t = time.time()
		print('Transforming {}...'.format(fn), end='')
		with open(fn) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ R_to_angle([float(value) for value in l.split(' ')]) for l in lines]  # list of pose (pose=list of 12 floats)
			return
			poses = np.array(poses)
			np.save(fn.split('.')[0]+'.npy', poses)
			print('elapsed time = {}'.format(time.time()-start_t))



if __name__ == "__main__":
	clean_unused_images()
	transform_poseGT_to_angles()