import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import time

__all__ = ['create_data_loader']
def create_data_loader(folder_list, batch_size, seq_len_range=[5,5]):
	# Read in data
	start_t = time.time()
	x_multi_videos, y_multi_videos = [], []
	image_shape = []
	for folder in folder_list:
		# for one seqence of frames, ex. 00
		x_one_video, y_one_video = [], []
		# Read ground truth poses
		with open('KITTI/pose_GT/{}.txt'.format(folder)) as f:
			lines = [line.split('\n')[0] for line in f.readlines()] 
			poses = [ [float(value) for value in l.split(' ')] for l in lines]  # list of pose (pose=list of 6 floats)

		y_one_video = poses[1:]
		# Read images and stack 2 images
		last_im = None
		fnames = glob.glob('KITTI/images/{}/image_03/data/*.png'.format(folder))
		for idx, fname in enumerate(fnames):
			im = plt.imread(fname)  #(370, 1226, 3)
			im = np.rollaxis(im, 2, 0)  #(3, 370, 1226)
			if im.shape != (3, 370, 1226):
				print('Debug: exist image with different shape in data: ', fname, im.shape)
			if idx == 0:
				last_im = im
			else:
				stacked_ims =  np.concatenate((last_im, im), axis=0)
				stacked_ims = np.expand_dims(stacked_ims, axis=0)  #(1, 6, 370, 1226)
				if idx == 1:
					x_one_video = stacked_ims
				else:
					x_one_video = np.concatenate((x_one_video, stacked_ims), axis=0)
		image_shape = [im.shape[1], im.shape[2]]
		print('Complete reading data in {}...'.format(folder))
		x_multi_videos.append(x_one_video)  # list (diff videos) of np arr [(n_frames, 6, 370, 1226), ...]
		y_multi_videos.append(y_one_video)  # list (diff videos) of list of pose (pose=list of 6 floats)
		del x_one_video, y_one_video

	# Create Data Loader
	X, Y = [], []
	if seq_len_range[0] == seq_len_range[1]:  # Fixed seq_len
		seq_len = seq_len_range[0]	
		for idx, x_one_video in enumerate(x_multi_videos):
			n_frames = len(x_one_video)
			res = n_frames % seq_len
			if res != 0:
				n_frames = n_frames - res
			x_segments = [x_one_video[i:i+seq_len] for i in range(0, n_frames, seq_len)]
			x_segments = np.array(x_segments)  #(n_segments_in_one_video, seq_len, 6, 370, 1226)
			y_segments = [y_multi_videos[idx][i:i+seq_len] for i in range(0, n_frames, seq_len)]
			X = x_segments if idx == 0 else np.concatenate((X, x_segments), axis=0)  #(n_segments_in_all_video, seq_len, 6, 370, 1226)
			Y += y_segments
			print('Complete segment data in {}...'.format(idx))
	else:
		min_len, max_len = seq_len_range[0], seq_len_range[1]
		for idx, x_one_video in enumerate(x_multi_videos):
			n_frames = len(x_one_video)
			start = 0
			while True:
				n = np.random.random_integers(min_len, max_len)
				if start + n < n_frames:
					seg = x_one_video[start:start+n]  #(random_n, 6, 370, 1226)
					pad_seg = np.zeros((max_len, 6, image_shape[0], image_shape[1]))  #(max_len, 6, 370, 1226)
					pad_seg[:n] = seg
					pad_seg = np.expand_dims(pad_seg, axis=0)
					X = pad_seg if  X == [] else np.concatenate((X, pad_seg), axis=0)  #(n_segs, max_seq_len, 6, 370, 1226)
					pad_y = [[0]*12] * max_len
					pad_y[:n] = y_multi_videos[idx][start:start+n]
					Y.append(pad_y)
				else:
					print('Last %d frames is not used' %n)
					break
				start += n
			print('Complete segment data in {}...'.format(idx))

	print('====================================================')
	print('X: ', X.shape)
	X = torch.from_numpy(X)
	Y = np.array(Y)  #(n_segments, seq_len, 12)
	print('Y: ', Y.shape)
	Y = torch.from_numpy(Y)
	torch_dataset = Data.TensorDataset(X, Y)
	data_loader = Data.DataLoader(
	    dataset=torch_dataset,
	    batch_size=batch_size ,
	    shuffle=True,
	)
	del X, Y
	print('Elapsed time for creating data loader: ', time.time()-start_t)

	return data_loader

