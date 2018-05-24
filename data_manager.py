import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import time
import psutil
from sys import getsizeof

import multiprocessing as mp
import threading
from queue import Queue



__all__ = ['prepare_sequence_data']


def prepare_sequence_data(folder_list, batch_size, seq_len_range=[5,5], mode='single', queue=None):
	# Read in data
	start_t = time.time()
	X, Y = [], []
	image_shape = (3, 370, 1226)
	for video_id, folder in enumerate(folder_list):
		print('processing %s'%folder, flush=True)
		x_one_video = []
		# Read ground truth poses
		poses = np.load('KITTI/pose_GT/{}.npy'.format(folder))
		#y_0_t = poses[0][-3:]
		# Read images and stack 2 images
		fnames = glob.glob('KITTI/images/{}/image_03/data/*.png'.format(folder))  #unorderd
		fnames = ['KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, i) for i in range(len(fnames))]
		for file_id, fname in enumerate(fnames):
			fname = 'KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, file_id)
			im = plt.imread(fname)  #(370, 1226, 3)
			im = np.rollaxis(im, 2, 0)  #(3, 370, 1226)
			if im.shape != image_shape:
				print('Debug: exist image with different shape in data: ', fname, im.shape)
			im = np.expand_dims(im, axis=0)  #(1, 3, 370, 1226)
			x_one_video = im if x_one_video == [] else np.concatenate((x_one_video, im), axis=0)

		n_frames = len(x_one_video)
		# Fixed seq_len
		if seq_len_range[0] == seq_len_range[1]:
			seq_len = seq_len_range[0]
			n_frames = len(x_one_video)
			res = n_frames % seq_len
			if res != 0:
				n_frames = n_frames - res
			x_segments = [x_one_video[i:i+seq_len] for i in range(0, n_frames, seq_len)]
			x_segments = np.array(x_segments)
			y_segments = [poses[i:i+seq_len] for i in range(0, n_frames, seq_len)]
			X = x_segments if video_id == 0 else np.concatenate((X, x_segments), axis=0)
			Y += y_segments
		# Random segment to sequences with diff lengths
		else:
			min_len, max_len = seq_len_range[0], seq_len_range[1]
			start = 0
			while True:
				n = np.random.random_integers(min_len, max_len)
				if start + n < n_frames:
					seg = x_one_video[start:start+n] 
					pad_x = np.zeros((max_len, 3, image_shape[1], image_shape[2]), dtype='float32')
					pad_x[:n] = seg
					pad_x = np.expand_dims(pad_x, axis=0)
					X = pad_x if  X == [] else np.concatenate((X, pad_x), axis=0)
					pad_zero = np.zeros((max_len-n, 6))
					pad_y = np.concatenate((poses[start:start+n], pad_zero))
					Y.append(pad_y)
				else:
					print('Last %d frames is not used' %n)
					break
				start += n
		print('Complete segment data in {}...'.format(video_id))

	Y = np.array(Y, dtype='float32')  #(n_segments, seq_len, 12)
	print('Finish {} in {} sec'.format(folder, time.time()-start_t), flush=True)
	if mode == 'single' or mode == 'multiprocessing':
		return X, Y
	elif mode == 'thread':
		queue.put((X,Y))


def segment_all_fnames(folder_list, batch_size, seq_len_range=[5,5], mode='single', queue=None):
# Segment to 100 images per process
	for video_id, folder in enumerate(folder_list):
		print('processing %s'%folder, flush=True)
		# for one seqence of frames, ex. 00
		x_one_video = []
		# Read ground truth poses
		poses = np.load('KITTI/pose_GT/{}.npy'.format(folder))
		# Segment file names into 100 files per seg
		segment_filename = []
		fnames = glob.glob('KITTI/images/{}/image_03/data/*.png'.format(folder))  #unorderd
		fnames = ['KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, i) for i in range(len(fnames))]
		for file_id, fname in enumerate(fnames):
			fname = 'KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, file_id)



if __name__=='__main__':

	mode = 'single'  # 'single' 'multiprocessing' 'thread'
	folder_list = ['04', '07']
	batch_size = 8
	seq_len_range = [3,5]

	if mode == 'single':
		folder = ['_04']
		print('start {}'.format(mode), flush=True)
		start_t = time.time()
		X, Y = prepare_sequence_data(folder, batch_size, seq_len_range, mode)

	elif mode == 'multiprocessing':
		print('start {}, cpu:{}'.format(mode, mp.cpu_count()), flush=True)
		args = [([folder], batch_size, seq_len_range, mode) for folder in folder_list]
		
		N_CPU = 2
		start_t = time.time()
		pool = mp.Pool(N_CPU)
		res = pool.starmap(prepare_sequence_data, iterable=args)
		X, Y = [], []
		for x,y in res:
			print('x.shape: ', x.shape)
			X = x if X == [] else np.concatenate(X, x)
			Y = y if Y == [] else np.concatenate(Y, y)

	elif mode == 'thread':
		print('start {}'.format(mode), flush=True)
		start_t = time.time()
		q = Queue()
		print('threading.active_count(): ',threading.active_count(),flush=True)
		print('threading.enumerate()\n', threading.enumerate(),flush=True)
		threads = []
		for f in folder_list:
			t = threading.Thread(target=prepare_sequence_data, args=([f], batch_size, seq_len_range, mode, q))
			t.start()
			threads.append(t)
		for t in threads:
			t.join()
		X, Y = [], []
		for _ in range(len(folder_list)):
			x, y = q.get()
			print('x.shape: ', x.shape)
			X = x if X == [] else np.concatenate(X, x)
			Y = y if Y == [] else np.concatenate(Y, y)
	print('=====================================')
	print('Job is done in {} sec'.format(time.time()-start_t))
	print('X.shape:', X.shape)
	print('Y.shape:', Y.shape)