import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data as Data
import time
from params import *
import psutil
from sys import getsizeof

import multiprocessing as mp
import threading
from queue import Queue



__all__ = ['prepare_sequence_data']


def prepare_sequence_data(folder_list, seq_len_range=[5,5], mode='single', sample_interval=None, queue=None):
	# Read in data
	X, Y = [], []
	start_t = time.time()
	image_shape = (3, 370, 1226)

	for video_id, folder in enumerate(folder_list):
		t_load = time.time()
		print('processing %s'%folder, flush=True)
		x_one_video = []
		# Read ground truth poses
		poses = np.load('KITTI/pose_GT/{}.npy'.format(folder))
		# Read images and stack 2 images
		fnames = glob.glob('KITTI/images/{}/image_03/data/*.png'.format(folder))  #unorderd
		fnames = ['KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, i) for i in range(len(fnames))]
		for file_id, fname in enumerate(fnames):
			fname = 'KITTI/images/{}/image_03/data/{:010d}.png'.format(folder, file_id)
			im = Image.open(fname)
			if im.size != (params.img_w, params.img_h):
				im = im.resize((params.img_w, params.img_h), Image.ANTIALIAS)
			im = np.array(im)  # (h, w, c)
			#im = plt.imread(fname)  #(h, w, c)
			im = np.rollaxis(im, 2, 0)  #(c, h, w)
			im = np.expand_dims(im, axis=0)  #(1, c, h, w)
			x_one_video = im if x_one_video == [] else np.concatenate((x_one_video, im), axis=0)

		print('Load images of video {} in {}...'.format(video_id, time.time()-t_load))

		# Fixed seq_len
		if seq_len_range[0] == seq_len_range[1]:
			if sample_interval:
				start_frames = list(range(0, seq_len_range[0], sample_interval))
				print('Sample start from frame {}'.format(start_frames))
			else:
				start_frames = [0]

			for st in start_frames:
				seq_len = seq_len_range[0]
				n_frames = len(x_one_video) - st
				res = n_frames % seq_len
				if res != 0:
					n_frames = n_frames - res
				x_segments = [x_one_video[i:i+seq_len] for i in range(st, n_frames, seq_len)]
				x_segments = np.array(x_segments)
				y_segments = [poses[i:i+seq_len] for i in range(st, n_frames, seq_len)]
				Y += y_segments
				X = x_segments if X == [] else np.concatenate((X, x_segments), axis=0)
				
		# Random segment to sequences with diff lengths
		else:
			n_frames = len(x_one_video)
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


if __name__=='__main__':

	sample_interval = 2  # None
	mode = 'single'  # 'single' 'multiprocessing' 'thread'
	folder_list = ['09']  # 01 07 09
	seq_len_range = params.seq_len
	save_name = 'KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}'.format('_'.join(folder_list), seq_len_range[0], seq_len_range[1], params.img_h, params.img_w)
	print('start {}'.format(mode), flush=True)
	print('Seq Len: {}, Image size: {}, {}'.format(params.seq_len, params.img_w, params.img_h))

	if mode == 'single': 
		save_name = 'KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}'.format(folder_list[0], 
			seq_len_range[0], seq_len_range[1],
			params.img_h, params.img_w)
		start_t = time.time()
		X, Y = prepare_sequence_data(folder_list, seq_len_range, mode, sample_interval)

	elif mode == 'multiprocessing':
		args = [([folder], seq_len_range, mode, sample_interval) for folder in folder_list]
		
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
		start_t = time.time()
		q = Queue()
		print('threading.active_count(): ',threading.active_count(),flush=True)
		print('threading.enumerate()\n', threading.enumerate(),flush=True)
		threads = []
		for f in folder_list:
			t = threading.Thread(target=prepare_sequence_data, args=([f], seq_len_range, mode, sample_interval, q))
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

	if save_name != None:
		np.savez(save_name, x=X, y=Y)

