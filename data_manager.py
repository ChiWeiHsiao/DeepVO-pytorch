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
#import threading
from queue import Queue



__all__ = ['prepare_sequence_data']


def prepare_sequence_data(folder_list, seq_len_range=[5,5], mode='single', overlap=0, sample_interval=None, queue=None):
	# Read in data
	X, Y = [], []
	start_t = time.time()
	image_shape = (3, 370, 1226)

	for video_id, folder in enumerate(folder_list):
		t_load = time.time()
		print('processing %s'%folder, flush=True)
		x_all = []
		# Read ground truth poses
		poses = np.load('KITTI/pose_GT/{}.npy'.format(folder))
		# Read images and stack 2 images
		fnames = glob.glob('KITTI/images/{}/*.png'.format(folder))  #unorderd
		fnames.sort()
		for fname in fnames:
			im = Image.open(fname)
			if im.size != (params.img_w, params.img_h):
				im = im.resize((params.img_w, params.img_h), Image.ANTIALIAS)
			im = np.array(im)  # (h, w, c)
			#im = plt.imread(fname)  #(h, w, c)
			im = np.rollaxis(im, 2, 0)  #(c, h, w)
			im = np.expand_dims(im, axis=0)  #(1, c, h, w)
			x_all = im if x_all == [] else np.concatenate((x_all, im), axis=0)

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
				n_frames = len(x_all) - st
				jump = seq_len - overlap
				res = n_frames % seq_len
				if res != 0:
					n_frames = n_frames - res
				x_segs = [x_all[i:i+seq_len] for i in range(st, n_frames, jump)]
				if len(x_segs[-1]) < seq_len:
					x_segs = x_segs[:-1]
				x_segs = np.array(x_segs)
				y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
				if len(y_segs[-1]) < seq_len:
					y_segs = y_segs[:-1]
				Y += y_segs
				X = x_segs if X == [] else np.concatenate((X, x_segs), axis=0)

				
		# Random segment to sequences with diff lengths
		else:
			n_frames = len(x_all)
			min_len, max_len = seq_len_range[0], seq_len_range[1]
			start = 0
			while True:
				n = np.random.random_integers(min_len, max_len)
				if start + n < n_frames:
					seg = x_all[start:start+n] 
					pad_x = np.zeros((max_len, 3, image_shape[1], image_shape[2]), dtype='float32')
					pad_x[:n] = seg
					pad_x = np.expand_dims(pad_x, axis=0)
					X = pad_x if X == [] else np.concatenate((X, pad_x), axis=0)
					
					pad_zero = np.zeros((max_len-n, 6))
					pad_y = np.concatenate((poses[start:start+n], pad_zero))
					Y.append(pad_y)
				else:
					print('Last %d frames is not used' %n)
					break
				start += n
		print('Complete segment data in {}...'.format(video_id))

	Y = np.array(Y, dtype='float32')
	print('Finish {} in {} sec'.format(folder, time.time()-start_t), flush=True)
	if seq_len_range[0] == seq_len_range[1]:
		return X, Y
	#if mode == 'thread':
	#	queue.put((X,Y))
	#	return


if __name__=='__main__':

	overlap = 1
	sample_interval = None  # None 2
	mode = 'single'  # 'single' 'multiprocessing'
	folder_list = ['04']  # 01 07 09
	seq_len_range = params.seq_len
	save_name = 'KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}'.format('_'.join(folder_list), seq_len_range[0], seq_len_range[1], params.img_h, params.img_w)
	print('start {}'.format(mode), flush=True)
	print('Seq Len: {}, Image size: {}, {}'.format(params.seq_len, params.img_w, params.img_h))
	print('Sample Overlap: {}, Sample starts interval: {}'.format(overlap, sample_interval))
	if mode == 'single': 
		save_name = 'KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}'.format(folder_list[0], 
			seq_len_range[0], seq_len_range[1],
			params.img_h, params.img_w)
		start_t = time.time()
		X, Y = prepare_sequence_data(folder_list, seq_len_range, mode, overlap, sample_interval)

	elif mode == 'multiprocessing':
		args = [([folder], seq_len_range, mode, overlap, sample_interval) for folder in folder_list]
		
		N_CPU = 2
		start_t = time.time()
		pool = mp.Pool(N_CPU)
		res = pool.starmap(prepare_sequence_data, iterable=args)
		X, Y = [], []
		for x, y in res:
			print('x.shape: ', x.shape)
			X = x if X == [] else np.concatenate(X, x)
			Y = y if Y == [] else np.concatenate(Y, y)

	#elif mode == 'thread':
		#start_t = time.time()
		#q = Queue()
		#print('threading.active_count(): ',threading.active_count(),flush=True)
		#print('threading.enumerate()\n', threading.enumerate(),flush=True)
		#threads = []
		#for f in folder_list:
			#t = threading.Thread(target=prepare_sequence_data, args=([f], seq_len_range, mode, overlap, sample_interval, q))
			#t.start()
			#threads.append(t)
		#for t in threads:
			#t.join()
		#X, Y = [], []
		#seq_lengths = []
		#for _ in range(len(folder_list)):
			#x, y = q.get()
			#print('x.shape: ', x.shape)
			#X = x if X == [] else np.concatenate(X, x)
			#Y = y if Y == [] else np.concatenate(Y, y)
			#seq_lengths = s if seq_lengths == [] else np.concatenate(seq_lengths, s)
	print('=====================================')
	print('Job is done in {} sec'.format(time.time()-start_t))
	print('X.shape:', X.shape)
	print('Y.shape:', Y.shape)

	if save_name != None:
		np.savez(save_name, x=X, y=Y)

