import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
from ast import literal_eval
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset


# Write all hyperparameters to record_path
mode = 'a' if par.resume else 'w'
with open(par.record_path, mode) as f:
	f.write('\n'.join("%s: %s" % item for item in vars(par).items()))
	f.write('\n'+'='*50 + '\n')

# Prepare Data
if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
	print('Load data info from {}'.format(par.train_data_info_path))
	train_df = pd.read_pickle(par.train_data_info_path, index_col=0, converters={'image_path': literal_eval})
	valid_df = pd.read_pickle(par.valid_data_info_path, index_col=0, converters={'image_path': literal_eval})
else:
	print('Create new data info')
	if par.partition != None:
		all_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times, shuffle=True, sort=False)
		all_len = len(all_df.index)
		partition = par.partition
		train_df = all_df[:int(all_len*partition)]
		train_df = train_df.sort_values(by=['seq_len'], ascending=False)
		valid_df = all_df[int(all_len*partition):]
		valid_df = valid_df.sort_values(by=['seq_len'], ascending=False)
	else:
		train_df = get_data_info(folder_list=par.train_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)	
		valid_df = get_data_info(folder_list=par.valid_video, seq_len_range=par.seq_len, overlap=1, sample_times=par.sample_times)
	# save
	train_df.to_pickle(par.train_data_info_path)
	valid_df.to_pickle(par.valid_data_info_path)

train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds, par.minus_point_5)
valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

print('Num of samples in training dataset: ', train_df.shape[0])
print('Num of samples in validation dataset: ', valid_df.shape[0])


# Model
M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()


# Load FlowNet weights pretrained with FlyingChairs
# NOTE: the pretrained model assumes image rgb values in range [-0.5, 0.5]
if par.pretrained_flownet and par.load_model_path == None:
	if use_cuda:
		pretrained_w = torch.load(par.pretrained_flownet)
	else:
		pretrained_w = torch.load(par.pretrained_flownet_flownet, map_location='cpu')
	print('Load FlowNet pretrained model')
	# Use only conv-layer-part of FlowNet as CNN for DeepVO
	model_dict = M_deepvo.state_dict()
	update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
	model_dict.update(update_dict)
	M_deepvo.load_state_dict(model_dict)


# Create optimizer
if par.optim['opt'] == 'Adam':
	optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
	T_iter = par.optim['T']*len(train_dl)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

# Load trained DeepVO model and optimizer
if par.resume:
	M_deepvo.load_state_dict(torch.load(par.load_model_path))
	optimizer.load_state_dict(torch.load(par.load_optimizer_path))
	print('Load model from: ', par.load_model_path)
	print('Load optimizer from: ', par.load_optimizer_path)


# Train
print('Record loss in: ', par.record_path)
min_loss_t = 1e10
min_loss_v = 1e10
before_train = True  # debug
M_deepvo.train()
for ep in range(par.epochs):
	st_t = time.time()
	print('='*50)
	M_deepvo.train()
	loss_mean = 0
	t_loss_list = []
	for _, t_x, t_y in train_dl:
		if use_cuda:
			t_x = t_x.cuda(non_blocking=par.pin_mem)
			t_y = t_y.cuda(non_blocking=par.pin_mem)
		if before_train:
			before_train = False
			print('\nDebug')
			print('Prediction before training')
			print('Predicted: ', M_deepvo.forward(t_x).data.cpu().numpy()[0])
			print('Target   :', t_y.cpu().numpy()[0])
			print('\n')
		ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
		t_loss_list.append(float(ls))
		loss_mean += float(ls)
		if par.optim == 'Cosine':
			lr_scheduler.step()
	print('Train take {:.1f} sec'.format(time.time()-st_t))
	#print(t_loss_list)
	loss_mean /= len(train_dl)

	st_t = time.time()
	M_deepvo.eval()
	loss_mean_valid = 0
	v_loss_list = []
	for _, v_x, v_y in valid_dl:
		if use_cuda:
			v_x = v_x.cuda(non_blocking=par.pin_mem)
			v_y = v_y.cuda(non_blocking=par.pin_mem)
		v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
		v_loss_list.append(float(v_ls))
		loss_mean_valid += float(v_ls)
	print('Valid take {:.1f} sec'.format(time.time()-st_t))
	#print(v_loss_list)
	loss_mean_valid /= len(valid_dl)

	f = open(par.record_path, 'a')
	f.write('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))
	print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

	# Save model
	check_interval = 1
	if loss_mean_valid < min_loss_v and ep % check_interval == 0:
		min_loss_v = loss_mean_valid
		print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))  # use 4.6 sec 
		torch.save(M_deepvo.state_dict(), par.save_model_path+'.valid')
		torch.save(optimizer.state_dict(), par.save_optimzer_path+'.valid')

	check_interval = 1
	if loss_mean < min_loss_t and ep % check_interval == 0:
		min_loss_t = loss_mean
		print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
		torch.save(M_deepvo.state_dict(), par.save_model_path+'.train')
		torch.save(optimizer.state_dict(), par.save_optimzer_path+'.train')
	f.close()
