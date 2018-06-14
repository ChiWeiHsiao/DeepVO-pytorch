import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from params import par
from model import DeepVO
from data_helper import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset


M_deepvo = DeepVO(par.img_h, par.img_w)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()


# Load FlowNet weights pretrained with FlyingChairs
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


# Prepare Data
train_df = get_data_info(folder_list=self.train_video, seq_len_range=par.seq_len, overlap=1, sample_interval=None)
train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, drop_last=True)
train_dataset = ImageSequenceDataset(train_df, (par.img_w, par.img_h), par.subtract_means)
train_dl = DataLoader(dataset, batch_sampler=train_sampler, num_workers=self.n_processors)

valid_df = get_data_info(folder_list=self.valid_video, seq_len_range=par.seq_len, overlap=1, sample_interval=None)
valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, drop_last=True)
valid_dataset = ImageSequenceDataset(valid_df, (par.img_w, par.img_h), par.subtract_means)
valid_dl = DataLoader(dataset, batch_sampler=valid_sampler, num_workers=self.n_processors)



if par.optim['opt'] == 'Adam':
	optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif par.optim['opt'] == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.optim['lr'])
elif par.optim['opt'] == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=par.optim['lr'])
	T_iter = par.optim['T']*len(train_dl)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)


if par.resume:
	M_deepvo.load_state_dict(torch.load(par.load_model_path))
	optimizer.load_state_dict(torch.load(par.load_optimzer_path))
	print('Load model from: ', par.load_model_path)
	print('Load optimizer from: ', par.load_optimizer_path)


print('Record loss in: ', par.record_path)
min_loss = 1e10
M_deepvo.train()
for ep in range(par.epochs):
	loss_mean = 0
	loss_mean_valid = 0
	f.write('Epoch {}'.format(ep+1))
	for it, (_, t_x, t_y) in enumerate(train_dl):
		if use_cuda:
			t_x = t_x.cuda(non_blocking=True)
			t_y = t_y.cuda(non_blocking=True)
		ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
		if par.optim == 'Cosine':
			lr_scheduler.step()
		loss_mean +=  float(ls)
	loss_mean /= (it+1)

	for it, (_, v_x, v_y) in enumerate(valid_dl):
		if use_cuda:
			v_x = v_x.cuda(non_blocking=True)
			v_y = v_y.cuda(non_blocking=True)
		ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
		loss_mean_valid +=  float(ls)
	loss_mean_valid /= (it+1)

	f = open(par.record_path, 'a')
	f.write('train loss mean: {}\nvalid loss nmean: {}\n\n'.format(loss_mean, loss_mean_valid))
	print('train loss mean: {}\nvalid loss nmean: {}\n\n'.format(loss_mean, loss_mean_valid))

	# Save model
	check_interval = 2
	if loss_mean_valid < min_loss and ep % check_interval == 0:
		min_loss = loss_mean_valid
		print('Save model at ep {}, mean of valid loss: {}'.format(ep, loss_mean_valid))  # use 4.6 sec 
		torch.save(M_deepvo.state_dict(), par.save_model_path)
		torch.save(optimizer.state_dict(), par.save_optimzer_path)
	f.close()
