from params import par
from model import DeepVO
#from Dataloader_loss import *

import numpy as np
import time
import torch

import torch.utils.data as Data
from data_manager import pregipare_sequence_data


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
#X, Y, seq_lengths = prepare_sequence_data(['07'], par.seq_len, 'single')
def load_data(data_path_list):
	use_pad = (par.seq_len[0] != par.seq_len[1])
	start_t = time.time()
	if len(data_path_list) == 1:
		data = np.load(data_path_list[0])
		X, Y = data['x'], data['y']
		if use_pad:
			seq_lengths = data['seq_lengths']
	elif len(data_path_list) > 1:
		X, Y = [], []
		seq_lengths = []
		for i, d in enumerate(data_path_list):
			data = np.load(d)
			x, y = data['x'], data['y']
			print('{}: x: {}, y: {}'.format(d, x.shape, y.shape))
			X = x if i == 0 else np.concatenate((X, x), axis=0)
			Y = y if i == 0 else np.concatenate((Y, y), axis=0)
			if use_pad:
				s = data['seq_lengths']
				seq_lengths = s if i == 0 else np.concatenate((seq_lengths, s), axis=0) 
	print('Load data use {} sec'.format(time.time()-start_t))
	print('X: {}, Y: {}'.format(X.shape, Y.shape))

	X, Y = torch.from_numpy(X), torch.from_numpy(Y)
	X = X.type(torch.FloatTensor)  # 0-255 is originally torch.uint8


	if use_pad:
		print('Seq_lengths: {}'.format(seq_lengths.shape))
		seq_lengths = troch.from_numpy(seq_lengths)
		return X, Y, seq_lengths
	else:
		return X, Y, None

train_X, train_Y, train_S = load_data(par.train_data_path)
valid_X, valid_Y, valid_S = load_data(par.valid_data_path)

# Preprocess, X subtract by the mean RGB values of training set
for c in range(3):
	train_X[:,:,c] -= par.RGB_means[c]
	valid_X[:,:,c] -= par.RGB_means[c]
	# Calculate proper mean_RGB and store to par.RGB_means
	#mean = torch.mean(X[:, :, c])
	#print('mean: ', mean.numpy())

#if use_pad:
#	train_dataset = Data.TensorDataset(X, Y, seq_lengths)  #AttributeError: 'list' object has no attribute 'size
train_dataset = Data.TensorDataset(train_X, train_Y)
train_dl = Data.DataLoader(
    dataset=train_dataset,
    batch_size=par.batch_size,
    shuffle=True,
)

#valid_dataset = Data.TensorDataset(valid_X, valid_Y)
#valid_dl = Data.DataLoader(
#    dataset=valid_dataset,
#    batch_size=par.batch_size,
#    shuffle=False,
#)


###############################################################
#     Prepare Data // By Yukun
#Dataloader read images folder one by one and return dataloader
#returned dataloader stores images address and will only read 
#images when take batches.
###############################################################
'''
seq_list = ['01']
for seq in seq_list:
    train_dl = DataLoader(,batch_size=par.batch_size,seq_len = par.seq_len,num_workers = 5) 
    
    #Custom loss function
    criterion = DeepvoLoss()
    #DEEPVO TRAINING PROCESS
    
'''

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
	#optimizer.state = defaultdict(dict, optimizer.state)
	print('Load model from: ', par.load_model_path)
	print('Load optimizer from: ', par.load_optimizer_path)


print('Record loss in: ', par.record_path)
min_loss = 1e10

M_deepvo.train()
for ep in range(par.epochs):
	loss_mean = 0
	loss_mean_valid = 0
	record_loss_list = []
	
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		ls = M_deepvo.step(batch_x, batch_y, optimizer).data.cpu().numpy()
		ls = float(ls)
		if par.optim == 'Cosine':
			lr_scheduler.step()
		record_loss_list.append(ls)
	# Record train loss of this epoch
	for l in record_loss_list:
		loss_mean += l
	loss_mean /= (it+1)

	# Calculate valid mean loss
	n_iter_valid = int(len(valid_Y) / par.batch_size)
	for it in range(n_iter_valid):
		v_x = valid_X[it*par.batch_size:it*par.batch_size+par.batch_size]
		v_y = valid_Y[it*par.batch_size:it*par.batch_size+par.batch_size]
		if use_cuda:
			v_y = v_y.cuda()
			v_x = v_x.cuda()
		ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
		loss_mean_valid += float(ls)
	loss_mean_valid /= (it+1)

	f = open(par.record_path, 'a')
	f.write('Epoch {}\nmean of train loss: {}\nmean of valid loss: {}\n\n'.format(ep, loss_mean, loss_mean_valid))
	print('Epoch {}\nmean of train loss: {}\nmean of valid loss: {}\n\n'.format(ep, loss_mean, loss_mean_valid))

	# Save model
	check_interval = 2
	if loss_mean_valid < min_loss and ep % check_interval == 0:
		min_loss = loss_mean_valid
		f.write('Save model!\n')
		print('Save model at ep {}, mean of valid loss: {}'.format(ep, loss_mean_valid))  # use 4.6 sec 
		torch.save(M_deepvo.state_dict(), par.save_model_path)
		torch.save(optimizer.state_dict(), par.save_optimzer_path)
	f.close()
