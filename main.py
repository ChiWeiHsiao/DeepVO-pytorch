from params import params
from model import DeepVO
#from Dataloader_loss import *

import numpy as np
import time
import torch

import torch.utils.data as Data
from data_manager import prepare_sequence_data


# Write all hyperparameters to record_path
p = vars(params)
with open(params.record_path, 'a') as f:
	f.write('\n'.join("%s: %s" % item for item in p.items()))
	f.write('\n'+'='*50 + '\n')


M_deepvo = DeepVO(params.img_h, params.img_w)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo = M_deepvo.cuda()


# Load FlowNet weights pretrained with FlyingChairs
if params.pretrained_flownet and params.load_model_path == None:
	if use_cuda:
		pretrained_w = torch.load(params.pretrained_flownet)
	else:
		pretrained_w = torch.load(params.pretrained_flownet_flownet, map_location='cpu')
	print('Load FlowNet pretrained model')
	# Use only conv-layer-part of FlowNet as CNN for DeepVO
	model_dict = M_deepvo.state_dict()
	update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
	model_dict.update(update_dict)
	M_deepvo.load_state_dict(model_dict)


# Prepare Data
#X, Y, seq_lengths = prepare_sequence_data(['07'], params.seq_len, 'single')
def load_data(data_path_list):
	use_pad = (params.seq_len[0] != params.seq_len[1])
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

train_X, train_Y, train_S = load_data(params.train_data_path)
valid_X, valid_Y, valid_S = load_data(params.valid_data_path)

# Preprocess, X subtract by the mean RGB values of training set
for c in range(3):
	train_X[:,:,c] -= params.RGB_means[c]
	valid_X[:,:,c] -= params.RGB_means[c]
	# Calculate proper mean_RGB and store to params.RGB_means
	#mean = torch.mean(X[:, :, c])
	#print('mean: ', mean.numpy())

#if use_pad:
#	train_dataset = Data.TensorDataset(X, Y, seq_lengths)  #AttributeError: 'list' object has no attribute 'size
train_dataset = Data.TensorDataset(train_X, train_Y)
train_dl = Data.DataLoader(
    dataset=train_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)

#valid_dataset = Data.TensorDataset(valid_X, valid_Y)
#valid_dl = Data.DataLoader(
#    dataset=valid_dataset,
#    batch_size=params.batch_size,
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
    train_dl = DataLoader(,batch_size=params.batch_size,seq_len = params.seq_len,num_workers = 5) 
    
    #Custom loss function
    criterion = DeepvoLoss()
    #DEEPVO TRAINING PROCESS
    
'''

if params.optim['opt'] == 'Adam':
	optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif params.optim['opt'] == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=params.optim['lr'])
elif params.optim['opt'] == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=params.optim['lr'])
	T_iter = params.optim['T']*len(train_dl)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

if params.resume:
	M_deepvo.load_state_dict(torch.load(params.load_model_path))
	optimizer.load_state_dict(torch.load(params.load_optimzer_path))
	#optimizer.state = defaultdict(dict, optimizer.state)
	print('Load model from: ', params.load_model_path)
	print('Load optimizer from: ', params.load_optimizer_path)


print('Record loss in: ', params.record_path)
min_loss = 1e10

M_deepvo.train()
for ep in range(params.epochs):
	loss_mean = 0
	loss_mean_valid = 0
	record_loss_list = []
	
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		ls = M_deepvo.step(batch_x, batch_y, optimizer).data.cpu().numpy()
		ls = float(ls)
		if params.optim == 'Cosine':
			lr_scheduler.step()
		record_loss_list.append(ls)
	# Record train loss of this epoch
	for l in record_loss_list:
		loss_mean += l
	loss_mean /= (it+1)

	# Calculate valid mean loss
	n_iter_valid = int(len(valid_Y) / params.batch_size)
	for it in range(n_iter_valid):
		v_x = valid_X[it*params.batch_size:it*params.batch_size+params.batch_size]
		v_y = valid_Y[it*params.batch_size:it*params.batch_size+params.batch_size]
		if use_cuda:
			v_y = v_y.cuda()
			v_x = v_x.cuda()
		ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
		loss_mean_valid += float(ls)
	loss_mean_valid /= (it+1)

	f = open(params.record_path, 'a')
	f.write('Epoch {}\nmean of train loss: {}\nmean of valid loss: {}\n\n'.format(ep, loss_mean, loss_mean_valid))
	print('Epoch {}\nmean of train loss: {}\nmean of valid loss: {}\n\n'.format(ep, loss_mean, loss_mean_valid))

	# Save model
	check_interval = 2
	if loss_mean_valid < min_loss and ep % check_interval == 0:
		min_loss = loss_mean_valid
		f.write('Save model!\n')
		print('Save model at ep {}, mean of valid loss: {}'.format(ep, loss_mean_valid))  # use 4.6 sec 
		torch.save(M_deepvo.state_dict(), params.save_model_path)
		torch.save(optimizer.state_dict(), params.save_optimzer_path)
	f.close()
