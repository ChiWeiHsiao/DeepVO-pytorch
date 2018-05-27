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
	f.write('\n')

M_deepvo = DeepVO(params.img_h, params.img_w)
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_deepvo.cuda()


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
use_pad = (params.seq_len[0] != params.seq_len[1])
start_t = time.time()
if len(params.load_data) == 1:
	data = np.load(params.load_data[0])
	X, Y = data['x'], data['y']
	if use_pad:
		seq_lengths = data['seq_lengths']
elif len(params.load_data) > 1:
	X, Y = [], []
	seq_lengths = []
	for i, d in enumerate(params.load_data):
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
if use_pad:
	print('Seq_lengths: {}'.format(seq_lengths.shape))
	seq_lengths = troch.from_numpy(seq_lengths)

X, Y = torch.from_numpy(X), torch.from_numpy(Y)
X = X.type(torch.FloatTensor)  # 0-255 is originally torch.uint8
print('max in tensor X:', X.max())

# Preprocess, X subtract by the mean RGB values of training set
for c in range(3):
	X[:,:,c] -= params.RGB_means[c]
	#mean = torch.mean(X[:, :, c])
	#print('mean: ', mean.numpy())
	#X[:,:,c] -= mean

train_dataset = Data.TensorDataset(X, Y)
#train_dataset = Data.TensorDataset(X, Y, seq_lengths)  #AttributeError: 'list' object has no attribute 'size
train_dl = Data.DataLoader(
    dataset=train_dataset,
    batch_size=params.batch_size,
    shuffle=True,
)

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
# toy data
#x = torch.randn(1, 3, 6, params.img_h, params.img_w).type(torch.FloatTensor))  # b_size, seq_len, channels(3*2andn(1, 3, 6).type(torch.FloatTensor)

if params.optim['opt'] == 'Adam':
	optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
elif params.optim['opt'] == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=params.optim['lr'])
elif params.optim['opt'] == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=params.optim['lr'])
	T_iter = params.optim['T']*len(train_dl)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_iter, eta_min=0, last_epoch=-1)

if params.resume:
	print('Load model')
	M_deepvo.load_state_dict(torch.load(params.load_model_path))
	print('Load optimizer')
	opt_state_dict = torch.load(params.load_optimzer_path)
	optimizer.load_state_dict(opt_state_dict)
	#optimizer.state = defaultdict(dict, optimizer.state)


print('Record loss in: ', params.record_path)
min_loss = self.min_loss

M_deepvo.train()
for ep in range(params.epochs):
	loss_mean = 0
	record_loss_list = []
	
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		# Train
		ls = M_deepvo.step(batch_x, batch_y, optimizer)
		if params.optim == 'Cosine':
			lr_scheduler.step()
		record_loss_list.append(ls)

	# Record mean loss of this epoch
	record_loss_list = [float(l.cpu().numpy()) for l in record_loss_list]
	for l in record_loss_list:
		loss_mean += l
	loss_mean /= (it+1)

	f = open(params.record_path, 'a')
	f.write('{}\n'.format(record_loss_list))
	f.write('===== Epoch {} mean of loss: {} =====\n'.format(ep, loss_mean))
	
	print('{}'.format(record_loss_list))
	print('===== Epoch {} mean of loss: {} =====\n'.format(ep, loss_mean))

	# Save model
	if loss_mean < min_loss and ep % 2 == 0:
		min_loss = loss_mean
		f.write('Save model!\n')
		print('Save model at ep {}, mean of loss: {}'.format(ep, loss_mean))  # use 4.6 sec 
		torch.save(M_deepvo.state_dict(), params.save_model_path)
		torch.save(optimizer.state_dict(), params.save_optimzer_path)
	f.close()
