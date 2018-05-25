# %load main.py
from params import *
from model import DeepVO
#from Dataloader_loss import *

import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from data_manager import prepare_sequence_data

save_path = '{},ep{},b{},lr{}'.format(
    params.solver,
    params.epochs,
    params.batch_size,
    params.lr)

print('torch.cuda.is_available(): ', torch.cuda.is_available())
print('======================================================')
use_cuda = torch.cuda.is_available()

# Load FlowNet weights pretrained with FlyingChairs
if params.pretrained:
	if use_cuda:
		pretrained_w = torch.load(params.pretrained)
	else:
		pretrained_w = torch.load(params.pretrained, map_location='cpu')
	print('load pretrained model')
else:
	pretrained_w = None


M_deepvo = DeepVO(params.img_h, params.img_w)#320, 96)


# Use only conv-layer-part of FlowNet as CNN for DeepVO
model_dict = M_deepvo.state_dict()
update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
model_dict.update(update_dict)
M_deepvo.load_state_dict(model_dict)
if use_cuda:
    print('CUDA used.')
    M_deepvo.cuda()

# Prepare Data
#X, Y = prepare_sequence_data(['07'], params.seq_len, 'single')
start_t = time.time()
if len(params.load_data) == 1:
	data = np.load(params.load_data[0])
	X, Y = data['x'], data['y']
	print('X: {}, Y: {}'.format(X.shape, Y.shape))
	X, Y = torch.from_numpy(X), torch.from_numpy(Y)
	X = X.type(torch.FloatTensor)  # 0-255 is originally torch.uint8
elif len(params.load_data) > 1:
	X, Y = [], []
	for i, d in enumerate(params.load_data):
		data = np.load(d)
		x, y = data['x'], data['y']
		print('{}: x: {}, y: {}'.format(d, x.shape, y.shape))
		x, y = torch.from_numpy(x), torch.from_numpy(y)
		x = x.type(torch.FloatTensor)  # 0-255 is originally torch.uint8
		X = x if i == 0 else torch.cat((X, x), dim=0)
		Y = y if i == 0 else torch.cat((Y, y), dim=0)
print('max in tensor X:', X.max())
print('X: {}, Y: {}'.format(X.shape, Y.shape))
print('Load data use {} sec'.format(time.time()-start_t))

train_dataset = Data.TensorDataset(X, Y)
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

# Train
optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=params.lr)
M_deepvo.train()

record_name = params.record_name
save_model_name = params.save_model_name
print('Record loss in: ', record_name)
print('Save model in: ', save_model_name)
min_loss = 1e9

for ep in range(params.epochs):
	loss_sum = 0
	record_loss_list = []
	f = open(record_name, 'w') if ep == 0 else open(record_name, 'a')
	f.write('===== Epoch {} =====\n'.format(ep))
	f.close()
	
	#print('============ Epoch {} ====================='.format(ep))
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		# Train
		ls = M_deepvo.step(batch_x, batch_y, optimizer)
		record_loss_list.append(ls)
		'''
		if it % 1 == 0:
			ls = ls.cpu().numpy()
			print('Iteration {}: loss = {}'.format(it, ls))

			f = open(record_name, 'a')
			f.write('{}\n'.format(ls))
			f.close()
			loss_sum += ls
		'''
	record_loss_list = [l.cpu().numpy() for l in record_loss_list]
	for l in record_loss_list:
		loss_sum += l
	f = open(record_name, 'a')
	f.write('{}'.format(record_loss_list))
	f.write('===== Epoch {} sum of loss: {} =====\n'.format(ep, loss_sum))
	f.close()
	print('{}'.format(record_loss_list))
	print('===== Epoch {} sum of loss: {} =====\n'.format(ep, loss_sum))

	if loss_sum < min_loss:
		min_loss = loss_sum
		print('Save model at ep {}, loss sum = {}'.format(ep, loss_sum))
		torch.save(M_deepvo.state_dict(), save_model_name)

#torch.save(M_deepvo.state_dict(), save_model_name)


