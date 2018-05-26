from params import *
from model import DeepVO
#from Dataloader_loss import *

import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from data_manager import prepare_sequence_data


M_deepvo = DeepVO(params.img_h, params.img_w)
use_cuda = torch.cuda.is_available()

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

if use_cuda:
    print('CUDA used.')
    M_deepvo.cuda()

# Prepare Data
#X, Y = prepare_sequence_data(['07'], params.seq_len, 'single')
start_t = time.time()
if len(params.load_data) == 1:
	data = np.load(params.load_data[0])
	X, Y = data['x'], data['y']
elif len(params.load_data) > 1:
	X, Y = [], []
	for i, d in enumerate(params.load_data):
		data = np.load(d)
		x, y = data['x'], data['y']
		print('{}: x: {}, y: {}'.format(d, x.shape, y.shape))
		X = x if i == 0 else np.concatenate((X, x), axis=0)
		Y = y if i == 0 else np.concatenate((Y, y), axis=0)

print('Load data use {} sec'.format(time.time()-start_t))
print('X: {}, Y: {}'.format(X.shape, Y.shape))

X, Y = torch.from_numpy(X), torch.from_numpy(Y)
X = X.type(torch.FloatTensor)  # 0-255 is originally torch.uint8
print('max in tensor X:', X.max())

# Preprocess, X subtract by the mean RGB values of training set
for c in range(3):
	mean = torch.mean(X[:, :, c])
	X[:,:,c] -= mean

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
M_deepvo.train()

if params.solver == 'Adagrad':
	optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=params.lr)
elif params.solver == 'Cosine':
	optimizer = torch.optim.SGD(M_deepvo.parameters(), lr=params.lr)

# Load trained model and optimizer
if params.load_model_path:
	M_deepvo.load_state_dict(torch.load(params.load_model_path))
if params.load_optimzer_path:
	optimizer.load_state_dict(torch.load(params.load_model_path)).
	#optimizer.state = defaultdict(dict, optimizer.state)

print('Record loss in: ', params.record_path)
min_loss = 1e10

for ep in range(params.epochs):
	loss_mean = 0
	record_loss_list = []
	
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		# Train
		ls = M_deepvo.step(batch_x, batch_y, optimizer)
		record_loss_list.append(ls)

	# Record mean loss of this epoch
	record_loss_list = [float(l.cpu().numpy()) for l in record_loss_list]
	for l in record_loss_list:
		loss_mean += l
	loss_mean /= (it+1)

	f = open(params.record_path, 'w') if ep == 0 else open(params.record_path, 'a')
	f.write('{}\n'.format(record_loss_list))
	f.write('===== Epoch {} mean of loss: {} =====\n'.format(ep, loss_mean))
	
	print('{}'.format(record_loss_list))
	print('===== Epoch {} mean of loss: {} =====\n'.format(ep, loss_mean))

	# Save model
	if loss_mean < min_loss:
		min_loss = loss_mean
		print('Save model at ep {}, mean of loss: {}'.format(ep, loss_mean))
		f.write('Save model!\n')
		torch.save(M_deepvo.state_dict(), params.save_model_path)
		torch.save(optimizer.state_dict(), params.save_optimzer_path)
	f.close()

