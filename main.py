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


M_deepvo = DeepVO(params.img_size[0], params.img_size[1])#320, 96)


# Use only conv-layer-part of FlowNet as CNN for DeepVO
model_dict = M_deepvo.state_dict()
update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
model_dict.update(update_dict)
M_deepvo.load_state_dict(model_dict)
if use_cuda:
    print('CUDA used.')
    M_deepvo.cuda()

# Prepare Data
params.img_size  # resize image to this size
params.seq_len  # prepare sequence of frames
seqs = ['01','04','06','07','09', '10']
X, Y = prepare_sequence_data(['_04'], params.batch_size, [10,10], 'single')
X, Y = torch.from_numpy(X), torch.from_numpy(Y)

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
#x = torch.randn(1, 3, 6, params.img_size[0], params.img_size[1]).type(torch.FloatTensor))  # b_size, seq_len, channels(3*2andn(1, 3, 6).type(torch.FloatTensor)

# Train
optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=params.lr)
M_deepvo.train()
for ep in range(params.epochs):
	print('============ Epoch {} ====================='.format(ep))
	for it, (batch_x, batch_y) in enumerate(train_dl):
		if use_cuda:
			batch_y = batch_y.cuda(non_blocking=True)
			batch_x = batch_x.cuda(non_blocking=True)
		
		# Train
		ls = M_deepvo.step(batch_x, batch_y, optimizer)
		ls = ls.cpu().numpy()
		print('Iteration {}: loss = {}'.format(it, ls))
