from params import *
from model import DeepVO

import numpy as np
import time
import torch
from torch.autograd import Variable
from data_manager import create_data_loader

save_path = '{},ep{},b{},lr{}'.format(
    params.solver,
    params.epochs,
    params.batch_size,
    params.lr)


# Load FlowNet weights pretrained with FlyingChairs
if params.pretrained:
	if params.cuda:
		pretrained_w = torch.load(params.pretrained)
	else:
		pretrained_w = torch.load(params.pretrained, map_location='cpu')
	print('load pretrained model')
else:
	pretrained_w = None


# Use only conv-layer-part of FlowNet as CNN for DeepVO
M_deepvo = DeepVO(params.img_size[0], params.img_size[1])#320, 96)
model_dict = M_deepvo.state_dict()
update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
model_dict.update(update_dict)
M_deepvo.load_state_dict(model_dict)
if params.cuda:
	M_deepvo = M_deepvo.cuda()


# Prepare Data
params.img_size  # resize image to this size
params.seq_len  # prepare sequence of frames
seqs = ['01','04','06','07','09', '10']
train_dl = create_data_loader(['_04'], batch_size=params.batch_size, seq_len_range=[3,5])


for it, (batch_x, batch_y) in enumerate(train_dl):
	print('Iteration %d' %it)
	print('batch x size: ', batch_x.size())
	print('batch y size: ', batch_y.size())
	#batch_x, batch_y = Variable(batch_x), Variable(batch_y)
	break

# Train
# toy data
x = Variable(torch.randn(1, 3, 6, params.img_size[0], params.img_size[1]).type(torch.FloatTensor))  # b_size, seq_len, channels(3*2andn(1, 3, 6).type(torch.FloatTensor))

#for i in range(params.epochs):
#	ls = M_deepvo.train(x, y)
#	print('loss:', ls)

