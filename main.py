from params import *
from model import DeepVO

import numpy as np
import time
import torch
from torch.autograd import Variable


img_size = (1280, 384)  #(320, 96)

save_path = '{},ep{},b{},lr{}'.format(
    params.solver,
    params.epochs,
    params.batch_size,
    params.lr)
###############################################################
# Load FlowNet weights pretrained with FlyingChairs
###############################################################
if params.pretrained:
	if params.cuda:
		pretrained_w = torch.load(params.pretrained)
	else:
		pretrained_w = torch.load(params.pretrained, map_location='cpu')
	print('load pretrained model')
else:
	pretrained_w = None

###############################################################
# Use only conv-layer-part of FlowNet as CNN for DeepVO
###############################################################
M_deepvo = DeepVO(img_size[0], img_size[1])#320, 96)
model_dict = M_deepvo.state_dict()
update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
model_dict.update(update_dict)
M_deepvo.load_state_dict(model_dict)
if params.cuda:
	M_deepvo = M_deepvo.cuda()

###############################################################
# Prepare Data
###############################################################



###############################################################
# Train
###############################################################
# toy data
x = Variable(torch.randn(1, 3, 6, img_size[0], img_size[1]).type(torch.FloatTensor))  # bat, seq, cha, im1, im2
y = Variable(torch.randn(1, 3, 6).type(torch.FloatTensor))

for i in range(params.epochs):
	ls = M_deepvo.train(x, y)
	print('loss:', ls)

