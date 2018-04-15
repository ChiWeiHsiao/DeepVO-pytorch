import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal
import numpy as np

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO,self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        print('\ntmp_size', __tmp.size())


        # RNN
        #self.rnn = nn.LSTM(input_size=1024*20*6, hidden_size=1000, num_layers=2, batch_first=True)  # IMG_SIZE = (1241, 376), RNN_INPUT_SIZE = (1024, 20, 6)
        self.rnn = nn.LSTM(input_size=int(np.prod(__tmp.size())), hidden_size=1000, num_layers=2, batch_first=True)  # IMG_SIZE = (1241, 376), RNN_INPUT_SIZE = (1024, 20, 6)
        self.linear = nn.Linear(in_features=1000, out_features=6)


    def forward(self, x):
        # CNN
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        flatten = x.view(batch_size, seq_len, x.size(1)*x.size(2)*x.size(3))   # IMG_SIZE = (1280, 384), RNN_INPUT_SIZE = (1024, 20, 6)
        print('flatten:', flatten.size())
        # RNN
        #h0 = Variable(torch.zeros(2, batch_size, 1000))
        #c0 = Variable(torch.zeros(2, batch_size, 1000))
        h_n, c_n = self.rnn(flatten)#, (h0, c0))
        out = self.linear(h_n)
        print('out:', out.size())
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
