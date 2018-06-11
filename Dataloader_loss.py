# %load Dataloader_loss.py


# In[ ]:


from params import par
from model import DeepVO
import cv2
import math

import numpy as np
import time
import torch

import os
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.modules import loss
from torch import functional as F


###############################################################
#Transform Rotation Matrix to Euler Angle
###############################################################
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

###############################################################
#DataLoader
###############################################################

#only fixed seq_len is used
class KITTI_Data(Dataset):
    def __init__(self,folder,seq_len): 
        
        #only store images address in dataloader 
        root_train = 'KITTI/images/{}/image_03/data'.format(folder)
        imgs = os.listdir(root_train)
        self.imgs = [os.path.join(root_train,img) for img in imgs]
        self.imgs.sort()
        self.GT = readGT('KITTI/pose_GT/{}.txt'.format(folder))
        
        self.seq_len = seq_len

    def __getitem__(self, index):
        #check index, CAUTION:the start/end of frames should be determined by GroundTruth file
        try:
        #Check the boundary, the lower boundary = frames-(seqs-1)-#StackNum
            self.GT[index+self.seq_len]
        except Exception:
            print("Error:Index OutofRange")
        filenames = []
        
        #load path to images,read image and resemble as RGB
        #NumofImgs = seqs + #StackNum
        filenames = [self.imgs[index+i] for i in range(self.seq_len+1)]
        images = [np.asarray(cv2.imread(img),dtype=np.float32) for img in filenames]

        #resemble images as RGB
        images = [img[:, :, (2, 1, 0)] for img in images]
        
        #Transpose the image that channels num. as first dimension
        images = [np.transpose(img,(2,0,1)) for img in images]
        images = [torch.from_numpy(img) for img in images]
        
        #stack per 2 images
        images = [np.concatenate((images[k],images[k+1]),axis = 0) for k in range(len(images)-1)]
                
        
        #prepare ground truth poses data

        #Stack the images for seqs
        return np.stack(images,axis = 0), self.GT[index:index+par.seq_len,:]

    def __len__(self):
        return self.GT.shape[0]-1-par.seq_len-1

#read groundtruth and return np.array
def readGT(root):
    with open(root, 'r') as posefile:
        #read ground truth
        GT = []
        for one_line in posefile:
            one_line = one_line.split(' ')
            one_line = [float(pose) for pose in one_line]
            #r = np.empty((3,3))
            #r[:] = ([one_line[0:3],one_line[4:7],one_line[8:11]])
            gt = np.append(rotationMatrixToEulerAngles(np.matrix([one_line[0:3],one_line[4:7],one_line[8:11]])),np.array([one_line[3],one_line[7],one_line[11]]))
            GT.append(gt)
    return np.array(GT,dtype=np.float32)
        
###############################################################
#Custom Loss Function
###############################################################

class DeepvoLoss(loss._Loss):
    def __init__(self, size_average=True, reduce=True):
        super(DeepvoLoss, self).__init__()    

    def forward(self, input,target):
        return F.mse_loss(input[0:3], target[0:3], size_average=self.size_average, reduce=self.reduce)+100 * F.mse_loss(input[3:6], target[3:6], size_average=self.size_average, reduce=self.reduce)
