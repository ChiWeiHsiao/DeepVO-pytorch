import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import time
from params import par


def get_data_info(folder_list, seq_len_range, overlap, sample_interval, pad_y=False):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load('{}{}.npy'.format(par.pose_dir, folder))  # (n_images, 6)
        fpaths = glob.glob('{}{}/*.png'.format(par.image_dir, folder))
        fpaths.sort()
        # Fixed seq_len
        if seq_len_range[0] == seq_len_range[1]:
            if sample_interval:
                start_frames = list(range(0, seq_len_range[0], sample_interval))
                print('Sample start from frame {}'.format(start_frames))
            else:
                start_frames = [0]

            for st in start_frames:
                seq_len = seq_len_range[0]
                n_frames = len(fpaths) - st
                jump = seq_len - overlap
                res = n_frames % seq_len
                if res != 0:
                    n_frames = n_frames - res
                x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
                if len(x_segs[-1]) < seq_len:
                    x_segs = x_segs[:-1]
                y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
                if len(y_segs[-1]) < seq_len:
                    y_segs = y_segs[:-1]
                Y += y_segs
                X_path += x_segs
                X_len.append(len(x_segs))
        # Random segment to sequences with diff lengths
        else:
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            start = 0
            while True:
                n = np.random.random_integers(min_len, max_len)
                if start + n < n_frames:
                    x_seg = fpaths[start:start+n] 
                    X_path.append(x_seg)
                    if not pad_y:
                        Y.append(poses[start:start+n])
                    else:
                        pad_zero = np.zeros((max_len-n, 6))
                        padded = np.concatenate((poses[start:start+n], pad_zero))
                        Y.append(padded.tolist())
                else:
                    print('Last %d frames is not used' %(start+n-n_frames))
                    break
                start += n - overlap
                X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Sort dataframe by seq_len
    df = df.sort_values(by=['seq_len'], ascending=False)
    return df


class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last=False):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)
        # Calculate number of sameples in each group (grouped by seq_len)
        list_batch_indexes = []
        start_idx = 0
        for v in unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return len(self.df)


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize=None, subtract_mean=None):
        # Transforms
        if resize != None:
            self.transformer = transforms.Compose([
                            transforms.Resize((resize[0], resize[1])),                                                                            
                            transforms.ToTensor(),
                            ])
        else:
            self.transformer = transforms.ToTensor()
        self.subtract_mean = subtract_mean
        self.data_info = info_dataframe
        self.seq_len_list = list(df.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        groundtruth_sequence = self.groundtruth_arr[index]
        groundtruth_sequence = torch.tensor(groundtruth_sequence)
        
        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])
        #sequence_len = torch.tensor(len(image_path_sequence))
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.subtract_mean != None:
                for c in range(3):
                    img_as_tensor[c] -= self.subtract_mean[c]
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)



if __name__ == '__main__':
    start_t = time.time()
    # Gernerate info dataframe
    overlap = 1
    sample_interval = None
    folder_list = ['00']
    seq_len_range = [5, 7]
    df = get_data_info(folder_list, seq_len_range, overlap, sample_interval)
    print('Elapsed Time (get_data_info): {} sec'.format(time.time()-start_t))
    # Customized Dataset, Sampler
    n_workers = 4
    resize = (150, 600)
    subtract_mean = (89.87475578450945/255, 94.48404712783562/255, 92.50648653696369/255)
    dataset = ImageSequenceDataset(df, resize, subtract_mean)
    sorted_sampler = SortedRandomBatchSampler(df, batch_size=4, drop_last=True)
    dataloader = DataLoader(dataset, 
                            batch_sampler=sorted_sampler, 
                            num_workers=n_workers)
    print('Elapsed Time (dataloader): {} sec'.format(time.time()-start_t))

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))
    
    print('Elapsed Time: {} sec'.format(time.time()-start_t))
    print('Number of workers = ', n_workers)

