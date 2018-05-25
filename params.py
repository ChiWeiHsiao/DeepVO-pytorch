__all__ = ['params', 'Parameters']

class Parameters():
	def __init__(self):
		self.lr = 0.001
		self.solver = 'Adagrad'  # Adagrad Cosine
		self.seq_len = [5, 5]
		self.batch_size = 16  # 64 8 128
		self.epochs = 1000
		self.rnn_hidden_size = 500  #1000
		self.pretrained = './pretrained/flownets_bn_EPE2.459.pth.tar'

		self.img_w = 613  # 613  1226
		self.img_h = 185  # 185  370

		self.use_video = ['01', '07', '09']
		self.load_data = ['KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}.npz'.format(video, self.seq_len[0], self.seq_len[1], self.img_h, self.img_w) for video in self.use_video]
		#['KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}.npz'.format('01', self.seq_len[0], self.seq_len[1], self.img_h, self.img_w),]  # 01_seq_11_11_im_370_1226.npz  01_seq_11_11_im_185_613.npz
		self.record_name = 'records/data_{}_im_{}_{}_s_{}_{}_b_{}_rnn_{}'.format('_'.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size)
		self.save_model_name = 'models/data_{}_im_{}_{}_s_{}_{}_b_{}_rnn_{}'.format('_'.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size)
params = Parameters()
