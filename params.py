import os

class Parameters():
	def __init__(self):
		self.optim = {'opt': 'Adam'}
					# {'opt': 'Adagrad', 'lr': 0.01}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

		self.img_w = 608  # 608  613  1226 
		self.img_h = 184  # 184  185  370 
		self.seq_len = [8, 8]  # [8, 10]

		self.batch_size = 8  # 64 8 128
		self.epochs = 500
		self.rnn_hidden_size = 500  #1000 500

		self.use_video = ['01', '07', '09']
		self.valid_video = ['04']
		self.RGB_means = [90.72659, 95.17316, 94.42595]  # caculated with videos 01+07+09

		self.train_data_path = ['KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}.npz'.format(video, self.seq_len[0], self.seq_len[1], self.img_h, self.img_w) for video in self.use_video]
		self.valid_data_path = ['KITTI/segmented_image/{}_seq_{}_{}_im_{}_{}.npz'.format(video, self.seq_len[0], self.seq_len[1], self.img_h, self.img_w) for video in self.valid_video]

		self.pretrained_flownet = './pretrained/flownets_bn_EPE2.459.pth.tar'  #'None'
		# Retrain
		self.resume = False
		self.load_model_path = 'models/v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.load_optimzer_path = 'models/v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))

		self.record_path = 'records/v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_model_path = 'models/v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_optimzer_path = 'models/v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.use_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		
		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
		if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
			os.makedirs(os.path.dirname(self.save_optimzer_path))

		# Write all hyperparameters to record_path
		p = vars(self)
		with open(self.record_path, 'a') as f:
			f.write('\n'.join("%s: %s" % item for item in p.items()))
			f.write('\n'+'='*50 + '\n')

params = Parameters()

