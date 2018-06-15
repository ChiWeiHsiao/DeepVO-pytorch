import os

class Parameters():
	def __init__(self):
		self.n_processors = 4
		# Path
		self.image_dir = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/images/'
		self.pose_dir = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/pose_GT/'
		self.train_video = ['00', '02', '08', '09']  # 09
		self.valid_video = ['06', '05']  #07 09

		# Data Preprocessing
		self.resize_mode = 'crop' # 'rescale' None
		self.img_w = 1200  # 608  613  1226  1200
		self.img_h = 360  # 184  185  370   360
		self.subtract_means = [89.87475578450945/255, 94.48404712783562/255, 92.50648653696369/255]  # caculated with video 00, 02, 08, 09
		self.seq_len = [5, 5]  # [8, 10]
		self.sample_interval = 2  # None (only for fixed seq_len)

		# Model
		self.rnn_hidden_size = 500  # 1000 500
		self.dropout = 0.5  # 0: no dropout
		self.clip = None # 5
		self.batch_norm = True
		# Training
		self.batch_size = 4  # 64 8 128
		self.pin_mem = False
		self.epochs = 200
		self.optim = {'opt': 'Adam'}
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
		
		# Pretrain, Retrain
		self.pretrained_flownet = None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'  
								# './pretrained/flownets_EPE1.951.pth.tar'
		self.resume = False
		self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.load_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))

		self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		
		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
		if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
			os.makedirs(os.path.dirname(self.save_optimzer_path))

		# Write all hyperparameters to record_path
		p = vars(self)
		mode = 'a' if self.resume else 'w'
		with open(self.record_path, mode) as f:
			f.write('\n'.join("%s: %s" % item for item in p.items()))
			f.write('\n'+'='*50 + '\n')

par = Parameters()

