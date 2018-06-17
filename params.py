import os

class Parameters():
	def __init__(self):
		self.n_processors = 4
		# Path
		self.image_dir = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/images/'
		self.pose_dir = '/nfs/nas12.ethz.ch/fs1201/infk_ivc_students/cvg-students/chsiao/KITTI/pose_GT/'
		self.train_video = ['00', '02', '08', '09', '01', '04', '05', '06', '07', '10']  # 09
		self.valid_video = []
		self.partition = 0.7   # None # partition videos in 'train_video' to train / valid dataset

		# Data Preprocessing
		self.resize_mode = 'rescale' # 'crop' 'rescale' None
		self.img_w = 608  # 608  613  1226  1024 896
		self.img_h = 184  # 184  185  370   312 272
		self.img_means = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)  #(0,0,0)
		self.img_stds = (1, 1, 1)  #(0.309122, 0.315710, 0.3226514)  #(1, 1, 1) 
		self.minus_point_5 = True
		#(1,1,1)
		#(0.31934219855028534, 0.3220230463601085, 0.32343616609004483)
		#(81.43226016669172, 82.11587565987408, 82.47622141551435)
		self.seq_len = [7, 9]  # [8, 10]
		self.sample_times = 3  # 1

		# Model
		self.rnn_hidden_size = 1000  # 1000 500
		#self.conv_dropout = 0.3
		self.rnn_dropout_in = 0.5
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0.5  # 0: no dropout
		self.clip = None # 5
		self.batch_norm = True
		# Training
		self.batch_size = 16 # 64 8 128
		self.pin_mem = True
		self.epochs = 100
		self.optim = {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}
		
		# Pretrain, Retrain
		self.pretrained_flownet = None
								# None
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

