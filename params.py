__all__ = ['params', 'Parameters']

class Parameters():
	def __init__(self):
		self.lr = 0.001
		self.solver = 'Adagrad'
		self.seq_len = [5, 5]
		self.batch_size = 8 #64
		self.epochs = 10
		self.rnn_hidden_size = 500  #1000
		self.pretrained = './pretrained/flownets_bn_EPE2.459.pth.tar'
		self.img_size = (370, 1226)
		#self.decrease_lr_at = [100,150,200]
		#self.decrease_lr = 0.5
		

params = Parameters()
