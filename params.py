class Parameters():
	def __init__(self):
		self.cuda = False
		self.lr = 0.001
		self.seq_len = 5
		self.img_size = (1280, 384)  #(320, 96)
		self.batch_size = 8
		self.epochs = 300
		self.pretrained = './pretrained/flownets_bn_EPE2.459.pth.tar'
		self.dropout
		#self.decrease_lr_at = [100,150,200]
		#self.decrease_lr = 0.5
		

params = Parameters()
