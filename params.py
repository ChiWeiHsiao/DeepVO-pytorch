class Parameters():
	def __init__(self):
		self.cuda = False
		self.solver = 'adam'
		self.lr = 0.0001
		self.batch_size = 8
		self.epochs = 300
		self.decrease_lr_at = [100,150,200]
		self.decrease_lr = 0.5
		self.pretrained = './pretrained/flownets_bn_EPE2.459.pth.tar'
	