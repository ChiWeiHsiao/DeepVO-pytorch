__all__ = ['params', 'Parameters']

class Parameters():
	def __init__(self):
		self.cuda = False
		self.lr = 0.001
		self.solver = 'Adagrad'
		self.seq_len = 5
		self.img_size = (320, 96) #(1280, 384)  #(320, 96)
		self.batch_size = 64
		self.epochs = 10
		self.pretrained = './pretrained/flownets_bn_EPE2.459.pth.tar'
		#self.decrease_lr_at = [100,150,200]
		#self.decrease_lr = 0.5
		

params = Parameters()
