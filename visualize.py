import matplotlib.pyplot as plt
import numpy as np
import time

def plot_route(gt, out):
	x_idx = 3
	y_idx = 5
	step = 30

	x = [v for v in gt[:, x_idx]]
	y = [v for v in gt[:, y_idx]]
	for i in range(0, len(x), step):
		plt.plot(x[i:i+step], y[i:i+step], color='k', ls='--')
		plt.scatter(x[i], y[i], s=6, color='g')

	x = [v for v in out[:, x_idx]]
	y = [v for v in out[:, y_idx]]
	for i in range(0, len(x), step):
		plt.plot(x[i:i+step], y[i:i+step], color=[0, 0, 1, 0.5])
		plt.scatter(x[i], y[i], s=6, color='b')
	plt.gca().set_aspect('equal', adjustable='datalim')

	'''
	x = [[v for v in gt[:, x_idx]]]
	y = [[v for v in gt[:, y_idx]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color='k', ls='--')
		#plt.scatter(x[i], y[i], color='y')

	x = [[v for v in out[:, x_idx]]]
	y = [[v for v in out[:, y_idx]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color=[0, 0, 1, 0.7])
		#plt.scatter(x[i], y[i], color='r')
	'''


# Load in GT and predicted pose
video = '05'
pose_GT_dir = 'KITTI/pose_GT/'
result_dir = 'result/'
overfit =  '' #'overfit_' #''
print('Testing video {}'.format(video))

GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
gt = np.load(GT_pose_path)
pose_result_path = '{}{}out_{}.txt'.format(result_dir, overfit, video)
with open(pose_result_path) as f_out:
	out = [l.split('\n')[0] for l in f_out.readlines()]
	#gt, out = gt[ignore_first:], out[ignore_first:]
	for i, line in enumerate(out):
		out[i] = [float(v) for v in line.split(',')]
	out = np.array(out)
	print('out shape', out.shape)
	print('gt shape', gt.shape)
	# show some of result
	for i in range(100, 105):
		print('==========')
		print('out: ', out[i][-3], out[i][-1])
		print('gt:  ', gt[i][-3], gt[i][-1])



plot_route(gt, out)
plt.title('Video {}'.format(video))
plt.savefig('{}{}route_video_{}.png'.format(result_dir, overfit, video))