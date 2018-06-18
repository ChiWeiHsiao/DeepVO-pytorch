import matplotlib.pyplot as plt
import numpy as np
import time

def plot_route(gt, out, c_gt='g', c_out='r'):
	x_idx = 3
	y_idx = 5
	x = [[v for v in gt[:, x_idx]]]
	y = [[v for v in gt[:, y_idx]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color=c_gt)
		#plt.scatter(x[i], y[i], color='b')

	x = [[v for v in out[:, x_idx]]]
	y = [[v for v in out[:, y_idx]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color=c_out)
		#plt.scatter(x[i], y[i], color='b')
	plt.gca().set_aspect('equal', adjustable='datalim')


# Load in GT and predicted pose
video_list = ['04', '05', '07', '10', '09']

for video in video_list:
	pose_GT_dir = 'KITTI/pose_GT/'
	result_dir = 'result_ep37/'
	print('Testing video {}'.format(video))

	GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
	gt = np.load(GT_pose_path)
	pose_result_path = '{}out_{}.txt'.format(result_dir, video)
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

	step = 200
	# plot gradient color
	plt.clf()
	for st in range(0, len(out), step):
		end = st + step
		g = max(0.2, st/len(out))
		c_gt = (0, g, 0)
		c_out = (1, g, 0)
		plot_route(gt[st:end], out[st:end], c_gt, c_out)
		plt.title('Video {}'.format(video))
		save_name = '{}route_video_{}.png'.format(result_dir, video)
	plt.savefig(save_name)

	# plot one color
	#plot_route(gt, out)
	#plt.title('Video {}'.format(video))
	#save_name = '{}route_video_{}.png'.format(result_dir, video)
	#plt.savefig(save_name)
