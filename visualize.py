import matplotlib.pyplot as plt
import numpy as np
import time

def plot_route(gt, out):
	x = [[v for v in gt[:, -3]]]
	y = [[v for v in gt[:, -1]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color='k')
		#plt.scatter(x[i], y[i], color='y')

	x = [[v for v in out[:, -3]]]
	y = [[v for v in out[:, -1]]]
	for i in range(len(x)):
		plt.plot(x[i], y[i], color='r')
		#plt.scatter(x[i], y[i], color='b')


# Load in GT and predicted pose
video = '09'  # 01 04 10
overfit =  '' #'overfit_' #''
print('Testing video {}'.format(video))

GT_pose_path = '{}{}.npy'.format(par.pose_dir, video)  #'KITTI/pose_GT/{}.npy'.format(video)
gt = np.load(GT_pose_path)
pose_result_path = 'result/{}out_{}.txt'.format(overfit, video)
#pose_result_path = 'result/out_{}.txt'.format(video)
with open(pose_result_path) as f_out:
	out = [l.split('\n')[0] for l in f_out.readlines()]
	#gt, out = gt[ignore_first:], out[ignore_first:]
	for i, line in enumerate(out):
		out[i] = [float(v) for v in line.split(',')]
	out = np.array(out)
	print('out shape', out.shape)
	print('gt shape', gt.shape)
	for i in range(0, 5):
		print('==========')
		print('out: ', out[i][-3], out[i][-1])
		print('gt:  ', gt[i][-3], gt[i][-1])


plot_route(gt, out)
plt.title('Video {}'.format(video))
plt.savefig('{}route_video_{}.png'.format(overfit, video))

'''
rs = [[i, i+400] for i in range(0, 700, 100)]
for r in rs:
	plot_route(gt[r[0]:r[1]], out[r[0]:r[1]])
	plt.title('%d - %d' %(r[0], r[1]))
	plt.savefig('{}_{}~{}.png'.format(video, r[0], r[1]))
	#input('Current %d %d,  Press to continue...\n' %(r[0], r[1]))
	plt.clf()
'''
