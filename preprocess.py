import os


def clean_unused_images():
	seq_frame = {'00': ['000000', '004540'],
				'01': ['000000', '001100'],
				'02': ['000000', '004660'],
				'03': ['000000', '000800'],
				'04': ['000000', '000270'],
				'05': ['000000', '002760'],
				'06': ['000000', '001100'],
				'07': ['000000', '001100'],
				'08': ['001100', '005170'],
				'09': ['000000', '001590'],
				'10': ['000000', '001200']
				}
	for dir_id, img_ids in seq_frame.items():
		dir_path = 'KITTI/images/{}/'.format(dir_id)
		if not os.path.exists(dir_path):
			continue

		print('Cleaning {} directory'.format(dir_id))
		start, end = img_ids
		start, end = int(start), int(end)
		for idx in range(0, start):
			img_name = '{:010d}.png'.format(idx)
			img_path = 'KITTI/images/{}/image_03/data/{}'.format(dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)
		for idx in range(end+1, 10000):
			img_name = '{:010d}.png'.format(idx)
			img_path = 'KITTI/images/{}/image_03/data/{}'.format(dir_id, img_name)
			if os.path.isfile(img_path):
				os.remove(img_path)


clean_unused_images()