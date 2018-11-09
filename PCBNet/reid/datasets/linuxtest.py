# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-31 20:37:26
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-01 09:39:49

import os.path as osp
import argparse

parser = argparse.ArgumentParser(description="Softmax loss classification")
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))

def main(args):
	root = args.data_dir
	image_dir = osp.join(root)
	train_path = 'bounding_box_train'
	print(osp.join(image_dir, train_path, '*.jpg'))


if __name__ == '__main__':
	main(parser.parse_args())