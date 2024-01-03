import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.exp_name = 'SYNTH' # 'POLED'  or 'TOLED' or 'SYNTH'

    def initialize(self):
        if self.exp_name == 'POLED': ## low-light  colorshift noise
            # Data and pre-trained weights for POLED dataset
            self.parser.add_argument('--dataroot', type=str, default='/data/UDCIE/dataset/POLED/train', help='path to images')
            self.parser.add_argument('--testroot', type=str, default='/data/UDCIE/dataset/POLED/test', help='path to images')
            self.parser.add_argument('--dataset', type=str, default='POLED', help='dataset_name')
            self.parser.add_argument('--name', type=str, default='POLED', help='name of the experiment. It decides where to store samples and models')
        elif self.exp_name == 'TOLED': ## blur noise
            # Data and pre-trained weights for TOLED dataset
            self.parser.add_argument('--dataroot', type=str, default='/data/UDCIE/dataset/TOLED/train', help='path to images')
            self.parser.add_argument('--testroot', type=str, default='/data/UDCIE/dataset/TOLED/test', help='path to images')
            self.parser.add_argument('--dataset', type=str, default='TOLED', help='dataset_name')
            self.parser.add_argument('--name', type=str, default='TOLED', help='name of the experiment. It decides where to store samples and models')
        elif self.exp_name == 'SYNTH': ## flare
            # Data and pre-trained weights for Synthetic dataset
            self.parser.add_argument('--dataroot', type=str, default='/data/UDCIE/dataset/synthetic_data', help='path to images')
            self.parser.add_argument('--testroot', type=str, default='/data/UDCIE/dataset/synthetic_data', help='path to images')
            self.parser.add_argument('--dataset', type=str, default='SYNTH', help='dataset_name')
            self.parser.add_argument('--name', type=str, default='SYNTH', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test
        
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.phase)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
