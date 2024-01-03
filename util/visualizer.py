import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.name = opt.name
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(self.web_dir, 'images')
        util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            util.save_image(image_numpy, img_path)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.4f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    # save image to the disk
    def save_images(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        for label, image_numpy in visuals.items():
            if label=='x_hat':
                image_name = '%s.png' % (name)
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                save_path = os.path.join(image_dir, image_name)
                util.save_image(image_numpy, save_path)
