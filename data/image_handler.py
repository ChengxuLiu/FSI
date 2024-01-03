import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
NPY_EXTENSIONS = ['.npy']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)




def make_dataset_image_synth_train(dir):
    imagesX = []
    imagesY = []
    for root, _, fnames in sorted(os.walk(dir)):
        if 'GT/train' in root:
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesX.append(path)
            imagesX = imagesX * 9
        elif os.path.basename(root) == 'train':
            if 'ZTE_new_' in root:
                for fname in sorted(fnames):
                    if is_image_file(fname) or is_npy_file(fname):
                        path = os.path.join(root, fname)
                        imagesY.append(path)
    return imagesX, imagesY



def make_dataset_image_synth_test(dir):
    imagesX = []
    imagesY = []
    for root, dir, fnames in sorted(os.walk(dir)):
        if 'GT/test' in root:
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesX.append(path)
        elif os.path.basename(root) == 'test':
            if 'ZTE_new_5' in root:
                for fname in sorted(fnames):
                    if is_image_file(fname) or is_npy_file(fname):
                        path = os.path.join(root, fname)
                        imagesY.append(path)
    return imagesX, imagesY



def make_dataset_image_toled(dir):
    imagesX = []
    imagesY = []
    for root, _, fnames in sorted(os.walk(dir)):
        if os.path.basename(root) == 'sharp' or os.path.basename(root) == 'GT' or os.path.basename(root) == 'HQ':
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesX.append(path)
        elif os.path.basename(root) == 'blur' or os.path.basename(root) == 'input' or os.path.basename(root) == 'LQ':
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesY.append(path)

    return imagesX, imagesY


def make_dataset_image_poled(dir):
    imagesX = []
    imagesY = []
    for root, _, fnames in sorted(os.walk(dir)):
        if os.path.basename(root) == 'sharp' or os.path.basename(root) == 'GT' or os.path.basename(root) == 'HQ':
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesX.append(path)
        elif os.path.basename(root) == 'blur' or os.path.basename(root) == 'input' or os.path.basename(root) == 'LQ_pre2':
            for fname in sorted(fnames):
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesY.append(path)
    return imagesX, imagesY

