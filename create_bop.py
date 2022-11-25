import os
import shutil
import cv2
import argparse
from dataset.dataset import get_loader
from solver import Solver
# For BOP test
# base_path = '/data/storage/BOP'
# datasets = ['ycbv', 'lmo']
# img_id = 0
# f = open('./data/BOP/test.lst', 'w')

# for dataset in datasets:
#     obj_path = os.path.join(base_path, dataset, 'test_video')
#     for obj in os.listdir(obj_path):
#         img_base = os.path.join(obj_path, obj, 'rgb', "000000.jpg")
#         new_name = './data/BOP/Imgs/{:06d}.jpg'.format(img_id)
#         shutil.copyfile(img_base, new_name)
#         f.write('{:06d}.jpg\n'.format(img_id))
#         img_id += 1

# f.close()

# For demo test

video_path = 'data/P1/1237.avi'
img_id = 0
obj_name = video_path.split('/')[-1][0:-4]
rgb_path = 'data/P1/{}/rgb'.format(obj_name)
save_folder = 'data/P1/{}/mask'.format(obj_name)
os.makedirs(rgb_path,exist_ok=True)
f = open('data/P1/{}/test.lst'.format(obj_name), 'w')

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
while success:
    framecount = "{:06d}.jpg".format(count)
    jpg_path = os.path.join(rgb_path, framecount)
    cv2.imwrite(jpg_path, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    f.write('{:06d}.jpg\n'.format(count))
    count += 1
f.close()


if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Get test set info
parser = argparse.ArgumentParser()

# Train data
parser.add_argument('--train_root', type=str, default='')
parser.add_argument('--train_list', type=str, default='')

# Testing settings
parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
parser.add_argument('--pretrained_model', type=str, default='checkpoint/resnet50_caffe.pth')
parser.add_argument('--model', type=str, default='checkpoint/final.pth') # Snapshot
parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
parser.add_argument('--num_thread', type=int, default=1)
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.add_argument('--lr', type=float, default=5e-5) # Learning rate resnet:5e-5, vgg:1e-4
parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
parser.add_argument('--load', type=str, default='')


config = parser.parse_args()
test_root = rgb_path
test_list = 'data/P1/{}/test.lst'.format(obj_name)

config.test_root, config.test_list = test_root, test_list
config.test_fold = save_folder
test_loader = get_loader(config, mode='test')
test = Solver(None, test_loader, config)
test.test()

        