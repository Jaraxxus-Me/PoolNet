import os
import shutil

base_path = '/data/storage/BOP'
datasets = ['ycbv', 'lmo']
img_id = 0
f = open('./data/BOP/test.lst', 'w')

for dataset in datasets:
    obj_path = os.path.join(base_path, dataset, 'test_video')
    for obj in os.listdir(obj_path):
        img_base = os.path.join(obj_path, obj, 'rgb', "000000.jpg")
        new_name = './data/BOP/Imgs/{:06d}.jpg'.format(img_id)
        shutil.copyfile(img_base, new_name)
        f.write('{:06d}.jpg\n'.format(img_id))
        img_id += 1

f.close()

        