"""
   Step 1.
   Prepare training and validation images and list files.
"""
import os
import random
import numpy as np

np.random.seed(1024)

image_root = '/data/zhenju/00_Project/00_VehicleSearch/data/VehicleID_V1.0/image/'
attr_root = '/data/zhenju/00_Project/00_VehicleSearch/data/VehicleID_V1.0/attribute/'
list_root = '/data/zhenju/00_Project/00_VehicleSearch/data/VehicleID_V1.0/train_test_split/'

train_ratio = 0.9

train_list = 'train_vehicleModel_list.txt'
f_train = open(train_list, 'w')
val_list = 'val_vehicleModel_list.txt'
f_val = open(val_list, 'w')

model_attr_lines = open(os.path.join(attr_root, 'model_attr.txt')).readlines()
dic_vehicleID_modelID = { }
for line in model_attr_lines:
    vehicleID, modelID = line.strip().split(' ')
    dic_vehicleID_modelID[vehicleID] = modelID
print('dic_vehicleID_modelID: # vehicles: {}'.format(len(dic_vehicleID_modelID)))


img2vid_lines = open(os.path.join(attr_root, 'img2vid.txt')).readlines()
dic_img_vehicleID = { }
dic_modelID_imgPath = { }
for line in img2vid_lines:
    image_name, vehicleID = line.strip().split(' ')
    dic_img_vehicleID[image_name] = vehicleID
    if vehicleID in dic_vehicleID_modelID:
        image_name += '.jpg'
        modelID = dic_vehicleID_modelID[vehicleID]
        image_path = os.path.join(image_root, image_name)
        dic_modelID_imgPath.setdefault(modelID, [ ]).append(image_path)
print('dic_img_vehicleID: # images: {}'.format(len(dic_img_vehicleID)))

print('# vehicle models: {}'.format(len(dic_modelID_imgPath)))
for modelID in dic_modelID_imgPath:
    images_list = dic_modelID_imgPath[modelID]

    # If the modelID is associated with more than 10 images,
    # we split the images into train and validation.
    # Otherwise, we put them into train.
    if len(images_list) > 20:
        tmp_nbr_train = int(train_ratio * len(images_list))
        tmp_train_list = random.sample(images_list, tmp_nbr_train)
        tmp_val_list = [w for w in images_list if w not in tmp_train_list]

        for image_path in tmp_train_list:
            to_write = '{} {}\n'.format(image_path, modelID)
            f_train.write(to_write)

        for image_path in tmp_val_list:
            to_write = '{} {}\n'.format(image_path, modelID)
            f_val.write(to_write)
    else:
        for image_path in images_list:
            to_write = '{} {}\n'.format(image_path, modelID)
            f_train.write(to_write)

print('train_vehicleModel_list.txt and val_vehicleModel_list.txt has been generated!')
