"""
   Step 1.
   Prepare training and validation images and list files.
"""
import os
import random
import numpy as np

np.random.seed(1024)

image_root = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/data/VehicleID_V1.0/image/'
attr_root = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/data/VehicleID_V1.0/attribute/'
list_root = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/data/VehicleID_V1.0/train_test_split/'

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

to_write_lines = [ ]
train_list_lines = open(os.path.join(list_root, 'train_list.txt')).readlines()
for line in train_list_lines:
    image_name, vehicleID = line.strip().split(' ')
    if vehicleID in dic_vehicleID_modelID:
        image_path = os.path.join(image_root, image_name + '.jpg')
        modelID = dic_vehicleID_modelID[vehicleID]
        to_write_lines.append('{} {}\n'.format(image_path, modelID))

nbr_train = int(train_ratio * len(to_write_lines))
random.shuffle(to_write_lines)

train_to_write_lines = to_write_lines[:nbr_train]
val_to_write_lines = to_write_lines[nbr_train:]

print('# train data: {}, # val data: {}'.format(len(train_to_write_lines), len(val_to_write_lines)))

for line in train_to_write_lines:
    f_train.write(line)

for line in val_to_write_lines:
    f_val.write(line)

f_train.close()
f_val.close()

print('train_vehicleModel_list.txt and val_vehicleModel_list.txt has been generated!')
