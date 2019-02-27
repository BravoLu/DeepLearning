"""
   Step 1.
   Prepare training and validation images and list files.
"""
import os
import random
import numpy as np

np.random.seed(1024)

image_root = '/root/shaohaolu/vehicleReid/data/VehicleID_V1.0/raw/image/'
attr_root = '/root/shaohaolu/vehicleReid/data/VehicleID_V1.0/raw/attribute/'
list_root = '/root/shaohaolu/vehicleReid/data/VehicleID_V1.0/raw/train_test_split/'

train_ratio = 0.9

train_list = '/root/shaohaolu/vehicleReid/data/VehicleID_V1.0/train_img_model_color.list'
f_train = open(train_list, 'w')
val_list = '/root/shaohaolu/vehicleReid/data/VehicleID_V1.0/val_img_model_color.list'
f_val = open(val_list, 'w')

model_attr_lines = open(os.path.join(attr_root, 'model_attr.txt')).readlines()
dic_vehicleID_modelID = { }
for line in model_attr_lines:
    vehicleID, modelID = line.strip().split(' ')
    dic_vehicleID_modelID[vehicleID] = modelID
print('dic_vehicleID_modelID: # vehicles: {}'.format(len(dic_vehicleID_modelID)))

color_attr_lines = open(os.path.join(attr_root, 'color_attr.txt')).readlines()
dic_vehicleID_colorID = { }
for line in color_attr_lines:
    vehicleID, colorID = line.strip().split(' ')
    dic_vehicleID_colorID[vehicleID] = colorID
print('dic_vehicleID_colorID: # vehicles: {}'.format(len(dic_vehicleID_colorID)))

vehicleIDs_list_ModelColor = [w for w in dic_vehicleID_modelID if w in dic_vehicleID_colorID]
print('# Vehicles which both has ModelIDs and ColorIDs: {}'.format(len(vehicleIDs_list_ModelColor)))

to_write_lines = [ ]
train_list_lines = open(os.path.join(list_root, 'train_list.txt')).readlines()
for line in train_list_lines:
    image_name, vehicleID = line.strip().split(' ')
    if vehicleID in vehicleIDs_list_ModelColor:
        # Is is for sure that the vehicleID is associated with
        # both modelIDs and colorIDs.
        image_path = os.path.join(image_root, image_name + '.jpg')
        modelID = dic_vehicleID_modelID[vehicleID]
        colorID = dic_vehicleID_colorID[vehicleID]
        to_write_lines.append('{} {} {} {}\n'.format(image_path, vehicleID, modelID, colorID))

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

print('{} and {}  has been generated!'.format(train_list, val_list))
