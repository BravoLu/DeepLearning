import os
import numpy as np 
import os.path as osp
import random

TRAIN_PATH='train_img_model_color.list'
VAL_PATH='val_img_model_color.list'

class Dataset(object):
    def __init__(self, root, test_size=800, split_id=0):
        self.root = root
        assert test_size in [800, 1600, 2400, 3200, 6000, 13164]
        self.tpath = osp.join(self.root,'raw/train_test_split', 'test_list_{}.txt'.format(str(test_size))) 
        self.train, self.val, self.query, self.gallery, self.test = [], [], [], [], []
        self.train_num = 0
        self.val_num = 0
        self.gallery_num = 0
        self.query_num = 0
    def load(self, verbose=True):   
        with open(os.path.join(self.root,TRAIN_PATH),'r') as train_file, open(os.path.join(self.root,VAL_PATH),'r') as val_file:     
            train_list = train_file.readlines()
            val_list = val_file.readlines()
            self.train_num = len(train_list)
            self.val_num   = len(val_list)
            for item in train_list:
                item = item.strip().split(' ')
                img = item[0]
                ID = int(item[1])
                model = int(item[2])
                color = int(item[3])
                self.train.append([img,ID,model,color])
            for item in val_list:
                item = item.strip().split(' ')
                img = item[0]
                ID  = int(item[1])
                model = int(item[2])
                color = int(item[3])
                self.val.append([img,ID,model,color])
        with open(self.tpath, 'r') as test_file:
            test_data_lines = test_file.readlines()
            self.test_num = len(test_data_lines)
            test_imgName_list = [w.strip().split(' ')[0] for w in test_data_lines]
            test_imgPath_list = [osp.join(self.root,'raw/image',w+'.jpg') for w in test_imgName_list]
            test_vehicleIDs_list = [w.strip().split(' ')[-1] for w in test_data_lines]
            dic_test_vehicleID_imgName = {}
            dic_test_imgName_vehicleID = {}
            
            for imgName, vehicleID in zip(test_imgName_list, test_vehicleIDs_list):
                dic_test_vehicleID_imgName.setdefault(vehicleID, []).append(imgName)
                dic_test_imgName_vehicleID[imgName] = vehicleID
        #query_imgNames = []
        #gallery_imgNames = []

        for vehicleID in dic_test_vehicleID_imgName:
            imgNames = dic_test_vehicleID_imgName[vehicleID]
            sampled_idx = random.randint(0,len(imgNames)-1)
            sampled_imgName = imgNames[sampled_idx]
            imgPath = osp.join(self.root,'raw/image', sampled_imgName+'.jpg')
            self.gallery.append([imgPath, sampled_imgName, vehicleID])
            self.test.append([imgPath, sampled_imgName, vehicleID])
            #gallery_imgNames.append(sampled_imgName)
            imgNames.remove(sampled_imgName) 
            #query_imgNames += imgNames
            for imgName in imgNames:
                imgPath = osp.join(self.root, 'raw/image', imgName+'.jpg')
                self.query.append([imgPath, imgName, vehicleID])
                self.test.append([imgPath, imgName, vehicleID])
            self.gallery_num = len(self.gallery)
            self.query_num   = len(self.query)
            self.test_num = len(self.test)
        if verbose:
            print(self.__class__.__name__,"dataset loaded")
            print("  set      | # images  ")
            print("  train    | # {}      ".format(self.train_num))
            print("  val      | # {}      ".format(self.val_num))
            print("  test     | # {}      ".format(self.test_num))
            print("  gallery  | # {}      ".format(self.gallery_num))
            print("  query    | # {}      ".format(self.query_num))
            print("  ----------------  ") 
    
