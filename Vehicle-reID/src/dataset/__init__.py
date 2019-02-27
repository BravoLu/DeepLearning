from __future__ import absolute_import 
import warnings
import transforms as T
from Preprocessor import *
from .vehicleID_V1 import VehicleID_V1
from torch.utils.data import DataLoader
NBR_ID = 85288 
NBR_MODELS = 250
NBR_COLORS = 7

__factory = {
    'vehicleid_v1.0':VehicleID_V1
}

def names():
    return sorted(__factory.keys())


def create(name, test_size):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)

    return __factory[name](test_size)

def get_data(name,
             batch_size,
             workers,
             test_size,
             img_width=224,
             img_height=224):
    
    data = create(name,test_size=test_size)
    
    train_set = data.train
    val_set   = data.val
    test_set  = data.test
    query_set = data.query
    gallery_set = data.gallery

    normalizer = T.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(img_height, img_width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])
    #train_transformer = T.RandomHorizontalFlip()
    
    test_transformer = T.Compose([
        T.RectScale(img_height, img_width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        #img id model color 
        Preprocessor(train_set, 
                     transform=train_transformer),
                     batch_size=batch_size,
                     num_workers=workers,
                     shuffle=True,
                     pin_memory=True,
    )
    

    val_loader = DataLoader(
        Preprocessor(val_set,
                     transform=test_transformer),
                     batch_size=batch_size,
                     num_workers=workers,
                     shuffle=False,
                     pin_memory=True,
    )

    test_loader = DataLoader(
        TestPreprocessor(test_set ,
                        transform=test_transformer),
                    batch_size=batch_size,
                    num_workers=workers,
                    shuffle=False,
                    pin_memory=True,
    )
    
    return train_set, val_set, query_set, gallery_set, train_loader,val_loader, test_loader
