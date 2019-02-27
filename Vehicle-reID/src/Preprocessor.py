from __future__ import absolute_import 
import os
from PIL import Image

class Preprocessor(object):
    def __init__(self, dataset, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, indices):
        if isinstance(indices, (tuple,list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)
        #pass

    def _get_single_item(self,index):
        imgpath, ID, model, color = self.dataset[index]
        img = Image.open(imgpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, ID, model, color
    
class TestPreprocessor(object):
    def __init__(self, dataset, transform=None):
        super(TestPreprocessor, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, indices):
        if isinstance(indices, (tuple,list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        imgpath, imgName, ID = self.dataset[index] 
        img = Image.open(imgpath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
    
        return img, imgName, ID    
