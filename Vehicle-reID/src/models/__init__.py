from __future__ import absolute_import

from .mobilenetv1 import *
from .mobilenetv2 import *
from .resnet import *
from .vgg import *
from .MDNet import *
from .RAJNet import *
from .ADFLNet import *
__factory = {
    'mobilenetv1': mobilenetv1,
    'mobilenetv2': mobilenetv2,
    'ResNet18'   : resnet18,
    'ResNet34'   : resnet34,
    'ResNet50'   : resnet50,
    'ResNet101'  : resnet101,
    'ResNet152'  : resnet152,
    'vgg'        : Vgg,
    'MDNet'      : MDNet,
    'RAJNet18'   : RAJNet18,
    'RAJNet34'   : RAJNet34,
    'RAJNet50'   : RAJNet50,
    'RAJNet101'  : RAJNet101,
    'RAJNet152'  : RAJNet152,
    'vgg16'      : Vgg,
    'ADFLNet'     : ADFLNet
}


def name():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
