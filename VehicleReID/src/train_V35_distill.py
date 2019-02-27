"""
   Step 2.
   Train a InceptionV3 ConvNet which predicts both vehicles and colors.
   It is a multi-task learning process.
   Add the triplet-loss branch.
"""
import os
GPUS = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
from math import ceil
import numpy as np
import copy
import keras.backend as K
from keras.applications import mobilenet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Input, concatenate, subtract, dot, Activation, add, merge, Lambda
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, RMSprop
from sklearn.utils import class_weight
from utils import generator_batch_triplet, generator_batch, generator_batch_distilling
# from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model
from loss import triplet_loss, identity_loss, MARGIN
#from models import model_from_json
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
np.random.seed(1024)

FINE_TUNE = True
LEARNING_RATE = 0.00001
NBR_EPOCHS = 200
BATCH_SIZE = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
RANDOM_SCALE = True
nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 86 

train_path = 'train_vehicleModelColor_list_filtered.txt_reid'
val_path='val_vehicleModelColor_list.txt_reid'

'''
    read feature map : adhoc code, to be remove by zju
'''
train_vehicle_id_file='feature/V35_0003/Train/id.idx'
train_feature_1024_file='feature/V35_0003/Train/1024_c.npy'
train_feature_256_file='feature/V35_0003/Train/256_c.npy'
train_vehicle_id_idx=open(train_vehicle_id_file,'r').readlines()
train_feature_1024=np.load(train_feature_1024_file)
train_feature_256=np.load(train_feature_256_file)
train_feature_1024_map={}
train_feature_256_map={}
for idx, line in enumerate(train_vehicle_id_idx):
    train_feature_1024_map[line[:-1]]=train_feature_1024[idx]
    train_feature_256_map[line[:-1]]=train_feature_256[idx]

def filter_data_list(data_list):
    # data_list  : a list of [img_path, vehicleID, modelID, colorID]
    # {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    # dic helps us to sample positive and negative samples for each anchor.
    # https://arxiv.org/abs/1708.02386
    # The original paper says that "only the hardest triplets in which the three images have exactly
    # the same coarse-level attributes (e.g. color and model), can be used for similarity learning."
    dic = { }
    # We construct a new data list so that we could sample enough positives and negatives.
    new_data_list = [ ]
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        dic.setdefault(modelID, { })
        dic[modelID].setdefault(colorID, { })
        dic[modelID][colorID].setdefault(vehicleID, [ ]).append(imgPath)

    # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        #print(imgPath, vehicleID, modelID, colorID)
        if modelID in dic and colorID in dic[modelID] and vehicleID in dic[modelID][colorID] and \
                                                      len(dic[modelID][colorID][vehicleID]) == 1:
            dic[modelID][colorID].pop(vehicleID, None)

    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        if modelID in dic and colorID in dic[modelID] and len(dic[modelID][colorID].keys()) == 1:
            dic[modelID].pop(colorID, None)

    for modelID in dic:
        for colorID in dic[modelID]:
            for vehicleID in dic[modelID][colorID]:
                for imgPath in dic[modelID][colorID][vehicleID]:
                    new_data_list.append('{} {} {} {}'.format(imgPath, vehicleID, modelID, colorID))

    print('The original data list has {} samples, the new data list has {} samples.'.format(
                                 len(data_list), len(new_data_list)))
    return new_data_list, dic

if __name__ == "__main__":

    if FINE_TUNE:
        teacher_model_path='###step2_model###'
        teacher= load_model(teacher_model_path,custom_objects={'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D, 'identity_loss': identity_loss, 'triplet_loss': triplet_loss, 'MARGIN': MARGIN})
        teacher.summary()
        model=Model(inputs=teacher.get_layer('model_1').get_input_at(node_index = 0), outputs=[teacher.get_layer('model_1').get_layer(name='f_acs_new').output, teacher.get_layer('model_1').get_layer(name = 'sls3').output])
        model.summary()
        #model=Model(input=teacher.input, output=teacher.output)
    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
    #optimizer = RMSprop(lr = LEARNING_RATE)

    if nbr_gpus > 1:
        print('Using {} GPUS.\n'.format(nbr_gpus))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    model.compile(loss=["mean_squared_error", "mean_squared_error"],
                  optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    model_file_saved = "./models/V35/V35-average_epoch={epoch:04d}-loss={loss:.4f}-val_loss={val_loss:.4f}.h5"
    # Define several callbacks

    checkpoint = ModelCheckpoint(model_file_saved, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=5, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    train_data_lines, dic_train_data_lines = filter_data_list(train_data_lines)
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))
    '''
    if SAVE_FILTERED_LIST:
        # Write filtered data lines into disk.
        filtered_train_list_path = './train_vehicleModelColor_list_filtered.txt'
        f_new_train_list = open(filtered_train_list_path, 'w')
        for line in train_data_lines:
            f_new_train_list.write(line + '\n')
        f_new_train_list.close()
        print('{} has been successfully saved!'.format(filtered_train_list_path))
    '''    
    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))
    
    model.fit_generator(generator_batch_distilling(train_data_lines,train_feature_1024_map , train_feature_256_map,
                        mode = 'train', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_distilling(val_data_lines, train_feature_1024_map , train_feature_256_map, 
                        mode = 'val', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 100, workers = 10, use_multiprocessing=True)
'''
    model.fit_generator(generator_batch_triplet(train_data_lines, dic_train_data_lines,
                        mode = 'train', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_triplet(val_data_lines, { },
                        mode = 'val', nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 100, workers = 10, use_multiprocessing=True)
'''
