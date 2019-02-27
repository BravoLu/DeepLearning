"""
   Step 2.
   Train a InceptionV3 ConvNet which predicts both vehicles and colors.
   It is a multi-task learning process.
   Add the triplet-loss branch.
"""
import os
import sys
GPUS = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import tensorflow as tf
config = tf.ConfigProto()
# 限制GPU使用率
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# 申请动态内存
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
from utils import generator_batch_triplet, generator_batch, generator_batch_triplet_id
# from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model
from loss import triplet_loss, identity_loss, MARGIN

np.random.seed(1024)

FINE_TUNE = False 
SAVE_FILTERED_LIST = False
FINE_TUNE_ON_ATTRIBUTES = False
CONTINUE_TRAINING=True
LEARNING_RATE = 0.00001
NBR_EPOCHS = 100
BATCH_SIZE = 4
IMG_WIDTH = 224
IMG_HEIGHT = 224
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
RANDOM_SCALE = True
nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 2
ID_NBR=85288
train_path = 'train_vehicleModelColor_list_filtered.txt_reid'
val_path = 'val_vehicleModelColor_list.txt_reid'
#val_path='val_vehicleModelColor_list.txt'

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
    # remove vehicleID and imageName = 1 
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
        model_path_input='####step1.model'
        basemodel=load_model(model_path_input, custom_objects={'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D, 'identity_loss': identity_loss, 'triplet_loss': triplet_loss, 'MARGIN': MARGIN})
        basemodel.get_layer(name = 'global_average_pooling2d_1').name = 'f_base'
        f_base = basemodel.get_layer(name = 'f_base').output
        anchor = basemodel.input
        positive = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='positive')
        negative = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='negative')
        # Attributes Branch 4       
        f_acs = basemodel.get_layer(name = 'f_acs_new').output
        f_id_prediction = basemodel.get_layer(name = 'predictions_id_new').output
        # Similarity Learning Branch
        f_sls1 = Dense(1024, name = 'sls1')(f_base)
        f_sls2 = concatenate([f_sls1, f_acs], axis = -1, name = 'sls1_concatenate')  # 1024-D
        # The author said that only layer ``SLS_2'' is applied ReLU since non-linearity
        # would disrupt the embedding learned in the layer ``SLS_1''.
        #f_sls2 = Activation('relu', name = 'sls1_concatenate_relu')(f_sls2)
        f_sls2 = Dense(1024, name = 'sls2')(f_sls2)
        f_sls2 = Activation('relu', name = 'sls2_relu')(f_sls2)
        # Non-linearity ?
        f_sls3 = Dense(256, name = 'sls3', activation='softsign')(f_sls2)
        sls_branch = Model(inputs = basemodel.input, outputs = f_sls3)
        f_sls3_anchor = sls_branch(anchor)
        f_sls3_positive = sls_branch(positive)
        f_sls3_negative = sls_branch(negative)

        loss = Lambda(triplet_loss,
                  output_shape=(1, ))([f_sls3_anchor, f_sls3_positive, f_sls3_negative])
        model = Model(inputs= [anchor, positive, negative], outputs = [f_id_prediction, loss])
  

    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
    #optimizer = RMSprop(lr = LEARNING_RATE)

    if nbr_gpus > 1:
        print('Using {} GPUS.\n'.format(nbr_gpus))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    model.compile(loss=["categorical_crossentropy",  identity_loss], #loss_weights=[0.05,0.05,0.9],
                  optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    model_file_saved = "./models/V35/V35-step2_epoch={epoch:04d}-loss={loss:.4f}-val_loss={val_loss:.4f}.h5"
    # Define several callbacks

    checkpoint = ModelCheckpoint(model_file_saved, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=5, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    # train_data_lines filtered , dic_train_data_lines not filtered
    train_data_lines, dic_train_data_lines = filter_data_list(train_data_lines)
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(generator_batch_triplet_id(train_data_lines, dic_train_data_lines,
                        mode = 'train', nbr_class_one = ID_NBR, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_triplet_id(val_data_lines, { },
                        mode = 'val', nbr_class_one = ID_NBR, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 100, workers = 10, use_multiprocessing=True)
