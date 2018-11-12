"""
   Step 2.
   Train a MobileNet-V1 ConvNet which predicts both vehicles and colors.
   It is a multi-task learning process.
"""
import os
GPUS = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
from math import ceil
import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.applications import mobilenet
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Conv2D, Convolution2D, Flatten, GlobalMaxPooling2D , concatenate, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.utils import class_weight
from utils import generator_batch_multitask_id
#from keras.utils.training_utils import multi_gpu_model
from keras.utils import multi_gpu_model

np.random.seed(1024)

FINE_TUNE = False
CONTINUE = False
LEARNING_RATE = 0.001
NBR_EPOCHS = 100
BATCH_SIZE = 16
IMG_WIDTH = 224
IMG_HEIGHT = 224
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
NBR_ID = 85288 
RANDOM_SCALE = True
nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 10

train_path = 'train_vehicleModelColor_list_filtered.txt_reid'
val_path = 'val_vehicleModelColor_list.txt_reid'


if __name__ == "__main__":

    if FINE_TUNE:
        print('Loading MobileNet-V1 Weights from ImageNet Pretrained ...')
        mobilenet = MobileNet(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), alpha = 1.0,
                          include_top = False, weights = 'imagenet', pooling='avg')
        f_base1 = mobilenet.get_layer(name = 'conv_pw_13_relu').output  
        f_base = GlobalAveragePooling2D()(f_base1)
        
        local_56 = mobilenet.get_layer(name = 'conv_pw_3_relu').output
        local_28 = mobilenet.get_layer(name = 'conv_pw_5_relu').output
        local_14 = mobilenet.get_layer(name = 'conv_pw_11_relu').output
        local_56_1 = Convolution2D(512, (1, 1))(local_56)
        local_56_2 = Convolution2D(512, (1, 1))(local_56_1)
        local_28_1 = Convolution2D(512, (1, 1))(local_28)
        local_28_2 = Convolution2D(512, (1, 1))(local_28_1)
        local_14_1 = Convolution2D(512, (1, 1))(local_14)
        local_14_2 = Convolution2D(512, (1, 1))(local_14_1)

        local_56_f = GlobalMaxPooling2D()(local_56_2)
        local_28_f = GlobalMaxPooling2D()(local_28_2)
        local_14_f = GlobalMaxPooling2D()(local_14_2)
        f_base = concatenate([f_base, local_56_f, local_28_f, local_14_f], axis = -1, name = 'f_base_concatenate')



        f_acs = Dense(1024, name='f_acs_new', activation='softsign')(f_base)

        f_id = Dense(NBR_ID, activation='softmax', name='predictions_id_new')(f_acs)

        model = Model(outputs = f_id, inputs = mobilenet.input)
        model.load_weights('./models/V28/V11_step1_epoch=0036-loss=0.0708-val_loss=1.6946.h5', by_name=True)
        model.summary()

    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)

    if nbr_gpus > 1:
        print('Using multiple GPUS: {}\n'.format(GPUS))
        model = multi_gpu_model(model, gpus = nbr_gpus)
        BATCH_SIZE *= nbr_gpus
    else:
        print('Using a single GPU.\n')
    model.compile(loss="categorical_crossentropy", 
                  optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    model_file_saved = "./models/V35/V35_step1_epoch={epoch:04d}-loss={loss:.4f}-val_loss={val_loss:.4f}.h5"
    # Define several callbacks

    checkpoint = ModelCheckpoint(model_file_saved, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(generator_batch_multitask_id(train_data_lines,
                        nbr_class_one = NBR_ID, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_multitask_id(val_data_lines,
                        nbr_class_one = NBR_ID, nbr_class_two = NBR_COLORS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint, reduce_lr], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)
