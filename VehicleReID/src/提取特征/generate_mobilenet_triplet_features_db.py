"""
   Step 3.
   Based on the trained deep ConvNets, we firstly extract the deep visual feature
   for each image in the training set. Then we build up index for these features
   in order to speed up the query processing.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
from keras.models import Model
from keras.models import load_model
from keras.applications import mobilenet
import keras.backend as K
from loss import triplet_loss, identity_loss, MARGIN
from utils import generator_batch_triplet, generator_batch
import numpy as np
from math import ceil
from sklearn.metrics import accuracy_score

BATCH_SIZE = 64
IMG_WIDTH = 224
IMG_HEIGHT = 224

np.random.seed(1024)

train_path = './train_vehicleModelColor_list_filtered.txt'
MODEL_PATH='models/V13/V13-step2_epoch=0024-loss=1.1622-val_loss=2.9339.h5'
train_path = 'feature/V13_0024/VehicleID/list'
f_acs_path='feature/V13_0024/VehicleID/1024.npy'
f_sls_3_path='feature/V13_0024/VehicleID/256.npy'

Model_Index = -2
Color_Index = -1

if __name__ == "__main__":
    print('Loading ConvNet model from {}'.format(MODEL_PATH))
    model = load_model(MODEL_PATH, custom_objects={'identity_loss': identity_loss,
                      'triplet_loss': triplet_loss, 'MARGIN': MARGIN,
                       'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    model.summary()

    # Use the F_ACS-1024 layer as features.
    #f_acs_extractor = Model(inputs = model.input, outputs = model.get_layer(name='f_acs').output)
    # Use the F_SLS3-256 layer as features.
    # layer with name 'model_1' is actually a model defined to represent the similarity learning branch
    f_sls3_extractor = Model(inputs = model.get_layer('model_1').get_input_at(node_index = 0),
                             outputs = [model.get_layer('model_1').get_layer(name = 'sls3').output, model.get_layer('model_1').get_layer(name = 'f_acs').output])
    #f_sls3_extractor = Model(inputs = model.input,
    #                         outputs = model.get_layer(name = 'sls3').output)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))
    print('Begin to extract F-SLS3 (256-D) for training data ...')
    f_sls3_features = f_sls3_extractor.predict_generator(
                            generator = generator_batch(train_data_lines,
                            nbr_classes = 7,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            img_height = IMG_HEIGHT, shuffle = False, augment = False),
                            steps = steps_per_epoch, verbose = 1)
    np.save(f_sls_3_path, f_sls3_features[0])
    np.save(f_acs_path, f_sls3_features[1])
    '''
    print('Feature extraction finished!')
    print('Begin to extract F-ACS (1024-D) for training data ...')

#f_acs_path = 'mobilenet_train_triplet_features_f-acs-1024.npy'
#f_sls_3_path = 'mobilenet_train_triplet_features_f-sls_3-256.npy'
Model_Index = -2
Color_Index = -1

if __name__ == "__main__":
    print('Loading ConvNet model from {}'.format(MODEL_PATH))
    model = load_model(MODEL_PATH, custom_objects={'identity_loss': identity_loss,
                      'triplet_loss': triplet_loss, 'MARGIN': MARGIN,
                       'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    #model.summary()

    # Use the F_ACS-1024 layer as features.
    #f_acs_extractor = Model(inputs = model.input, outputs = model.get_layer(name='f_acs').output)
    # Use the F_SLS3-256 layer as features.
    # layer with name 'model_1' is actually a model defined to represent the similarity learning branch
    f_sls3_extractor = Model(inputs = model.get_layer('model_1').get_input_at(node_index = 0),
                             outputs = [model.get_layer('model_1').get_layer(name = 'sls3').output, model.get_layer('model_1').get_layer(name = 'f_acs_new').output])
    #f_sls3_extractor = Model(inputs = model.input,
    #                         outputs = model.get_layer(name = 'sls3').output)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))
    print('Begin to extract F-SLS3 (256-D) for training data ...')
    f_sls3_features = f_sls3_extractor.predict_generator(
                            generator = generator_batch(train_data_lines,
                            nbr_classes = 7,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            img_height = IMG_HEIGHT, shuffle = False, augment = False),
                            steps = steps_per_epoch, verbose = 1)
    np.save(f_sls_3_path, f_sls3_features[0])
    np.save(f_acs_path, f_sls3_features[1])
    print('Feature extraction finished!')
    print('Begin to extract F-ACS (1024-D) for training data ...')
    f_acs_features = f_acs_extractor.predict_generator(
                            generator = generator_batch_triplet(train_data_lines, { },
                            mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            img_height = IMG_HEIGHT, shuffle = False, augment = False),
                            steps = steps_per_epoch, verbose = 1)
    np.save(f_acs_path, f_acs_features)
    # Evaludate the model on the whole validation set.
    val_data_lines = open(val_path).readlines()
    val_data_model_labels = [int(w.strip().split(' ')[Model_Index]) for w in val_data_lines]
    val_data_model_colors = [int(w.strip().split(' ')[Color_Index]) for w in val_data_lines]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))
    print('Evaludating the performance on validation data ...')
    val_model_predictions, val_color_predictions, _ = model.predict_generator(
                            generator = generator_batch_triplet(val_data_lines, { },
                            mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            img_height = IMG_HEIGHT, shuffle = False, augment = False),
                            steps = validation_steps, verbose = 1)

    val_models_pred = list(np.argmax(val_model_predictions, axis = -1))
    val_colors_pred = list(np.argmax(val_color_predictions, axis = -1))
    print('# predictions: {}'.format(len(val_models_pred)))
    print('Accuracy of validation models: {}'.format(accuracy_score(val_data_model_labels, val_models_pred)))
    print('Accuracy of validation colors: {}'.format(accuracy_score(val_data_model_colors, val_colors_pred)))
    '''
