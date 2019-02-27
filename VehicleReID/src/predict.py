import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.models import Model
from keras.models import load_model
from loss import triplet_loss, identity_loss, MARGIN
from utils import generator_batch, generator_batch_triplet
import numpy as np
from math import ceil
import cv2

IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 1

data_list = ['/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/input_and_output/0000002.jpg 0 0 0']

MODEL_PATH = './models/triplet_models_backup/InceptionV3_Triplet_epoch=0008-loss=1.1226-modelAcc=0.9782-colorAcc=0.9404-val_loss=1.2868-val_modelAcc=0.9859-val_colorAcc=0.9282.h5'

print('Loading ConvNet model from {}'.format(MODEL_PATH))
model = load_model(MODEL_PATH, custom_objects={'identity_loss': identity_loss, 'triplet_loss': triplet_loss, 'MARGIN': MARGIN})

# Use the F_ACS-1024 layer as features.
f_acs_extractor = Model(inputs = model.input, outputs = model.get_layer(name='f_acs').output)
# Use the F_SLS3-256 layer as features.
# layer with name 'model_1' is actually a model defined to represent the similarity learning branch
f_sls3_extractor = Model(inputs = model.get_layer('model_1').get_input_at(node_index = 0),
                         outputs = model.get_layer('model_1').get_layer(name = 'sls3').output)

f_acs = f_acs_extractor.predict_generator(
                        generator = generator_batch_triplet(data_list, { },
                        mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, return_label = False,
                        save_to_dir = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/input_and_output',
                        #save_network_input = 'network_example_input.txt',
                        shuffle = False, augment = False),
                        steps = 1, verbose = 1)
print(f_acs.shape)
np.savetxt('f_acs_1024d.txt', f_acs, delimiter = ' ')

f_sls3 = f_sls3_extractor.predict_generator(
                            generator = generator_batch(data_list,
                            nbr_classes = 7,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            save_network_input = 'network_example_input.txt',
                            img_height = IMG_HEIGHT, shuffle = False, augment = False),
                            steps = 1, verbose = 1)

print(f_sls3.shape)
np.savetxt('f_sls3_256d.txt', f_sls3, delimiter = ' ')

# model_pre, color_pred, loss_pred = model.predict_generator(
#                         generator = generator_batch_triplet(data_list, { },
#                         mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
#                         batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
#                         img_height = IMG_HEIGHT,
#                         save_to_dir = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/input_and_output',
#                         save_network_input = 'network_example_input.txt',
#                         shuffle = False, augment = False),
#                         steps = 1, verbose = 1)
#
# print(model_pre, color_pred, loss_pred)
