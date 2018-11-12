"""
   Step 4.
   Given the visual features extracted beforehand,
   we construct an index tree and preprocess the K-NN queries.
"""
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
import random
from sklearn.neighbors import BallTree
from sklearn import preprocessing
import cPickle
import shutil
from keras.models import Model
from keras.models import load_model
from keras.applications import mobilenet
from loss import triplet_loss, identity_loss, MARGIN
from utils import generator_batch, generator_batch_triplet
import numpy as np
from math import ceil
import cv2
from re_ranking_colerzhang import spatial_rerank
from globalreranking import local_rerank
from re_ranking_k_reciprocal import re_ranking
import time 

random.seed(2017)

topK = 50
topKReranking = 5000
toTestLines = 200000000  # Debug: only focus on a few queries.
FEATURE_MODE = 0 #   0: F_ACS-1024, 1: F_SLS3-256, 2: F_ACS-1024 + F_SLS3-256
RERANK = False
K_RECIPROCAL_RERANK = True  # whether use the k-reciprocal nearest neighbours for reranking.
DENSE_FEATURE=False
USE_MOBILENET = True 

if USE_MOBILENET:
    from generate_mobilenet_triplet_features_db import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
    from generate_mobilenet_triplet_features_db import train_path, MODEL_PATH
    from generate_mobilenet_triplet_features_db import f_acs_path, f_sls_3_path

else:
    from generate_triplet_features_db import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
    from generate_triplet_features_db import train_path, MODEL_PATH
    from generate_triplet_features_db import f_acs_path, f_sls_3_path

np.random.seed(2017)
sticker_root='./InspectionSticker/'
ROOT='/data/data1/zhenju/01_Project/00_VehicleReID/04_TmpWorkshop'
attr_root = './data/VehicleID_V1.0/attribute/'
image_root = './data/VehicleID_V1.0/image/'
test_path = './data/VehicleID_V1.0/train_test_split/test_list_13164.txt'
test_path = './data/VehicleID_V1.0/train_test_split/test_list_800.txt'

def precision_at_k(relevance_score, k):
    # relevance_score: a list of relevance scores, e.g. [1, 0, 0, 1, 1, 1, 0]
    # return precision@k, 1 <= k <= len(relevance_score)
    # precision@4 = 2 / 4 = 0.5
    relevance_score = np.array(relevance_score, dtype = float)
    pak = relevance_score[k - 1] * relevance_score[:k].sum() / k
    return pak
def recall_at_k(relevance_score, k, num):
	pak = np.array(relevance_score[:k], dtype = float).sum() / min(num, k)
	return pak
def average_precision(relevance_score, top_k = 1, epsilon = 0.00001):
    # As explained in https://medium.com/@pds.bangalore/mean-average-precision-abd77d0b9a7e
    # https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
    # ap([1, 0, 0, 1, 1, 1, 0]) = 0.69
    relevance_score = relevance_score[:top_k]
    precision_list = [precision_at_k(relevance_score, i) for i in range(1, top_k + 1)]
    ap = sum(precision_list) / (sum(relevance_score) + epsilon)
    return ap

img2vid_lines = open(os.path.join(attr_root, 'img2vid.txt')).readlines()
dic_img_vehicleID = { }
for line in img2vid_lines:
    image_name, vehicleID = line.strip().split(' ')
    dic_img_vehicleID[image_name] = vehicleID
sticker_path2vid = { }
for filename in os.listdir(sticker_root):
	if dic_img_vehicleID.has_key(filename[:-4]):
		sticker_path = filename
		vid = dic_img_vehicleID[filename[:-4]]
		sticker_path2vid[sticker_path] = vid
model_attr_lines = open(os.path.join(attr_root, 'model_attr.txt')).readlines()
dic_vehicleID_modelID = { }
for line in model_attr_lines:
    vehicleID, modelID = line.strip().split(' ')
    dic_vehicleID_modelID[vehicleID] = modelID
print('dic_vehicleID_modelID: # vehicles: {}'.format(len(dic_vehicleID_modelID)))

train_data_lines = open(train_path).readlines()
# Check if image path exists.
train_imgPath_list = [w.strip().split(' ')[0] for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
train_imgName_list = [os.path.basename(w)[:-4] for w in train_imgPath_list]
train_veModel_list = [w.strip().split(' ')[-1] for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]

print('Loading F-ACS (1024-D) Features from {}'.format(f_acs_path))
f_acs_features = np.load(f_acs_path)
print('Loading F-SLS3 (256-D) Features from {}'.format(f_sls_3_path))
f_sls3_features = np.load(f_sls_3_path)

if FEATURE_MODE == 0:
    print('Only using attributed features (1024-D)')
    train_features = f_acs_features
elif FEATURE_MODE == 1:
    print('Only using triplet features (256-D)')
    train_features = f_sls3_features
elif FEATURE_MODE == 2:
    print('Using both attributed features and triplet features (1280-D)')
    train_features = np.hstack((f_acs_features, f_sls3_features))
else:
    print('FEATURE_MODE has to be only 0, 1 or 2!!!')

print('train_features.shape: {}'.format(train_features.shape))

# L2-normalize the visual features to be Unit-norm.
train_features = preprocessing.normalize(train_features, norm='l2')

test_data_lines = open(test_path).readlines()[:toTestLines]
test_imgName_list = [w.strip().split(' ')[0] for w in test_data_lines]
test_imgPath_list = [os.path.join(image_root, w + '.jpg') for w in test_imgName_list]
test_vehicleIDs_list = [w.strip().split(' ')[-1] for w in test_data_lines]
dic_test_vehicleID_imgName = { }
dic_test_imgName_vehicleID = {}
for imgName, vehicleID in zip(test_imgName_list, test_vehicleIDs_list):
    dic_test_vehicleID_imgName.setdefault(vehicleID, [ ]).append(imgName)
    dic_test_imgName_vehicleID[imgName] = vehicleID
print('# test vehicle IDs: {}'.format(len(dic_test_vehicleID_imgName)))
vehicleIDs_above_6 = [k for k in dic_test_vehicleID_imgName if len(dic_test_vehicleID_imgName[k]) > 6]
print('# test vehicle IDs with more than 6 images: {}'.format(len(vehicleIDs_above_6)))

# For those vehicle IDs which are associated with more than 6 images, we randomly select 1 image
# as a query, and put the rest into the database (gallery set).
query_imgNames = [ ]
gallery_imgNames = [ ]
query_imgName2Num={}
f_write_query = open('/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/query_list/{}'.format(
                 os.path.basename(test_path[:-4]) + '_query.txt'), 'w')
f_write_gallery = open('/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/query_list/{}'.format(
                 os.path.basename(test_path[:-4]) + '_gallery.txt'), 'w')
for vehicleID in vehicleIDs_above_6:
    imgNames = dic_test_vehicleID_imgName[vehicleID]
    sampled_idx = random.randint(0, len(imgNames) - 1)
    sampled_imgName = imgNames[sampled_idx]
    # Sampling a query
    #print('We sampled an image: {}'.format(sampled_imgName))
    query_imgNames.append(sampled_imgName)
    f_write_query.write('{} {}\n'.format(sampled_imgName, vehicleID))
    imgNames.remove(sampled_imgName)
    query_imgName2Num[sampled_imgName] = len(imgNames)
    for db_imgName in imgNames:
        f_write_gallery.write('{} {}\n'.format(db_imgName, vehicleID))
    #print('Add the rest into gallery: {}'.format(imgNames))
    # Adding the rest into gallery set
    gallery_imgNames += imgNames
#print('Gallery image names: {}'.format(gallery_imgNames))
f_write_query.close()
f_write_gallery.close()

query_imgPath_list = [os.path.join(image_root, w + '.jpg') for w in query_imgNames]
# '0' is a psudo label which doesn't matter, which are placeholders for
# vehicleID, modelID, colorID
query_data_list = [w + ' 0 0 0\n' for w in query_imgPath_list]
gallery_imgPath_list = [os.path.join(image_root, w + '.jpg') for w in gallery_imgNames]
gallery_data_list = [w + ' 0 0 0\n' for w in gallery_imgPath_list]

print('Loading ConvNet model from {}'.format(MODEL_PATH))
if USE_MOBILENET:
    model = load_model(MODEL_PATH, custom_objects={'identity_loss': identity_loss,
                      'triplet_loss': triplet_loss, 'MARGIN': MARGIN,
                       'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
else:
    model = load_model(MODEL_PATH, custom_objects={'identity_loss': identity_loss, 'triplet_loss': triplet_loss, 'MARGIN': MARGIN})
# Use the F_ACS-1024 layer as features.
f_acs_extractor = Model(inputs = model.input, outputs = model.get_layer(name='f_acs').output)
# Use the F_SLS3-256 layer as features.
# layer with name 'model_1' is actually a model defined to represent the similarity learning branch
f_sls3_extractor = Model(inputs = model.get_layer('model_1').get_input_at(node_index = 0),
                         outputs = model.get_layer('model_1').get_layer(name = 'sls3').output)

nbr_queries = len(query_data_list)
print('# Selected Queries: {}.'.format(nbr_queries))
steps_per_epoch_query = int(ceil(nbr_queries * 1. / BATCH_SIZE))
print('Begin to extract deep features for Queries ...')

# obtain the F_ACS-1024  features for queries
query_features_acs = f_acs_extractor.predict_generator(
                        generator = generator_batch_triplet(query_data_list, { },
                        mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, shuffle = False, augment = False),
                        steps = steps_per_epoch_query, verbose = 1)
# obtain the F_SLS3-256  features for queries
query_features_sls3 = f_sls3_extractor.predict_generator(
                        generator = generator_batch(query_data_list,
                        nbr_classes = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, shuffle = False, augment = False),
                        steps = steps_per_epoch_query, verbose = 1)

if FEATURE_MODE == 0:
    print('Only using attributed features (1024-D)')
    query_features = query_features_acs
elif FEATURE_MODE == 1:
    print('Only using triplet features (256-D)')
    query_features = query_features_sls3
elif FEATURE_MODE == 2:
    print('Using both attributed features and triplet features (1280-D)')
    query_features = np.hstack((query_features_acs, query_features_sls3))
else:
    print('FEATURE_MODE has to be only 0, 1 or 2!!!')

print('query_features.shape: {}'.format(query_features.shape))
query_features = preprocessing.normalize(query_features, norm='l2')

# obtain the vehicle model predictions for queries
print('Begin to predict the vehicle model probabilities for Queries ...')
'''
query_vehicle_model_predictions, query_vehicle_color_predictions, _ = model.predict_generator(
                        generator = generator_batch_triplet(query_data_list, { },
                        mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, shuffle = False, augment = False),
                        steps = steps_per_epoch_query, verbose = 1)

query_vehicle_models = np.argmax(query_vehicle_model_predictions, axis = -1)
'''
# obtain the features for gallery set
nbr_gallery = len(gallery_imgNames)
steps_per_epoch_gallery = int(ceil(nbr_gallery * 1. / BATCH_SIZE))

# obtain the F_ACS-1024  features for gallery images
gallery_features_acs = f_acs_extractor.predict_generator(
                        generator = generator_batch_triplet(gallery_data_list, { },
                        mode = 'val', nbr_class_one = 250, nbr_class_two = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, shuffle = False, augment = False),
                        steps = steps_per_epoch_gallery, verbose = 1)
# obtain the F_SLS3-256  features for gallery images
gallery_features_sls3 = f_sls3_extractor.predict_generator(
                        generator = generator_batch(gallery_data_list,
                        nbr_classes = 7,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, shuffle = False, augment = False),
                        steps = steps_per_epoch_gallery, verbose = 1)

if FEATURE_MODE == 0:
    print('Only using attributed features (1024-D)')
    gallery_features = gallery_features_acs
elif FEATURE_MODE == 1:
    print('Only using triplet features (256-D)')
    gallery_features = gallery_features_sls3
elif FEATURE_MODE == 2:
    print('Using both attributed features and triplet features (1280-D)')
    gallery_features = np.hstack((gallery_features_acs, gallery_features_sls3))
else:
    print('FEATURE_MODE has to be only 0, 1 or 2!!!')

print('gallery_features.shape: {}'.format(gallery_features.shape))
gallery_features = preprocessing.normalize(gallery_features, norm='l2')

# Add gallery set into the database.
db_features = np.vstack((train_features, gallery_features))
db_imgNames = train_imgName_list + gallery_imgNames
#db_features = gallery_features
#db_imgNames = gallery_imgNames
db_imgPaths = [os.path.join(image_root, w) for w in db_imgNames]

print('constructing index ...')
tree = BallTree(db_features, leaf_size = 200)

if os.path.exists(os.path.join(ROOT, 'queries_noranking2')):
    shutil.rmtree(os.path.join(ROOT, 'queries_noranking2'))
else:
    os.makedirs(os.path.join(ROOT, 'queries_noranking2'))

# process each query
t1 = time.time()
avg_precision_list = [ ]
avg_recall_20_list = [ ]
avg_recall_50_list = [ ]
top_one_acc_list = [ ]
top_five_acc_list = [ ]
top_10_acc_list = [ ]
top_twenty_acc_list = [ ]
top_30_acc_list = [ ]
top_40_acc_list = [ ]
top_50_acc_list = [ ]
if RERANK:
	siamese_model = load_model('./models/NewSiamese_epoch=0027-loss=0.1645-acc=0.9640=val_loss=0.2464-val_acc=0.9286.h5')
for query_idx, query_imgPath in enumerate(query_imgPath_list):
    query_feature = query_features[query_idx]
    query_imgName = query_imgNames[query_idx]
    query_veID = dic_img_vehicleID[query_imgName]
    #queryModelID_predicted =query_vehicle_models[query_idx]
    queryModelID = dic_vehicleID_modelID[query_veID] if query_veID in dic_vehicleID_modelID \
                   else 'None'

    print('Query image name: {}, Query image vehicle ID: {}'.format(query_imgName, query_veID))
    retrieved_image_veIDs = [ ]
    retrieved_image_modelIDs = [ ]
    directory = os.path.join(ROOT, 'queries_noranking2', query_imgName)
    if not os.path.exists(directory):
        os.makedirs(directory)
        r_distances, r_index = tree.query([query_feature], k = topKReranking)
        print os.path.basename(query_imgPath)
        if RERANK and os.path.basename(query_imgPath) in sticker_path2vid.keys():
            print "AAAAAAAAAAAAAAAA"
            initial_ranked_list = [db_imgPaths[r_idx] for r_idx in r_index[0]]
            #sorted_idx, reranked_list = spatial_rerank(query_imgPath, initial_ranked_list)
            sorted_idx, reranked_list = local_rerank(query_imgPath, initial_ranked_list, r_distances,0.3,siamese_model, sticker_path2vid)
            r_index = [[r_index[0][idx] for idx in sorted_idx]]
            r_distances = [[r_distances[0][idx] for idx in sorted_idx]]
        if K_RECIPROCAL_RERANK:
            probFea = query_feature.reshape(1, -1)
            galFea = np.zeros((topKReranking, db_features.shape[1]))
            for i in range(topKReranking):
                galFea[i] = db_features[r_index[0][i]]

            dist = re_ranking(probFea, galFea, k1 = 10, k2 = 3, lambda_value = 0.5)
            sorted_distances, sorted_index = zip(*sorted(zip(dist.flatten(), r_index.flatten())))
            r_distances = np.array([sorted_distances])
            r_index = np.array([sorted_index])

        for sub_idx, r_idx in enumerate(r_index[0]):
            source_imgName = db_imgNames[r_idx]
            source_path = os.path.join(image_root, source_imgName + '.jpg')
            source_veID = dic_img_vehicleID[source_imgName] if source_imgName in dic_img_vehicleID \
                          else 'None'
            source_modelID = dic_vehicleID_modelID[source_veID] if source_veID in dic_vehicleID_modelID \
                             else 'None'
            retrieved_image_veIDs.append(source_veID)
            retrieved_image_modelIDs.append(source_modelID)

            # if the query image vehicle ID is equal to the retrieved image vehicle ID,
            # draw a green bounding box, else draw a red one.
            # if the retrieved image has not been identified, do nothing.
            # BGR - (blue, green, red)
            if source_veID == None:
                color_tuple = (0, 0, 0)
            elif source_veID == query_veID:
                color_tuple = (0, 255, 0)
            else:
                color_tuple = (0, 0, 255)
            img = cv2.imread(source_path)
            #print source_path
            height, width = img.shape[:2]
            cv2.rectangle(img, (0, 0), (width, height), color_tuple, 10)
            to_save_imgPath = os.path.join(directory, '{}-ModelID={}-distance={}-imgName={}.jpg').format(
                            sub_idx + 1, source_modelID, r_distances[0][sub_idx], source_imgName)
            cv2.imwrite(to_save_imgPath, img)

        relevance_score = (np.array(retrieved_image_veIDs, dtype=str) == query_veID).astype(int).tolist()
        ap = average_precision(relevance_score, top_k = topK)
        print('Averaged Precision: {}\n'.format(ap))
        avg_precision_list.append(ap)
        print('MAP: {}'.format(np.mean(avg_precision_list)))
        query_gallery_num = len(dic_test_vehicleID_imgName[query_veID])
        recall_20 = recall_at_k(relevance_score, 20, query_gallery_num)
        avg_recall_20_list.append(recall_20)
        recall_50 = recall_at_k(relevance_score, 50, query_gallery_num)
        avg_recall_50_list.append(recall_50)
        query_gallery_num = len(dic_test_vehicleID_imgName[query_veID])
        print('Averaged Recall: {}\n'.format(recall_50))
        topOne_acc = 1. if query_veID == retrieved_image_veIDs[0] else 0.
        top_one_acc_list.append(topOne_acc)
        topFive_acc = 1. if query_veID in retrieved_image_veIDs[:5] else 0.
        top_five_acc_list.append(topFive_acc)
        top10_acc = 1. if query_veID in retrieved_image_veIDs[:10] else 0.
        top_10_acc_list.append(top10_acc)
        topTwenty_acc = 1. if query_veID in retrieved_image_veIDs[:20] else 0
        top_twenty_acc_list.append(topTwenty_acc)
        top30_acc = 1. if query_veID in retrieved_image_veIDs[:30] else 0.
        top_30_acc_list.append(top30_acc)
        top40_acc = 1. if query_veID in retrieved_image_veIDs[:40] else 0.
        top_40_acc_list.append(top40_acc)
        top50_acc = 1. if query_veID in retrieved_image_veIDs[:50] else 0.
        top_50_acc_list.append(top50_acc)
        target_path = os.path.join(directory, '0-modelID={}-predictedModelID={}-#{}-AP={}-imgName={}.jpg'.format(
              queryModelID, 1, len(dic_test_vehicleID_imgName[query_veID]) - 1, ap, query_imgName))
        shutil.copy(query_imgPath, target_path)
print('MAP: {}'.format(np.mean(avg_precision_list)))
print('Top-1 Accuracy: {}'.format(np.mean(top_one_acc_list)))
print('Top-5 Accuracy: {}'.format(np.mean(top_five_acc_list)))
print('Top-10 Accuracy: {}'.format(np.mean(top_10_acc_list)))
print('Top-20 Accuracy: {}'.format(np.mean(top_twenty_acc_list)))
print('Top-30 Accuracy: {}'.format(np.mean(top_30_acc_list)))
print('Top-40 Accuracy: {}'.format(np.mean(top_40_acc_list)))
print('Top-50 Accuracy: {}'.format(np.mean(top_50_acc_list)))
print('Top-20 Recall: {}'.format(np.mean(avg_recall_20_list)))
print('Top-50 Recall: {}'.format(np.mean(avg_recall_50_list)))
print('{} seconds / query'.format((time.time() - t1) / len(query_imgPath_list)))
