import cv2
import os
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np

def local_rerank(query_img_path, initial_ranked_list, r_distances, lambda_value, model, sticker_path2vid):
    """
    query_img_path     : the global image path of query image
    initial_ranked_list: a intial ranked list, each element in the list is a
                         gobal image path in the database.
    """
    final_score = []
    local_score = []

    sticker_root = './InspectionSticker/'
    #sticker_root = '/data1/colerzhang/VehicleSearch/data/Jiangsu_test_images/cropstickers/'

    queryName = os.path.basename(query_img_path)
    query_sticker_path = os.path.join(sticker_root, queryName)
    query_img = image.load_img(query_sticker_path, target_size=(64, 64))
    query_img = image.img_to_array(query_img) # shape[80,80,3]
    query_img = np.expand_dims(query_img, axis=0) # shape[1,80,80,3]
    query_img = preprocess_input(query_img)  # shape[1,80,80,3]

    top1Name = os.path.basename(initial_ranked_list[0])+'.jpg'
    if top1Name not in sticker_path2vid.keys():
        local_dist = 1
        local_score.append(local_dist)
    for sub_idx, imgPath in enumerate(initial_ranked_list):
        global_dist = 1 - r_distances[0][sub_idx]
        imgName = os.path.basename(imgPath)+'.jpg'
        if imgName in sticker_path2vid.keys():
            stickerPath =  os.path.join(sticker_root,imgName)
            img1 = image.load_img(stickerPath, target_size=(64, 64))
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 = preprocess_input(img1)
            local_dist = model.predict([query_img,img1])[0][0]
        else:
            local_dist = np.mean(local_score)
            #local_dist = np.min(local_score)
        final_dist = global_dist*(1-lambda_value) + local_dist*lambda_value
        local_score.append(local_dist)
        final_score.append(final_dist)

    sorted_idx = np.argsort(final_score)[::-1]
    reranked_list = [initial_ranked_list[idx] for idx in sorted_idx]

    return sorted_idx, reranked_list
