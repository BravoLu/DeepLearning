import cv2
import os
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import numpy as np
# folder_path = './stickers_example/0025468stickers'
# folder_path = './stickers_example/0007155stickers'

# initial_ranked_list = []
# for filename in os.listdir(folder_path):
#     filename = folder_path + '/' + filename
#     initial_ranked_list.append(filename)
#
# query_img_path = os.path.join(folder_path, '0-modelID=None-predictedModelID=247-#8-AP=0.92515329304.jpg')
# #query_img_path = os.path.join(folder_path, '0-modelID=None-predictedModelID=246-#6-AP=0.483233956679.jpg')
#
# #model = load_model('./models/SiameseforSticker_epoch=0035-loss=0.1477-acc=0.9745=val_loss=0.2703-val_acc=0.9224.h5',custom_objects={'W_init': W_init,'b_init': b_init})
# siamese_model = load_model('./models/SiameseforSticker_epoch=0034-loss=0.1570-acc=0.9701=val_loss=0.2706-val_acc=0.9204.h5')

def sticker_rerank(query_img_path, initial_ranked_list, model, sticker_path2vid):
    """
    query_img_path     : the global image path of query image
    initial_ranked_list: a intial ranked list, each element in the list is a
                         gobal image path in the database.
    """
    score = []
    sticker_root = '/data/data1/zhenju/01_Project/00_VehicleReID/00_TestWorkshop/data/VehicleID_V1.0/cropstikers/'
    queryName = os.path.basename(query_img_path)
    query_sticker_path = os.path.join(sticker_root, queryName)
    query_img = image.load_img(query_sticker_path, target_size=(80, 80))
    query_img = image.img_to_array(query_img) # shape[80,80,3]
    query_img = np.expand_dims(query_img, axis=0) # shape[1,80,80,3]
    query_img = preprocess_input(query_img)  # shape[1,80,80,3]

    top1Name = os.path.basename(initial_ranked_list[0]) + '.jpg'
    if top1Name not in sticker_path2vid.keys():
        sim = 1
        score.append(sim)
    for imgPath in initial_ranked_list:
        imgName = os.path.basename(imgPath) + '.jpg'
        if imgName in sticker_path2vid.keys():
            stickerPath =  os.path.join(sticker_root,imgName)
            img1 = image.load_img(stickerPath, target_size=(80, 80))
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 = preprocess_input(img1)
            sim = model.predict([query_img,img1])[0][0]
        else:
            sim = sim - 0.00001
        score.append(sim)

    sorted_idx = np.argsort(score)[::-1]
    reranked_list = [initial_ranked_list[idx] for idx in sorted_idx]

    return sorted_idx, reranked_list

# sorted_idx, reranked_list = spatial_rerank(query_img_path, initial_ranked_list)
# print sorted_idx, reranked_list




