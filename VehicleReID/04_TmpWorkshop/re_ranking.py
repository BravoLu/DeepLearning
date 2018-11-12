import cv2
import os
import numpy as np
# path = '0007155'
# initial_ranked_list = []
# for filename in os.listdir(path):
#     filename = path+ '/' + filename
#     initial_ranked_list.append(filename)
# query_img_path = '0007155/0-modelID=None-predictedModelID=247-#8-AP=0.92515329304.jpg'
# query_img_path = '0007155/1-ModelID=None-distance=0.361145520984.jpg'
def spatial_rerank(query_img_path, initial_ranked_list, thresh = 0.75):
    """
    query_img_path     : the global image path of query image
    initial_ranked_list: a intial ranked list, each element in the list is a
                         gobal image path in the database.
    """
    query_img = cv2.imread(query_img_path, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    query_kp, query_des = sift.detectAndCompute(query_img, None)
    nbr_matched_list = [ ]
    for imgName in initial_ranked_list:
        img_path = imgName + '.jpg'
        img = cv2.imread(img_path, 0)
        kp, des = sift.detectAndCompute(img, None)
        matches = bf.knnMatch(query_des, des, k = 2)
        good = [ ]
        for m,n in matches:
            if m.distance < thresh * n.distance:
                good.append([m])

        nbr_matched_list.append(len(good))

    sorted_idx = np.argsort(nbr_matched_list)[::-1]
    reranked_list = [initial_ranked_list[idx] for idx in sorted_idx]

    return sorted_idx, reranked_list

# def spatial_rerank(query_img_path, initial_ranked_list):
#     """
#     query_img_path     : the global image path of query image
#     initial_ranked_list: an initial ranked list, each element in the list is a
#                          global image path in the database.
#     """
#     score = []
#     img1 = cv2.imread(query_img_path,0)

#     for _, img_path in enumerate(initial_ranked_list):
#         img2 = cv2.imread(img_path,0)
#         # Initiate ORB detector
#         # orb = cv2.ORB_create()
#         orb = cv2.xfeatures2d.SIFT_create()     
#         # find the key points and descriptors with ORB
#         kp1, des1 = orb.detectAndCompute(img1,None)
#         kp2, des2 = orb.detectAndCompute(img2,None)
#         bf = cv2.BFMatcher() 
#         # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# 	    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
#         matches = bf.knnMatch(des1,des2,k=2)
#         # print 'matches...',len(matches)

#         # Apply ratio test
#         good = []
#         for m,n in matches:
#             if m.distance < 0.7*n.distance:
#                 good.append(m)
#         # print 'good',len(good)
#         score.append(len(good))

#     score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)

#     reranked_list = []

#     for i in range(len(score)):
#         index = score[i][0]
#         reranked_list.append(initial_ranked_list[index])

#     return reranked_list  # each element is a global image path


# sorted_idx, reranked_list = spatial_rerank(query_img_path, initial_ranked_list)
# print sorted_idx, reranked_list

