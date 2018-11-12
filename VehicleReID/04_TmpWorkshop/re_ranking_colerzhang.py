import cv2
import os
# path = '0007155'
# initial_ranked_list = []
# for filename in os.listdir(path):
#     filename = path+ '/' + filename
#     initial_ranked_list.append(filename)
# query_img_path = '0007155/0-modelID=None-predictedModelID=247-#8-AP=0.92515329304.jpg'
# query_img_path = '0007155/1-ModelID=None-distance=0.361145520984.jpg'

def spatial_rerank(query_img_path, initial_ranked_list):
    """
    query_img_path     : the global image path of query image
    initial_ranked_list: an initial ranked list, each element in the list is a
                         global image path in the database.
    """
    score = []
    img1 = cv2.imread(query_img_path,0)

    for _, img_path in enumerate(initial_ranked_list):
        img2 = cv2.imread(img_path,0)
        # Initiate ORB detector
        # orb = cv2.ORB_create()
        orb = cv2.xfeatures2d.SIFT_create()     
        # find the key points and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        bf = cv2.BFMatcher() 
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)
        # print 'matches...',len(matches)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        # print 'good',len(good)
        score.append(len(good))

    score = sorted(enumerate(score), key=lambda x: x[1], reverse=True)

    reranked_list = []

    for i in range(len(score)):
        index = score[i][0]
        reranked_list.append(initial_ranked_list[index])

    return reranked_list  # each element is a global image path


# reranked_list = spatial_rerank(query_img_path, initial_ranked_list)
# print reranked_list

