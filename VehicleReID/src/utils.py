from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import Sequence
import PIL
from PIL import Image
import os
import numpy as np
import random
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import time

np.random.seed(1024)

# Refer to 7th slide in http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf
def balanced_sample(data_lines):
    # data_lines is a list of lines which has the following format: image_path, label id
    print('# Lines Before Unique: {}'.format(len(data_lines)))
    data_lines = list(set(data_lines))
    print('# Lines After Unique: {}'.format(len(data_lines)))
    dic = { }
    for line in data_lines:
        img_path, label = line.strip().split(' ')
        label = int(label)
        dic.setdefault(label, [ ]).append(line)

    # find the key/label with max length/images
    label_dominate = max(dic, key=lambda k: len(dic[k]))
    max_length = len(dic[label_dominate])
    new_data_lines = [ ]
    for k in dic.keys():
        length = len(dic[k])
        randomID = range(max_length)
        np.random.shuffle(randomID)
        randomID = [w % length for w in randomID]
        tmp_lines = [dic[k][w] for w in randomID]
        new_data_lines += tmp_lines

    np.random.shuffle(new_data_lines)

    data_lines = new_data_lines

    return data_lines

def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    #print centerw - halfw
    #print halfw
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped
def center_crop_padding(img_path, ratio=1.0, return_width=299):
	img = cv2.imread(img_path)
	h = img.shape[0]
	w = img.shape[1]
	shorter = min(w, h)
	longer = max(w, h)
	new_image = Image.new("RGB", (longer, longer))
	centerx = longer // 2
	halfw = img.shape[0] // 2
	halfh = img.shape[1] // 2
	new_image[centerx - halfw : centerx + halfw,centerx - halfh : centerx + halfh,:] = img
	image_resized = cv2.resize(new_image, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
	img_rgb = img_resized
	img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

	return img_rgb

def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array
    if crop_method==center_crop_padding:
        return center_crop_padding(img_path, ratio, return_width)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb

def padding_byRatio(img_path, return_width=299):
    # Given an image path, return a scaled array
    # We scale the original image with longer-side to 299,
    # then padding the scaled image with zeros to (299, 299).
    img = load_img(img_path)
    w, h = img.size
    shorter = min(w, h)
    longer = max(w, h)

    longer_side = return_width
    shorter_side = int(longer_side * 1. / longer * shorter )

    img_width = shorter_side if w < h else longer_side
    img_height = longer_side if w < h else shorter_side

    img = load_img(img_path, target_size = (img_width, img_height))

    # New an image with all zeros, and pase the scaled image
    # to this new image, i.e. padding zeros.
    new_img = Image.new("RGB", (return_width, return_width))

    if img_width >= img_height:
        new_img.paste(img, ((longer_side - shorter_side)/2, 0))
    else:
        new_img.paste(img, (0, (longer_side - shorter_side)/2))

    return img_to_array(new_img)


def dense_byStride(img_path, ratio = 1.15, stride_x = 1, stride_y = 1, return_width = 299):
    # Given an image path, return a scaled array
    img = load_img(img_path)
    w, h = img.size
    shorter = min(w, h)
    longer = max(w, h)

    shorter_side = int(ratio * 1. * return_width)
    longer_side = int(longer * 1. / shorter * shorter_side)

    img_width = shorter_side if w < h else longer_side
    img_height = longer_side if w < h else shorter_side

    img = load_img(img_path, target_size = (img_width, img_height))
    crop_width = return_width if ratio >=1.0 else shorter_side
    w = crop_width
    h = crop_width
    left_r = (img_width - w) // 2
    left = left_r * stride_x
    top_r = (img_height - h) // 2
    top  = top_r * stride_y
    right = left + w
    bottom = top + h
    img =  img.crop((left, top, right, bottom))

    if ratio >= 1.0:
        # If the image size if larger than cropping size
        return img_to_array(img)
    else:
        # In this case, we need to pad with zeros
        new_img = Image.new("RGB", (return_width, return_width))
        new_img.paste(img, ((return_width - crop_width)/2, (return_width - crop_width)/2))
        return img_to_array(new_img)

def scale_byRatio_crop_biasRatio(img_path, biasRatio=1.0, return_width=299):
    # Given an image path, return a scaled array baised by a specific biasRatio
    # Note that the left coordinate absolute bias range is (0, w - 299)
    img = load_img(img_path)
    w, h = img.size
    shorter = min(w, h)
    longer = max(w, h)

    shorter_side = return_width
    longer_side = int(longer * 1. / shorter * shorter_side)

    img_width = shorter_side if w < h else longer_side
    img_height = longer_side if w < h else shorter_side

    img = load_img(img_path, target_size = (img_width, img_height))

    if img_width >= img_height:
        left = int((img_width - return_width) * biasRatio)
        top = 0
        right = left + return_width
        bottom = return_width
    else:
        left = 0
        top = int((img_width - return_width) * biasRatio)
        right = return_width
        bottom = top + return_width

    return img_to_array(img.crop((left, top, right, bottom)))

# Data Agumentation: https://github.com/aleju/imgaug

"""
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.15))), # crop images by 0-15% of their height/width
        #st(iaa.GaussianBlur((0, 2.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.85, 1.15), per_channel=0.5)), # change brightness of images (75-125% of original value)
        st(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            #translate_px={"x": (-10, 10), "y": (-10, 10)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-15, 15), # rotate by -10 to +10 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
    ],
    random_order=True # do all of the above in random order
)
"""

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.5)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 3),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(1, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),

                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
                ]),
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)

def generator_batch(data_list, nbr_classes=3, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            if return_label:
                label = int(line[-1])%nbr_classes
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                Y_batch[i - current_index, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if save_network_input:
            print('X_batch.shape: {}'.format(X_batch.shape))
            X_to_save = X_batch.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_to_save[:, :, 2], delimiter = ' ')

        img = X_batch[0,:,:,:]
        img = np.reshape(img, (-1))
        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch

def generator_batch_multitask(data_list, nbr_class_one=250, nbr_class_two=7, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):

    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                label_one = int(line[-2])
                label_two = int(line[-1])
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if preprocess:
            for i in range(current_batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image=True,
                                               result_size=(img_width, img_height))
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if return_label:
            yield ([X_batch], [Y_batch_one, Y_batch_two])
        else:
            yield X_batch

def generator_batch_multitask_total(data_list, nbr_class_one=250, nbr_class_two=7, nbr_class_three=1000, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):

    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0

    #  current_batch_size处理不能整除的情况
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))
        Y_batch_three = np.zeros((current_batch_size, nbr_class_three))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                label_one = int(line[-2])%nbr_class_one
                label_two = int(line[-1])%nbr_class_two
                label_three = int(line[-3])%nbr_class_three
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1
                Y_batch_three[i - current_index, label_three] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if preprocess:
            for i in range(current_batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image=True,
                                               result_size=(img_width, img_height))
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if return_label:
            yield ([X_batch], [Y_batch_one, Y_batch_two, Y_batch_three])
        else:
            yield X_batch


def generator_batch_triplet(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    if shuffle:
        random.shuffle(data_list)

    N = len(data_list)
    dic = dic_data_list

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_anchor = np.zeros((current_batch_size, img_width, img_height, 3))
        X_positive = np.zeros((current_batch_size, img_width, img_height, 3))
        X_negative = np.zeros((current_batch_size, img_width, img_height, 3))

        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))
        Y_batch_fake = np.zeros((current_batch_size, 1))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print(line)
            anchor_path, vehicleID, modelID, colorID = line

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)

            anchor = scale_byRatio(anchor_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            if mode == 'train':
                # Find the same modelID, note that it is still a dictionary.
                # In this dictionary, the keys are colorIDs. In other words,
                # those images with the same modelID and the same colorID, the same
                # vehicleID are positives. Negatives are the ones with same modelID,
                # the same colorID but different vehicleIDs. Same modelID but different
                # colorIDs may also be considered as negatives.

                assert len(dic[modelID][colorID][vehicleID]) > 1, 'vehicleID: {} has only ONE image! The list is  {}'.format(vehicleID, dic[modelID][colorID][vehicleID])

                # copy a list of image paths with same vehicleID
                positive_list = dic[modelID][colorID][vehicleID][:]

                positive_list.remove(anchor_path)
                positive_path = random.choice(positive_list)
                positive = scale_byRatio(positive_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                negative_vehicleID_list = dic[modelID][colorID].keys()[:]
                negative_vehicleID_list.remove(vehicleID)
                assert negative_vehicleID_list !=[ ], 'vehicleID_list is [ ], {}'.format(dic[modelID][colorID].keys())
                negative_vehicleID = random.choice(negative_vehicleID_list)
                negative_path = random.choice(dic[modelID][colorID][negative_vehicleID])
                negative = scale_byRatio(negative_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = positive
                X_negative[i - current_index] = negative

            elif mode == 'val':
                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = anchor
                X_negative[i - current_index] = anchor

            if return_label:
                label_one = int(line[-2])%nbr_class_one
                label_two = int(line[-1])%nbr_class_two
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1

        if augment:
            X_anchor = X_anchor.astype(np.uint8)
            X_positive = X_positive.astype(np.uint8)
            X_negative = X_negative.astype(np.uint8)
            X_anchor = seq.augment_images(X_anchor)
            X_positive = seq.augment_images(X_positive)
            X_negative = seq.augment_images(X_negative)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_anchor[i - current_index])
                img.save(img_to_save_path)

        X_anchor = X_anchor.astype(np.float64)
        X_positive = X_positive.astype(np.float64)
        X_negative = X_negative.astype(np.float64)
        X_anchor = preprocess_input(X_anchor)
        X_positive = preprocess_input(X_positive)
        X_negative = preprocess_input(X_negative)

        if save_network_input:
            print('X_anchor.shape: {}'.format(X_anchor.shape))
            X_anchor_to_save = X_anchor.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_anchor_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_anchor_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_anchor_to_save[:, :, 2], delimiter = ' ')

        if return_label:
            yield ([X_anchor, X_positive, X_negative], [Y_batch_one, Y_batch_two, Y_batch_fake])
        else:
            if mode == 'feature_extraction':
                yield X_anchor
            else:
                yield [X_anchor, X_positive, X_negative]

def generator_batch_triplet_total(data_list, dic_data_list, nbr_class_one=250, nbr_class_two=7, nbr_class_three = 100,
                    batch_size=32, return_label=True, mode='train',
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, save_network_input=None, augment=False):
    if shuffle:
        random.shuffle(data_list)

    N = len(data_list)
    dic = dic_data_list

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_anchor = np.zeros((current_batch_size, img_width, img_height, 3))
        X_positive = np.zeros((current_batch_size, img_width, img_height, 3))
        X_negative = np.zeros((current_batch_size, img_width, img_height, 3))

        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))
        Y_batch_two = np.zeros((current_batch_size, nbr_class_two))
        Y_batch_three = np.zeros((current_batch_size, nbr_class_three))
        Y_batch_fake = np.zeros((current_batch_size, 1))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print(line)
            anchor_path, vehicleID, modelID, colorID = line

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)

            anchor = scale_byRatio(anchor_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            if mode == 'train':
                # Find the same modelID, note that it is still a dictionary.
                # In this dictionary, the keys are colorIDs. In other words,
                # those images with the same modelID and the same colorID, the same
                # vehicleID are positives. Negatives are the ones with same modelID,
                # the same colorID but different vehicleIDs. Same modelID but different
                # colorIDs may also be considered as negatives.

                assert len(dic[modelID][colorID][vehicleID]) > 1, 'vehicleID: {} has only ONE image! The list is  {}'.format(vehicleID, dic[modelID][colorID][vehicleID])

                # copy a list of image paths with same vehicleID
                positive_list = dic[modelID][colorID][vehicleID][:]

                positive_list.remove(anchor_path)
                positive_path = random.choice(positive_list)
                positive = scale_byRatio(positive_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                negative_vehicleID_list = dic[modelID][colorID].keys()[:]
                negative_vehicleID_list.remove(vehicleID)
                assert negative_vehicleID_list !=[ ], 'vehicleID_list is [ ], {}'.format(dic[modelID][colorID].keys())
                negative_vehicleID = random.choice(negative_vehicleID_list)
                negative_path = random.choice(dic[modelID][colorID][negative_vehicleID])
                negative = scale_byRatio(negative_path, ratio=scale_ratio, return_width=img_width,
                                    crop_method=crop_method)

                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = positive
                X_negative[i - current_index] = negative

            elif mode == 'val':
                X_anchor[i - current_index] = anchor
                X_positive[i - current_index] = anchor
                X_negative[i - current_index] = anchor

            if return_label:
                label_one = int(line[-2])%nbr_class_one
                label_two = int(line[-1])%nbr_class_two
                label_three = int(line[-3])%nbr_class_three
                Y_batch_one[i - current_index, label_one] = 1
                Y_batch_two[i - current_index, label_two] = 1
                Y_batch_three[i - current_index, label_three] = 1

        if augment:
            X_anchor = X_anchor.astype(np.uint8)
            X_positive = X_positive.astype(np.uint8)
            X_negative = X_negative.astype(np.uint8)
            X_anchor = seq.augment_images(X_anchor)
            X_positive = seq.augment_images(X_positive)
            X_negative = seq.augment_images(X_negative)

        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_anchor[i - current_index])
                img.save(img_to_save_path)

        X_anchor = X_anchor.astype(np.float64)
        X_positive = X_positive.astype(np.float64)
        X_negative = X_negative.astype(np.float64)
        X_anchor = preprocess_input(X_anchor)
        X_positive = preprocess_input(X_positive)
        X_negative = preprocess_input(X_negative)

        if save_network_input:
            print('X_anchor.shape: {}'.format(X_anchor.shape))
            X_anchor_to_save = X_anchor.reshape((299, 299, 3))
            to_save_base_name = save_network_input[:-4]
            np.savetxt(to_save_base_name + '_0.txt', X_anchor_to_save[:, :, 0], delimiter = ' ')
            np.savetxt(to_save_base_name + '_1.txt', X_anchor_to_save[:, :, 1], delimiter = ' ')
            np.savetxt(to_save_base_name + '_2.txt', X_anchor_to_save[:, :, 2], delimiter = ' ')

        if return_label:
            yield ([X_anchor, X_positive, X_negative], [Y_batch_one, Y_batch_two, Y_batch_three, Y_batch_fake])
        else:
            if mode == 'feature_extraction':
                yield X_anchor
            else:
                yield [X_anchor, X_positive, X_negative]



class VehicleID_Triplet_Sequence(Sequence):

    def __init__(self, data_list, dic_data_list, nbr_models=250, nbr_colors=7,
                        batch_size=32, return_label=True,
                        crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                        img_width=299, img_height=299, shuffle=True,
                        save_to_dir=None, augment=False):
        self.data_list = self.dic_data_list
        self.nbr_models = nbr_models
        self.nbr_colors = nbr_colors
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)




def generator_batch_multitask_id(data_list, nbr_class_one=250, nbr_class_two=7, batch_size=32, return_label=True,
                    crop_method=center_crop, scale_ratio=1.0, random_scale=False,
                    preprocess=False, img_width=299, img_height=299, shuffle=True,
                    save_to_dir=None, augment=False):
    '''
    A generator that yields a batch of (data, class_one, class_two).

    Input:
        data_list  : a MxNet style of data list, e.g.
                     "/data/workspace/dataset/Cervical_Cancer/train/Type_1/0.jpg 0"
        shuffle    : whether shuffle rows in the data_llist
        batch_size : batch size

    Output:
        (X_batch, Y1_batch, Y2_batch)
    '''

    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_one = np.zeros((current_batch_size, nbr_class_one))

        for i in range(current_index, current_index + current_batch_size):
            # ./data/VehicleID_V1.0/image/0058148.jpg 1891 216 2
            line = data_list[i].strip().split(' ')
            #print line
            img_path = line[0]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img

            if return_label:
                label_one = int(line[-3])%nbr_class_one
                # one hot 编码 , nbr_class_one 指class_one的值
                Y_batch_one[i - current_index, label_one] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        if preprocess:
            for i in range(current_batch_size):
                X_batch[i] = preprocessing_eye(X_batch[i], return_image=True,
                                               result_size=(img_width, img_height))
        if save_to_dir:
            for i in range(current_index, current_index + current_batch_size):
                tmp_path = data_list[i].strip().split(' ')[0]
                basedir = tmp_path.split(os.sep)[-2:]
                image_name = '_'.join(basedir)
                img_to_save_path = os.path.join(save_to_dir, image_name)
                img = array_to_img(X_batch[i - current_index])
                img.save(img_to_save_path)

        X_batch = X_batch.astype(np.float64)
        X_batch = preprocess_input(X_batch)

        if return_label:
            yield ([X_batch], Y_batch_one)
        else:
            yield X_batch