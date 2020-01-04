import cv2
import os
import pickle
import numpy as np
from . import utils

LANDMARK_POINT_NUM = 5
BBOX_PARAM_NUM = 4
MIN_IMAGE_WIDTH = 12
MIN_IMAGE_HEIGHT = 12

NEGATIVE_UPPER_BOUNCE = 0.3
POSITIVE_LOWER_BOUNCE = 0.6


def save_file(data, filename, dir_name="./"):
    with open(os.path.join(dir_name, filename), "wb") as fout:
        pickle.dump(data, fout)


def load_file(filename, dir_name="./"):
    with open(os.path.join(dir_name, filename), "rb") as fin:
        return pickle.load(fin)


def get_image_path_list(dir_name):
    """
    param: dir_name: string, the path of image
    return: list: the path of all image
    """
    result = []
    for filename in os.listdir(dir_name):
        result.append(os.path.join(dir_name, filename))
    return result


def load_image(images_path):
    """
    param: images_path: list of the path of images
    return: dictionary: image_name -> image
    """
    result = {}
    for path in images_path:
        image = cv2.imread(path)
        image_name = os.path.split(path)[-1]
        result[image_name] = image
    return result


def processing_images(images):
    """
    transform all image into (channel, height, width) and normalize it
    param: images: list of image, every image shape =
    (height, width, channel)
    return: list of transformed image, every image shape =
    (channel, height, width)
    """
    result = []
    for image in images:
        trans_image = np.transpose(image, [2, 0, 1])
        normal_image = (trans_image-127.5)/255
        result.append(normal_image)
    return result


def split_image_into_train_and_test(images_path, train_ratio=0.95):
    count = int(len(images_path)*train_ratio)
    return images_path[:count], images_path[count:]


def load_image_bbox_and_landmark(filename, dir_name="./"):
    """
    param: filename: the file load gt bbox and landmark
    param: dir_name: the path of the file
    return: bbox: dictionary: the bbox of the target, image_name -> rectangle
    (x,y,width,height)
    return: landmark: dictionary: the landmark of the target, image_name ->
    point(x,y)
    """
    bboxes = {}
    landmarks = {}
    with open(os.path.join(dir_name, filename), "r") as fin:
        for line in fin.readlines():
            d = line.split()
            image_name = os.path.split(d[0])[-1]
            rectangle = [int(d[1]), int(d[3]), int(
                d[2])-int(d[1]), int(d[4])-int(d[3])]
            ld = list(map(float, d[5:]))
            bboxes[image_name] = [rectangle]
            landmarks[image_name] = [ld]
    return bboxes, landmarks


def combine_image_and_result(images, bboxes, landmarks):
    """
    param: images: dict: image
    param: bboxes: dict: bbox of the image
    param: landmarks: dict: landmark of the image
    return: result: list: images_list, bboxes_list, landmarks_list
    """
    images_list = []
    bboxes_list = []
    landmarks_list = []
    for name in images:
        images_list.append(images[name])
        bboxes_list.append(bboxes[name])
        landmarks_list.append(landmarks[name])
    return images_list, bboxes_list, landmarks_list


def iterate_image_batches(images_path, bboxes, landmarks, batch_size):
    assert isinstance(images_path, list)
    assert isinstance(bboxes, dict)
    assert isinstance(landmarks, dict)

    offset = 0
    while True:
        s = offset*batch_size
        e = min((offset+1)*batch_size, len(images_path))
        if s >= e:
            break
        batch = combine_image_and_result(
            load_image(images_path[s:e]), bboxes, landmarks)
        yield batch
        offset += 1


def generate_region(width, height, max_scale):
    region = [np.random.randint(0, max(1, int(width*(1-max_scale))), 1)[0],
              np.random.randint(0, max(1, int(height*(1-max_scale))), 1)[0],
              0, 0]
    region[2] = np.random.randint(
        MIN_IMAGE_WIDTH, int(width*(max_scale)), 1)[0]
    region[3] = np.random.randint(
        MIN_IMAGE_HEIGHT, int(height*(max_scale)), 1)[0]
    return region


def resize_image_bboxes_landmarks(image, bboxes, landmarks, size):
    """
    resize image, bboxes, landmarks into size
    param: image: numpy array
    param: bboxes: bboxes of the image
    param: landmarks: landmarks of the image
    param: size: target size, (width, height)
    return: (image, bboxes, landmarks) wanted to resize
    """
    bboxes = np.array(bboxes)
    landmarks = np.array(landmarks)
    image = np.array(image)

    origin_width = image.shape[1]
    origin_height = image.shape[0]

    scale_width = origin_width/size[0]
    scale_height = origin_height/size[1]

    resized_bboxes = bboxes.copy()
    for i in range(len(bboxes)):
        resized_bboxes[i][0] /= scale_width
        resized_bboxes[i][1] /= scale_height
        resized_bboxes[i][2] /= scale_width
        resized_bboxes[i][3] /= scale_height

    resized_landmarks = landmarks.copy()
    for i in range(len(landmarks)):
        for j in range(len(landmarks[i])):
            if i % 2 == 0:
                resized_landmarks[i][j] /= scale_width
            else:
                resized_landmarks[i][j] /= scale_height
    resized_image = cv2.resize(image, tuple(size))
    return resized_image, resized_bboxes, resized_landmarks


def generate_images(images_list, bboxes_list, landmarks_list,
                    dest_width, dest_height, num=15,
                    max_scale=0.7):
    """
    generate positive, negative and partial face data from an image
    param: images_list: numpy array, shape = (batch, height, width, channel)
    param: bboxes_list: numpy array, shape = (batch, num, 4)
    param: num: num of dataset should be generated
    param: landmarks_list: numpt array, shape = (batch, num, 10)
    return: images: list, shape = (num, height, width, channel)
    return: labels: list, the label of the cropped image, shape =
    (num, 1)
    return: bboxes: list, the bbox of the target in image, element is a
    bbox list
    return: landmarks: list, the landmarks of target in image, the element
    is landmark
    """

    labels = []
    images = []
    bboxes_result = []
    landmarks_result = []
    zero_landmarks = list(np.zeros(LANDMARK_POINT_NUM*2))
    zero_region = [0, 0, 0, 0]

    for image, bboxes, landmarks in zip(images_list, bboxes_list,
                                        landmarks_list):
        width = image.shape[1]
        height = image.shape[0]
        dest_image, dest_bboxes, dest_landmarks = \
            resize_image_bboxes_landmarks(
                image, bboxes, landmarks, (dest_width, dest_height))
        images.append(dest_image)
        labels.append([1.0])
        bboxes_result.append(list(dest_bboxes))
        landmarks_result.append(list(dest_landmarks))
        if int(width*max_scale) <= MIN_IMAGE_WIDTH or \
                int(height*max_scale) <= MIN_IMAGE_HEIGHT:
            continue
        for i in range(num):
            region = generate_region(width, height, max_scale)
            dest_image = image[region[1]:(region[1]+region[3]),
                               region[0]:(region[0]+region[2]), :]

            intersect_region = []
            origin_intersect_region = []
            for bbox in bboxes:
                xmin = max(bbox[0], region[0])
                ymin = max(bbox[1], region[1])
                xmax = min(bbox[0] + bbox[2], region[0]+region[2])
                ymax = min(bbox[1] + bbox[3], region[1]+region[3])

                if xmin >= xmax or ymin >= ymax:
                    intersect_region.append(zero_region)
                    origin_intersect_region.append(zero_region)
                else:
                    intersect_region.append(
                        [xmin-region[0], ymin-region[1], xmax-xmin, ymax-ymin])
                    origin_intersect_region.append(
                        [xmin, ymin, xmax-xmin, ymax-ymin])
            iou_list = [utils.IOU(oir, bbox)
                        for oir in origin_intersect_region]
            target_landmarks = []
            target_region = []
            target_label = []
            for j, iou in enumerate(iou_list):
                tl = [(lm-region[k % 2]) for k, lm in enumerate(landmarks[j])]
                if iou < POSITIVE_LOWER_BOUNCE and \
                        iou >= NEGATIVE_UPPER_BOUNCE:
                    target_landmarks.append(tl)
                    target_region.append(intersect_region[j])
                    target_label.append(0.8)
                elif iou >= POSITIVE_LOWER_BOUNCE:
                    target_label.append(1.0)
                    target_landmarks.append(tl)
                    target_region.append(intersect_region[j])
            if len(target_landmarks) == 0:
                target_landmarks.append(zero_landmarks)
                target_region.append(zero_region)
                target_label.append(0)
            dest_image, dest_bboxes, dest_landmarks = \
                resize_image_bboxes_landmarks(
                    dest_image, target_region, target_landmarks,
                    (dest_width, dest_height))
            images.append(dest_image)
            landmarks_result.append(dest_landmarks)
            labels.append(target_label)
            bboxes_result.append(dest_bboxes)
    return images, labels, bboxes_result, landmarks_result


def generate_pyramid(images_list, bboxes_list, landmarks_list, scale=0.7,
                     max_num=10, min_width=MIN_IMAGE_WIDTH,
                     min_height=MIN_IMAGE_HEIGHT):
    """
    generate data pyramid
    param: images_list: image to generate pyramid, numpy array
    param: bboxes_list: bbox to generate pyramid, numpy array
    param: landmarks_list: landmark to generate pyramid, numpy array
    param: scale: scale to shrink the image
    param: max_num: max image num
    return: list: image pyramid
    return: list: bboxes pyramid
    return: list: landmarks pyramid
    """
    images_result = []
    bboxes_result = []
    landmarks_result = []
    for image, bboxes, landmarks in zip(images_list, bboxes_list,
                                        landmarks_list):
        th = int(image.shape[0]*scale)
        tw = int(image.shape[1]*scale)
        num = 0

        images_result.append(image)
        bboxes_result.append(bboxes)
        landmarks_result.append(landmarks)

        while th > min_height and tw > min_width and num < max_num and\
                scale > 0:
            new_image = cv2.resize(image, (th, tw))
            new_bboxes = [[int(b*scale) for b in bbox]
                          for bbox in bboxes_result[-1]]
            new_landmarks = [[lr*scale for lr in landmark]
                             for landmark in landmarks_result[-1]]
            images_result.append(new_image)
            bboxes_result.append(new_bboxes)
            landmarks_result.append(new_landmarks)
            th = int(th*scale)
            tw = int(tw*scale)
            num += 1
    return images_result, bboxes_result, landmarks_result
