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
    max_width = max(MIN_IMAGE_WIDTH, int(width*max_scale))
    max_height = max(MIN_IMAGE_HEIGHT, int(height*max_scale))
    region = [np.random.randint(0, (width-max_width)+1, 1)[0],
              np.random.randint(0, (height-max_height)+1, 1)[0],
              0, 0]
    region[2] = np.random.randint(
        MIN_IMAGE_WIDTH, max_width+1, 1)[0]
    region[3] = np.random.randint(
        MIN_IMAGE_HEIGHT, max_height+1, 1)[0]
    return region


def transform_bbox(bbox, start_point, scale):
    bbox = np.array(bbox)
    bbox_scale = np.tile(scale, [1, 2])
    bbox = bbox * bbox_scale

    bbox_translation = np.zeros([1, 4])
    bbox_translation[:, :2] = start_point
    bbox = bbox + bbox_translation
    return list(bbox)


def transform_landmark(landmark, start_point, scale):
    landmark = np.array(landmark)
    landmark_scale = np.tile(scale, [1, 5])
    landmark = landmark * landmark_scale

    landmark_translation = np.tile(start_point, [1, 5])
    landmark = landmark + landmark_translation
    return list(landmark)


def generate_random_data(images_list, bboxes_list,
                         landmarks_list):
    """
    random crop the image to generate new data
    param: images_list: list of image
    param: bboxes_list: bboxes of the image
    param: landmarks_list: landmarks of the image
    param: size: target size, (width, height)
    return: (image, labels, bboxes, landmarks) generated data
    """

    cropped_images_list = []
    cropped_bboxes_list = []
    cropped_landmarks_list = []
    cropped_labels_list = []

    width = images_list[0].shape[1]
    height = images_list[0].shape[0]

    cropped_region = generate_region(width, height, 0.5)
    zero_bbox = [0]*4
    zero_landmark = [0]*10

    for image, bboxes, landmarks in zip(images_list, bboxes_list,
                                        landmarks_list):
        image = image[cropped_region[1]:(cropped_region[1]+cropped_region[3]),
                      cropped_region[0]:(cropped_region[0]+cropped_region[2]),
                      :]
        intersect_bbox = [utils.intersect_region(
            b, cropped_region) for b in bboxes]
        iou_list = [utils.IOU(intersect_bbox[i], bboxes[i])
                    for i in range(len(intersect_bbox))]

        iou_list = np.array(iou_list)
        indices = np.where(iou_list > NEGATIVE_UPPER_BOUNCE)[0]
        max_iou = np.argmax(iou_list)
        if len(indices) > 0:
            intersect_bbox = np.array(intersect_bbox)
            bboxes = list(intersect_bbox[indices])

            landmarks = np.array(landmarks)
            landmarks = list(landmarks[indices])

            bboxes = transform_bbox(bboxes, cropped_region[:2], (1, 1))
            landmarks = transform_landmark(
                landmarks, cropped_region[:2], (1, 1))
            label = 1 if max_iou > POSITIVE_LOWER_BOUNCE else 0.8
        else:
            bboxes = [zero_bbox]
            landmarks = [zero_landmark]
            label = 0
        cropped_images_list.append(image)
        cropped_bboxes_list.append(bboxes)
        cropped_landmarks_list.append(landmarks)
        cropped_labels_list.append(label)
    return cropped_images_list, cropped_labels_list, \
        cropped_bboxes_list, cropped_landmarks_list


def resize_image_bboxes_landmarks(images_list, bboxes_list,
                                  landmarks_list, size):
    """
    resize image, bboxes, landmarks into size
    param: images_list: list of image
    param: bboxes_list: bboxes of the image
    param: landmarks_list: landmarks of the image
    param: size: target size, (width, height)
    return: (image, bboxes, landmarks) wanted to resize
    """
    resized_image_list = []
    resized_bboxes_list = []
    resized_landmarks_list = []
    for image, bboxes, landmarks in zip(images_list, bboxes_list,
                                        landmarks_list):

        origin_width = image.shape[1]
        origin_height = image.shape[0]

        scale_width = origin_width/size[0]
        scale_height = origin_height/size[1]

        resized_image = cv2.resize(image, tuple(size))

        resized_bboxes = []
        for b in bboxes:
            resized_b = [b[0]/scale_width, b[1]/scale_height,
                         b[2]/scale_width, b[3]/scale_height]
        resized_bboxes.append(resized_b)

        resized_landmarks = []
        for lr in landmarks:
            resized_lr = []
            for j, lr_value in enumerate(lr):
                if j % 2 == 0:
                    resized_lr.append(lr_value/scale_width)
                else:
                    resized_lr.append(lr_value/scale_height)
            resized_landmarks.append(resized_lr)
        resized_image_list.append(resized_image)
        resized_bboxes_list.append(resized_bboxes)
        resized_landmarks_list.append(resized_landmarks)

    return resized_image_list, resized_bboxes_list, resized_landmarks_list


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
