import torch
import torch.nn

from facelib import utils, mtcnn, data

import numpy as np

import logging

NEGATIVE_UPPER_BOUNCE = 0.3
POSITIVE_LOWER_BOUNCE = 0.6

TRAIN_PNET = "P_Net"
TRAIN_RNET = "R_Net"
TRAIN_ONET = "O_Net"

log = logging.getLogger("train")


def generate_ground_truth_data(labels, bboxes, landmarks,
                               res_bboxes):
    res_num = res_bboxes.shape[0]
    gt_labels = []
    gt_bboxes = []
    gt_landmarks = []
    for i in range(res_num):
        num = labels.shape[0]
        iou_list = [utils.IOU(bboxes[j], res_bboxes[i])
                    for j in range(num)]
        max_iou_idx = np.argmax(iou_list)
        gt_labels.append(labels[max_iou_idx])
        gt_bboxes.append(bboxes[max_iou_idx])
        gt_landmarks.append(landmarks[max_iou_idx])
    gt_labels = torch.stack(gt_labels, dim=0)
    gt_bboxes = torch.stack(gt_bboxes, dim=0)
    gt_landmarks = torch.stack(gt_landmarks, dim=0)
    return gt_labels, gt_bboxes, gt_landmarks


def train(net, optimizer, images_list, labels_list,
          bboxes_list, landmarks_list,
          filter, train_type=TRAIN_ONET):
    loss_list = []
    for image, labels, bboxes, landmarks in zip(images_list,
                                                labels_list,
                                                bboxes_list,
                                                landmarks_list):
        if train_type == TRAIN_PNET:
            classifications_result, bboxes_result, landmarks_result = \
                net.forward_to_pnet(image, filter)
        elif train_type == TRAIN_RNET:
            classifications_result, bboxes_result, landmarks_result = \
                net.forward_to_rnet(image, filter)
        elif train_type == TRAIN_ONET:
            classifications_result, bboxes_result, landmarks_result = \
                net.forward_to_onet(image, filter)
        else:
            log.error("error train type %s" % (train_type))
            return

        classifications_result = classifications_result[0]
        bboxes_result = bboxes_result[0]
        landmarks_result = landmarks_result[0]
        width = image.shape[-1]
        height = image.shape[-2]
        bboxes = mtcnn.scale_bounding_box(bboxes, width, height)
        landmarks = mtcnn.scale_landmarks(landmarks,
                                          width, height)
        bboxes = bboxes.clamp(min=0.0, max=1.0)
        landmarks = landmarks.clamp(min=0.0, max=1.0)
        if labels.numel() > 1:
            gt_labels, gt_bboxes, gt_landmarks = \
                generate_ground_truth_data(
                    labels, bboxes, landmarks, bboxes_result)
        else:
            gt_labels = labels[0]
            gt_bboxes = bboxes[0]
            gt_landmarks = landmarks[0]
        res_labels_positive = torch.clamp(
            classifications_result[:, 0], min=1e-3, max=0.999)
        res_labels_negative = torch.clamp(
            classifications_result[:, 1], min=1e-3, max=0.999)
        label_loss = -(gt_labels*torch.log(res_labels_positive) +
                       (1-gt_labels)*(torch.log(res_labels_negative)))
        bbox_loss = torch.sum((bboxes_result - gt_bboxes)**2, -1)
        landmark_loss = torch.sum((landmarks_result - gt_landmarks)**2, -1)

        optimizer.zero_grad()
        if train_type == TRAIN_PNET or train_type == TRAIN_RNET:
            loss = label_loss + bbox_loss*0.5 + landmark_loss*0.5
        elif train_type == TRAIN_ONET:
            loss = label_loss + bbox_loss*0.5 + landmark_loss
        else:
            loss = label_loss + bbox_loss + landmark_loss
        loss = torch.mean(loss, -1)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return np.mean(loss_list)


def train_Pnet(net, optimizer, images_list, labels_list,
               bboxes_list, landmarks_list,
               filter=False):
    train(net, optimizer, images_list, labels_list,
          bboxes_list, landmarks_list, filter, "P_Net")


def train_Rnet(net, optimizer, images_list, labels_list,
               bboxes_list, landmarks_list,
               filter=False):
    train(net, optimizer, images_list, labels_list,
          bboxes_list, landmarks_list, filter, "R_Net")


def train_Onet(net, optimizer, images_list, labels_list,
               bboxes_list, landmarks_list,
               filter=False):
    train(net, optimizer, images_list, labels_list,
          bboxes_list, landmarks_list, filter, "O_Net")


def calc_bbox_metrics(res, gt):
    """
    Compute the metric between result and ground-truth
    param: res: tensor, shape = (res_num, 4), result data
    param: gt: tensor, shape = (gt_num, 4), ground-truth data
    return: TP, FP, TN, FN
    """
    BOUNDCE = 0.5
    res_num = res.shape[0]
    gt_num = gt.shape[0]

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(res_num):
        for j in range(gt_num):
            intersect = utils.Intersect(res[i], gt[j])
            res_area = utils.Area(res[i])
            gt_area = utils.Area(gt[j])
            if intersect/res_area > BOUNDCE and intersect/gt_area > BOUNDCE:
                TP += 1
            elif intersect/res_area < BOUNDCE and intersect/gt_area > BOUNDCE:
                FN += 1
            elif intersect/res_area > BOUNDCE and intersect/gt_area < BOUNDCE:
                FP += 1
            elif intersect/res_area < BOUNDCE and intersect/gt_area < BOUNDCE:
                TN += 1
    return TP, FP, TN, FN


def calc_clasification_metrics(res, gt):
    """
    Compute the metric between result and ground-truth
    param: res: tensor, shape = (res_num, 10), result data
    param: gt: tensor, shape = (gt_num, 10), ground-truth data
    return: TP, FP, TN, FN
    """
    BOUNDCE = 0.1
    res_num = res.shape[0]
    gt_num = gt.shape[0]

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(res_num):
        for j in range(gt_num):
            true_num = torch.sum(torch.abs(res[i]-gt[j]) < BOUNDCE)
            TP += true_num
            FP += 10-true_num
            TN += true_num
            FN += 10-true_num
    return TP, FP, TN, FN


def calc_landmark_metrics(res, gt):
    """
    Compute the metric between result and ground-truth
    param: res: tensor, shape = (res_num, 2), result data
    param: gt: tensor, shape = (gt_num, 2), ground-truth data
    return: TP, FP, TN, FN
    """
    BOUNDCE = 0.5
    res_num = res.shape[0]
    gt_num = gt.shape[0]

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(res_num):
        for j in range(gt_num):
            if res[i, 0] > BOUNDCE and gt[i, 0] > BOUNDCE:
                TP += 1
            elif res[i, 0] < BOUNDCE and gt[i, 0] > BOUNDCE:
                FN += 1
            elif res[i, 0] > BOUNDCE and gt[i, 0] < BOUNDCE:
                FP += 1
            elif res[i, 0] < BOUNDCE and gt[i, 0] < BOUNDCE:
                TN += 1
    return TP, FP, TN, FN


def get_metrics(net, test_path, bboxes_data, landmarks_data, device):
    gt_bboxes_result_list = []
    gt_labels_result_list = []
    gt_landmarks_result_list = []

    res_bboxes_result_list = []
    res_labels_result_list = []
    res_landmarks_result_list = []
    k = 0
    for images_list, bboxes_list, landmarks_list in \
            data.iterate_image_batches(test_path,
                                       bboxes_data, landmarks_data, 1):
        # prepare data

        images_list = data.processing_images(images_list)

        images_list = [torch.FloatTensor([image]).to(device)
                       for image in images_list]
        bboxes_list = [torch.FloatTensor(bboxes).to(device)
                       for bboxes in bboxes_list]
        labels_list = [bboxes.new_ones(bboxes.shape[0]).to(device)
                       for bboxes in bboxes_list]
        landmarks_list = [torch.FloatTensor(landmarks).to(
            device) for landmarks in landmarks_list]
        # get the net ouput
        for image, labels, bboxes, landmarks in zip(images_list,
                                                    labels_list,
                                                    bboxes_list,
                                                    landmarks_list):

            classifications_result, bboxes_result, landmarks_result = \
                net.forward(image)

            classifications_result = classifications_result[0]
            bboxes_result = bboxes_result[0]
            landmarks_result = landmarks_result[0]
            width = image.shape[-1]
            height = image.shape[-2]
            bboxes = mtcnn.scale_bounding_box(bboxes, width, height)
            landmarks = mtcnn.scale_landmarks(landmarks,
                                              width, height)
            bboxes = bboxes.clamp(min=0.0, max=1.0)
            landmarks = landmarks.clamp(min=0.0, max=1.0)
            if labels.numel() > 1:
                gt_labels, gt_bboxes, gt_landmarks = \
                    generate_ground_truth_data(
                        labels, bboxes, landmarks, bboxes_result)
            else:
                gt_labels = labels[0]
                gt_bboxes = bboxes[0]
                gt_landmarks = landmarks[0]

            gt_labels_result_list.append(gt_labels)
            gt_bboxes_result_list.append(gt_bboxes)
            gt_landmarks_result_list.append(gt_landmarks)

            res_labels_result_list.append(classifications_result.squeeze(0))
            res_bboxes_result_list.append(bboxes_result.squeeze(0))
            res_landmarks_result_list.append(landmarks_result.squeeze(0))

            k += 1
            if k > 100:
                break
        if k > 100:
            break

    gt_labels = torch.stack(gt_labels_result_list, dim=0)
    gt_bboxes = torch.stack(gt_bboxes_result_list, dim=0)
    gt_landmarks = torch.stack(gt_landmarks_result_list, dim=0)
    res_labels = torch.stack(res_labels_result_list, dim=0)
    res_bboxes = torch.stack(res_bboxes_result_list, dim=0)
    res_landmarks = torch.stack(res_landmarks_result_list, dim=0)

    classification_recall = utils.Recall(
        res_bboxes, gt_bboxes, calc_bbox_metrics)
    bbox_recall = utils.Recall(res_labels, gt_labels,
                               calc_clasification_metrics)
    landmark_recall = utils.Recall(
        res_landmarks, gt_landmarks, calc_landmark_metrics)

    return classification_recall, bbox_recall, landmark_recall
