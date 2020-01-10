import torch
import torch.nn

from facelib import utils, mtcnn, data, model_utils
import numpy as np

import logging

NEGATIVE_UPPER_BOUNCE = 0.3
POSITIVE_LOWER_BOUNCE = 0.6

TRAIN_PNET = "P_Net"
TRAIN_RNET = "R_Net"
TRAIN_ONET = "O_Net"

log = logging.getLogger("train")


def generate_pnet_ground_truth_data(images_list, bboxes_list, landmarks_list):
    """
    generate_pnet_ground_truth_data
    param: images_list: list of image
    param: bboxes_list: list of all bounding-boxes of the image
    param: landmarks_list: list of all landmarks of the image
    return: (images_list, labels_list, new_bboxes_list, new_landmarks_list)
    """
    num = len(images_list)

    new_bboxes_list = []
    new_landmarks_list = []
    labels_list = []

    for i in range(num):
        width = images_list[i].shape[1]
        height = images_list[i].shape[0]
        start_points_list = utils.start_points_for_conv(
            [width, height], mtcnn.PNET_INPUT_SIZE, (2, 2))

        present_bboxes = []
        present_landmarks = []
        present_labels = []
        for start_point in start_points_list:

            iou_list = [
                utils.IOU([start_point[0], start_point[1], 12, 12], b)
                for b in bboxes_list[i]]
            intersect = [utils.intersect_region(
                [start_point[0], start_point[1], 12, 12], b) for b in bboxes_list[i]]

            intersect = [(start_point[0], start_point[1], 0, 0)
                         if b[2] == 0 or b[3] == 0 else b for b in intersect]

            inters_landmark = [lm for lm in landmarks_list[i]]

            max_iou = np.max(iou_list)
            max_iou_idx = np.argmax(iou_list)

            if max_iou < NEGATIVE_UPPER_BOUNCE:
                zero_landmark = [start_point[0] if i %
                                 2 == 0 else start_point[1] for i in range(10)]
                present_bboxes.append([start_point[0], start_point[1], 0, 0])
                present_labels.append(0)
                present_landmarks.append(zero_landmark)
            elif max_iou < POSITIVE_LOWER_BOUNCE:
                present_bboxes.append(intersect[max_iou_idx])
                present_labels.append(0.8)
                present_landmarks.append(inters_landmark[max_iou_idx])
            else:
                present_bboxes.append(intersect[max_iou_idx])
                present_labels.append(1.0)
                present_landmarks.append(inters_landmark[max_iou_idx])
        new_bboxes_list.append(present_bboxes)
        new_landmarks_list.append(present_landmarks)
        labels_list.append(present_labels)
    return images_list, labels_list, new_bboxes_list, new_landmarks_list


def train(net, optimizer, images, labels,
          bboxes, landmarks,
          filter, train_type=TRAIN_ONET):

    if train_type == TRAIN_PNET:
        batch_output = net.forward_to_pnet(images, filter)
        if not filter:
            batch_output = batch_output[:3]
    elif train_type == TRAIN_RNET:
        batch_output = net.forward_to_rnet(images, filter)
    else:
        batch_output = net.forward_to_onet(images, filter)

    lable_loss_list = []
    bbox_loss_list = []
    landmark_loss_list = []

    for i, output in enumerate(zip(*batch_output)):
        classifications_result, bboxes_result, landmarks_result = \
            output[:3]
        gt_labels = labels[i]
        gt_bboxes = bboxes[i]
        gt_landmarks = landmarks[i]
        if len(output) == 4:
            indices = output[3]
            gt_labels = gt_labels.index_select(0, indices)
            gt_bboxes = gt_bboxes.index_select(0, indices)
            gt_landmarks = gt_landmarks.index_select(0, indices)

        lable_loss = -(gt_labels*torch.log(classifications_result[:, 0])+(
            1-gt_labels)*torch.log(classifications_result[:, 1]))
        bbox_loss = torch.sum((bboxes_result - gt_bboxes)**2, -1)
        landmark_loss = torch.sum((landmarks_result - gt_landmarks)**2, -1)
        lable_loss_list.append(lable_loss)
        bbox_loss_list.append(bbox_loss)
        landmark_loss_list.append(landmark_loss)

    lable_loss_list = torch.cat(lable_loss_list)
    bbox_loss_list = torch.cat(bbox_loss_list)
    landmark_loss_list = torch.cat(landmark_loss_list)

    if train_type == TRAIN_RNET or train_type == TRAIN_PNET:
        loss = lable_loss_list + bbox_loss_list*0.5 + landmark_loss_list*0.5
    else:
        loss = lable_loss_list + bbox_loss_list*0.5 + landmark_loss_list

    loss = torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


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
