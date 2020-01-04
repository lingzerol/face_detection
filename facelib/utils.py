import numpy as np
import cv2


def iterate_batches(data, batch_size):
    assert isinstance(data, list)
    assert isinstance(batch_size, int)

    offset = 0
    while True:
        s = offset * batch_size
        e = min((offset + 1) * batch_size, len(data))
        if e <= s:
            break
        batch = data[s:e]
        yield batch
        offset += 1


def Area(r):
    """
    Compute the area between rectangle1 and rectangle2
    param: r: (x, y, w, h)
    return: a number: area
    """
    return r[2]*r[3]


def Intersect(r1, r2):
    """
    Compute the intersect between rectangle1 and rectangle2
    param: r1: (x, y, width, height)
    param: r2: (x, y, width, height)
    return: a number: area of intersect
    """
    xmin_i = max(r1[0], r2[0])
    ymin_i = max(r1[1], r2[1])

    xmax_i = min(r1[2]+r1[0], r2[2]+r2[0])
    ymax_i = min(r1[3]+r1[1], r2[3]+r2[1])

    w = max(0, xmax_i-xmin_i)
    h = max(0, ymax_i-ymin_i)

    return w*h


def Union(r1, r2):
    """
    Compute the union between rectangle1 and rectangle2
    param: r1: (x, y, width, height)
    param: r2: (x, y, width, height)
    return: a number: area of union
    """

    area1 = Area(r1)
    area2 = Area(r2)
    inters = Intersect(r1, r2)

    return area1+area2-inters


def IOU(r1, r2):
    union_area = float(Union(r1, r2))
    inters_area = float(Intersect(r1, r2))
    return float(inters_area/union_area)


def calc_metrics(res, gt):
    """
    Compute the metric between result and ground-truth
    param: res: result data
    param: gt: ground-truth data
    return: TP, FP, TN, FN
    """
    res = np.array(res)
    gt = np.array(gt)
    assert res.shape[0] == gt.shape[0]

    num = res.shape[0]

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(num):
        if not res[i] and gt[i]:
            FN += 1
        elif res[i] and gt[i]:
            TP += 1
        elif not res[i] and not gt[i]:
            TN += 1
        elif res[i] and not gt[i]:
            FP += 1
    return TP, FP, TN, FN


def Recall(res, gt, calc=calc_metrics):
    """
    Compute the recall between result and ground-truth
    param: res: result data
    param: gt: ground-truth data
    param: calc: function to calculate TP, FP, TN, FN
    return: a number: recall
    """
    TP, FP, TN, FN = calc(res, gt)
    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return -1


def Precision(res, gt, calc=calc_metrics):
    """
    Compute the precision between result and ground-truth
    param: res: result data
    param: gt: ground-truth data
    param: calc: function to calculate TP, FP, TN, FN
    return: a number: precision
    """
    TP, FP, TN, FN = calc(res, gt)
    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return -1
