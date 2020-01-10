import numpy as np
from multiprocessing.dummy import Pool


class MultiThreadPool:
    def __init__(self, pool_size=1):
        self.pool_size = pool_size
        self._pool = Pool(self.pool_size)

    def start(self, target, args):
        self._pool.apply_async(func=target, agrs=args)

    def join(self):
        self._pool.close()
        self._pool.join()

        self._pool = Pool(self.pool_size)

    def map(self, target, args_list):
        self._pool.map(target, args_list)

    def __del__(self):
        self._pool.close()
        self._pool.join()


def intersect_region(r1, r2):
    xmin_i = max(r1[0], r2[0])
    ymin_i = max(r1[1], r2[1])

    xmax_i = min(r1[2]+r1[0], r2[2]+r2[0])
    ymax_i = min(r1[3]+r1[1], r2[3]+r2[1])

    w = max(0, xmax_i-xmin_i)
    h = max(0, ymax_i-ymin_i)

    if w == 0 or h == 0:
        xmin_i = 0
        ymin_i = 0

    return (xmin_i, ymin_i, w, h)


def start_points_for_conv(size, kernel_size, stride, padding=(0, 0)):
    """
    get the start_points of every result of the conv
    param: size: (width, height), image size
    param: kernel_size, stride, padding: conv param
    return: start_points: list of the start_point of every result,
    len = (height*width)
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    width = size[0] + 2*padding[0]
    height = size[1] + 2*padding[1]
    start_points = []
    for i in range(0, int((height-kernel_size[0]+stride[0])/stride[0])):
        for j in range(0, int((width-kernel_size[1]+stride[1])/stride[1])):
            start_points.append([j*stride[0], i*stride[1]])
    return start_points


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
    r = intersect_region(r1, r2)

    return r[2]*r[3]


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
