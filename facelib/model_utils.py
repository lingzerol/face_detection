import torch
import torch.nn.functional as F


def pack_to_conv2d_sequence(data):
    """
    pack data list of tenor, the shape of (every tensor
    = (num, channel, height, width)), and batch = len(data)
    into shape = (batch*num, channel, height, width)
    param: data: list of tensor, data wanted to pack
    return: packed_data, batch_sizes
    packed_data: packed data, shape = (batch*num, channel, height, width)
    batch_sizes: list, every element is the num of each batch
    """
    batch_sizes = [d.shape[0] for d in data]
    packed_data = torch.cat(data)
    return packed_data, batch_sizes


def unpack_conv2d_sequence(data, batch_sizes):
    """
    unpack tenor , shape = (batch*num,
    channel, height, width)
    to list of tensor, the shape of every tensor
     = (num, channel, height, width)
    param: data: tensor, data wanted to unpack
    param: batch_sizes: list, every element is the num of each batch
    return: unpacked_data
    unpacked_data: list of tensor
    """
    offset = 0
    unpacked_data = []
    for num in batch_sizes:
        s = min(offset, data.shape[0])
        e = min(offset+num, data.shape[0])
        unpacked_data.append(data[s:e])
    return unpacked_data


def NMS(bboxes, scores, overlap=0.5):
    """
    nms algorithm to filter bboxes
    param: bboxes: list: rectangle of target, element: (x, y, width, height)
    param: scores: list: confident score of the rectangle
    return: list: index of box after filtering
    """
    assert isinstance(scores, torch.Tensor)
    assert isinstance(bboxes, torch.Tensor)

    keep = scores.new_zeros(scores.size(0)).long()
    if bboxes.numel() == 0:
        return keep

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2] + bboxes[:, 0]
    ymax = bboxes[:, 3] + bboxes[:, 1]

    area = torch.mul(xmax-xmin, ymax-ymin).to(torch.float32)

    _, idx = scores.sort()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        idx = idx[:-1]
        if idx.numel() <= 0:
            break

        xmin_t = xmin.index_select(0, idx)
        ymin_t = ymin.index_select(0, idx)
        xmax_t = xmax.index_select(0, idx)
        ymax_t = ymax.index_select(0, idx)

        xmin_t = xmin_t.clamp(min=float(xmin[i]))
        ymin_t = ymin_t.clamp(min=float(ymin[i]))
        xmax_t = xmax_t.clamp(max=float(xmax[i]))
        ymax_t = ymax_t.clamp(max=float(ymax[i]))

        w = xmax_t - xmin_t
        h = ymax_t - ymin_t
        w = w.clamp(min=0.0).to(torch.float32)
        h = h.clamp(min=0.0).to(torch.float32)
        inter = w*h

        rem_area = area.index_select(0, idx)
        union = rem_area + area[i] - inter
        iou = inter/union

        idx = idx[iou.le(overlap)]

    return keep[:count]


def bounding_box_regression(net, features, rectangles):
    """
    param: net: a net to get the shift, in_features = feature_len,
    out_feature = 4
    param: features: the feature of the rectangles
    param: rectangles: the rectangle of the target
    return: tensor: new rectangles
    """

    t = net(features)
    t = torch.sigmoid(t)
    tx = t[:, 0]
    ty = t[:, 1]
    tw = t[:, 2]
    th = t[:, 3]
    x = rectangles[:, 0]
    y = rectangles[:, 1]
    w = rectangles[:, 2]
    h = rectangles[:, 3]
    x = tx*w + x
    y = ty*h + y
    w = w*torch.exp(tw)
    h = h*torch.exp(th)

    x = x.clamp(min=0)
    y = y.clamp(min=0)

    result = rectangles.new_zeros(rectangles.size())

    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = w
    result[:, 3] = h

    return result


def crop_and_resize_image(image, regions, size):
    """
    crop the region of the image, and resize it into dest size
    param: image: tensor, shape = (channel, height, width)
    param: region: tensor, shape = (num, 4)
    param: size: dest size wanted to resize
    return: images: tensor, image after cropping and resizing,
    shape = (num, channel, size[1], size[0])
    return: start_points: tensor, start_point of the region,
    shape = (num, 2)
    return: scales: tensor, scale between origin size and dest size
    shape = (num, 2)
    """
    num = regions.shape[0]
    width = image.shape[-1]
    height = image.shape[-2]

    dest_image = image.unsqueeze(0)
    dest_image = F.interpolate(dest_image, size=size)
    dest_image = dest_image.squeeze(0)
    images = [dest_image]
    start_points = [image.new([0, 0])]
    scales = [image.new([width/size[0], height/size[1]])]

    regions = torch.clamp(regions, min=0.0)

    for i in range(num):
        xmin = int(regions[i][0])
        ymin = int(regions[i][1])
        xmax = int(regions[i][2] + regions[i][0])
        ymax = int(regions[i][3] + regions[i][1])

        xmax = max(xmin, min(width, xmax))
        ymax = max(ymin, min(height, ymax))

        if ymin < ymax and xmin < xmax:
            dest_image = image[:, ymin:ymax, xmin:xmax]
            dest_image = dest_image.unsqueeze(0)
            dest_image = F.interpolate(dest_image, size=size)
            dest_image = dest_image.squeeze(0)
            start_points.append(image.new([xmin, ymin]))
            scales.append(
                image.new([(xmax-xmin)/float(size[0]),
                           (ymax-ymin)/float(size[1])]))
            images.append(dest_image)
    images = torch.stack(images, dim=0)
    start_points = torch.stack(start_points, dim=0)
    scales = torch.stack(scales, dim=0)
    return images, start_points, scales


def transform_result(data, start_points, scales):
    """
    transform the network output into the right coordinate
    param: network ouput: (classification, bbox, landmark),
    classification, shape = (num, 2),
    bbox, shape = (num, 4),
    landmark, shape = (num, 10)
    param: start_points: tensor, start_point for every result,
    shape = (num, 2)
    param: scale: tensor, scale for every result,
    shape = (num, 10)
    return: result: tensor, data after moving into right coordinate
    """
    classification, bbox, landmark = data
    num = classification.shape[0]

    bbox_scale = scales.repeat(1, 2)
    landmark_scale = scales.repeat(1, 5)
    bbox = bbox * bbox_scale
    landmark = landmark * landmark_scale

    bbox_translation = classification.new_zeros([num, 4])
    bbox_translation[:, :2] = start_points
    landmark_translation = start_points.repeat(1, 5)
    bbox = bbox + bbox_translation
    landmark = landmark + landmark_translation

    return classification, bbox, landmark
