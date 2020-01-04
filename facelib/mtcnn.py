import torch
import torch.nn as nn
import torch.nn.functional as F
PNET_INPUT_SIZE = (12, 12)
RNET_INPUT_SIZE = (24, 24)
ONET_INPUT_SZIE = (48, 48)
FEATURE_LEN = 10
RECTANGLE_POINT_NUM = 4
SCORE_INDEX = 0


def pad_image(images, divisors):
    h = images.shape[-2]
    w = images.shape[-1]
    if isinstance(divisors, int):
        divisors = [divisors, divisors]
    hp = 0
    wp = 0
    if h % divisors[0] != 0:
        hp = divisors[0] - h % divisors[0]
    if w % divisors[1] != 0:
        wp = divisors[1] - w % divisors[1]
    if wp != 0 or hp != 0:
        images = F.pad(images, pad=(0, wp, 0, hp))
    return images


def split_image_into_batch(images, size):
    """
    split an image into a batch small image, and its size is (h, w)
    param: image: tensor, store the value of image
    param: size: size of image should be splited
    return: tensor: batch of small image
    """
    images = pad_image(images, size)
    channel = images.shape[-3]
    start_idx = len(images.shape)-3
    images = images.unfold(start_idx, channel, channel).unfold(
        start_idx+1, size[0], size[0]).unfold(start_idx+2, size[1], size[1])
    images = images.squeeze(start_idx)
    return images


def conv_image_into_batch(image, size):
    """
    make a conv to the image
    param: image: the image want to conv, shape = (channel, height, width)
    param: size: conv size
    return: tensor: shape = (num, channel, size[0], size[1])
    """

    result = []
    channel, height, width = image.shape
    for i in range(height - size[0] + 1):
        for j in range(width - size[1] + 1):
            result.append(image[:, i:(i+size[0]), j:(j+size[1])])
    result = torch.stack(result, dim=0)
    return result


def image_into_batch(image, size, conv=False):
    if isinstance(size, int):
        size = [size, size]

    if conv:
        images = conv_image_into_batch(image, size)
    else:
        images = split_image_into_batch(image, size)
        images = images.reshape(-1,
                                images.shape[-3], images.shape[-2],
                                images.shape[-1])
    return images


def crop_and_split_image(image, region, size):
    """
    crop the image into region, and split it
    param: image: tensor, shape = (channel, heigt, width)
    param: region: tensor, (x, y, width, height)
    param: divisior: list, the width and the height of the region should
    be divided
    return: tensor: shape = (batch, channel, size[0], size[1])
    """
    if isinstance(size, int):
        size = [size, size]

    xmin = int(region[0])
    ymin = int(region[1])
    xmax = int(region[2]+region[0])
    ymax = int(region[3]+region[1])
    xp = 0
    yp = 0

    if (xmax-xmin) % size[0] != 0:
        xp = size[0] - (xmax-xmin) % size[0]
    if (ymax-ymin) % size[1] != 0:
        yp = size[0] - (ymax-ymin) % size[0]

    xmax = max(xmin, min(xmax+xp, image.shape[2]))
    ymax = max(ymin, min(ymax+yp, image.shape[1]))

    if ymin < ymax and xmin < xmax:
        image = image[:, ymin:ymax, xmin:xmax]
        images = image_into_batch(image, size)
        return images
    else:
        return None


def crop_and_resize_image(image, region, size):
    """
    crop the image into region, and split it
    param: image: tensor, shape = (channel, heigt, width)
    param: region: tensor, (x, y, width, height)
    param: divisior: list, the width and the height of the region should
    be divided
    return: tensor: shape = (channel, size[0], size[1])
    """
    if isinstance(size, int):
        size = [size, size]

    xmin = int(region[0])
    ymin = int(region[1])
    xmax = int(region[2]+region[0])
    ymax = int(region[3]+region[1])

    xmax = max(xmin, min(xmax, image.shape[2]))
    ymax = max(ymin, min(ymax, image.shape[1]))

    if ymin < ymax and xmin < xmax:
        image = image[:, ymin:ymax, xmin:xmax]
        scale_width = image.shape[-1] / size[1]
        scale_height = image.shape[-2] / size[0]
        image = image.unsqueeze(0)
        image = F.interpolate(image, size)
        image = image.squeeze(0)
        return image, image.new([xmin, ymin]), \
            image.new((scale_width, scale_height))
    else:
        return None, (0, 0), (0, 0)


def split_pnet_output(data):
    """
    split the pnet output (after reshaping) into a list,
    and every element of the list is (classification_result, bbox_result,
    landmark_result)
    param: data: pnet output after reshaping
    return: list:  every element is (classification_result, bbox_result,
    landmark_result)
    """

    result = []
    batch_num = data[0].shape[0]
    output_num = data[0].shape[1]
    for i in range(batch_num):
        r = []
        for j in range(output_num):
            r.append([data[0][i][j], data[1][i][j], data[2][i][j]])
        result.append(r)
    return result


def reshape_pnet_output(data):
    """
    reshape p_net output (batch, num, h, w) into (batch, h*w, num)
    param: data: tuple: p_net ouput
    return: list: p_net output after reshaping
    """
    result = []
    for d in data:
        d = d.transpose(1, -1)
        d = d.reshape(d.shape[0], -1, d.shape[-1])
        result.append(d)
    return result


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


def iterator_pnet_output(bbr, images_list, p_out):
    width = images_list.shape[-1]
    height = images_list.shape[-2]
    p_out = reshape_pnet_output(p_out)
    classification_result, bbox_result, landmark_result = p_out
    batch_num = p_out[0].shape[0]

    for i in range(batch_num):
        image = images_list[i]
        image = image.unsqueeze(0)
        image = F.interpolate(image, RNET_INPUT_SIZE)
        image = image.squeeze(0)

        images = [image]
        scales = [image.new([1.0, 1.0])]
        start_points = [image.new([0, 0])]
        indices = NMS(bbox_result[i], classification_result[i, :, SCORE_INDEX])
        regions = bbox_result[i].index_select(0, indices)
        features = landmark_result[i].index_select(0, indices)
        regions = bounding_box_regression(bbr, features, regions)
        regions = transform_bounding_box(
            regions, regions.new([[0, 0]]),
            width, height, regions.new([[1, 1]]))
        for j in range(regions.shape[0]):
            image, start_point, scale = crop_and_resize_image(
                images_list[i], regions[j], RNET_INPUT_SIZE)
            if image is not None:
                images.append(image)
                scales.append(scale)
                start_points.append(start_point)
        images = torch.stack(images, dim=0)
        scales = torch.stack(scales, dim=0)
        start_points = torch.stack(start_points, dim=0)
        yield images_list[i], images, start_points, \
            scales


def get_rnet_output_region(image, regions):
    # image_batch = image_into_batch(image, ONET_INPUT_SZIE)
    # image_batch = image_batch.reshape(-1,
    #                                   image_batch.shape[-3],
    #                                   image_batch.shape[-2],
    #                                   image_batch.shape[-1])

    image_batch = image
    image_batch = image_batch.unsqueeze(0)
    image_batch = F.interpolate(image_batch, ONET_INPUT_SZIE)
    image_batch = image_batch.squeeze(0)

    result = [image_batch]
    scales = [image.new([1, 1])]
    start_points = [image.new([0, 0])]
    for i in range(regions.shape[0]):
        image_batch, start_point, scale = \
            crop_and_resize_image(image, regions[i],
                                  ONET_INPUT_SZIE)
        if image_batch is not None:
            result.append(image_batch)
            scales.append(scale)
            start_points.append(start_point)
    result = torch.stack(result, dim=0)
    scales = torch.stack(scales, dim=0)
    start_points = torch.stack(start_points, dim=0)
    return result, start_points, scales


def rnet_output_to_onet_input(bbr, r_out, image, start_points, scales):
    """
    transform rnet output to onet input, only process one image per time
    param: bbr: the bounding-box-regression layer
    param: r_out: tuple: rnet output
    param: image: the image process
    return: tensor: onet input
    """
    width = image.shape[-1]
    height = image.shape[-2]

    classification_result, bbox_result, landmark_result = r_out

    indices = NMS(bbox_result, landmark_result[:, SCORE_INDEX])
    features = landmark_result.index_select(0, indices)
    scales = scales.index_select(0, indices)
    start_points = start_points.index_select(0, indices)
    regions = bbox_result.index_select(0, indices)
    regions = bounding_box_regression(bbr, features, regions)
    regions = transform_bounding_box(
        regions, start_points, width, height, scales)
    result, start_points, scales = get_rnet_output_region(image, regions)
    return result, start_points, scales


def onet_output_process(bbr, o_out, start_points, width,
                        height, scales, filter):
    """
    process onet output
    param: bbr: the bounding-box-regression layer
    param: r_out: tuple: onet output
    return: tensor: result after processing
    """
    classification_result, bbox_result, landmark_result = o_out
    if filter:
        indices = NMS(bbox_result, landmark_result[:, SCORE_INDEX])
        landmark_result = landmark_result.index_select(0, indices)
        classification_result = classification_result.index_select(0, indices)
        bbox_result = bbox_result.index_select(0, indices)
        bbox_result = bounding_box_regression(bbr, landmark_result,
                                              bbox_result)
        start_points = start_points.index_select(0, indices)
        scales = scales.index_select(0, indices)
    bbox_result = transform_bounding_box(
        bbox_result, start_points, width, height, scales)
    bbox_result = scale_bounding_box(bbox_result, width, height)
    landmark_result = transform_lanmarks(
        landmark_result, start_points, width, height, scales)
    landmark_result = scale_landmarks(landmark_result,
                                      width, height)
    return classification_result, bbox_result, landmark_result


def transform_bounding_box(bboxes, start_points,
                           width, height, scales):
    result = bboxes.new_zeros(bboxes.size())
    if len(result.shape) == 2:
        result[:, 0] = (bboxes[:, 0]*width)/scales[:, 0]+start_points[:, 0]
        result[:, 1] = (bboxes[:, 1]*height)/scales[:, 1]+start_points[:, 1]

        result[:, 2] = bboxes[:, 2]*width/scales[:, 0]
        result[:, 3] = bboxes[:, 3]*height/scales[:, 1]
    else:
        result[:, :, 0] = (bboxes[:, :, 0]*width) / \
            scales[:, 0]+start_points[:, 0]
        result[:, :, 1] = (bboxes[:, :, 1]*height) / \
            scales[:, 1]+start_points[:, 1]

        result[:, :, 2] = bboxes[:, :, 2]*width/scales[:, 0]
        result[:, :, 3] = bboxes[:, :, 3]*height/scales[:, 1]
    return result


def transform_lanmarks(landmarks, start_points,
                       width, height, scales):

    result = landmarks.new_zeros(landmarks.size())
    for i in range(FEATURE_LEN):
        if i % 2:
            if len(result.shape) == 2:
                result[:, i] = (landmarks[:, i]*height) / \
                    scales[:, 1]+start_points[:, 1]
            else:
                result[:, :, i] = (landmarks[:, :, i] *
                                   height)/scales[:, 1]+start_points[:, 1]
        else:
            if len(result.shape) == 2:
                result[:, i] = (landmarks[:, i]*width) / \
                    scales[:, 0]+start_points[:, 0]
            else:
                result[:, :, i] = (landmarks[:, :, i]*width) / \
                    scales[:, 0]+start_points[:, 0]
    return result


def scale_bounding_box(bboxes, width, height):
    result = bboxes.new_zeros(bboxes.size())
    if len(result.shape) == 1:
        result[0] = bboxes[0]/width
        result[1] = bboxes[1]/height

        result[2] = bboxes[2]/width
        result[3] = bboxes[3]/height
    elif len(result.shape) == 2:
        result[:, 0] = bboxes[:, 0]/width
        result[:, 1] = bboxes[:, 1]/height

        result[:, 2] = bboxes[:, 2]/width
        result[:, 3] = bboxes[:, 3]/height
    else:
        result[:, :, 0] = bboxes[:, :, 0]/width
        result[:, :, 1] = bboxes[:, :, 1]/height

        result[:, :, 2] = bboxes[:, :, 2]/width
        result[:, :, 3] = bboxes[:, :, 3]/height
    return result


def scale_landmarks(landmarks, width, height):
    result = landmarks.new_zeros(landmarks.size())
    for i in range(FEATURE_LEN):
        if i % 2:
            if len(result.shape) == 1:
                result[i] = landmarks[i]/height
            elif len(result.shape) == 2:
                result[:, i] = landmarks[:, i]/height
            else:
                result[:, :, i] = (landmarks[:, :, i] /
                                   height)
        else:
            if len(result.shape) == 1:
                result[i] = landmarks[i]/width
            elif len(result.shape) == 2:
                result[:, i] = landmarks[:, i]/width
            else:
                result[:, :, i] = (landmarks[:, :, i]/width)
    return result


class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=(1, 1))
        self.prelu1 = nn.PReLU(num_parameters=10)
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=16, kernel_size=(3, 3), stride=(1, 1))
        self.prelu2 = nn.PReLU(num_parameters=16)

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.prelu3 = nn.PReLU(num_parameters=32)

        self.front = nn.Sequential(
            self.conv1, self.prelu1, self.pooling1, self.conv2, self.prelu2,
            self.conv3, self.prelu3)

        self.classification = nn.Conv2d(
            in_channels=32, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.bbox = nn.Conv2d(
            in_channels=32, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.landmark = nn.Conv2d(
            in_channels=32, out_channels=10, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height and width are not limited.
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result = (batch, 2, h, w),
        bbox_result = (batch, 4, h, w), landmark_result = (batch, 10, h, w)
        """
        front_out = self.front(x)

        classification_result = self.softmax(self.classification(front_out))
        bbox_result = self.bbox(front_out)
        bbox_result = self.sigmoid(bbox_result)
        landmark_result = self.landmark(front_out)
        landmark_result = self.sigmoid(landmark_result)

        return classification_result, bbox_result, landmark_result


class R_Net(nn.Module):
    def __init__(self):
        super(R_Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=28, kernel_size=(3, 3), stride=(1, 1))
        self.prelu1 = nn.PReLU(num_parameters=28)
        self.pooling1 = nn.MaxPool2d(kernel_size=(
            2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=28, out_channels=48, kernel_size=(3, 3), stride=(1, 1))
        self.prelu2 = nn.PReLU(num_parameters=48)
        self.pooling2 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=64, kernel_size=(2, 2), stride=(1, 1))
        self.prelu3 = nn.PReLU(num_parameters=64)

        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)

        self.fcl = nn.Linear(in_features=576, out_features=128)
        self.prelu4 = nn.PReLU(num_parameters=128)

        self.front = nn.Sequential(self.conv1, self.prelu1,
                                   self.pooling1, self.conv2,
                                   self.prelu2, self.pooling2,
                                   self.conv3, self.prelu3,
                                   self.flatten, self.fcl,
                                   self.prelu4)

        self.classification = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=-1)
        self.bbox = nn.Linear(in_features=128, out_features=4)
        self.sigmoid = nn.Sigmoid()
        self.landmark = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height = width = 24
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result = (batch, 2),
        bbox_result = (batch, 4), landmark_result = (batch, 10)
        """
        front_out = self.front(x)

        classification_result = self.softmax(self.classification(front_out))
        bbox_result = self.bbox(front_out)
        bbox_result = self.sigmoid(bbox_result)
        landmark_result = self.landmark(front_out)
        landmark_result = self.sigmoid(landmark_result)

        return classification_result, bbox_result, landmark_result


class O_Net(nn.Module):
    def __init__(self):
        super(O_Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.prelu1 = nn.PReLU(num_parameters=32)
        self.pooling1 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1))

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.prelu2 = nn.PReLU(num_parameters=64)
        self.pooling2 = nn.MaxPool2d(kernel_size=(
            3, 3), stride=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.prelu3 = nn.PReLU(num_parameters=64)
        self.pooling3 = nn.MaxPool2d(kernel_size=(
            2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 2),
            stride=(1, 1))
        self.prelu4 = nn.PReLU(num_parameters=128)

        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)

        self.fcl = nn.Linear(in_features=1152, out_features=256)
        self.prelu5 = nn.PReLU(num_parameters=256)

        self.front = nn.Sequential(
            self.conv1, self.prelu1,
            self.pooling1, self.conv2,
            self.prelu2, self.pooling2,
            self.conv3, self.prelu3,
            self.pooling3, self.conv4,
            self.prelu4, self.flatten,
            self.fcl, self.prelu5)

        self.classification = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=-1)
        self.bbox = nn.Linear(in_features=256, out_features=4)
        self.sigmoid = nn.Sigmoid()
        self.landmark = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height = width = 48
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result = (batch, 2),
        bbox_result = (batch, 4), landmark_result = (batch, 10)
        """
        front_out = self.front(x)

        classification_result = self.softmax(self.classification(front_out))
        bbox_result = self.bbox(front_out)
        bbox_result = self.sigmoid(bbox_result)
        landmark_result = self.landmark(front_out)
        landmark_result = self.sigmoid(landmark_result)

        return classification_result, bbox_result, landmark_result


class Mtcnn(nn.Module):
    def __init__(self, p_net=P_Net(), r_net=R_Net(), o_net=O_Net(),
                 bbr=nn.Linear(in_features=FEATURE_LEN,
                               out_features=RECTANGLE_POINT_NUM)):
        super(Mtcnn, self).__init__()

        self.p_net = p_net
        self.r_net = r_net
        self.o_net = o_net
        if not isinstance(bbr, list):
            self.pnet_bbr = bbr
            self.rnet_bbr = bbr
            self.onet_bbr = bbr
        else:
            self.pnet_bbr = bbr[0]
            self.rnet_bbr = bbr[1]
            self.onet_bbr = bbr[2]

    def forward(self, x):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height and width are not limited.
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result: list: the classfication of
        every image
        bbox_result = the bbox_result of every image, landmark_result = the
        landmark_result of every image
        """
        return self.forward_to_onet(x)

    def forward_to_pnet(self, x, filter=True):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height and width are not limited.
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result: list: the classfication of
        every image
        bbox_result = the bbox_result of every image, landmark_result = the
        landmark_result of every image
        """
        p_out = self.p_net(x)
        p_out = reshape_pnet_output(p_out)
        classifications, bboxes, landmarks = p_out
        if not filter:
            return classifications, bboxes, landmarks

        num = classifications.shape[0]
        classifications_result = []
        bboxes_result = []
        landmarks_result = []
        for i in range(num):
            indices = NMS(bboxes[i], landmarks[i][:, SCORE_INDEX])

            lr = landmarks[i].index_select(0, indices)
            br = bboxes[i].index_select(0, indices)
            cr = classifications[i].index_select(0, indices)

            br = bounding_box_regression(self.pnet_bbr, lr,
                                         br)
            classifications_result.append(cr)
            bboxes_result.append(br)
            landmarks_result.append(lr)
        return classifications_result, bboxes_result, landmarks_result

    def forward_to_rnet(self, x, filter=True):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height and width are not limited.
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result: list: the classfication of
        every image
        bbox_result = the bbox_result of every image, landmark_result = the
        landmark_result of every image
        """
        p_out = self.p_net(x)
        classifications_result = []
        bboxes_result = []
        landmarks_result = []
        for origin_image, images, start_points, scales in \
                iterator_pnet_output(self.pnet_bbr, x, p_out):
            cr, br, lr = self.r_net(images)
            if filter:
                indices = NMS(br, lr[:, SCORE_INDEX])

                lr = lr.index_select(0, indices)
                br = br.index_select(0, indices)
                cr = cr.index_select(0, indices)

                br = bounding_box_regression(self.pnet_bbr, lr,
                                             br)
            width = origin_image.shape[-1]
            height = origin_image.shape[-2]
            br = transform_bounding_box(
                br, start_points, width, height, scales)
            br = scale_bounding_box(br, width, height)
            lr = transform_lanmarks(lr, start_points, width, height, scales)
            lr = scale_landmarks(lr, width, height)
            classifications_result.append(cr)
            bboxes_result.append(br)
            landmarks_result.append(lr)

        return classifications_result, bboxes_result, landmarks_result

    def forward_to_onet(self, x, filter=True):
        """
        param: x: tensor: shape = (batch, channel, height, width),
        height and width are not limited.
        return: (tensor, tensor, tensor): (classification_result, bbox_result,
        landmark_result), classification_result: list: the classfication of
        every image
        bbox_result = the bbox_result of every image, landmark_result = the
        landmark_result of every image
        """
        p_out = self.p_net(x)
        classification_result = []
        bbox_result = []
        landmark_result = []
        for origin_image, images, start_points, scales in \
                iterator_pnet_output(self.pnet_bbr, x, p_out):
            r_out = self.r_net(images)
            o_in, start_points, scales = rnet_output_to_onet_input(
                self.rnet_bbr, r_out, origin_image, start_points, scales)
            o_out = self.o_net(o_in)
            width = origin_image.shape[-1]
            height = origin_image.shape[-2]
            cr, br, lr = onet_output_process(
                self.onet_bbr, o_out, start_points, width, height,
                scales, filter)

            classification_result.append(cr)
            bbox_result.append(br)
            landmark_result.append(lr)

        return classification_result, bbox_result, landmark_result


def NMS(bboxes, scores, overlap=0.5):
    """
    nms algorithm to filter bboxes
    param: bboxes: list: rectangle of target, element: (xmin, ymin, xmax, ymax)
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
