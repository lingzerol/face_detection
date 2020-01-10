import torch.nn as nn

from . import model_utils, utils

PNET_INPUT_SIZE = (12, 12)
RNET_INPUT_SIZE = (24, 24)
ONET_INPUT_SZIE = (48, 48)
FEATURE_LEN = 10
BBOX_FEATURE_NUM = 4
SCORE_INDEX = 0

# pnet process


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


def filter_and_translation_output(data, start_points_list, scales_list,
                                  bbr_net, filter=True):
    """
    filter pnet output, and translation output into right coordinate
    param: data: tensor, pnet output, list of (
        classfication - (batch, num, 2),
        bbox - (batch, num, 4),
        landmark - (batch, num, 10)
    )
    param: start_points_list: list of tensor, every tensor is (num, 2),
    start_point of every_result
    param: bounding_box_regression network
    return: list of data after filtered
    """
    classification, bbox, landmark = data

    batch_num = len(classification)

    classification_list = []
    bbox_list = []
    landmark_list = []
    indices_list = []
    for i in range(batch_num):
        num = classification[i].shape[0]
        bbox_scale = scales_list[i].repeat(1, 2)
        landmark_scale = scales_list[i].repeat(1, 5)

        bbox_translation = classification[0].new_zeros([num, 4])
        bbox_translation[:, :2] = start_points_list[i]
        landmark_translation = start_points_list[i].repeat(
            1, int(FEATURE_LEN/2))

        sub_bbox = bbox[i]*bbox_scale + bbox_translation
        sub_landmark = landmark[i]*landmark_scale + landmark_translation
        sub_classification = classification[i]
        if filter:
            indices = model_utils.NMS(sub_bbox, classification[i][:, 0])

            sub_classification = sub_classification.index_select(0, indices)
            sub_bbox = sub_bbox.index_select(0, indices)
            sub_landmark = sub_landmark.index_select(0, indices)

            sub_bbox = model_utils.bounding_box_regression(
                bbr_net, sub_landmark, sub_bbox)

            indices_list.append(indices)

        classification_list.append(sub_classification)
        bbox_list.append(sub_bbox)
        landmark_list.append(sub_landmark)
    return classification_list, bbox_list, landmark_list, indices_list

# pnet ouput to rnet input


def trainsform_output_to_net_input(x, p_out, size):
    """
    transform output into net_input
    param: x: tensor original data, shape =
    (batch, channel, height, width)
    param: p_out: list of reshaped p_net output tensor,
    len(p_out) = batch, the shape of every element is
    (num, height, width)
    """
    classification, bbox, landmark = p_out
    images_list = []
    start_points_list = []
    scales_list = []
    for i, image_all_bbox in enumerate(bbox):
        cropped_image, start_points, scales = \
            model_utils.crop_and_resize_image(
                x[i], image_all_bbox, size)
        images_list.append(cropped_image)
        start_points_list.append(start_points)
        scales_list.append(scales)
    return images_list, start_points_list, scales_list


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
        landmark_result = self.landmark(front_out)

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
        landmark_result = self.landmark(front_out)

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
        landmark_result = self.landmark(front_out)

        return classification_result, bbox_result, landmark_result


class Mtcnn(nn.Module):
    def __init__(self, p_net=P_Net(), r_net=R_Net(), o_net=O_Net(),
                 bbr=nn.Linear(in_features=FEATURE_LEN,
                               out_features=BBOX_FEATURE_NUM)):
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
        width = x.shape[-1]
        height = x.shape[-2]

        p_out = self.p_net(x)
        p_out = reshape_pnet_output(p_out)

        start_points = utils.start_points_for_conv(
            (width, height), PNET_INPUT_SIZE, (2, 2))
        start_points = x.new(start_points)
        start_points_list = [start_points]*x.shape[0]
        scales = x.new([1.0, 1.0])
        scales_list = [scales]*x.shape[0]
        p_out_list = filter_and_translation_output(
            p_out, start_points_list, scales_list, self.pnet_bbr, filter)
        return p_out_list

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
        width = x.shape[-1]
        height = x.shape[-2]

        p_out = self.p_net(x)
        p_out = reshape_pnet_output(p_out)

        start_points = utils.start_points_for_conv(
            (width, height), PNET_INPUT_SIZE, (2, 2))
        start_points = x.new(start_points)
        start_points_list = [start_points]*x.shape[0]
        scales = x.new([1.0, 1.0])
        scales_list = [scales]*x.shape[0]
        p_out_list = filter_and_translation_output(
            p_out, start_points_list, scales_list, self.pnet_bbr)
        p_out_list = list(p_out_list[:3])

        r_in, start_points_list, scales_list = trainsform_output_to_net_input(
            x, p_out_list, RNET_INPUT_SIZE)

        r_in, batch_sizes = model_utils.pack_to_conv2d_sequence(r_in)
        r_out = list(self.r_net(r_in))
        r_out[0] = model_utils.unpack_conv2d_sequence(r_out[0], batch_sizes)
        r_out[1] = model_utils.unpack_conv2d_sequence(r_out[1], batch_sizes)
        r_out[2] = model_utils.unpack_conv2d_sequence(r_out[2], batch_sizes)
        r_out_list = filter_and_translation_output(
            r_out, start_points_list, scales_list, self.rnet_bbr)
        r_out_list = list(r_out_list[:3])
        return r_out_list

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
        width = x.shape[-1]
        height = x.shape[-2]

        p_out = self.p_net(x)
        p_out = reshape_pnet_output(p_out)

        start_points = utils.start_points_for_conv(
            (width, height), PNET_INPUT_SIZE, (2, 2))
        start_points = x.new(start_points)
        start_points_list = [start_points]*x.shape[0]
        scales = x.new([1.0, 1.0])
        scales_list = [scales]*x.shape[0]
        p_out_list = filter_and_translation_output(
            p_out, start_points_list, scales_list, self.pnet_bbr)
        p_out_list = list(p_out_list[:3])

        r_in, start_points_list, scales_list = trainsform_output_to_net_input(
            x, p_out_list, RNET_INPUT_SIZE)

        r_in, batch_sizes = model_utils.pack_to_conv2d_sequence(r_in)
        r_out = list(self.r_net(r_in))
        r_out[0] = model_utils.unpack_conv2d_sequence(r_out[0], batch_sizes)
        r_out[1] = model_utils.unpack_conv2d_sequence(r_out[1], batch_sizes)
        r_out[2] = model_utils.unpack_conv2d_sequence(r_out[2], batch_sizes)
        r_out_list = filter_and_translation_output(
            r_out, start_points_list, scales_list, self.rnet_bbr)
        r_out_list = list(r_out_list[:3])

        o_in, start_points_list, scales_list = trainsform_output_to_net_input(
            x, r_out_list, ONET_INPUT_SZIE)
        o_in, batch_sizes = model_utils.pack_to_conv2d_sequence(o_in)
        o_out = list(self.o_net(o_in))
        o_out[0] = model_utils.unpack_conv2d_sequence(o_out[0], batch_sizes)
        o_out[1] = model_utils.unpack_conv2d_sequence(o_out[1], batch_sizes)
        o_out[2] = model_utils.unpack_conv2d_sequence(o_out[2], batch_sizes)
        o_out_list = filter_and_translation_output(
            o_out, start_points_list, scales_list, self.onet_bbr, filter)
        o_out_list = list(o_out_list[:3])
        return o_out_list
