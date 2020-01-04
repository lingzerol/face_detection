import numpy as np
import cv2
import torch

from facelib import data, mtcnn
import train

import argparse
import os
import logging

from tensorboardX import SummaryWriter

log = logging.getLogger("main")

SAVE_DIR = "./saves"


def load_data(dirname):
    images_path = []
    bboxes_list = {}
    landmarks_list = {}
    for name in os.listdir(dirname):
        path = os.path.join(dirname, name)
        if os.path.isdir(path):
            images_path.extend(data.get_image_path_list(path))
        elif name.endswith("txt"):
            br, lr = data.load_image_bbox_and_landmark(name, dirname)
            bboxes_list.update(br)
            landmarks_list.update(lr)
    return images_path, bboxes_list, landmarks_list


def experiment(net):
    image = cv2.imread("./data/train/net_7876/_0_0_0.jpg")
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor([image])
    height = image.shape[-2]
    width = image.shape[-1]
    classifications, bboxes, landmarks = net.forward_to_pnet(image, False)
    bboxes = mtcnn.transform_bounding_box(bboxes, torch.FloatTensor(
        [[0, 0]]), width, height, torch.FloatTensor([[1, 1]]))
    landmarks = mtcnn.transform_lanmarks(landmarks, torch.FloatTensor(
        [[0, 0]]), width, height, torch.FloatTensor([[1, 1]]))
    bboxes = bboxes.detach().numpy()[0]
    landmarks = landmarks.detach().numpy()[0]


def main(args):
    images_path, bboxes_data, landmarks_data = load_data(args.data)
    train_path, test_path = data.split_image_into_train_and_test(images_path)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    net = mtcnn.Mtcnn().to(device)
    args.load = "D:/Downloads/epoch_1000_loss_0.983.dat"
    if args.load:
        net.load_state_dict(torch.load(
            args.load, map_location=torch.device('cpu')))
    args.experiment = True
    if args.experiment:
        experiment(net)

    writer = SummaryWriter(comment="-"+args.name)

    p_net_optimizer = torch.optim.Adam(params=net.p_net.parameters(), lr=1e-4)
    p_net_and_filter_parameter = list(
        net.p_net.parameters()) + list(net.pnet_bbr.parameters())
    p_net_and_filter_optimizer = torch.optim.Adam(
        params=p_net_and_filter_parameter, lr=1e-4)

    r_net_optimizer = torch.optim.Adam(params=net.r_net.parameters(), lr=1e-4)
    r_net_and_filter_parameter = list(
        net.r_net.parameters()) + list(net.rnet_bbr.parameters())
    r_net_and_filter_optimizer = torch.optim.Adam(
        params=r_net_and_filter_parameter, lr=1e-4)

    o_net_optimizer = torch.optim.Adam(params=net.o_net.parameters(), lr=1e-4)
    o_net_and_filter_parameter = list(
        net.o_net.parameters()) + list(net.onet_bbr.parameters())
    o_net_and_filter_optimizer = torch.optim.Adam(
        params=o_net_and_filter_parameter, lr=1e-4)

    epoch = 0
    max_classification_recall = 0
    max_bbox_recall = 0
    max_landmark_recall = 0
    loss_list = []
    saves_path = os.path.join(SAVE_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    train_types = [train.TRAIN_PNET, train.train_Rnet, train.train_Onet]
    optimizers_list = [[p_net_optimizer, p_net_and_filter_optimizer], [
        r_net_optimizer, r_net_and_filter_optimizer],
        [o_net_optimizer, o_net_and_filter_optimizer]]
    iter_times = [args.pt, args.rt, args.ot]

    for train_type, optimizers, iter_time in \
            zip(train_types, optimizers_list, iter_times):
        for images_list, bboxes_list, landmarks_list in \
                data.iterate_image_batches(train_path,
                                           bboxes_data, landmarks_data, 1):
            images_list, bboxes_list, landmarks_list = data.generate_pyramid(
                images_list, bboxes_list, landmarks_list)
            images_list, labels_list, bboxes_list, landmarks_list = \
                data.generate_images(
                    images_list, bboxes_list, landmarks_list,
                    mtcnn.PNET_INPUT_SIZE[0], mtcnn.PNET_INPUT_SIZE[1])
            images_list = data.processing_images(images_list)

            images_list = [torch.FloatTensor([image]).to(device)
                           for image in images_list]
            labels_list = [torch.FloatTensor(labels).to(device)
                           for labels in labels_list]
            bboxes_list = [torch.FloatTensor(bboxes).to(device)
                           for bboxes in bboxes_list]
            landmarks_list = [torch.FloatTensor(landmarks).to(
                device) for landmarks in landmarks_list]

            log.info("Image num: %d" % (len(images_list)))

            filter = np.random.random() > 0.5
            for _ in range(iter_time):
                loss = train.train(net, optimizers[int(filter)], images_list,
                                   labels_list, bboxes_list, landmarks_list,
                                   filter=filter, train_type=train_type)
                loss_list.append(loss)
                writer.add_scalar("train_loss", loss, epoch)
                log.info("Epoch: %d, mean loss: %.3f" % (epoch, loss))
                if epoch % 1000 == 0:
                    torch.save(net.state_dict(),
                               os.path.join(saves_path,
                                            "epoch_%d_loss_%.3f.dat" % (
                                                epoch,
                                                loss)))

                if epoch % 10000 == 0:
                    classification_recall, bbox_recall, landmark_recall = \
                        train.get_metrics(
                            net, test_path, bboxes_data, landmarks_data,
                            device)
                    if max_classification_recall < classification_recall or \
                            max_bbox_recall < bbox_recall or \
                            max_landmark_recall < landmark_recall:
                        max_classification_recall = classification_recall
                        max_bbox_recall = bbox_recall
                        max_landmark_recall = landmark_recall
                        torch.save(net.state_dict(),
                                   os.path.join(saves_path,
                                                "epoch_%d_cr_%.3f_br_%.3f"
                                                "_lr_%.3f.dat" % (
                                                    epoch,
                                                    max_classification_recall,
                                                    max_bbox_recall,
                                                    max_landmark_recall)))
                        log.info("Epoch: %d, classification recall: %.3f, "
                                 "bboxes recall: %.3f,"
                                 "landmark recall: %.3f"
                                 % (epoch,
                                     classification_recall,
                                     bbox_recall, landmark_recall))
                        writer.add_scalar(
                            "classification_recall", max_classification_recall,
                            epoch)
                        writer.add_scalar(
                            "landmark_recall", max_landmark_recall, epoch)
                        writer.add_scalar(
                            "bbox_recall", max_bbox_recall, epoch)
                epoch += 1


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s",
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true",
                        default=False, help="using cuda")
    parser.add_argument("-d", "--data", required=True,
                        default="./data/train", help="the"
                        "path of training data")
    parser.add_argument("--pt", required=False, default=1000,
                        type=int, help="training pnet times")
    parser.add_argument("--rt", required=False, default=1000,
                        type=int, help="training rnet times")
    parser.add_argument("--ot", required=False, default=1000,
                        type=int, help="training onet times")

    parser.add_argument("-e", "--experiment", required=False,
                        default=False, action="store_true",
                        help="experiment the model")
    parser.add_argument("-l", "--load", required=False,
                        help="load the model")

    parser.add_argument("-n", "--name", required=True, help="Name of the run")

    args = parser.parse_args()
    main(args)
