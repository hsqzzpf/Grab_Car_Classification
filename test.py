import time
import torch
from dataset import CustomDataset, TestDataSet
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from utils import accuracy
import os
from torch.utils.data import DataLoader
from PIL import Image
import tarfile
from scipy.io import loadmat
from torchvision.datasets.utils import download_url, list_dir, list_files
from os.path import join


def download_test_folder():

    if os.path.exists('Dataset/cars_test') and os.path.exists('Dataset/cars_test_annos_withlabels.mat'):
        if len(os.listdir('Dataset/cars_test')) == len(
                loadmat('Dataset/cars_test_annos_withlabels')['annotations'][0]):
            print('Files already downloaded and verified')
            return

    car_url = 'http://imagenet.stanford.edu/internal/car196/'
    test_imgs = 'cars_test.tgz'
    test_annos = 'cars_test_annos_withlabels.mat'

    for filename in [test_imgs, test_annos]:
        url = car_url + filename
        download_url(url, 'Dataset', filename, None)
        print('download file {} already'.format(filename))

    with tarfile.open('Dataset/cars_test.tgz', 'r') as tar_file:
        tar_file.extractall('Dataset')
    os.remove('Dataset/cars_test.tgz')


def classifier(img_addr, net, boundary=None):
    """
    :param img_addr: a string that contains the address to the image
    :param boundary:  optional, used for cropping the function
                should be a tuple of 4 int elements: (x_min, y_min, x_max, y_max)
     :param net: well-trained model
    :returns: a tuple (predict class, confidence score)
              predict class: from 1 to 196
    """

    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

    input_img = Image.open(img_addr).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if boundary:
        input_img = input_img.crop(boundary)

    input_img = transform(input_img)
    input_img = torch.unsqueeze(input_img, 0)
    X = input_img
    X = X.to(torch.device("cuda"))
    #
    # print(type(X))
    # print(X.shape)

    ##################################
    # Raw Image
    ##################################
    y_pred_raw, feature_matrix, attention_map = net(X)

    ##################################
    # Object Localization and Refinement
    ##################################
    # crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
    crop_mask = F.interpolate(attention_map, size=(X.size(2), X.size(3))) > theta_c
    crop_images = []
    for batch_index in range(crop_mask.size(0)):
        nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
        height_min = nonzero_indices[:, 0].min()
        height_max = nonzero_indices[:, 0].max()
        width_min = nonzero_indices[:, 1].min()
        width_max = nonzero_indices[:, 1].max()
        crop_images.append(F.upsample_bilinear(
            X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
    crop_images = torch.cat(crop_images, dim=0)

    y_pred_crop, _, _ = net(crop_images)

    # final prediction
    y_pred = (y_pred_raw + y_pred_crop) / 2

    _, pred_idx = y_pred[0].topk(1, 0, True, True)

    s = torch.nn.Softmax(dim=0)
    confidence_score = max(s(y_pred[0])).item()

    y_pred = pred_idx.item() + 1

    return y_pred, confidence_score


def test(data_loader, txtfile):

    net = torch.load('trained_models/resnet152_94.pkl')

    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

    net.eval()

    with open(txtfile, 'w') as file:
        with torch.no_grad():
            for i, X in enumerate(data_loader):

                # obtain data
                X = X.to(torch.device("cuda"))
                print(type(X))
                print(X.shape)

                ##################################
                # Raw Image
                ##################################
                y_pred_raw, feature_matrix, attention_map = net(X)

                ##################################
                # Object Localization and Refinement
                ##################################
                # crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
                crop_mask = F.interpolate(attention_map, size=(X.size(2), X.size(3))) > theta_c
                crop_images = []
                for batch_index in range(crop_mask.size(0)):
                    nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                    height_min = nonzero_indices[:, 0].min()
                    height_max = nonzero_indices[:, 0].max()
                    width_min = nonzero_indices[:, 1].min()
                    width_max = nonzero_indices[:, 1].max()
                    crop_images.append(F.upsample_bilinear(
                        X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
                crop_images = torch.cat(crop_images, dim=0)

                y_pred_crop, _, _ = net(crop_images)

                # final prediction
                y_pred = (y_pred_raw + y_pred_crop) / 2

                for y in y_pred:
                    _, pred_idx = y.topk(1, 0, True, True)
                    file.write(str(pred_idx.item()+1) + ' ')

                    s = torch.nn.Softmax(dim=0)
                    confidence_score = max(s(y)).item()
                    file.write(str(confidence_score) + '\n')


def test_sample(annos_file, predict_text):
    annos = loadmat(annos_file)['annotations'][0]

    net = torch.load('trained_models/resnet152_94.pkl')

    with open(predict_text, 'w') as file:
        for item in annos:
            boxes = [item[0][0][0], item[1][0][0], item[2][0][0], item[3][0][0]]
            filenames = join('Dataset', 'cars_test', item[5][0])

            y_pred, _ = classifier(filenames, net, boundary=boxes)
            file.write(str(y_pred) + '\n')


if __name__ == '__main__':

    testset = TestDataSet('./Dataset', cropped=True)

    test_loader = DataLoader(testset, 64, shuffle=False)

    download_test_folder()
    # test(test_loader, 'predict.txt')
    test_sample('Dataset/cars_test_annos_withlabels.mat', 'pred.txt')