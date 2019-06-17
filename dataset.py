from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, list_dir, list_files

from scipy.io import loadmat
from os.path import join

import os


class CustomDataset(Dataset):

    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 download=True,
                 shape=(224, 224)):

        self.root = root
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.shape = shape

        if download:
            self.download()

        self.annotation_file = loadmat(join(root, 'cars_annos'))
        self._annotations = self.annotation_file['annotations'][0]
        self._class_names = self.annotation_file['class_names'][0]

        self._breeds = [name[0] for name in self._class_names]

        split = self.load_split()

        if self.cropped:
            self._breed_annos = [[(annos, box, idx)] for annos, box, idx in split]
            self._flat_breed_annos = sum(self._breed_annos, [])
            self._flat_breed_images = [(annotation, idx) for annotation, box, idx in self._flat_breed_annos]
        else:
            self._breed_images = [(annotation, idx) for annotation, _, idx in split]

            self._flat_breed_images = self._breed_images

        # transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize(size=(self.shape[0], self.shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, item):
        image_name, class_id = self._flat_breed_images[item]
        image = Image.open(join(self.root, image_name)).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annos[item][1])
        if self.transform:
            image = self.transform(image)

        return image, class_id

    def __len__(self):
        return len(self._flat_breed_images)
        # return len(self.)

    def load_split(self):

        if self.train:
            labels = [item[5][0][0]-1 for item in self._annotations if item[6][0] == 0]
            boxes = [[item[1][0][0], item[2][0][0], item[3][0][0], item[4][0][0]] for item in self._annotations
                     if item[6][0] == 0]
            filenames = [item[0][0] for item in self._annotations if item[6][0] == 0]

        else:
            labels = [item[5][0][0]-1 for item in self._annotations if item[6][0] == 1]
            boxes = [[item[1][0][0], item[2][0][0], item[3][0][0], item[4][0][0]] for item in self._annotations
                     if item[6][0] == 1]
            filenames = [item[0][0] for item in self._annotations if item[6][0] == 1]

        # annos: 0->min x; 1->max x; 2->min y; 3-> max y; 4->class; 5->file name
        # print(list(zip(filenames, labels)))
        return list(zip(filenames, boxes, labels))

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'car_ims')) and os.path.exists(join(self.root, 'cars_annos.mat')):
            if len(os.listdir(join(self.root, 'car_ims'))) == len(loadmat(join(self.root, 'cars_annos.mat'))['annotations'][0]):
                print('Files already downloaded and verified')
                return

        if os.path.exists(self.root):
            os.makedirs(self.root)

        car_url = 'http://imagenet.stanford.edu/internal/car196/'
        imgs = 'car_ims.tgz'
        annos = 'cars_annos.mat'

        for filename in [imgs, annos]:
            url = car_url + filename
            download_url(url, self.root, filename, None)
            print('download file {} already'.format(filename))

        with tarfile.open(join(self.root, imgs), 'r') as tar_file:
            tar_file.extractall(self.root)
        os.remove(join(self.root, imgs))


class TestDataSet(Dataset):

    def __init__(self,
                 root,
                 cropped=False,
                 transform=None,
                 shape=(224, 224)):

        self.root = root
        self.cropped = cropped
        self.transform = transform
        self.shape = shape

        self.annotation_file = loadmat(join(root, 'cars_test_annos_withlabels'))
        self._annotations = self.annotation_file['annotations'][0]

        split = self.load_split()

        if self.cropped:
            self._breed_annos = [[(annos, box)] for annos, box in split]
            self._flat_breed_annos = sum(self._breed_annos, [])
            self._flat_breed_images = [annotation for annotation, box in self._flat_breed_annos]
        else:
            self._breed_images = [annotation for annotation, _, in split]

            self._flat_breed_images = self._breed_images

        # transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize(size=(self.shape[0], self.shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, item):
        image_name = 'cars_test/' + self._flat_breed_images[item]
        image = Image.open(join(self.root, image_name)).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annos[item][1])
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self._flat_breed_images)

    def load_split(self):

        boxes = [[item[0][0][0], item[1][0][0], item[2][0][0], item[3][0][0]] for item in self._annotations]
        filenames = [item[5][0] for item in self._annotations]

        return list(zip(filenames, boxes))

