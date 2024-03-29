import numpy as np
import pandas as pd
import tensorflow as tf


class MRNetDataset():
    def __init__(self, root_dir, task, plane, train=True, transform=None, weights=None):
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = [1, neg / pos]
        else:
            self.weights = weights

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        # label = torch.FloatTensor([label])
        label = tf.constant(label, tf.float32)
        # print "label torch is ", label.numpy()
        if self.transform:
            array = self.transform(array)

        else:
            array = np.stack((array,)*3, axis=1)
            # array1 = torch.FloatTensor(array)
            array = tf.constant(array, tf.float32)

        if label.numpy() == 1:
            weight = np.array([self.weights[1]])
            # weight = torch.FloatTensor(weight)
            weight = tf.constant(weight, tf.float32)

        else:
            weight = np.array([self.weights[0]])
            # weight = torch.FloatTensor(weight)
            weight = tf.constant(weight, tf.float32)

        return array, label, weight

