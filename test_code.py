from __future__ import print_function, division
from sklearn.model_selection import train_test_split
import cv2
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os, csv
import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z =Input(shape=self.img_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(32, activation="relu", input_shape=(256, 256, 1)))
        # model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        # noise = Input(shape=(self.latent_dim,))
        noise = Input(shape=self.img_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, save_interval=50):

        # Load the dataset


        # Rescale -1 to 1


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':

    train_acl_lbl = []
    train_abnormal_lbl = []
    train_meniscus_lbl = []

    valid_acl_lbl = []
    valid_abnormal_lbl = []
    valid_meniscus_lbl = []

    i = 0
    csvs = ["train-acl.csv", "train-abnormal.csv", "train-meniscus.csv", "valid-acl.csv", "valid-abnormal.csv",
            "valid-meniscus.csv"]
    for c in csvs:
        with open('C:/Fall19/AI/Project/MRNet-v1.0/' + c, 'r') as csvfile:
            read = csv.reader(csvfile, delimiter=' ', quotechar='|')
            if i == 0:
                for row in read:
                    train_acl_lbl.append(', '.join(row)[5])
                    # print "reaching here"
                i += 1
            elif i == 1:
                for row in read:
                    train_abnormal_lbl.append(', '.join(row)[5])
                i += 1
            elif i == 2:
                for row in read:
                    train_meniscus_lbl.append(', '.join(row)[5])
                i += 1
            elif i == 3:
                for row in read:
                    valid_acl_lbl.append(', '.join(row)[5])
                i += 1
            elif i == 4:
                for row in read:
                    valid_abnormal_lbl.append(', '.join(row)[5])
                i += 1
            elif i == 5:
                for row in read:
                    valid_meniscus_lbl.append(', '.join(row)[5])
                i += 1


    def getTheLabels(a, b, c):
        # print "length of each is ", len(a), len(b), len(c)
        labels = [0] * len(a)
        for i in range(len(labels)):
            # print "val is ", str(a[i]) + str(c[i])
            labels[i] = int(str(a[i]) + str(b[i]) + str(c[i]), 2)
        return np.array(labels)


    # print "train acl lbl ", train_acl_lbl
    # Encoding all labels to be a number from (0-7) (Abnormal,ACL,Meniscus)
    # GAN doesn't look like using labels
    train_label = getTheLabels(train_abnormal_lbl, train_acl_lbl, train_meniscus_lbl)
    valid_label = getTheLabels(valid_abnormal_lbl, valid_acl_lbl, valid_meniscus_lbl)
    del (train_abnormal_lbl)
    del (train_acl_lbl)
    del (train_meniscus_lbl)
    del (valid_abnormal_lbl)
    del (valid_acl_lbl)
    del (valid_meniscus_lbl)

    WIDTH = 256
    HEIGHT = 256

    # load x_train
    train_axial = np.zeros([38778, WIDTH, HEIGHT], dtype='uint8')
    train_coronal = np.zeros([33649, WIDTH, HEIGHT], dtype='uint8')
    train_sagittal = np.zeros([34370, WIDTH, HEIGHT], dtype='uint8')

    train_axial_lbl = np.zeros([38778], dtype='uint8')
    train_coronal_lbl = np.zeros([33649], dtype='uint8')
    train_sagittal_lbl = np.zeros([34370], dtype='uint8')

    train_axial_idx = 0
    train_sagittal_idx = 0
    train_coronal_idx = 0

    dir_train = "C:/Fall19/AI/Project/dataset/MRNet-v1.0/train"


    def to_rgb(img, wid, hei):  # -> Resizing image to fit as (WIDTH,HEIGHT)
        img = cv2.resize(img, (wid, hei), interpolation=cv2.INTER_AREA)
        return img


    def getTheDataLabelPerView_(obj, save_in, idx):
        global train_axial_idx, train_sagittal_idx, train_coronal_idx
        for j in range(len(obj)):  # 0 -> s (For every view)
            if (save_in == 'train_axial'):
                train_axial[train_axial_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
                train_axial_lbl[train_axial_idx] = train_label[idx]  # -> Giving all images the same label as patient.
                train_axial_idx += 1
            elif (save_in == 'train_coronal'):
                train_coronal[train_coronal_idx] = to_rgb(obj[j], WIDTH,
                                                          HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
                train_coronal_lbl[train_coronal_idx] = train_label[
                    idx]  # -> Giving all images the same label as patient.
                train_coronal_idx += 1
            else:
                train_sagittal[train_sagittal_idx] = to_rgb(obj[j], WIDTH,
                                                            HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
                train_sagittal_lbl[train_sagittal_idx] = train_label[
                    idx]  # -> Giving all images the same label as patient.
                train_sagittal_idx += 1


    i = 0
    for folder in sorted(os.listdir(dir_train)):
        idx = 0
        if folder == ".DS_Store" or folder == 'DG1__DS_DIR_HDR':
            continue
        type_dir = os.path.join(dir_train, folder)
        os.chdir(type_dir)
        for img in sorted(os.listdir(type_dir)):
            if img == ".DS_Store" or img == 'DG1__DS_DIR_HDR':
                continue
            img_dir = os.path.join(type_dir, img)
            if i == 0:
                getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_axial', idx)
            elif i == 1:
                getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_coronal', idx)
            elif i == 2:
                getTheDataLabelPerView_(np.load(img_dir).astype('uint8'), 'train_sagittal', idx)

            idx += 1
        i += 1

    # load y_train
    valid_ = []

    dir_valid = "C:/Fall19/AI/Project/MRNet-v1.0/valid"
    i = 0
    for folder in sorted(os.listdir(dir_valid)):
        if folder == ".DS_Store" or folder == 'DG1__DS_DIR_HDR':
            continue
        type_dir = os.path.join(dir_valid, folder)

        os.chdir(type_dir)
        for img in sorted(os.listdir(type_dir)):
            if img == ".DS_Store" or img == "DG1__DS_DIR_HDR":
                continue
            img_dir = os.path.join(type_dir, img)

            if i == 0:
                valid_.append(np.load(img_dir).astype('uint8'))
            elif i == 1:
                valid_.append(np.load(img_dir).astype('uint8'))

        i += 1


    def convert_to_one_hot(Y, C):
        Y = np.eye(C)[Y.reshape(-1)].T
        return Y


    train_axial_lbl = train_axial_lbl.reshape(38778, 1)
    print("label shape ", train_axial_lbl.shape)

    x_train, x_test, y_train, y_test = train_test_split(train_axial[0:100, :, :], train_axial_lbl[0:100, :],
                                                        test_size=0.2, random_state=42)

    x_train = np.stack([x_train], axis=-1)
    x_test = np.stack([x_test], axis=-1)

    ROWS, COLS, CHANNELS = x_train.shape[1:]
    CLASSES = 8

    train_set_x, train_set_y = x_train, y_train
    test_set_x, test_set_y = x_test, y_test

    X_train = train_set_x / 255
    X_test = test_set_x / 255

    Y_train = convert_to_one_hot(train_set_y, CLASSES).T
    Y_test = convert_to_one_hot(test_set_y, CLASSES).T

    dcgan = DCGAN()
    dcgan.train(X_train, epochs=1, batch_size=32, save_interval=50)
