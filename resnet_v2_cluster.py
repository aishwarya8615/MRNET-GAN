import os
import cv2
import numpy as np
from keras import layers
import csv
from sklearn.model_selection import train_test_split
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import multi_gpu_model
# from keras.utils import plot_model
# # from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# import keras.backend as K
# import tensorflow as tf
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
    with open('MRNet-v1.0/' + c, 'r') as csvfile:
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

dir_train = "MRNet-v1.0/train"


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
            train_coronal[train_coronal_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
            train_coronal_lbl[train_coronal_idx] = train_label[idx]  # -> Giving all images the same label as patient.
            train_coronal_idx += 1
        else:
            train_sagittal[train_sagittal_idx] = to_rgb(obj[j], WIDTH, HEIGHT)  # -> save each image as (WIDTH,HEIGHT)
            train_sagittal_lbl[train_sagittal_idx] = train_label[idx]  # -> Giving all images the same label as patient.
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

dir_valid = "MRNet-v1.0/valid"
i = 0
for folder in sorted(os.listdir(dir_valid)):
    if folder == ".DS_Store":
        continue
    type_dir = os.path.join(dir_valid, folder)

    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store":
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

X_train = train_set_x/255
X_test = test_set_x/255

Y_train = convert_to_one_hot(train_set_y, CLASSES).T
Y_test = convert_to_one_hot(test_set_y, CLASSES).T



print ("number of training examples =", X_train.shape[0])
print ("number of test examples =", X_test.shape[0])
print ("X_train shape:", X_train.shape)
print ("Y_train shape:", Y_train.shape)
print ("X_test shape:", X_test.shape)
print ("Y_test shape:", Y_test.shape)

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(256, 256, 1), classes=8):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

model = ResNet50(input_shape = (ROWS, COLS, CHANNELS), classes = CLASSES)
parallel_model=multi_gpu_model(model,gpus=2)

parallel_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
parallel_model.fit(X_train, Y_train, epochs = 1, batch_size = 8)
preds = parallel_model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
model.summary()
model.save('ResNet50.h5')