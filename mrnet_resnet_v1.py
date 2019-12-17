from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Activation, Flatten
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import cv2
import csv
from tqdm import tqdm
from keras.datasets import cifar10
from keras.models import Sequential
from sklearn.model_selection import train_test_split

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
    with open('/home/vignesh/PycharmProjects/AI_Course/Project/MRNet-v1.0/' + c, 'r') as csvfile:
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

'''
In the dataset, labels are given in 3 files:
train-abnormal.csv/valid-abnormal.csv  - contains labels for each patient for train/validation dataset: 1 if abormal       , 0 if not abormal
train-acl.csv/valid-acl.csv           - contains labels for each patient for train/validation dataset: 1 if acl tear      , 0 if no acl tear
train-meniscus.csv/valid-meniscus.csv - contains labels for each patient for train/validation dataset: 1 if meniscus tear , 0 if no meniscus tear
This function bit-packs three conditions into a single number from 0-7.
Bit 0 -> meniscus tear
Bit 1 -> acl tear
Bit 2 -> abnormal
'''
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
'''
The mapping of issue for each patient in train_lablel and valid_label is as follows:
0 -> No issues
1 -> meniscus tear
2 -> acl tear
3 -> meniscus tear + acl tear
4 -> abnormal
5 -> abnormal + meniscus tear
6 -> abnormal + acl tear
7 -> abnormal + meniscus tear + acl tear  
'''
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

dir_train = "/home/vignesh/PycharmProjects/AI_Course/Project/MRNet-v1.0/train"


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

dir_valid = "/home/vignesh/PycharmProjects/AI_Course/Project/MRNet-v1.0/valid"
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

# dataset in desired format is available now
# print('dataset shape: ', train_axial.shape)
# plt.imshow(train_axial[0, :, :], 'rgb')
# plt.show()
# plt.imshow(train_axial[1, :, :], 'gray')
# plt.show()
# plt.imshow(train_axial[2, :, :], 'gray')
# plt.show()



# Training parameters
batch_size = 2  # orig paper trained all networks with batch_size=128
epochs = 1
data_augmentation = True
num_classes = 8

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
# cut_train_axial = []
# cut_train_axial_lbl = []
# for j in range(0, 1000):
#     cut_train_axial.append(train_axial[j, :, :])
#     cut_train_axial_lbl.append(train_axial_lbl[j, :, :])
train_axial_lbl = train_axial_lbl.reshape(38778, 1)
print("label shape ", train_axial_lbl.shape)
# Load the CIFAR10 data.
# x_train, x_test, y_train, y_test = train_test_split(np.asarray(cut_train_axial), np.asarray(cut_train_axial_lbl),
#                                                                 test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_axial[0:100, :, :], train_axial_lbl[0:100, :],
                                                                test_size=0.2, random_state=42)
print("The shape is at start", x_train.shape, y_train.shape)
print("type of data is ", type(x_train), type(y_train))
# x_train = x_train.reshape(31022, 256, 256, 2)
# y_train = y_train.reshape(31022, 1)
x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)

# Input image dimensions.
input_shape = x_train.shape[1:]
# input_shape = (256, 256, 1)
# print("Input shape is ", input_shape)

# Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=8):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=8):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])