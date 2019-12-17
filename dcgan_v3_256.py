from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

tf.__version__
# To generate GIFs

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from keras.applications import ResNet50
import numpy as np
import cv2
import csv
import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from tqdm import tqdm
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.xception import Xception
from keras.preprocessing.image import array_to_img

from IPython import display

# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

WIDTH = 256
HEIGHT = 256

# load y_train and y_test
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
    with open('/home/vvarier/ai_project/MRNet-v1.0/' + c, newline='') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar='|')
        if i == 0:
            for row in read:
                train_acl_lbl.append(', '.join(row)[5])
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
    labels = [0] * len(a)
    for i in tqdm(range(len(labels))):
        labels[i] = int(str(a[i]) + str(b[i]) + str(c[i]), 2)
    return np.array(labels)


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

# load x_train
train_axial = np.zeros([38778, WIDTH, HEIGHT], dtype='uint8')
train_axial.nbytes
train_coronal = np.zeros([33649, WIDTH, HEIGHT], dtype='uint8')
train_sagittal = np.zeros([34370, WIDTH, HEIGHT], dtype='uint8')

train_axial_lbl = np.zeros([38778], dtype='uint8')
train_coronal_lbl = np.zeros([33649], dtype='uint8')
train_sagittal_lbl = np.zeros([34370], dtype='uint8')

i = 0

train_axial_idx = 0
train_sagittal_idx = 0
train_coronal_idx = 0

dir_train = "/home/vvarier/ai_project/MRNet-v1.0/train"


def to_rgb(img, wid, hei):  # -> Resizing image to fit as (WIDTH,HEIGHT,3)
    img = cv2.resize(img, (wid, hei), interpolation=cv2.INTER_AREA)
    return img


def getTheDataLabelPerView_(obj, save_in, idx):
    global train_axial_idx, train_sagittal_idx, train_coronal_idx
    for j in tqdm(range(len(obj))):  # 0 -> s (For every view)
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


for folder in sorted(os.listdir(dir_train)):
    idx = 0
    if folder == ".DS_Store"or folder == 'DG1__DS_DIR_HDR':
        continue
    type_dir = os.path.join(dir_train, folder)
    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store"or img == 'DG1__DS_DIR_HDR':
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

dir_valid =  "/home/vvarier/ai_project/MRNet-v1.0/train"
i = 0
for folder in sorted(os.listdir(dir_valid)):
    if folder == ".DS_Store"or folder == 'DG1__DS_DIR_HDR':
        continue
    type_dir = os.path.join(dir_valid, folder)

    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store"or img == 'DG1__DS_DIR_HDR':
            continue
        img_dir = os.path.join(type_dir, img)

        if i == 0:
            valid_.append(np.load(img_dir).astype('uint8'))

        elif i == 1:
            valid_.append(np.load(img_dir).astype('uint8'))
    i += 1

BUFFER_SIZE = 60000
BATCH_SIZE = 256


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None,16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 256, 256, 8)
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 256, 256, 1)

    return model


generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)


# plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[256, 256, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
#The model will be trained to output positive values for real images, and negative values for fake images.
print("This is the decision",decision)
# This method returns a helper function to compute cross entropy loss
if decision < 0:
    print("The image is fake")
else:
    print("The image is real")
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    #It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
    print("This is the total Loss: ",total_loss)
    return total_loss
'''
The generator's loss quantifies how well it was able to trick the discriminator.
if the generator is performing well, the discriminator will classify the fake images as real (or 1). 
Here, we will compare the discriminators decisions on the generated images to an array of 1s.

'''

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 1
noise_dim = 100
num_examples_to_generate = 8

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        print("This is gen loss:",gen_loss)
        disc_loss = discriminator_loss(real_output, fake_output)
        print("This is disc loss:",disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

            # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


#    plt.show()


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


train_datasets = np.array([train_axial, train_coronal, train_sagittal])
for i in range(train_datasets.shape[0]):
    train_dataset = train_datasets[i].reshape(train_datasets[i].shape[0], 256, 256, 1).astype('float32')
    train_dataset = (train_dataset - 127.5) / 127.5  # Normalize the images to [-1, 1]
    dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train(dataset, EPOCHS)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython

if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=anim_file)

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(anim_file)
