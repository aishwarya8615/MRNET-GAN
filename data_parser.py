import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import csv
from tqdm import tqdm

from IPython import display
#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

#load y_train and y_test
train_acl_lbl = []
train_abnormal_lbl = []
train_meniscus_lbl = []

valid_acl_lbl = []
valid_abnormal_lbl = []
valid_meniscus_lbl = []

i = 0
csvs = ["train-acl.csv", "train-abnormal.csv","train-meniscus.csv","valid-acl.csv", "valid-abnormal.csv","valid-meniscus.csv"]
for c in csvs:
    with open('MRNet-v1.0/'+ c, newline='') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar='|')
        if i == 0:
            for row in read:
                train_acl_lbl.append(', '.join(row)[5])
            i += 1
        elif i == 1:
            for row in read:
                train_abnormal_lbl.append(', '.join(row)[5])
            i+=1
        elif i == 2:
            for row in read:
                train_meniscus_lbl.append(', '.join(row)[5])
            i+=1
        elif i == 3:
            for row in read:
                valid_acl_lbl.append(', '.join(row)[5])
            i+=1
        elif i == 4:
            for row in read:
                valid_abnormal_lbl.append(', '.join(row)[5])
            i+=1
        elif i == 5:
            for row in read:
                valid_meniscus_lbl.append(', '.join(row)[5])
            i+=1    

def getTheLabels(a,b,c):
  labels = [0] * len(a)
  for i in tqdm(range(len(labels))):
    labels[i] = int(str(a[i]) + str(b[i]) + str(c[i]),2)
  return np.array(labels)

#Encoding all labels to be a number from (0-7) (Abnormal,ACL,Meniscus)
#GAN doesn't look like using labels
train_label = getTheLabels(train_abnormal_lbl,train_acl_lbl,train_meniscus_lbl)
valid_label = getTheLabels(valid_abnormal_lbl,valid_acl_lbl,valid_meniscus_lbl)
del(train_abnormal_lbl)
del(train_acl_lbl)
del(train_meniscus_lbl)
del(valid_abnormal_lbl)
del(valid_acl_lbl)
del(valid_meniscus_lbl)

WIDTH = 256
HEIGHT = 256

#load x_train
train_axial = np.zeros([38778,WIDTH,HEIGHT], dtype='uint8')
train_coronal = np.zeros([33649,WIDTH,HEIGHT], dtype='uint8') 
train_sagittal = np.zeros([34370,WIDTH,HEIGHT],dtype='uint8')

train_axial_lbl = np.zeros([38778], dtype='uint8')
train_coronal_lbl = np.zeros([33649], dtype='uint8')
train_sagittal_lbl = np.zeros([34370],dtype='uint8')

train_axial_idx = 0
train_sagittal_idx = 0
train_coronal_idx = 0

dir_train="C:/admin/masters/AI/gan/MRNet-v1.0/train"

def to_rgb(img,wid,hei):# -> Resizing image to fit as (WIDTH,HEIGHT)
    img = cv2.resize(img, (wid,hei), interpolation = cv2.INTER_AREA) 
    return img
    
def getTheDataLabelPerView_(obj,save_in,idx):
    global train_axial_idx , train_sagittal_idx , train_coronal_idx
    for j in tqdm(range(len(obj))): # 0 -> s (For every view)
        if (save_in == 'train_axial'):
            train_axial[train_axial_idx] = to_rgb(obj[j],WIDTH,HEIGHT) # -> save each image as (WIDTH,HEIGHT)
            train_axial_lbl[train_axial_idx] = train_label[idx] # -> Giving all images the same label as patient.
            train_axial_idx += 1
        elif(save_in == 'train_coronal'):
            train_coronal[train_coronal_idx] = to_rgb(obj[j],WIDTH,HEIGHT) # -> save each image as (WIDTH,HEIGHT)
            train_coronal_lbl[train_coronal_idx] = train_label[idx] # -> Giving all images the same label as patient.
            train_coronal_idx += 1
        else:
            train_sagittal[train_sagittal_idx] = to_rgb(obj[j],WIDTH,HEIGHT) # -> save each image as (WIDTH,HEIGHT)
            train_sagittal_lbl[train_sagittal_idx] = train_label[idx] # -> Giving all images the same label as patient.
            train_sagittal_idx += 1

i = 0 
for folder in sorted(os.listdir(dir_train)):
    idx = 0
    if folder == ".DS_Store" or folder == 'DG1__DS_DIR_HDR':
        continue;
    type_dir = os.path.join(dir_train,folder)
    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store" or img == 'DG1__DS_DIR_HDR':
            continue
        img_dir=os.path.join(type_dir,img)
        if i == 0:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'),'train_axial',idx)
        elif i == 1:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'),'train_coronal',idx)
        elif i == 2:
            getTheDataLabelPerView_(np.load(img_dir).astype('uint8'),'train_sagittal',idx)

        idx += 1
    i+=1      
            
#load y_train
valid_ = []

dir_valid="C:/admin/masters/AI/gan/MRNet-v1.0/valid"
i = 0
for folder in sorted(os.listdir(dir_valid)):
    if folder == ".DS_Store":
        continue;
    type_dir=os.path.join(dir_valid,folder)

    os.chdir(type_dir)
    for img in sorted(os.listdir(type_dir)):
        if img == ".DS_Store":
            continue;
        img_dir=os.path.join(type_dir,img)  

        if i==0:
            valid_.append(np.load(img_dir).astype('uint8'))
        elif i==1:
            valid_.append(np.load(img_dir).astype('uint8'))

    i+=1       

# dataset in desired format is available now
print('dataset shape: ', train_axial.shape)
plt.imshow(train_axial[0, :, :], 'gray')
plt.show()
