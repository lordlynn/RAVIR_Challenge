########################################################################
# Title: Final Project - DataAugmentation
# Filename: dataAugmentation.py
# Author: Zac Lynn
# Date: 4/2/2023
# Instructor: Dr. Rhodes
# Description: This code reads in the RAVIR dataset and generates an 
#                   augmented and normalized output data set.
########################################################################
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import os

global training_image_batch, training_mask_batch, training_image_filenames

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()



def read_training_data():
    global training_image_batch, training_mask_batch, training_image_filenames
    classes = {"background": 0, "artery": 128, "vein": 255}

    training_image_dir = "./RAVIRDataset/train/training_images/"
    training_mask_dir = "./RAVIRDataset/train/training_masks/"

    training_image_filenames = os.listdir(training_image_dir)

    # Load the training images
    training_image_batch = []
    for file in training_image_filenames:
        training_image_batch.append(cv2.imread(training_image_dir + file, 0))
    training_image_batch = np.array(training_image_batch, dtype=np.uint8)

    # Load the training masks
    training_mask_batch = []
    for file in training_image_filenames:
        training_mask_batch.append(cv2.imread(training_mask_dir + file, 0))
    training_mask_batch = np.array(training_mask_batch, dtype=np.uint8)



read_training_data()


original_height = 768
original_width = 768


aug = A.Compose([  

    # A.Normalize(p=1),
    A.OneOf([
        A.VerticalFlip(p=1),        
        A.HorizontalFlip(p=1),          
        A.Rotate(limit=180, p=1),        
        A.Rotate(limit=90, p=1),
        A.Rotate(limit=270, p=1),    
        A.HorizontalFlip(A.Rotate(limit=90, p=1), p=1),
        A.VerticalFlip(A.Rotate(limit=90, p=1), p=1)], p=0.875),

    A.CLAHE(p=0.07),
    A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.25], p=0.9),
    
     ])

output_dir = "./autoencoderDataBE/train/"

num_samples = 18 # number of augmented samples to generate
augmented_images = []
augmented_masks = []

for idx in range(len(training_image_filenames)):
    for i in range(num_samples):
        image = training_image_batch[idx]
        mask = training_mask_batch[idx]
        
        # Apply augmentation pipeline to image and mask
        random.seed(i)  # set seed for reproducibility
        augmented = aug(image=image, mask=mask)


        # visualize(augmented['image'], augmented['mask'], original_image=image, original_mask=mask)
        print(output_dir + str(training_image_filenames[idx][:-4]) + "_" + str(i) + ".png" )

        cv2.imwrite(output_dir + "training_images/" + str(training_image_filenames[idx][:-4]) + "_" + str(i) + ".png", augmented['image'])
        cv2.imwrite(output_dir + "training_masks/" + str(training_image_filenames[idx][:-4]) + "_" + str(i) + ".png", augmented['mask'])

# Convert lists to arrays
# augmented_images = np.array(augmented_images)
# augmented_masks = np.array(augmented_masks)


