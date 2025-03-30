########################################################################
# Title: Final Project - Encoder Network
# Filename: encoder_R4.py
# Author: Zac Lynn
# Date: 4/2/2023
# Instructor: Dr. Rhodes
# Description: This code defines the encoder architecture and trains the
#                   network. The Learning rate and Batch size should be 
#                   updated at periods of 50 epochs.
########################################################################
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

global training_image_batch, training_mask_batch
global autoencoder

img_x = 768
img_y = 768

# Learning rate = 0.001 * 0.5 ^ epoch / 50
LR = 0.001 * 0.5 ** 0
print("LEARNING RATE: " + str(LR), end="\n\n\n")

def hybridLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)

    # Create a tensor for each prediction plane. Values are the raw predictions
    y_pred_class1, y_pred_class2, y_pred_class3 = tf.split(y_pred, num_or_size_splits=3, axis=-1)
    y_pred_class1 = K.flatten(tf.squeeze(y_pred_class1, axis=-1))
    y_pred_class2 = K.flatten(tf.squeeze(y_pred_class2, axis=-1))
    y_pred_class3 = K.flatten(tf.squeeze(y_pred_class3, axis=-1))

    # Create a tensor for each class. Tensors values are 1 or 0
    y_true_class1 = tf.where(tf.equal(y_true_f, 0), tf.ones_like(y_true_f), tf.zeros_like(y_true_f))
    y_true_class2 = tf.where(tf.equal(y_true_f, 1), tf.ones_like(y_true_f), tf.zeros_like(y_true_f))
    y_true_class3 = tf.where(tf.equal(y_true_f, 2), tf.ones_like(y_true_f), tf.zeros_like(y_true_f))

    intersection1 = K.sum(y_true_class1 * y_pred_class1)
    intersection2 = K.sum(y_true_class2 * y_pred_class2)
    intersection3 = K.sum(y_true_class3 * y_pred_class3)

    pixelSum1 = K.sum(y_pred_class1)
    pixelSum2 = K.sum(y_pred_class2)
    pixelSum3 = K.sum(y_pred_class3)

    dice1 = (intersection1) / (pixelSum1 + K.sum(y_true_class1))
    dice2 = (intersection2) / (pixelSum2 + K.sum(y_true_class2))
    dice3 = (intersection3) / (pixelSum3 + K.sum(y_true_class3))

    diceLoss = 1.0 - (2.0 / 3.0) * (dice1 + dice2 + dice3)


    ceLoss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    ceLoss = K.mean(ceLoss)

    return 1.0 * diceLoss  + 1.0 * ceLoss

def read_training_data():
    global training_image_batch, training_mask_batch
    classes = {"background": 0, "artery": 128, "vein": 255}

    training_image_dir = "./autoencoderDataBE/train/training_images/"
    training_mask_dir = "./autoencoderDataBE/train/training_masks/"

    training_image_filenames = os.listdir(training_image_dir)

    # Load the training images
    training_image_batch = []
    for file in training_image_filenames:
        training_image_batch.append(cv2.imread(training_image_dir + file, 0) / 255.0)
    training_image_batch = np.array(training_image_batch, dtype=np.float32)

    # Load the training masks
    training_mask_batch = []
    for file in training_image_filenames:
        training_mask_batch.append(cv2.imread(training_mask_dir + file, 0))
    training_mask_batch = np.array(training_mask_batch, dtype=np.float32)

    # Threshold any mask values that changed with image rotation
    training_mask_batch[(training_mask_batch == classes["artery"])] = 1
    training_mask_batch[(training_mask_batch == classes["vein"])] = 2

    training_mask_batch[(training_mask_batch == classes["artery"]-1)] = 1
    training_mask_batch[(training_mask_batch == classes["vein"]-1)] = 2

def define_model():
    global autoencoder

    encoder_input = keras.Input(shape=(img_x, img_y, 1), name="image_in")
    
    # -------------- INPUT -------------- 
    x = keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same') (encoder_input) 
    block1 = keras.layers.BatchNormalization()(x)
    
    # -------------- ENCODER-------------- 
    x = keras.layers.Conv2D(32, (7, 7), strides=(2,2), padding='same') (block1)                                 
    block2 = keras.layers.Dropout(0.15)(x)                                                                             
    
    x = keras.layers.Conv2D(64, (5, 5), strides=(2,2), padding='same') (block2)                                 
    block3 = keras.layers.Dropout(0.15)(x)                         
    
    x = keras.layers.Conv2D(128, (3, 3), strides=(2,2), padding='same') (block3)                                 
    block4 = keras.layers.Dropout(0.15)(x)                  
    block4 = keras.layers.BatchNormalization()(block4)  #??

    encoder = keras.Model(encoder_input, block4)

    # -------------- DECODER-------------- 
    decoder_input = keras.layers.Conv2D(128, (3, 3), padding='same')(block4)        
    x = keras.layers.Dropout(0.15)(decoder_input)
    x = keras.layers.BatchNormalization()(x)            #??

    x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = keras.layers.add([x, block4])
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same') (x)                        
    x = keras.layers.add([x, block3])                                         
    x = keras.layers.Dropout(0.15)(x)
    
    x = keras.layers.Conv2DTranspose(32, (7, 7), strides=(2, 2), padding='same') (x)                        
    x = keras.layers.add([x, block2])                                          
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Conv2DTranspose(16, (7, 7), strides=(2, 2), padding='same') (x)                        
    x = keras.layers.add([x, block1])                                            
    x = keras.layers.Dropout(0.15)(x)

    # -------------- OUTPUT -------------- 
    x = keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same') (x)           
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Conv2D(16, (7, 7), activation='relu', padding='same') (x)         
    x = keras.layers.BatchNormalization()(x)

    # Kernel is 1,1 for non blur version
    decoder_output = keras.layers.Conv2D(3, (7, 7), activation='softmax', padding='same') (x) 

    opt = keras.optimizers.Adam(lr=LR)

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")

    autoencoder.compile(opt, loss=hybridLoss) 

    autoencoder.summary()
 

define_model()
read_training_data()

# Load model - only load after first 50 epochs
# autoencoder.load_weights('./model/R4_BE_50')

# Train model
autoencoder.fit(training_image_batch, training_mask_batch, epochs=50, batch_size=3, shuffle=True) 

# Save Model
# autoencoder.save_weights("./model/R4_BE_50")


# Test image
testImg = np.array(cv2.imread("./RAVIRDataset/test/IR_Case_060.png", 0) / 255.0, dtype=np.float32)
testImg = testImg.reshape(-1, 768, 768, 1)
ex = autoencoder.predict(testImg)[0]


r = np.zeros((img_x, img_y))
g = np.zeros((img_x, img_y))
b = np.zeros((img_x, img_y))


for row in range(len(ex)):
    for col in range(len(ex[row])):
        r[row][col] = ex[row][col][0]
        g[row][col] = ex[row][col][1]
        b[row][col] = ex[row][col][2]

        ex[row][col][0] = 0


plt.figure(1)
plt.imshow(testImg[0], cmap='gray')
plt.title("Original")

plt.figure(2)
plt.imshow(r, cmap='gray')
plt.title("Background")

plt.figure(3)
plt.imshow(g, cmap='gray')
plt.title("Artery")

plt.figure(4)
plt.imshow(b, cmap='gray')
plt.title("Vein")

plt.figure(5)
plt.imshow(ex, cmap='gray')
plt.title("Prediction")

plt.show()

