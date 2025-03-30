########################################################################
# Title: Final Project - Make Predictions
# Filename: MakePredictions.py
# Author: Zac Lynn
# Date: 4/2/2023
# Instructor: Dr. Rhodes
# Description: This code reads in previously trained network weights and
#                   makes predictions based on test images.
########################################################################
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np
import cv2
import os

global model, test_image_batch, test_image_filenames
img_x = 768
img_y = 768


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

    # KErnel is 1,1 for non blur version
    decoder_output = keras.layers.Conv2D(3, (7, 7), activation='softmax', padding='same') (x) 

    opt = keras.optimizers.Adam(lr=0.001)

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")

    autoencoder.compile(opt, loss=hybridLoss) 

    autoencoder.summary()


def read_test_data():
    global test_image_batch, test_image_filenames
    
    test_image_dir =  "./RAVIRDataset/test/" 
    test_image_filenames = os.listdir(test_image_dir)

    # Load the training images
    test_image_batch = []
    for file in test_image_filenames:
        test_image_batch.append(cv2.imread(test_image_dir + file, 0) / 255.0)
    test_image_batch = np.array(test_image_batch, dtype=np.float32)




def make_predictions(output_dir):
    prediction = []
    for image in test_image_batch:
        # Make a prediction with the model
        prediction.append(autoencoder.predict(np.expand_dims(image, axis=0))[0])

    # Threshold predictions
    threshold = 0.01

    for image in range(len(prediction)):
        imageOut = np.zeros((img_x,img_y))
        for row in range(len(prediction[image])):
            for col in range(len(prediction[image][row])):

                if (prediction[image][row][col][0] >= prediction[image][row][col][1] and prediction[image][row][col][0] >= prediction[image][row][col][2]):
                    # Background is 0
                    imageOut[row][col] = 0
                elif (prediction[image][row][col][1] >= prediction[image][row][col][0] and prediction[image][row][col][1] >= prediction[image][row][col][2]):
                    if (prediction[image][row][col][1] >= threshold):
                        # Arteries are 128
                        imageOut[row][col] = 128
                    else:
                        imageOut[row][col] = 0
                    
                elif (prediction[image][row][col][2] >= prediction[image][row][col][0] and prediction[image][row][col][2] >= prediction[image][row][col][1]):
                    if (prediction[image][row][col][2] >= threshold):
                        # Veins are 255
                        imageOut[row][col] = 256
                    else:
                        imageOut[row][col] = 0

        cv2.imwrite(output_dir + test_image_filenames[image], imageOut)


define_model()
autoencoder.load_weights("./model/R4_BE_150")

read_test_data()
make_predictions("./predictions/")