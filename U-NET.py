import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy.misc
import sklearn.feature_extraction
import numpy as np
import scipy.ndimage
import cv2
from skimage import measure, io
from skimage import transform
import train_UNET_2
import skimage
import json
import datetime
import pickle
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

epoch_cb_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Epoch_Callback'
save_results = '/home/rhys/Demosaicing_Training/Models'
load_training = '/home/rhys/Demosaicing_Training/Data/Training_Data/Gharbi_hdrvdp'
# load_validation = r'C:\Users\buggyr\Mosaic_Experiments\data\external\Val_data'

import keras
from keras.models import *
from keras.models import Sequential
from keras.layers import Conv2D, Activation, UpSampling2D, Reshape, MaxPooling2D, Dropout, Cropping2D, merge, Input, \
    concatenate, Conv2DTranspose, Lambda
from keras.optimizers import Adadelta, Nadam, RMSprop
from keras.models import load_model

kernel_size = (5, 5)
# kernel_size = (7, 7)
chnl4_input = Input(shape=(None, None, 4))
chnl3_input = Input(shape=(None, None, 3))

conv1 = Conv2D(32, kernel_size, activation='relu', padding='same')(chnl4_input)
conv2 = Conv2D(32, kernel_size, strides=(2, 2), activation='relu', padding='same')(conv1)

conv5 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv2)
conv6 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv5)

up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
conv7 = Conv2D(64, kernel_size, activation='relu', padding='same')(up1)

conv8 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv7)
conv9 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv8)

conv11 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv9)
conv12 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv11)

up3 = concatenate([UpSampling2D(size=(2, 2))(conv12), chnl3_input], axis=-1)
conv13 = Conv2D(67, kernel_size, activation='relu', padding='same')(up3)

conv14 = Conv2D(67, kernel_size, activation='relu', padding='same')(conv13)
conv15 = Conv2D(32, kernel_size, activation='relu', padding='same')(conv14)
conv16 = Conv2D(3, kernel_size, activation='relu', padding='same')(conv15)

out = Lambda(train_UNET_2.clipper, name='clipper')(conv16)

model = Model(inputs=[chnl4_input, chnl3_input], outputs=[out])

# model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
optimizer_func = Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
loss_func = 'mse'
# model1.compile(optimizer = optimizer_func, loss = loss_func)
model.compile(optimizer=optimizer_func, loss=loss_func)

model.summary()

model = keras.models.load_model('/home/rhys/Demosaicing_Training/Models/2018-05-21 19-12_UNET_2_layer_patch_64x64_kernel_5x5_mse_loss_2_input_8_mb/Epoch_Models/model.02-0.00.hdf5')


keyname = "_UNET_2_layer_patch_64x64_kernel_5x5_mse_loss_2_input_8_mb"
print(keyname)
now = datetime.datetime.now()
save_file = os.path.join(save_results, now.strftime("%Y-%m-%d %H-%M") + keyname)
os.mkdir(save_file)
save_pred = os.path.join(save_file, 'Epoch_Predictions')
os.mkdir(save_pred)
save_model = os.path.join(save_file, 'Epoch_Models')
os.mkdir(save_model)
save_test = os.path.join(save_file, 'Test_Results')
os.mkdir(save_test)

with open(os.path.join(save_file, 'Model_Summary.txt'), 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
tbCallBack = keras.callbacks.TensorBoard(log_dir=os.path.join(save_file, 'TNSR_BRD'), histogram_freq=0,
                                         write_graph=True, write_images=True, write_grads=True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(save_file, 'training.log'), separator=',', append=False)
epoch_predict = train_UNET_2.Save_predictions(save_pred, epoch_cb_dir)
model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_model, 'model.{epoch:02d}-{loss:.2f}.hdf5'),
                                                   monitor='loss')

model.compile(optimizer=optimizer_func, loss=loss_func)

fls = len(os.listdir(load_training))
# val_steps = len(os.listdir(load_validation))

# train_dir, patch_size, batch_size
train_generator = train_UNET_2.TrainSeq(load_training, 64, 8, save_file=save_file, rotate=False)
# val_generator = train_UNET_2.val_generator_rgb(load_validation)

history = model.fit_generator(generator=train_generator, epochs=50,
                              callbacks=[tbCallBack, csv_logger, epoch_predict, model_checkpoint],
                              workers=8, use_multiprocessing=True)
print(history.history)
