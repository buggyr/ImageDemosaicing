import train_UNET_2
import keras
from keras.models import *
from keras.layers import Conv2D, Activation, UpSampling2D, Reshape, MaxPooling2D, Dropout, Cropping2D, merge, Input, \
    concatenate, Conv2DTranspose, Lambda, Add, BatchNormalization
from keras.optimizers import Adadelta, Nadam, RMSprop


class UNET:
    
    def __init__(self, filters=64, kernel_size=7, loss_func='mse',
                 optimizer_func=Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)):

        self.filters = filters
        self.kernel_size = kernel_size
        self.loss_func = loss_func
        self.optimizer_func = optimizer_func
        self.model = None

    def build_model(self):

        chnl4_input = Input(shape=(None, None, 4))
        chnl3_input = Input(shape=(None, None, 3))

        conv1 = Conv2D(32, self.kernel_size, activation='relu', padding='same')(chnl4_input)
        conv2 = Conv2D(32, self.kernel_size, strides=(2, 2), activation='relu', padding='same')(conv1)

        conv5 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv2)
        conv6 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv5)

        up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
        conv7 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(up1)

        conv8 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv7)
        conv9 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv8)

        conv11 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv9)
        conv12 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv11)

        up3 = concatenate([UpSampling2D(size=(2, 2))(conv12), chnl3_input], axis=-1)
        conv13 = Conv2D(67, self.kernel_size, activation='relu', padding='same')(up3)

        conv14 = Conv2D(67, self.kernel_size, activation='relu', padding='same')(conv13)
        conv15 = Conv2D(32, self.kernel_size, activation='relu', padding='same')(conv14)
        conv16 = Conv2D(3, self.kernel_size, activation='relu', padding='same')(conv15)

        out = conv16

        self.model = Model(inputs=[chnl4_input, chnl3_input], outputs=[out])

        self.model.compile(optimizer=self.optimizer_func, loss=self.loss_func)
        self.model.name = 'UNET'

        return self.model

    def build_residual_model(self):

        chnl4_input = Input(shape=(None, None, 4))
        chnl3_input = Input(shape=(None, None, 3))

        conv1 = Conv2D(32, self.kernel_size, activation='relu', padding='same')(chnl4_input)
        conv2 = Conv2D(32, self.kernel_size, strides=(2, 2), activation='relu', padding='same')(conv1)

        conv5 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv2)
        conv6 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv5)

        up1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
        conv7 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(up1)

        conv8 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv7)
        conv9 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv8)

        conv11 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv9)
        conv12 = Conv2D(64, self.kernel_size, activation='relu', padding='same')(conv11)

        up3 = concatenate([UpSampling2D(size=(2, 2))(conv12), chnl3_input], axis=-1)
        conv13 = Conv2D(67, self.kernel_size, activation='relu', padding='same')(up3)

        conv14 = Conv2D(67, self.kernel_size, activation='relu', padding='same')(conv13)
        conv15 = Conv2D(32, self.kernel_size, activation='relu', padding='same')(conv14)
        conv16 = Conv2D(3, self.kernel_size, activation='relu', padding='same')(conv15)

        out = Add()([chnl3_input, conv16])

        self.model = Model(inputs=[chnl4_input, chnl3_input], outputs=[out])

        self.model.compile(optimizer=self.optimizer_func, loss=self.loss_func)
        self.model.name = 'UNET'

        return self.model

    def compile_model(self):

        self.model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        return self.model


class ChangRESNET:

    def __init__(self, model_dir, filters=64, kernel_size=7, num_layers=20, loss_func='mse',
                 optimizer_func=Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)):
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.model_dir = model_dir

        self.loss_func = loss_func
        self.optimizer_func = optimizer_func
        self.model = None

    def build_layer(self, x):

        x = Conv2D(self.filters, (self.kernel_size, self.kernel_size), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('selu')(x)

        return x
    
    def build_model_layers(self, x):
        
        for i in range(self.num_layers - 1):
            x = self.build_layer(x)

        x = Conv2D(3, (self.kernel_size, self.kernel_size), padding='same')(x)
            
        return x

    def build_model(self):
        chnl3_input = Input(shape=(None, None, 3))

        body_out = self.build_model_layers(chnl3_input)

        add1 = Add()([chnl3_input, body_out])

        out = Lambda(train_UNET_2.clipper, name='clipper')(add1)

        self.model = Model(inputs=[chnl3_input], outputs=[out])

        self.model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        self.model.name = 'Chang'

        with open(os.path.join(self.model_dir, 'Model_Summary.txt'), 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\\n'))

        return self.model


class DCNN:

    def __init__(self, filters=32, kernel_size=3, activation='relu', num_layers=10, loss_func='mse',
                 optimizer_func=Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)):

        self.filters = filters
        self.activation = activation
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.loss_func = loss_func
        self.optimizer_func = optimizer_func
        self.model = None

    def build_model_layers(self, x):

        for i in range(self.num_layers):
            x = Conv2D(self.filters, (self.kernel_size, self.kernel_size), activation=self.activation, padding='same')(x)

        return x

    def build_model(self):
        chnl3_input = Input(shape=(None, None, 3))

        body_out = self.build_model_layers(chnl3_input)

        out = Lambda(train_UNET_2.clipper, name='clipper')(body_out)

        self.model = Model(inputs=[chnl3_input], outputs=[out])

        self.model.compile(optimizer=self.optimizer_func, loss=self.optimizer_func)

        return self.model


class RESNET:
    def __init__(self, filters=64, kernel_size=7, num_layers=20, loss_func='mse',
                 optimizer_func=Nadam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)):
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.loss_func = loss_func
        self.optimizer_func = optimizer_func
        self.model = None

    def build_layer(self, x):
        x1 = Conv2D(self.filters, (self.kernel_size, self.kernel_size), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('selu')(x1)
        x1 = Add()([x, x1])

        return x1

    def build_model_layers(self, x):
        for i in range(self.num_layers - 1):
            x = self.build_layer(x)

        x = Conv2D(3, (self.kernel_size, self.kernel_size), padding='same')(x)

        return x

    def build_model(self):
        chnl3_input = Input(shape=(None, None, 3))

        body_out = self.build_model_layers(chnl3_input)

        out = Lambda(train_UNET_2.clipper, name='clipper')(body_out)

        self.model = Model(inputs=[chnl3_input], outputs=[out])

        self.model.compile(optimizer=self.optimizer_func, loss=self.loss_func)

        return self.model


def get_model_memory_usage(batch_size, model):

    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def load_model(model_path):
    return keras.models.load_model(model_path)
