import cv2
import numpy as np
import math
from sklearn.feature_extraction import image
from keras.applications.vgg16 import VGG16
from keras.models import *
import keras
import skimage.measure
import tensorflow as tf
from keras.utils import Sequence
from keras import backend as K
import random
import logging


# Keras Callback for epoch end process - predictions
class SavePredictions(keras.callbacks.Callback):
    def __init__(self, save_file, open_dir):
        self.savf = save_file
        self.open_dir = open_dir

    def on_epoch_end(self, epoch, logs={}):
        predict_gen = predict_generator_rgb(self.open_dir, self.model.name)
        if self.model.name == 'Chang':
            scale = -1
        else:
            scale = 0

        for i in (os.listdir(self.open_dir)):
            p_img_gen, orig_gen, img_name = next(predict_gen)

            pred_img = self.model.predict(p_img_gen, batch_size=1)[0]
            pred_img[pred_img > 1] = 1

            # pred_img = addBayer(orig_gen, pred)

            print(img_name, skimage.measure.compare_psnr(pred_img, orig_gen, 1))

            sv = os.path.join(self.savf, str(epoch) + '_' + img_name)
            cv2.imwrite(sv, denormalise1(pred_img, scale=scale))


# Sequence Generator for tiled image input
class TrainSeqTiled(Sequence):

    def __init__(self, train_dir, patch_size, batch_size, save_file, rotate=True, shuffle=True):

        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_names = os.listdir(train_dir)
        self.btch_start = np.zeros(len(os.listdir(train_dir))).astype(np.uint8)
        self.rotate = rotate
        self.file_errors = []
        self.save_file = save_file
        logging.basicConfig(filename=os.path.join(save_file, 'example.log'), level=logging.DEBUG)

    def __len__(self):
        return math.ceil(len(self.img_names))

    def __getitem__(self, idx):

        train_dir = self.train_dir
        batch_size = self.batch_size
        patch_size = self.patch_size

        img = self.img_names[idx]

        n_dim = 128
        m_dim = 64
        tile_n = 8
        orig = np.ones((128, 128, 6))
        input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
        input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))

        try:

            if self.rotate:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 6))
            else:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
            num_btchs = int(64 / batch_size)

            sc = os.path.join(train_dir, img)

            ptch = cv2.imread(sc)

            # ptch_mos4 = mosaic1(ptch, 4)
            # ptch_mos3 = mosaic1(ptch, 3)

            btch_begin = self.btch_start[idx]
            if btch_begin == 0:
                self.btch_start[idx] = 4
            else:
                self.btch_start[idx] = 0
            x = 0
            for i in range(btch_begin, tile_n):
                for j in range(tile_n):
                    i_dim = n_dim * i
                    j_dim = n_dim * j

                    im_dim = m_dim * i
                    jm_dim = m_dim * j

                    if self.rotate:

                        ptch_rotated = rotate_random(ptch[i_dim:i_dim + n_dim,
                                                     j_dim:j_dim + n_dim, :], random.randint(0, 90))

                        ptch_mos4 = mosaic1(ptch_rotated, 4)
                        ptch_mos3 = mosaic1(ptch_rotated, 3)

                        orig[:, :, 0:3] = normalise1(ptch_rotated)
                        orig[:, :, 3:6] = (orig[:, :, 0:3] > 0).astype(np.float32)

                        input_orig[x, :, :, :] = orig
                        input_mos4[x, :, :, :] = ptch_mos4
                        input_mos3[x, :, :, :] = ptch_mos3
                    else:

                        ptch_mos4 = mosaic1(ptch, 4)
                        ptch_mos3 = mosaic1(ptch, 3)

                        input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                        input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
                        input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
                    x += 1
                    if x % batch_size == 0:
                        x = 0

                        return [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)

        except Exception as e:
            print('Generator error: ' + str(e))
            self.file_errors.append(sc)
            logging.warning('Generator Error: ' + str(e) + ' Image: ' + sc)
            return [np.zeros((batch_size, patch_size, patch_size, 4)),
                    np.zeros((batch_size, patch_size * 2, patch_size * 2, 3))], np.zeros(
                (batch_size, patch_size * 2, patch_size * 2, 3))


# Sequence Generator for addition of rotations
class TrainSeq1Input(Sequence):

    def __init__(self, train_dir, patch_size, batch_size, save_file, rotate=False, shuffle=True):

        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_names = os.listdir(train_dir)
        self.rotate = rotate
        self.file_errors = []
        self.save_file = save_file
        logging.basicConfig(filename=os.path.join(save_file, 'Generator.log'), level=logging.DEBUG)

    def __len__(self):
        return math.floor(len(self.img_names) / self.batch_size)

    def __getitem__(self, idx):

        train_dir = self.train_dir
        batch_size = self.batch_size
        patch_size = self.patch_size

        indexes = self.img_names[idx * self.batch_size: (idx + 1) * self.batch_size]

        orig = np.ones((128, 128, 6))

        img_path = '/'
        img = '...'

        try:

            img_patches = np.zeros((self.batch_size, 2 * self.patch_size, 2 * self.patch_size, 3))
            for i, img in enumerate(indexes):
                img_patches[i, :, :, :] = normalise1(cv2.imread(os.path.join(train_dir, img)), scale=0)

            if self.rotate:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 6))
            else:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))

            if self.rotate:

                patch_rotated = rotate_random(img_patches, random.randint(0, 90))

                input_mos3 = mosaic(patch_rotated, 3)

                input_orig[:, :, 0:3] = patch_rotated
                input_orig[:, :, 3:6] = (orig[:, :, 0:3] > 0).astype(np.float32)

            else:

                input_mos3 = mosaic(img_patches, 3)
                input_orig = img_patches

            return input_mos3, input_orig

        except Exception as e:
            print('Generator error: ' + str(e))
            self.file_errors.append(img_path)
            logging.warning('Generator Error: ' + str(e) + ' Image: ' + img)
            return [np.zeros((batch_size, patch_size * 2, patch_size * 2, 3))], np.zeros(
                (batch_size, patch_size * 2, patch_size * 2, 3))


# Sequence Generator for addition of noise
class TrainSeq2Input(Sequence):

    def __init__(self, train_dir, patch_size, batch_size, save_file, rotate=False,
                 shuffle=True, noise=False, noise_std=4):

        self.train_dir = train_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_names = os.listdir(train_dir)
        self.rotate = rotate
        self.file_errors = []
        self.save_file = save_file
        self.noise = noise
        self.noise_std = noise_std
        logging.basicConfig(filename=os.path.join(save_file, 'Generator.log'), level=logging.DEBUG)
        self.scale = 0

    def __len__(self):
        return math.floor(len(self.img_names) / self.batch_size)

    def __getitem__(self, idx):

        train_dir = self.train_dir
        batch_size = self.batch_size
        patch_size = self.patch_size

        indexes = self.img_names[idx * self.batch_size: (idx + 1) * self.batch_size]

        orig = np.ones((128, 128, 6))
        g_noise = np.zeros((128, 28, 3))

        img_path = '/'
        img = '...'

        try:

            if self.rotate:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 6))
            else:
                input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))

            img_patches = np.zeros((self.batch_size, 2 * self.patch_size, 2 * self.patch_size, 3))
            noisy_patches = np.zeros((self.batch_size, 2 * self.patch_size, 2 * self.patch_size, 3))

            if self.noise:
                for i, img in enumerate(indexes):
                    img_patches[i, :, :, :] = normalise1(cv2.imread(os.path.join(train_dir, img)))
                    g_noise = cv2.randn(g_noise, 0, np.array([1, 1, 1]))
                    noisy_patches[i, :, :, :] = img_patches[i, :, :, :]

            else:
                for i, img in enumerate(indexes):
                    img_patches[i, :, :, :] = normalise1(cv2.imread(os.path.join(train_dir, img)))

            if self.rotate:

                if self.noise:
                    patch_rotated = rotate_random(noisy_patches, random.randint(0, 90))
                    orig_rotated = rotate_random(img_patches, random.randint(0, 90))

                    input_mos4 = mosaic(patch_rotated, 4)
                    input_mos3 = mosaic(patch_rotated, 3)

                    input_orig[:, :, 0:3] = orig_rotated
                    input_orig[:, :, 3:6] = (orig[:, :, 0:3] > 0).astype(np.float32)

                else:
                    patch_rotated = rotate_random(img_patches, random.randint(0, 90))

                    input_mos4 = mosaic(patch_rotated, 4)
                    input_mos3 = mosaic(patch_rotated, 3)

                    input_orig[:, :, 0:3] = patch_rotated
                    input_orig[:, :, 3:6] = (orig[:, :, 0:3] > 0).astype(np.float32)

            else:

                if self.noise:

                    input_mos4 = mosaic(noisy_patches, 4)
                    input_mos3 = mosaic(noisy_patches, 3)
                    input_orig = img_patches

                else:
                    input_mos4 = mosaic(img_patches, 4)
                    input_mos3 = mosaic(img_patches, 3)
                    input_orig = img_patches

            return [input_mos4, input_mos3], input_orig

        except Exception as e:
            print('Generator error: ' + str(e))
            self.file_errors.append(img_path)
            logging.warning('Generator Error: ' + str(e) + ' Image: ' + img)
            return [np.zeros((batch_size, patch_size, patch_size, 4)),
                    np.zeros((batch_size, patch_size * 2, patch_size * 2, 3))], np.zeros(
                (batch_size, patch_size * 2, patch_size * 2, 3))


# Display image with opencv
def show_image(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Generate UNET friendly input (for downsampling/upsampling evenly)
def get_unet_input(array):
    y = []
    h, w, _ = array.shape
    for x in range(0, max(h, w)):
        x1 = x
        if x % 32 == 0:
            y.append(x1)

    h1 = max(int(t) for t in y if h >= t)
    w1 = max(int(t) for t in y if w >= t)
    return array[:h1, :w1:, :]


# random extraction of image patches
def extractPatches(im, patch_size, max_patches=96):
    return image.extract_patches_2d(im, (patch_size * 2, patch_size * 2), max_patches)


# Random rotation of image patches
def rotate_random(img, x):
    if len(img.shape) > 3:
        _, rows, cols, _ = img.shape
    else:
        rows, cols, _ = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), x, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


# Batch application of the Bayer pattern to full colour image (mosaic) - return various formats
def mosaic(im_ptchs, num_channels=4):
    if num_channels == 4:
        b = im_ptchs[:, ::2, ::2, 0]
        # print(g1.shape)
        g1 = im_ptchs[:, ::2, 1::2, 1]
        # print(r.shape)
        r = im_ptchs[:, 1::2, 1::2, 2]
        # print(g2.shape)
        g2 = im_ptchs[:, 1::2, ::2, 1]
        # print(b.shape)
        out = np.stack((g1, b, g2, r), -1)
        # print("Mos4 Shape:" + str(out.shape))
        return out

    elif num_channels == 9:
        n, h, w, c = im_ptchs.shape

        b = np.zeros((n, h, w))
        g = np.zeros((n, h, w))
        r = np.zeros((n, h, w))

        g[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 1]

        b[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 0]

        r[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 2]

        g[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 1]

        out = np.stack((b, g, r), -1)

        return out

    elif num_channels == 3:
        n, h, w, c = im_ptchs.shape

        b = np.zeros((n, h, w))
        g = np.zeros((n, h, w))
        r = np.zeros((n, h, w))

        g[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 1]

        b[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 0]

        r[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 2]

        g[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 1]

        # out = np.stack((b,g,r),-1), im_ptchs
        out = np.stack((b, g, r), -1)

        return out

    elif num_channels == 6:
        n, h, w, c = im_ptchs.shape

        b = np.zeros((n, h, w))
        g = np.zeros((n, h, w))
        r = np.zeros((n, h, w))
        bm = np.zeros((n, h, w))
        gm = np.zeros((n, h, w))
        rm = np.zeros((n, h, w))

        g[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 1]
        # print(b.shape)
        b[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 0]
        r[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 2]
        # print(g.shape)
        g[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 1]
        # print(r.shape)
        gm[:, 1::2, ::2] = 1
        # print(b.shape)
        bm[:, ::2, ::2] = 1
        rm[:, 1::2, 1::2] = 1
        # print(g.shape)
        gm[:, ::2, 1::2] = 1

        return np.stack((b, g, r, bm, gm, rm), -1), im_ptchs

    elif num_channels == 1:
        n, h, w, c = im_ptchs.shape

        bgr = np.zeros((n, h, w))

        bgr[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 1]
        # print(b.shape)
        bgr[:, ::2, ::2] = im_ptchs[:, ::2, ::2, 0]
        bgr[:, 1::2, 1::2] = im_ptchs[:, 1::2, 1::2, 2]
        # print(g.shape)
        bgr[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 1]

        return bgr, im_ptchs

    elif num_channels == 0:
        n, h, w, c = im_ptchs.shape

        g = np.zeros((n, h, w))

        g[:, 1::2, ::2] = im_ptchs[:, 1::2, ::2, 1]
        g[:, ::2, 1::2] = im_ptchs[:, ::2, 1::2, 1]

        return g, im_ptchs[:, :, :, 1]

    return 0


# Application of the Bayer pattern to full colour image (mosaic) - return various formats
def mosaic1(im_ptchs, num_channels=4):
    # print("Input Dims: ",str(im_ptchs.shape))
    if num_channels == 4:
        b = im_ptchs[::2, ::2, 0]
        # print(g1.shape)
        g1 = im_ptchs[::2, 1::2, 1]
        # print(r.shape)
        r = im_ptchs[1::2, 1::2, 2]
        # print(g2.shape)
        g2 = im_ptchs[1::2, ::2, 1]
        # print(b.shape)
        out = np.stack((g1, b, g2, r), -1)
        # print("Mos4 Shape:" + str(out.shape))
        return out

    elif num_channels == 3:

        h, w, c = im_ptchs.shape

        b = np.zeros((h, w))

        g = np.zeros((h, w))

        r = np.zeros((h, w))

        g[1::2, ::2] = im_ptchs[1::2, ::2, 1]

        b[::2, ::2] = im_ptchs[::2, ::2, 0]

        r[1::2, 1::2] = im_ptchs[1::2, 1::2, 2]

        g[::2, 1::2] = im_ptchs[::2, 1::2, 1]

        # out = np.stack((b,g,r),-1), im_ptchs

        out = np.stack((b, g, r), -1)

        return out

    elif num_channels == 6:

        h, w, c = im_ptchs.shape

        b = np.zeros((h, w))

        g = np.zeros((h, w))

        r = np.zeros((h, w))

        bm = np.zeros((h, w))

        gm = np.zeros((h, w))

        rm = np.zeros((h, w))

        g[1::2, ::2] = im_ptchs[1::2, ::2, 1]

        # print(b.shape)

        b[::2, ::2] = im_ptchs[::2, ::2, 0]

        r[1::2, 1::2] = im_ptchs[1::2, 1::2, 2]

        # print(g.shape)

        g[::2, 1::2] = im_ptchs[::2, 1::2, 1]

        # print(r.shape)

        gm[1::2, ::2] = 1

        # print(b.shape)

        bm[::2, ::2] = 1

        rm[1::2, 1::2] = 1

        # print(g.shape)

        gm[::2, 1::2] = 1

        return np.stack((b, g, r, bm, gm, rm), -1), im_ptchs

    elif num_channels == 1:
        h, w, c = im_ptchs.shape

        bgr = np.zeros((h, w))

        bgr[1::2, ::2] = im_ptchs[1::2, ::2, 1]
        # print(b.shape)
        bgr[::2, ::2] = im_ptchs[::2, ::2, 0]
        bgr[1::2, 1::2] = im_ptchs[1::2, 1::2, 2]
        # print(g.shape)
        bgr[::2, 1::2] = im_ptchs[::2, 1::2, 1]

        return bgr, im_ptchs

    elif num_channels == 0:
        h, w, c = im_ptchs.shape

        g = np.zeros((h, w))
        g[1::2, ::2] = im_ptchs[1::2, ::2, 1]
        g[::2, 1::2] = im_ptchs[::2, 1::2, 1]

        return g

    return 0

# Rescale pixel values
def normalise1(array, scale=0):
    if scale == 0:
        return (array / 255).astype(np.float32)
    elif scale == -1:
        return ((array / 127.5) - 1).astype(np.float32)
    else:
        return -1


# Rescale pixel values
def denormalise1(array, scale=0):
    if scale == 0:
        im = (array * 255).astype(np.uint8)
        im[im > 255] = 255
        return im
    elif scale == -1:
        im = ((array + 1) * 127.5).astype(np.uint8)
        im[im > 255] = 255
        return im
    else:
        return -1


# Add original bayer pattern information to predicted image
def addBayer(orig, predicted):
    predicted[::2, ::2, 0] = orig[::2, ::2, 0]

    predicted[::2, 1::2, 1] = orig[::2, 1::2, 1]

    predicted[1::2, 1::2, 2] = orig[1::2, 1::2, 2]

    predicted[1::2, ::2, 1] = orig[1::2, ::2, 1]

    print("Bayer Difference:", str(np.average(abs(predicted - orig))))

    return predicted


# clip pixel values (tensor values through Keras backend)
def clipper(x):
    x = K.clip(x, -1, 1)
    return x


# VGG perceptual loss - MSE of throughput of trained VGG
def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


# SSIM loss - NOT FUNCTIONAL
def loss_SSIM(y_true, y_pred):
    y_true = tf.transpose(y_true, [0, 3, 1, 2])
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])

    u_true = K.mean(y_true, axis=3)
    u_pred = K.mean(y_pred, axis=3)
    var_true = K.var(y_true, axis=3)
    var_pred = K.var(y_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    return K.mean(((1.0 - ssim) / 2))

# MSE loss in YUV colour space - incorporates rotation loss
def loss_yuv(y_true, y_pred):
    bgr2yuv = K.constant([[0.11400, 0.436, -0.1],
                          [0.58700, -0.289, -0.515],
                          [0.29900, -0.147, 0.615]], name='bgr2yuv')

    y_true_yuv = K.dot(y_true[:, :, :, 0:3], bgr2yuv)
    y_pred_yuv = K.dot(y_pred, bgr2yuv)

    # rotation mask
    if K.int_shape(y_true)[2] is not None:
        if K.int_shape(y_true)[2] > 3:
            return K.mean(K.square(y_pred_yuv - y_true_yuv) * y_true[:, :, :, 3:6], axis=-1)

    else:
        return K.mean(K.square(y_pred_yuv - y_true_yuv), axis=-1)


# Test custom loss
def loss_custom(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


# Helper for loss_fourier
def matlab_style_gauss2D(shape=(128, 128), sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma)) + 0.01
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return np.stack((h * 100, h * 100, h * 100), axis=-1)


# Loss in Fourier domain - NOT FUNCTIONAL
def loss_fourier():
    g_tensor = tf.convert_to_tensor(matlab_style_gauss2D(), dtype=tf.float32)

    bgr2yuv = K.constant([[0.11400, 0.436, -0.1],
                          [0.58700, -0.289, -0.515],
                          [0.29900, -0.147, 0.615]], dtype=tf.float32, name='bgr2yuv')

    def loss_func(y_true, y_pred):
        y_true_yuv = K.dot(y_true, bgr2yuv)
        y_true_windowed = tf.multiply(y_true_yuv - K.mean(y_true_yuv), g_tensor)
        yt = tf.cast(y_true_windowed, tf.complex64)
        yt_fmag = tf.abs(tf.spectral.fft2d(yt))

        y_pred_yuv = K.dot(y_pred, bgr2yuv)
        y_pred_windowed = tf.multiply(y_pred_yuv - K.mean(y_pred_yuv), g_tensor)
        yp = tf.cast(y_pred_windowed, tf.complex64)
        yp_fmag = tf.abs(tf.spectral.fft2d(yp))

        return K.mean(K.square(yp_fmag - yt_fmag), axis=-1)

    return loss_func


# Generate input suitable for UNET
def generate_model_input(im, model='UNET'):

    if model == 'UNET':
        im = get_unet_input(im)

        mos4 = np.array([mosaic1(im, 4)])
        mos3 = np.array([mosaic1(im, 3)])

        return [normalise1(mos4), normalise1(mos3)], normalise1(im)

    elif model == 'Chang':

        mos3 = np.array([mosaic1(im, 3)])

        return normalise1(mos3, scale=0), normalise1(im, scale=0)

    else:

        return -1


def predict_generator_rgb(train_dir, model_name):
    """
    Python generator that loads imgs and batches
    """

    while True:
        img_name = os.listdir(train_dir)

        for img in enumerate(img_name):

            sc = os.path.join(train_dir, img[1])
            # print(sc)

            try:
                if model_name == 'UNET':
                    im = cv2.imread(sc)

                    im = get_unet_input(im)

                    mos4 = np.array([mosaic1(im, 4)])
                    mos3 = np.array([mosaic1(im, 3)])

                    yield [normalise1(mos4), normalise1(mos3)], normalise1(im), img[1]

                elif model_name == 'Chang':
                    im = cv2.imread(sc)

                    mos3 = np.array([mosaic1(im, 3)])

                    yield [normalise1(mos3, scale=0)], normalise1(im, scale=0), img[1]

                else:
                    return -1

            except Exception as e:
                print('file open error: ' + str(e))


# def train_generator_rgb(train_dir, patch_size, batch_size, scale, num_patches):
#     """
#     Python generator that loads imgs and batches
#     """
#
#     while True:
#         img_name = os.listdir(train_dir)
#
#         input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
#         input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#         input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#
#         for img in enumerate(img_name):
#
#             sc = os.path.join(train_dir, img[1])
#             # print(sc)
#             try:
#                 im = cv2.imread(sc)
#                 x, y, _ = np.shape(im)
#                 fx = int(x / scale)
#                 fy = int(y / scale)
#                 # im = cv2.resize(im, (fy, fx), interpolation = cv2.INTER_LANCZOS4)
#
#                 ptch = extractPatches(im, patch_size, num_patches)
#                 # print(img_ptch.shape)
#
#                 ptch_mos4 = mosaic(ptch, 4)
#                 ptch_mos3 = mosaic(ptch, 3)
#
#                 num_ptchs = ptch_mos4.shape[0]
#                 # print(num_ptchs)
#                 btchs = int(num_ptchs / batch_size)
#                 # print(str(btchs))
#                 for i in range(btchs):
#                     b = i * batch_size
#
#                     input_mos4 = ptch_mos4[b:b + batch_size]
#
#                     input_mos3 = ptch_mos3[0][b:b + batch_size]
#
#                     input_orig = ptch[b:b + batch_size]
#
#                     yield [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)
#
#             except Exception as e:
#                 print('file open error: ' + str(e))
#
#
# def train_generator_rgb_patches(train_dir, patch_size, batch_size, input_channels=4):
#     """
#     Python generator that loads imgs and batches
#     """
#
#     if (input_channels == 4):
#         input_mos = np.zeros((batch_size, patch_size, patch_size, 4))
#         input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#     else:
#         input_mos = np.zeros((batch_size, patch_size, patch_size, input_channels))
#         input_orig = np.zeros((batch_size, patch_size, patch_size, 3))
#     read = 0
#
#     while True:
#         img_name = os.listdir(train_dir)
#
#         for img in enumerate(img_name):
#
#             # print(img[0])
#             sc = os.path.join(train_dir, img[1])
#
#             try:
#                 ptch = cv2.imread(sc)
#
#                 ptch_mos = mosaic1(ptch, input_channels)
#
#                 input_mos[read] = ptch_mos
#                 input_orig[read] = ptch
#
#                 read += 1
#
#                 if (read == batch_size):
#                     yield normalise1(input_mos), normalise1(input_orig)
#                     read = 0
#
#             except Exception as e:
#                 print('file open error: ' + str(e) + str(img[1]))
#
#
# def train_generator_rgb_tiled(train_dir, patch_size, batch_size, input_channels=4):
#     """
#     Python generator that loads imgs and batches
#     """
#
#     while True:
#         img_name = os.listdir(train_dir)
#         n_dim = 128
#         m_dim = 64
#         tile_n = 8
#         input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
#         input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#         input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#         num_btchs = int(64 / batch_size)
#
#         for img in enumerate(img_name):
#
#             sc = os.path.join(train_dir, img[1])
#             # print(sc)
#             try:
#                 ptch = cv2.imread(sc)
#
#                 ptch_mos4 = mosaic1(ptch, 4)
#                 ptch_mos3 = mosaic1(ptch, 3)
#
#                 x = 0
#                 for i in range(tile_n):
#                     for j in range(tile_n):
#                         i_dim = n_dim * i
#                         j_dim = n_dim * j
#
#                         im_dim = m_dim * i
#                         jm_dim = m_dim * j
#
#                         input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
#                         input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
#                         input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
#                         x += 1
#                         if (x % batch_size == 0):
#                             x = 0
#                             yield [normalise1(input_mos4), normalise1(input_mos3)], normalise1(input_orig)
#
#             except Exception as e:
#                 print('file open error: ' + str(e))
#
#
# def train_generator_ad(train_dir, patch_size, batch_size, gen_model):
#     while True:
#         print("CUDA Devices ", os.environ["CUDA_VISIBLE_DEVICES"])
#         img_name = os.listdir(train_dir)
#         n_dim = 128
#         m_dim = 64
#         tile_n = 8
#         input_mos4 = np.zeros((batch_size, patch_size, patch_size, 4))
#         input_mos3 = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#         input_orig = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3))
#         num_btchs = int(64 / batch_size)
#
#         for img in enumerate(img_name):
#
#             sc = os.path.join(train_dir, img[1])
#             # print(sc)
#             try:
#                 ptch = cv2.imread(sc)
#
#                 ptch_mos4 = mosaic1(ptch, 4)
#                 ptch_mos3 = mosaic1(ptch, 3)
#
#                 x = 0
#                 for i in range(tile_n):
#                     for j in range(tile_n):
#                         i_dim = n_dim * i
#                         j_dim = n_dim * j
#
#                         im_dim = m_dim * i
#                         jm_dim = m_dim * j
#
#                         input_orig[x, :, :, :] = ptch[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
#                         input_mos4[x, :, :, :] = ptch_mos4[im_dim:im_dim + m_dim, jm_dim:jm_dim + m_dim, :]
#                         input_mos3[x, :, :, :] = ptch_mos3[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :]
#                         x += 1
#                         if x % batch_size == 0:
#                             gen_btch = gen_model.predict_on_batch([normalise1(input_mos4), normalise1(input_mos3)])
#                             gen_label = [0] * batch_size
#
#                             orig_label = [1] * batch_size
#                             x = 0
#
#                             yield np.concatenate((gen_btch, input_orig), axis=0), gen_label + orig_label
#
#             except Exception as e:
#                 print('file open error: ' + str(e))
#
#
# def val_generator_rgb(train_dir):
#     while True:
#         img_name = os.listdir(train_dir)
#
#         for img in enumerate(img_name):
#
#             sc = os.path.join(train_dir, img[1])
#             # print(sc)
#
#             try:
#                 im = cv2.imread(sc)
#
#                 im = get_unet_input(im)
#
#                 mos4 = np.array([mosaic1(im, 4)])
#                 mos3 = np.array([mosaic1(im, 3)])
#                 # print(normalise1(mos).shape)
#
#                 yield [normalise1(mos4), normalise1(mos3)], normalise1(np.array([im]))
#                 # print('Batched Input')
#
#             except Exception as e:
#                 print('file open error: ' + str(e))
#
#
#
#
#
# def predict_generator(model, k_gen, steps, data, bp, save_file):
#     psnr = []
#     ssim = []
#
#     for i in range(steps):
#         p_img_gen, orig_gen, img_name = next(k_gen)
#
#         pred_img = (model.predict(p_img_gen, batch_size=1))[0]
#         pred_img[(pred_img > 1)] = 1
#         # pred_img = denormalise1(pred_img)
#         # print("Output max:", str(pred_img.min()))
#         # print("Output min", str(pred_img.max()))
#         print("Predicted Model Shape: " + str(pred_img.shape))
#         # cv2.imshow('image',pred_img)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#
#         pred_img = addBayer(orig_gen, pred_img)
#
#         pred_psnr = skimage.measure.compare_psnr(pred_img[bp:-bp, bp:-bp, :], orig_gen[bp:-bp, bp:-bp, :], 1)
#         print(img_name + " PSNR: " + str(pred_psnr))
#
#         cv2.imshow("Original " + img_name, orig_gen)
#         cv2.imshow("Difference " + img_name, (np.floor((abs(orig_gen - pred_img)) * 255)).astype(np.uint8) + 100)
#         cv2.imshow("Predicted " + img_name, pred_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#         cv2.imwrite(os.path.join(save_file, ('predicted_' + img_name)), denormalise1(pred_img))
#
#         psnr.append(pred_psnr)
#         ssim.append(
#             skimage.measure.compare_ssim(pred_img[bp:-bp, bp:-bp, :], orig_gen[bp:-bp, bp:-bp, :], multichannel=True))
#
#     psnr_avg = sum(psnr) / steps
#     ssim_avg = sum(ssim) / steps
#     print('Kodak PSNR Average: ' + str(psnr_avg))
#     print('Kodak SSIM Average: ' + str(ssim_avg))
#
#     return psnr, ssim, psnr_avg, ssim_avg
#
#
# def predict_chal_patchs(model, psnr_threshold, k_gen, steps, save_file):
#     target = []
#     num_btch = 0
#     chlng = 0
#     test_chlng = 0
#
#     for i in range(steps):
#
#         print(str(i) + " of " + str(steps))
#
#         input_ptchs, target_patchs = next(k_gen)
#
#         pred_btch = model.predict_on_batch(input_ptchs)
#
#         # print(target_patchs.dtype)
#         # print(pred_btch.dtype)
#
#         for j in range(pred_btch.shape[0]):
#             pred_psnr = skimage.measure.compare_psnr(pred_btch[j], target_patchs[j])
#             test_chlng += 1
#
#             if pred_psnr < psnr_threshold:
#                 print(pred_psnr)
#                 target.append(target_patchs[j])
#                 chlng += 1
#
#             if len(target) > 63:
#                 chnlg_target = np.stack(target, 0)
#                 tile_image(chnlg_target, save_file, str(num_btch))
#
#                 target = []
#
#                 num_btch += 1
#
#     return [chlng, test_chlng]
#
#
# def predict_chal_patchs_1(model, psnr_threshold, patch_size, data_dir, save_file, save_name):
#
#     model = keras.models.load_model(model)
#
#     imgs = os.listdir(data_dir)
#
#     write_idx = 0
#     for img in imgs:
#         # print(str(i) + " of " + str(len(imgs)))
#
#         img_n = cv2.imread(os.path.join(data_dir, img))
#
#         img_n = cv2.resize(img_n, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#         h, w, c = img_n.shape
#
#         i = j = 0
#         while (h - j) > 128:
#             while (w - i) > 128:
#
#                 img_patch = img_n[j:j + 128, i:i + 128, :]
#                 img_input = generate_model_input(img_patch, model='UNET')
#                 pred_img_patch = (model.predict(img_input[0], batch_size=1))[0]
#                 pred_psnr = skimage.measure.compare_psnr(img_input[1], pred_img_patch)
#                 print(pred_psnr)
#
#                 i += 64
#
#                 if (pred_psnr <= 40):
#                     cv2.imwrite(os.path.join(save_file, save_name + '_{}.png'.format(write_idx)), img_patch)
#                     write_idx += 1
#                     print('Written ', save_name + '_{}.png'.format(write_idx))
#
#             j += 64
#     print('Written: ', str(write_idx))
#     return
#
#
# def tile_image(target, save_dir, s_name):
#     n_dim = 128
#     tile_n = 8
#
#     im_tile = np.zeros((n_dim * tile_n, n_dim * tile_n, 3))
#
#     x = 0
#
#     print("Tiling ")
#     for i in range(tile_n):
#         for j in range(tile_n):
#             im = target[x]
#             # cv2.imshow(im_name[x], im)
#             i_dim = n_dim * i
#             j_dim = n_dim * j
#             im_tile[i_dim:i_dim + n_dim, j_dim:j_dim + n_dim, :] = im
#             x += 1
#             cv2.imwrite(os.path.join(save_dir, os.path.split(save_dir)[1] + s_name + '.png'), denormalise1(im_tile))
#
#     # cv2.imshow("Tiled ", denormalise1(im_tile))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return
