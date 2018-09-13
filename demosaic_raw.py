import argparse
from subprocess import call
import time
import cv2
import numpy as np
import os
import keras
import train_UNET_2

os.environ["CUDA_VISIBLE_DEVICES"] = '5'


def get_inputs(raw, bayer_pattern='RGGB'):

    raw = np.rot90(raw)

    if bayer_pattern == 'RGGB':

        raw = train_UNET_2.get_unet_input(raw)
        print(raw.shape)
        h, w, _ = raw.shape

        b = np.zeros((h, w))

        g = np.zeros((h, w))

        r = np.zeros((h, w))

        g[1::2, ::2] = raw[1::2, ::2, 0]

        r[::2, ::2] = raw[::2, ::2, 0]

        b[1::2, 1::2] = raw[1::2, 1::2, 0]

        g[::2, 1::2] = raw[::2, 1::2, 0]

        mos3 = np.stack((b, g, r), -1)

        r = raw[::2, ::2, 0]

        g1 = raw[::2, 1::2, 0]

        b = raw[1::2, 1::2, 0]

        g2 = raw[1::2, ::2, 0]

        mos4 = np.stack((g1, b, g2, r), -1)

        return [train_UNET_2.normalise1(np.array([mos4])), train_UNET_2.normalise1(np.array([mos3]))]


def main(args):
    model_path = os.path.join(args.model)

    if args.gpu:
        print('Using GPU')
    else:
        print('Using CPU')

    if os.path.isdir(args.input):
        inputs = [f for f in os.listdir(args.input)]
        inputs = [os.path.join(args.input, f) for f in inputs]
    else:
        inputs = [args.input]

    print(args)
    avg_psnr = 0
    n = 0
    t1 = 0
    t2 = 0
    for fname in inputs:
        print('Processing {}'.format(fname))

        call(["dcraw -T -d -w -6 " + fname], shell=True)

        bayer = cv2.imread(fname.split('.')[0] + '.tiff')

        model_input = get_inputs(bayer, bayer_pattern='RGGB')

        model = keras.models.load_model(args.model, custom_objects={'loss_yuv': train_UNET_2.loss_yuv})

        t1 = time.time()

        pred_img = model.predict(model_input, batch_size=1)[0]

        t2 = time.time()

        cv2.imwrite(os.path.join(args.output, os.path.basename(fname).split('.')[0] + '.png'), train_UNET_2.denormalise1(pred_img))

    print('Complete in ', (t2-t1), ' seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='/home/rhys/Demosaicing_Training/Data/Raw_Images', help='path to input image or folder.')
    parser.add_argument('--output', type=str, default='/home/rhys/Demosaicing_Training/Data/Demosaiced_Raw_Images', help='path to output folder.')
    parser.add_argument('--model', type=str, default='/home/rhys/Demosaicing_Training/Models/2018-02-07 20-22_UNET_2_layer_64x64_logcosh_normal1_Patterns+Gharbi_2_input_strides+upsample/model.48-0.00.hdf5', help='path to trained model.')
    parser.add_argument('--tile_size', type=int, default=512, help='split the input into tiles of this size.')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='use the GPU for processing.')

    parser.set_defaults(gpu=False)

    args = parser.parse_args()

    main(args)