# Model Train and Test
import train_UNET_2
import keras
import cv2
import skimage.measure
from keras.models import *


# GPU selection
GPU = '1'
if GPU == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU


class ModelTrain:
    def __init__(self, model, train_dir, val_dir, model_dir, epochs, epoch_predict_dir, new=True, initial_epoch=0):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.model_dir = model_dir
        self.epochs = epochs
        self.epoch_predict_dir = epoch_predict_dir
        self.model_dir = model_dir
        self.model = model
        self.initial_epoch = initial_epoch

        ep_models_folder = os.path.join(model_dir, 'Epoch_Models')
        self.ep_models_folder = ep_models_folder

        ep_predictions_folder = os.path.join(model_dir, 'Epoch_Predictions')
        self.ep_predictions_folder = ep_predictions_folder

        tnsr_brd_folder = os.path.join(model_dir, 'TNSR_BRD')
        self.tnsr_brd_folder = tnsr_brd_folder

        if new:
            os.mkdir(ep_models_folder)
            os.mkdir(ep_predictions_folder)
            os.mkdir(tnsr_brd_folder)

        self.k_callbacks = []

    def callbacks(self):
        tbCallBack = keras.callbacks.TensorBoard(log_dir=self.tnsr_brd_folder, histogram_freq=0,
                                                 write_graph=True, write_images=True, write_grads=True)
        csv_logger = keras.callbacks.CSVLogger(os.path.join(self.model_dir, 'training.log'),
                                               separator=',', append=False)
        epoch_predict = train_UNET_2.SavePredictions(self.ep_predictions_folder, self.epoch_predict_dir)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(self.ep_models_folder, 'model.{epoch:02d}.hdf5'), monitor='loss')

        self.k_callbacks = [tbCallBack, csv_logger, epoch_predict, model_checkpoint]

    def model_train(self, batch_size=32, patch_size=64, rotate=False, noise=False):

        self.callbacks()
        history = None

        if self.model.name == 'UNET':
            train_generator = train_UNET_2.TrainSeq2Input(self.train_dir, patch_size, batch_size,
                                                          save_file=self.model_dir, rotate=rotate, noise=noise)
            validation_generator = train_UNET_2.TrainSeq2Input(self.val_dir, patch_size, batch_size,
                                                               save_file=self.model_dir, rotate=rotate, noise=noise)
        elif self.model.name == 'Chang':
            train_generator = train_UNET_2.TrainSeq1Input(self.train_dir, patch_size, batch_size,
                                                          save_file=self.model_dir, rotate=rotate)
            validation_generator = train_UNET_2.TrainSeq1Input(self.val_dir, patch_size, batch_size,
                                                               save_file=self.model_dir, rotate=rotate)
        else:
            return -1

        try:

            history = self.model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                               epochs=self.epochs, callbacks=self.k_callbacks,
                                               workers=8, use_multiprocessing=True, initial_epoch=self.initial_epoch)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping early and Saving Out.")
            self.model.save(os.path.join(self.model_dir, 'interrupt_model.h5'))
            return self.model, history

        self.model.save(os.path.join(self.model_dir, 'final_model.h5'))

        return self.model, history


class ModelTest:

    def __init__(self, model_dir, model, test_dirs, model_type, test_border=5):

        self.model_dir = model_dir
        self.model = model
        self.test_dirs = test_dirs
        self.model_type = model_type
        self.tb = test_border

        self.parent_test_dir = test_dirs
        self.test_dirs = os.listdir(test_dirs)

        model_test_folder = os.path.join(model_dir, 'Model_Test')
        if not os.path.exists(model_test_folder):
            os.mkdir(model_test_folder)
        self.model_test_folder = model_test_folder

    def test_model(self):

        data = []

        for test_dir in self.test_dirs:

            sub_test_dir = os.path.join(self.parent_test_dir, test_dir)
            test_imgs = sorted(os.listdir(sub_test_dir))
            psnr = []
            cpsnr = []
            ssim = []

            for idx, img in enumerate(test_imgs):
                img_name, img_ext = os.path.splitext(img)

                im = cv2.imread(os.path.join(sub_test_dir, img))

                img_input, img_target = train_UNET_2.generate_model_input(im, self.model_type)

                pred_img = (self.model.predict(img_input, batch_size=1))[0]
                # print('Range -1:1: ', 'Max: {}'.format(pred_img.max()), 'Min: {}'.format(pred_img.min()))
                pred_img[(pred_img > 1)] = 1

                # pred_img = denormalise1(pred_img)
                # print("Output max:", str(pred_img.min()))
                # print("Output min", str(pred_img.max()))
                # print("Predicted Model Shape: " + str(pred_img.shape))
                # cv2.imshow('image',pred_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                pred_img = train_UNET_2.addBayer(img_target, pred_img)

                pred_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, :], 1)
                blue_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 0],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, 0], 1)
                green_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 1],
                                                          img_target[self.tb:-self.tb, self.tb:-self.tb, 1], 1)
                red_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 2],
                                                        img_target[self.tb:-self.tb, self.tb:-self.tb, 2], 1)
                print(img + " PSNR: " + str(pred_psnr))

                # cv2.imshow("Original " + img_name, img_target)
                # cv2.imshow("Difference " + img_name,
                #            (np.floor((abs(img_target - pred_img)) * 255)).astype(np.uint8) + 100)
                # cv2.imshow("Predicted " + img_name, pred_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                pred_n = train_UNET_2.denormalise1(pred_img)
                # print('Range 0:255: ', 'Max: {}'.format(pred_n.max()), 'Min: {}'.format(pred_n.min()))

                cv2.imwrite(os.path.join(self.model_test_folder, ('predicted_' + img)),
                            train_UNET_2.denormalise1(pred_img))

                cv2.imwrite(os.path.join(self.model_test_folder, ('difference' + img)),
                            (np.floor((abs(img_target - pred_img)) * 255)).astype(np.uint8) + 100)

                psnr.append(pred_psnr)
                cpsnr.append([blue_psnr, green_psnr, red_psnr])
                ssim.append(skimage.measure.compare_ssim(pred_img[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         multichannel=True))

            # imgs, psnrs = zip(*psnr)
            # imgs, ssims = zip(*ssim)
            # imgs, cpsnrs = zip(*cpsnr)
            # bpsnr, gpsnr, rpsnr = zip(*cpsnrs)
            psnr_avg = sum(psnr) / len(psnr)
            ssim_avg = sum(ssim) / len(psnr)
            bpsnr, gpsnr, rpsnr = zip(*cpsnr)
            bpsnr_avg = sum(bpsnr) / len(bpsnr)
            gpsnr_avg = sum(gpsnr) / len(gpsnr)
            rpsnr_avg = sum(rpsnr) / len(rpsnr)
            cpsnr_avg = [bpsnr_avg, gpsnr_avg, rpsnr_avg]
            print(bpsnr_avg)
            print(cpsnr_avg)

            print(test_dir, ' PSNR Average: ' + str(psnr_avg))
            print(test_dir, ' SSIM Average: ' + str(ssim_avg))

            data.append({test_dir + ' Results':
                [
                    {'psnr': dict(zip(test_imgs, psnr))},
                    {'ssim': dict(zip(test_imgs, ssim))},
                    {'average_psnr': psnr_avg},
                    {'average _ssim': ssim_avg},
                    {'cpsnr': dict(zip(test_imgs, cpsnr))},
                    {'average_cpsnr': [bpsnr_avg, gpsnr_avg, rpsnr_avg]},
                ]
            })

            data.append({'Model Training Parameters': {
                'Loss Function': str(self.model.loss),
                'Optimizer': str(self.model.optimizer)
            }
            })

        with open(os.path.join(self.model_dir, 'Results.txt'), 'w') as outfile:
            json.dump(data, outfile, indent=4)


class ModelEval:

    def __init__(self, model_dir, model_type, test_dir, test_border=5):

        self.model_dir = model_dir
        self.model_type = model_type
        self.tb = test_border

        ep_models_dir = os.path.join(model_dir, 'Epoch_Models')
        models = os.listdir(ep_models_dir)
        latest_epoch_model = models[len(models) - 1]

        self.model = keras.models.load_model(os.path.join(ep_models_dir, latest_epoch_model))
        if self.model.name == 'Chang':
            self.scale = -1
        else:
            self.scale = 0

        self.parent_test_dir = test_dir
        self.test_dirs = os.listdir(test_dir)

        model_test_folder = os.path.join(model_dir, 'Model_Test')
        if not os.path.exists(model_test_folder):
            os.mkdir(model_test_folder)
        self.model_test_folder = model_test_folder

    def test_model(self):

        data = []

        for test_dir in self.test_dirs:

            sub_test_dir = os.path.join(self.parent_test_dir, test_dir)
            test_imgs = sorted(os.listdir(sub_test_dir))
            psnr = []
            cpsnr = []
            ssim = []

            for idx, img in enumerate(test_imgs):
                img_name, img_ext = os.path.splitext(img)

                im = cv2.imread(os.path.join(sub_test_dir, img))

                img_input, img_target = train_UNET_2.generate_model_input(im, self.model_type)

                pred_img = (self.model.predict(img_input, batch_size=1))[0]
                pred_img[(pred_img > 1)] = 1

                # pred_img = denormalise1(pred_img)
                # print("Output max:", str(pred_img.min()))
                # print("Output min", str(pred_img.max()))
                # print("Predicted Model Shape: " + str(pred_img.shape))
                # cv2.imshow('image',pred_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                pred_img = train_UNET_2.addBayer(img_target, pred_img)

                pred_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, :], 1)
                blue_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 0],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, 0], 1)
                green_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 1],
                                                          img_target[self.tb:-self.tb, self.tb:-self.tb, 1], 1)
                red_psnr = skimage.measure.compare_psnr(pred_img[self.tb:-self.tb, self.tb:-self.tb, 2],
                                                        img_target[self.tb:-self.tb, self.tb:-self.tb, 2], 1)
                print(img + " PSNR: " + str(pred_psnr))

                # cv2.imshow("Original " + img_name, img_target)
                # cv2.imshow("Difference " + img_name,
                #            (np.floor((abs(img_target - pred_img)) * 255)).astype(np.uint8) + 100)
                # cv2.imshow("Predicted " + img_name, pred_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                pred_n = train_UNET_2.denormalise1(pred_img)
                # print('Range 0:255: ', 'Max: {}'.format(pred_n.max()), 'Min: {}'.format(pred_n.min()))

                cv2.imwrite(os.path.join(self.model_test_folder, ('predicted_' + img)),
                            train_UNET_2.denormalise1(pred_img))

                cv2.imwrite(os.path.join(self.model_test_folder, ('difference' + img)),
                            (np.floor((abs(img_target - pred_img)) * 255)).astype(np.uint8) + 100)

                psnr.append(pred_psnr)
                cpsnr.append([blue_psnr, green_psnr, red_psnr])

                ssim.append(skimage.measure.compare_ssim(pred_img[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         img_target[self.tb:-self.tb, self.tb:-self.tb, :],
                                                         multichannel=True))

                # imgs, psnrs = zip(*psnr)
                # imgs, ssims = zip(*ssim)
                # imgs, cpsnrs = zip(*cpsnr)
                # bpsnr, gpsnr, rpsnr = zip(*cpsnrs)
            psnr_avg = sum(psnr) / len(psnr)
            ssim_avg = sum(ssim) / len(psnr)
            bpsnr, gpsnr, rpsnr = zip(*cpsnr)
            bpsnr_avg = sum(bpsnr) / len(bpsnr)
            gpsnr_avg = sum(gpsnr) / len(gpsnr)
            rpsnr_avg = sum(rpsnr) / len(rpsnr)
            cpsnr_avg = [bpsnr_avg, gpsnr_avg, rpsnr_avg]
            print(bpsnr_avg)
            print(cpsnr_avg)

            print(test_dir, ' PSNR Average: ' + str(psnr_avg))
            print(test_dir, ' SSIM Average: ' + str(ssim_avg))

            data.append({test_dir + ' Results':
                [
                    {'psnr': dict(zip(test_imgs, psnr))},
                    {'ssim': dict(zip(test_imgs, ssim))},
                    {'average_psnr': psnr_avg},
                    {'average _ssim': ssim_avg},
                    {'cpsnr': dict(zip(test_imgs, cpsnr))},
                    {'average_cpsnr': [bpsnr_avg, gpsnr_avg, rpsnr_avg]},
                ]
            })

            data.append({'Model Training Parameters': {
                'Loss Function': str(self.model.loss),
                'Optimizer': str(self.model.optimizer)
            }
            })

        with open(os.path.join(self.model_dir, 'Results.txt'), 'w') as outfile:
            json.dump(data, outfile, indent=4)
