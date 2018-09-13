import datetime
import os
import model_train
import models

# Directories
epoch_cb_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Epoch_Callback'
parent_models_dir = '/home/rhys/Demosaicing_Training/Models'
training_data_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Gharbi_hdrvdp'
test_dirs = '/home/rhys/Demosaicing_Training/Data/Test_Data'
validation_data_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Flickr500_Challenging_Patches'


# Model Directory
key_name = "_UNET_2_layer_residual_patch_64x64_kernel_3x3_mae_loss"
now = datetime.datetime.now()
model_dir = os.path.join(parent_models_dir, now.strftime("%Y-%m-%d %H-%M") + key_name)
os.mkdir(model_dir)
# model_dir='/home/rhys/Demosaicing_Training/Models/2018-05-25 18-16_UNET_2_layer_patch_64x64_kernel_7x7_mse_loss_2_input_noisy_train'


# Make Model
model = models.UNET(kernel_size=3, loss_func='mean_absolute_error').build_model()
# model = models.load_model('/home/rhys/Demosaicing_Training/Models/2018-05-25 '
#                           '18-16_UNET_2_layer_patch_64x64_kernel_7x7_mse_loss_2_input_noisy_train/interrupt_model.h5')
model.name = 'UNET'

# Train Model
unet_trainer = model_train.ModelTrain(model=model, train_dir=training_data_dir, val_dir=validation_data_dir,
                                      model_dir=model_dir, epochs=100, epoch_predict_dir=epoch_cb_dir, new=True,
                                      initial_epoch=0)

model, history = unet_trainer.model_train(patch_size=64, batch_size=32, rotate=False, noise=False)

# Test Model
unet_test = model_train.ModelTest(model_dir=model_dir, model=model, test_dirs=test_dirs, model_type='UNET')
unet_test.test_model()

# # Eval Model
# model_train.ModelEval(model_dir=model_dir, model_type='UNET', test_dir='/home/rhys/Demosaicing_Training/Data/Test_Data').test_model()
