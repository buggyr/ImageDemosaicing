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
key_name = "_RESNET_patch_64x64_kernel_5x5_mse_loss_2_input_8_mb"
now = datetime.datetime.now()
model_dir = os.path.join(parent_models_dir, now.strftime("%Y-%m-%d %H-%M") + key_name)
os.mkdir(model_dir)


# Make Model
model = models.RESNET().build_model()
# model = models.load_model(model_path)


# Train Model
resnet_trainer = model_train.ModelTrain(model=model, train_dir=training_data_dir, val_dir=validation_data_dir, model_dir=model_dir, epochs=50,
                                        epoch_predict_dir=epoch_cb_dir)
model, history = resnet_trainer.model_train(patch_size=64, batch_size=32)


# Test Model
resnet_test = model_train.ModelTest(model_dir=model_dir, model=model, test_dirs=test_dirs, model_type='UNET')
resnet_test.test_model()
