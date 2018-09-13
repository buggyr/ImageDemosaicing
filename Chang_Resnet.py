import datetime
import os
import model_train
import models

# Directories
epoch_cb_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Epoch_Callback'
parent_models_dir = '/home/rhys/Demosaicing_Training/Models'
# training_data_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Flickr500_Challenging_Patches'
training_data_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Gharbi_hdrvdp'
test_dirs = '/home/rhys/Demosaicing_Training/Data/Test_Data'
validation_data_dir = '/home/rhys/Demosaicing_Training/Data/Training_Data/Flickr500_Challenging_Patches'


# # Model Directory
# key_name = "_Chang_RESNET_patch_64x64_kernel_3x3_mse_loss"
# now = datetime.datetime.now()
# model_dir = os.path.join(parent_models_dir, now.strftime("%Y-%m-%d %H-%M") + key_name)
# os.mkdir(model_dir)
model_dir = '/home/rhys/Demosaicing_Training/Models/2018-06-05 18-29_Chang_RESNET_patch_64x64_kernel_3x3_mse_loss'
model_path = os.path.join(model_dir, 'interrupt_model.h5')

# # Make Model
# model = models.ChangRESNET(model_dir=model_dir, num_layers=15, kernel_size=3).build_model()
model = models.load_model(model_path)


# Train Model
chang_trainer = model_train.ModelTrain(model=model, train_dir=training_data_dir, val_dir=validation_data_dir, model_dir=model_dir,
                                       epochs=50, epoch_predict_dir=epoch_cb_dir, initial_epoch=4, new=False)
model, history = chang_trainer.model_train(patch_size=64, batch_size=16)


# Test Model
chang_test = model_train.ModelTest(model_dir=model_dir, model=model, test_dirs=test_dirs, model_type='Chang')
chang_test.test_model()

# # Eval_Model
#
# chang_eval = model_train.ModelEval(model_dir=model_dir, test_dir=test_dirs, model_type='Chang')
# chang_eval.test_model()

