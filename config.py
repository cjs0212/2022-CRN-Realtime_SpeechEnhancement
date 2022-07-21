#################################################################################
#                               configuration                                   #
#################################################################################
BATCH_SIZE = 1
EPOCH = 10
start_epoch = 0
encoder_filter = [16, 32, 64]
# encoder_filter = [16, 32, 64, 128, 128, 128]
decoder_filter = [32, 16, 1]
# decoder_filter = [128, 128, 64, 32, 16, 1]
#################################################################################
#                                    STFT                                      #
#################################################################################
win_len = 512
stride = 128
fft_len = 512
win_type = 'hann'
fs = 16000

#################################################################################
#                                  Data Load                                    #
#################################################################################
# path_train = "./Dataset/SE-CRN-Train.npy" # train
# path_train = "./Dataset/SE-CRN-Train.npy" # train
path_train = "./Dataset/timit_SNR_3_11088.npy"  # train
path_validation = "./Dataset/SE-CRN-Validation.npy"  # validation
path_test = "./Dataset/test.npy"

import numpy as np

input_train = np.load(path_train)  # (300, 2, 48000)
# input_validation = np.load(path_validation)  # (100, 2, 48000)

noisy_train = input_train[:9000, 0]  # noisy
clean_train = input_train[:9000, 1]  # clean

noisy_validation = input_train[9000:10000, 0]  # noisy
clean_validation = input_train[9000:10000, 1]  # clean
#################################################################################
#                                    Path                                       #
#################################################################################
import time

file_name = "crn_valid_timit_E10_v2.h5"
saved_model_dir = './saved_model/'
saved_model = 'new_model.h5'
saved_model_path = saved_model_dir + saved_model
logdir = 'logs/'
cur_time = time.strftime('%Y-%m-%d-%H-%M-', time.localtime(time.time()))
logdir = logdir + cur_time + file_name
chpt_dir = 'models/'
chpt_file = '/checkpoint-{epoch:03d}.h5'
tflite_dir = './tflite'
tflite_file = tflite_dir + '/21_03_07_test.tflite'

#################################################################################
#                                  compile                                      #
#################################################################################
import tensorflow as tf

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
