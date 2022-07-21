import numpy as np
import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import config as cfg
import time
from scipy.signal import get_window
from keras.layers import Lambda
from keras import layers, Sequential, Input, Model
from tools_for_estimate import pesq
from tools_for_compile import CustomTensorBoard, reduce_lr, early_stopping
# from tools_for_optimize import print_model_weights_sparsity, print_model_weight_clusters
# from tensorflow_model_optimization.python.core.clustering.keras.experimental import (cluster, )
import os
import random


class CRN_model():
    def __init__(self):
        # self.model = []
        self.BATCH_SIZE = cfg.BATCH_SIZE
        self.EPOCH = cfg.EPOCH
        self.start_epoch = cfg.start_epoch
        self.noisy_train = cfg.noisy_train
        self.noisy_validation = cfg.noisy_validation
        self.clean_train = cfg.clean_train
        self.clean_validation = cfg.clean_validation
        self.optimizer = cfg.optimizer
        self.loss_fn = cfg.loss_fn
        self.pruned_model = []
        self.clusterd_model = []
        self.pcqat_model = []
        self.win_type = cfg.win_type
        self.feature_type = 'real'
        self.stride = cfg.stride  # 128
        self.win_len = cfg.win_len  # 512
        self.fft_len = cfg.fft_len  # 512
        self.kernel, self.window = self.init_kernels(self.win_len, self.fft_len, self.win_type)  # [400, 1, 514]
        self.enframe = tf.eye(self.win_len)[:, None, :]
        self.encoder_filter = cfg.encoder_filter
        self.decoder_filter = cfg.decoder_filter
        self.callback = []
        self.tensorboard_callback = CustomTensorBoard(validation_data=(cfg.noisy_validation, cfg.clean_validation),
                                                      log_dir=cfg.logdir,
                                                      batch_size=cfg.BATCH_SIZE)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.chpt_dir + cfg.chpt_file,
                                                                   save_weights_only=True,
                                                                   save_best_only=True)
        self.callback.append(self.tensorboard_callback)
        self.callback.append(self.model_checkpoint)
        self.callback.append(reduce_lr)
        self.callback.append(early_stopping)
        # SEED = 42
        # os.environ['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)
        # random.seed(SEED)

    def stftLayer(self, x):  # [None, None]
        frames = tf.signal.frame(x, self.win_len, self.stride)  # [None, None, 512]
        stft_dat = tf.signal.rfft(frames)  # [None, None, 257]
        # stft_dat = tf.signal.fft(frames)
        mag = tf.abs(stft_dat)  # [None, None, 257]
        phase = tf.math.angle(stft_dat)  # [None, None, 257]

        return mag, phase

    def fftLayer(self, x):  # [1, 512]
        with tf.GradientTape() as tape:
            tape.watch(x)
            x = tf.expand_dims(x, axis=0)  # [1, 1, 512]

            stft_dat = tf.signal.rfft(x)  # [1, 1, 257]
            mag = tf.abs(stft_dat)
            phase = tf.math.angle(stft_dat)
            transformed_mags = tf.expand_dims(mag, axis=3)  # [B, 483, 257, 1]
            transformed_mags = transformed_mags[:, :, :-1]  # [B, 483, 256, 1]
            # transformed_mags = transformed_mags[:, :, 1:, :]  # [B, 483, 256, 1]

        return mag, phase, transformed_mags

    def encoder(self, x):  # [None, None, 256, 1]
        conv_out = []
        for idx, filter in enumerate(self.encoder_filter):
            x = layers.ZeroPadding2D(padding=[[1, 0], [3, 0]])(x)
            x = layers.Conv2D(filter, (2, 5), strides=(1, 2), padding='valid')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            conv_out.append(x)
            self.conv_out = conv_out
        return x

    def lstm(self, x):
        # reshape error
        # https://github.com/matterport/Mask_RCNN/issues/1070
        shape = np.shape(x)
        if shape[1] == None:
            x = layers.Reshape((-1, shape[2] * shape[3]))(x)
        else:
            x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
        # x = layers.LSTM(128, return_sequences=True)(x)
        # x = layers.Dense(512)(x)

        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dense(2048)(x)
        if shape[1] == None:
            x = layers.Reshape((-1, shape[2], shape[3]))(x)
        else:
            x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
        return x

    def decoder(self, x):
        for idx, filter in enumerate(self.decoder_filter):
            x = layers.concatenate([x, self.conv_out[-idx - 1]], 3)
            x = layers.Conv2DTranspose(filter, (2, 5), strides=(1, 2), padding='valid')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = x[:, 1:, 3:, :]

        return x

    def tfmasking(self, tensor):
        x = tensor[0]
        mags = tensor[1]
        phase = tensor[2]
        out = tf.squeeze(x, axis=3)  # [None, 483, 256]
        paddings = tf.constant([[0, 0], [0, 0], [0, 1]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
        out = tf.pad(out, paddings, mode='CONSTANT')  # [None, 483, 257]

        mask_mags = tf.tanh(out)  # [None, 483, 257]
        est_mags = mask_mags * mags  # [None, 483, 257]
        # out_real = est_mags * tf.cos(phase)  # [None, 483, 257]
        # out_imag = est_mags * tf.sin(phase)  # [None, 483, 257]
        #
        # out_spec = tf.concat([out_real, out_imag], axis=2)  # [None, 483, 514]

        return est_mags  # out_spec

    def ifftLayer(self, x):
        s1_stft = (tf.cast(x[0], tf.complex64) *
                   tf.exp((1j * tf.cast(x[1], tf.complex64))))  # [None, None, 257]
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):

        return tf.signal.overlap_and_add(x, self.stride)

    def build_model(self):
        x = Input(shape=48000, batch_size=self.BATCH_SIZE)
        # mags, phase, transformed_mags = Lambda(self.stft)(x)  # [B, 483, 257]
        mags, phase = Lambda(self.stftLayer, name="stft")(x)  # [None, None, 257]
        transformed_mags = tf.expand_dims(mags, axis=3)  # [B, 483, 257, 1]
        transformed_mags = transformed_mags[:, :, 1:, :]  # [B, 483, 256, 1]

        encoder_out = self.encoder(transformed_mags)  # [B, 483, 4, 128]
        rnn_out = self.lstm(encoder_out)  # [B, 483, 4, 128]
        decoder_out = self.decoder(rnn_out)  # [B, 483, 256, 1]
        decoder_out = tf.squeeze(decoder_out, axis=3)  # [None, 483, 256]
        paddings = tf.constant([[0, 0], [0, 0], [1, 0]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
        out = tf.pad(decoder_out, paddings, mode='CONSTANT')  # [None, 483, 257]
        mask_mags = tf.tanh(out)  # [None, 483, 257]
        est_mags = mask_mags * mags  # [None, 483, 257]
        y = Lambda(self.ifftLayer)([est_mags, phase])
        y = Lambda(self.overlapAddLayer)(y)  # [None, None]
        self.model = tf.keras.Model(inputs=x, outputs=y)
        self.model.summary()
        model = self.model
        return model

    def train_model(self, path):
        self.build_model()
        multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        print("### Start training original model #############################################################")
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[pesq], run_eagerly=True)

        start = time.time()
        with tf.device('/device:GPU:0'):
            self.model.fit(x=self.noisy_train, y=self.clean_train,
                           validation_data=(self.noisy_validation, self.clean_validation),
                           batch_size=self.BATCH_SIZE,
                           epochs=self.EPOCH,
                           initial_epoch=self.start_epoch,
                           callbacks=self.callback,
                           verbose=1,
                           # max_queue_size=50,
                           # workers=5,
                           # use_multiprocessing=True
                           )
        print("takse {:.2f} seconds".format(time.time() - start))

        self.model.save_weights(path)
        tf.keras.backend.clear_session()

    def converter_rt(self, weight, conv1, conv2, conv3, lstm, conv_t1, conv_t2, conv_t3):
        self.build_model()
        self.model.load_weights(weight)
        # padding = 3
        # Convolution Layer 1
        conv1_in = Input(batch_shape=(1, 2, (self.fft_len // 2), 1))  # [1, 2, 259, 1]
        # conv1_in = Input(batch_shape=(1, 6, (self.fft_len // 2) + 3, 1))  # [1, 2, 259, 1]
        conv1_out = layers.Conv2D(16, (2, 5), strides=(1, 2), padding='same')(conv1_in)  # [1, 1, 128, 16]
        # conv1_out = layers.Conv2D(16, (2, 5), strides=(1, 2), padding='valid')(conv1_in)  # [1, 1, 128, 16]
        conv1_out = layers.BatchNormalization()(conv1_out)
        conv1_out = layers.Activation('relu')(conv1_out)
        conv1_tfl = Model(inputs=conv1_in, outputs=conv1_out)

        # Convolution Layer 2
        conv2_in = Input(batch_shape=(1, 2, 128, 16))
        # conv2_in = Input(batch_shape=(1, 5, 128 + 3, 16))
        conv2_out = layers.Conv2D(32, (2, 5), strides=(1, 2), padding='same')(conv2_in)  # [1, 1 ,64, 32]
        # conv2_out = layers.Conv2D(32, (2, 5), strides=(1, 2), padding='valid')(conv2_in)  # [1, 1 ,64, 32]
        conv2_out = layers.BatchNormalization()(conv2_out)
        conv2_out = layers.Activation('relu')(conv2_out)
        conv2_tfl = Model(inputs=conv2_in, outputs=conv2_out)

        # Convolution Layer 3
        conv3_in = Input(batch_shape=(1, 2, 64, 32))
        # conv3_in = Input(batch_shape=(1, 2, 64 + 3, 32))
        conv3_out = layers.Conv2D(64, (2, 5), strides=(1, 2), padding='same')(conv3_in)  # [1, 1, 32, 64]
        # conv3_out = layers.Conv2D(64, (2, 5), strides=(1, 2), padding='valid')(conv3_in)  # [1, 1, 32, 64]
        conv3_out = layers.BatchNormalization()(conv3_out)
        conv3_out = layers.Activation('relu')(conv3_out)
        conv3_tfl = Model(inputs=conv3_in, outputs=conv3_out)
        # conv3_out = conv3_out[:,1:2,:,:]
        # conv3_out = conv3_out[:,0:1,:,:]

        # LSTM Layer
        lstm_in = tf.reshape(conv3_out,
                             (conv3_out.shape[0], conv3_out.shape[1],
                              conv3_out.shape[2] * conv3_out.shape[3]))  # [1,1,2048]
        lstm_out = layers.LSTM(128, return_sequences=True)(lstm_in)  # [1, 1, 128]
        lstm_out = layers.Dense(2048)(lstm_out)  # [1,1,2048]
        lstm_tfl = Model(inputs=lstm_in, outputs=lstm_out)

        # Convolution Transpose Layer 1
        conv_t1_in = Input(batch_shape=(1, 2, 32, 64 * 2))
        conv_t1_out = layers.Conv2DTranspose(32, (2, 5), strides=(1, 2), padding='same')(
            conv_t1_in)
        conv_t1_out = layers.BatchNormalization()(conv_t1_out)
        conv_t1_out = layers.Activation('relu')(conv_t1_out)
        conv_t1_tfl = Model(inputs=conv_t1_in, outputs=conv_t1_out)

        # Convolution Transpose Layer 2
        conv_t2_in = Input(batch_shape=(1, 2, 64, 32 * 2))
        conv_t2_out = layers.Conv2DTranspose(16, (2, 5), strides=(1, 2), padding='same')(
            conv_t2_in)
        conv_t2_out = layers.BatchNormalization()(conv_t2_out)
        conv_t2_out = layers.Activation('relu')(conv_t2_out)
        conv_t2_tfl = Model(inputs=conv_t2_in, outputs=conv_t2_out)

        # Convolution Transpose Layer 3
        conv_t3_in = Input(batch_shape=(1, 2, 128, 16 * 2))
        conv_t3_out = layers.Conv2DTranspose(1, (2, 5), strides=(1, 2), padding='same')(conv_t3_in)
        conv_t3_out = layers.BatchNormalization()(conv_t3_out)
        conv_t3_out = layers.Activation('relu')(conv_t3_out)
        conv_t3_tfl = Model(inputs=conv_t3_in, outputs=conv_t3_out)

        conv1_tfl.summary()
        conv2_tfl.summary()
        conv3_tfl.summary()
        lstm_tfl.summary()
        conv_t1_tfl.summary()
        conv_t2_tfl.summary()
        conv_t3_tfl.summary()

        # set weights to submodels
        weights = self.model.get_weights()
        conv1_tfl.set_weights(weights[:6])
        conv2_tfl.set_weights(weights[6:12])
        conv3_tfl.set_weights(weights[12:18])
        lstm_tfl.set_weights(weights[18:23])
        conv_t1_tfl.set_weights(weights[23:29])
        conv_t2_tfl.set_weights(weights[29:35])
        conv_t3_tfl.set_weights(weights[35:])

        # Create the Conv1
        converter = tf.lite.TFLiteConverter.from_keras_model(conv1_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(conv1, 'wb') as f:
            f.write(tflite_model)

        # Create the Conv2
        converter = tf.lite.TFLiteConverter.from_keras_model(conv2_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(conv2, 'wb') as f:
            f.write(tflite_model)

        # Create the Conv3
        converter = tf.lite.TFLiteConverter.from_keras_model(conv3_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(conv3, 'wb') as f:
            f.write(tflite_model)

        # Create the LSTM
        converter = tf.lite.TFLiteConverter.from_keras_model(lstm_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(lstm, 'wb') as f:
            f.write(tflite_model)

        # Create the Conv-T-1
        converter = tf.lite.TFLiteConverter.from_keras_model(conv_t1_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(conv_t1, 'wb') as f:
            f.write(tflite_model)

        # Create the Conv-T-2
        converter = tf.lite.TFLiteConverter.from_keras_model(conv_t2_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(conv_t2, 'wb') as f:
            f.write(tflite_model)

        # Create the Conv-T-3
        converter = tf.lite.TFLiteConverter.from_keras_model(conv_t3_tfl)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops. # 필수
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(conv_t3, 'wb') as f:
            f.write(tflite_model)
