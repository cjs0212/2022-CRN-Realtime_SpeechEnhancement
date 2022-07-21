import numpy as np
import tensorflow as tf
from tensorflow.summary import scalar, audio
from tools_for_estimate import pesq
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import config as cfg


###############################################################################
#                             Custom TensorBoard                              #
###############################################################################
# execute tensorboard : tensorboard --logdir=./logs
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self,
                 validation_data,
                 batch_size=1,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 write_steps_per_second=False,
                 update_freq='epoch',
                 profile_batch=0,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        super().__init__(log_dir,
                         histogram_freq,
                         write_graph,
                         write_images,
                         write_steps_per_second,
                         update_freq,
                         profile_batch,
                         embeddings_freq,
                         embeddings_metadata,
                         **kwargs)
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.batch_size = batch_size

    def set_model(self, model):
        super().set_model(model)
        self.file_writer = tf.summary.create_file_writer(self.log_dir + '/validation')  # Logging custom scalars

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        batch = self.create_datasets().batch(self.batch_size)
        progbar = tf.keras.utils.Progbar(len(list(batch)), stateful_metrics=['loss'], unit_name='step')
        pesq_scores = []
        stoi_scores = []
        for b in batch:
            y_valid, y_pred = b[:, 0, :], b[:, 1, :]  # [1, 48000]
            values = []
            # need to connect every valid_metrics_list
            pesq_score = pesq(y_valid, y_pred)
            # pesq_score = pesq(self.sr, y_valid[0].numpy(), y_pred[0].numpy(), self.mode)
            pesq_scores.append(pesq_score)
            values.append(('val_pesq', np.nanmean(pesq_scores)))

            # stoi_score = stoi(y_valid, y_pred)
            # stoi_scores.append(stoi_score)
            # values.append(('val_stoi', np.nanmean(stoi_scores)))

            progbar.add(1, values=values)

        X_valid, y_valid = self.validation_data
        y_pred = self.model.predict(X_valid, batch_size=self.batch_size)
        y_pred = tf.cast(tf.reshape(y_pred[0], (1, 48000, 1)), dtype=tf.float32)
        X_valid = tf.cast(tf.reshape(X_valid[0], (1, 48000, 1)), dtype=tf.float32)
        y_valid = tf.cast(tf.reshape(y_valid[0], (1, 48000, 1)), dtype=tf.float32)

        with tf.summary.record_if(True):
            with self.file_writer.as_default():
                scalar('epoch_pesq', np.nanmean(pesq_scores), step=epoch)
                # summary_ops_v2.scalar('epoch_stoi', np.nanmean(stoi_scores), step=epoch)
                if epoch % 5 == 0:
                    audio('clean_target_wav', y_valid, 16000, max_outputs=1, step=epoch, encoding=None)
                    audio('estimated_wav', y_pred, 16000, max_outputs=1, step=epoch)
                    audio('mixed_wav', X_valid, 16000, max_outputs=1, step=epoch)

    def create_datasets(self):
        X_valid, y_valid = self.validation_data  # [100,48000]
        y_pred = self.model.predict(X_valid, batch_size=self.batch_size)
        y_valid, y_pred = np.expand_dims(y_valid, axis=1), np.expand_dims(y_pred, axis=1)
        data = np.concatenate((y_valid, y_pred), 1)
        return tf.data.Dataset.from_tensor_slices(data)


tensorboard_callback = CustomTensorBoard(validation_data=(cfg.noisy_validation, cfg.clean_validation),
                                         log_dir=cfg.logdir,
                                         batch_size=cfg.BATCH_SIZE, )
model_checkpoint = ModelCheckpoint(filepath=cfg.chpt_dir + cfg.chpt_file,
                                   save_weights_only=True,
                                   save_best_only=True,
                                   verbose=1,
                                   mode='auto',
                                   save_freq='epoch'
                                   )

# create callback for the adaptive learning rate
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, verbose=1, min_lr=10 ** (-10), cooldown=1)

# create callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=15, verbose=1, mode='auto', baseline=None)