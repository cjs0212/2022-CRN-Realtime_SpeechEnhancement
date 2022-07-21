from model import CRN_model
import soundfile as sf
import numpy as np
import time
from pesq import pesq as get_pesq
import tensorflow as tf

m = CRN_model()
model = m.build_model()

model.load_weights("./saved_model/crn_valid_timit_E10_v2.h5")
weights = model.get_weights()
conv1 = model.layers[5]
conv1.set_weights(weights[:2])

batch_norm1 = model.layers[6]
batch_norm1.set_weights(weights[2:6])

actv1 = model.layers[7]

conv2 = model.layers[9]
conv2.set_weights(weights[6:8])

batch_norm2 = model.layers[10]
batch_norm2.set_weights(weights[8:12])

actv2 = model.layers[11]

conv3 = model.layers[13]
conv3.set_weights(weights[12:14])

batch_norm3 = model.layers[14]
batch_norm3.set_weights(weights[14:18])

actv3 = model.layers[15]

reshape1 = model.layers[16]
lstm = model.layers[17]
lstm.set_weights(weights[18:21])

dense = model.layers[18]
dense.set_weights(weights[21:23])

reshape2 = model.layers[19]

convT1 = model.layers[21]
convT1.set_weights(weights[23:25])

batch_norm4 = model.layers[22]
batch_norm4.set_weights(weights[25:29])

actv4 = model.layers[23]

convT2 = model.layers[26]
convT2.set_weights(weights[29:31])

batch_norm5 = model.layers[27]
batch_norm5.set_weights(weights[31:35])

actv5 = model.layers[28]

convT3 = model.layers[31]
convT3.set_weights(weights[35:37])

batch_norm6 = model.layers[32]
batch_norm6.set_weights(weights[37:])

actv6 = model.layers[33]

audio, fs = sf.read('noisy3.wav')  # [48000,] / 16000
audio = np.concatenate([audio, np.zeros(256)], axis=0)

block_len = 512
stride = 128

if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
in_buffer = np.zeros((block_len)).astype('float32')  # [512,]
# out_file = np.zeros((len(audio)))  # [48000,]
out_file = []
out_test = []
time_array = []
etime_array = []
ltime_array = []
dtime_array = []

outputs = []
conv1_preframe = np.zeros((1, 1, 256, 1), dtype=np.float32)
conv2_preframe = np.zeros((1, 1, 128, 16), dtype=np.float32)
conv3_preframe = np.zeros((1, 1, 64, 32), dtype=np.float32)
conv_t1_preframe = np.zeros((1, 1, 32, 64 * 2), dtype=np.float32)
conv_t2_preframe = np.zeros((1, 1, 64, 32 * 2), dtype=np.float32)
conv_t3_preframe = np.zeros((1, 1, 128, 16 * 2), dtype=np.float32)
paddings = np.array([[0, 0], [0, 0], [3, 0], [0, 0]])

for idx in range(len(audio) // stride):  # 372
    start_time = time.time()
    in_buffer[:-stride] = in_buffer[stride:]
    in_buffer[-stride:] = audio[idx * stride:(idx * stride) + stride]
    # rfft
    in_block_fft = np.fft.rfft(in_buffer)  # [257,]
    # Get mags and phase
    in_mag = np.abs(in_block_fft)  # [257,]
    in_phase = np.angle(in_block_fft)  # [257,]
    # Reshape mags for input
    expand_mag = np.reshape(in_mag, (1, 1, -1, 1)).astype('float32')  # [1,1,257,1]
    sliced_mag = expand_mag[:, :, 1:]  # [1, 1, 256, 1]

    encoder_time = time.time()
    # Convolution Layer 1
    conv1_frame2 = np.concatenate([conv1_preframe, sliced_mag], axis=1)
    conv1_frame2 = np.pad(conv1_frame2, paddings)
    conv1_out = conv1(conv1_frame2)
    conv1_out = batch_norm1(conv1_out)
    conv1_out = actv1(conv1_out)
    conv1_preframe = sliced_mag
    conv2_curframe = conv1_out

    # Convolution Layer 2
    conv2_frame2 = np.concatenate([conv2_preframe, conv2_curframe], axis=1)
    conv2_in = np.pad(conv2_frame2, paddings)
    conv2_out = conv2(conv2_in)
    conv2_out = batch_norm2(conv2_out)
    conv2_out = actv2(conv2_out)
    conv2_preframe = conv2_curframe
    conv3_curframe = conv2_out

    # Convolution Layer 3
    conv3_frame2 = np.concatenate([conv3_preframe, conv3_curframe], axis=1)
    conv3_in = np.pad(conv3_frame2, paddings)
    conv3_out = conv3(conv3_in)
    conv3_out = batch_norm3(conv3_out)
    conv3_out = actv3(conv3_out)
    conv3_preframe = conv3_curframe
    etime_array.append(time.time() - encoder_time)

    # LSTM Layer
    shape = conv3_out.shape
    lstm_in = tf.reshape(conv3_out, [shape[0], shape[1], shape[2] * shape[3]])
    lstm_time = time.time()
    lstm_out = lstm(lstm_in)
    lstm_out = dense(lstm_out)
    ltime_array.append(time.time() - lstm_time)
    lstm_out = tf.reshape(lstm_out, [shape[0], shape[1], shape[2], shape[3]])
    conv_t1_curframe = lstm_out

    # Convolution Transpose Layer 1
    decoder_time = time.time()
    conv_t1_curframe = np.concatenate([lstm_out, conv3_out], axis=3)  # Skip connection
    conv_t1_in = np.concatenate([conv_t1_preframe, conv_t1_curframe], axis=1)
    conv_t1_out = convT1(conv_t1_in)
    conv_t1_out = batch_norm4(conv_t1_out)
    conv_t1_out = actv4(conv_t1_out)
    conv_t1_out = conv_t1_out[:, 1:, 3:, :]
    conv_t1_preframe = conv_t1_curframe

    # Convolution Transpose Layer 2
    conv_t2_curframe = np.concatenate([conv_t1_out, conv3_frame2], axis=3)  # Skip connection
    # conv_t2_in = np.concatenate([conv_t2_preframe, conv_t2_curframe], axis=1)
    conv_t2_out = convT2(conv_t2_curframe)
    conv_t2_out = batch_norm5(conv_t2_out)
    conv_t2_out = actv5(conv_t2_out)
    conv_t2_out = conv_t2_out[:, 1:, 3:, :]
    conv_t2_preframe = conv_t2_curframe

    # Convolution Transpose Layer 3
    conv_t3_curframe = np.concatenate([conv_t2_out, conv2_frame2], axis=3)  # Skip connection
    # conv_t3_in = np.concatenate([conv_t3_preframe, conv_t3_curframe], axis=1)
    conv_t3_out = convT3(conv_t3_curframe)
    conv_t3_out = batch_norm6(conv_t3_out)
    conv_t3_out = actv6(conv_t3_out)
    conv_t3_out = conv_t3_out[:, 2:3, 3:, :]
    dtime_array.append(time.time() - decoder_time)

    # T-F Masking
    mag_padding = np.array([[0, 0], [0, 0], [1, 0], [0, 0]])  # pad. 3차원 Time 축으로 한칸 제로패딩.
    out_mask = np.pad(conv_t3_out, mag_padding)  # [1, 1, 257]
    out_mask = np.squeeze(out_mask, axis=3)  # [1, 1, 256]
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)  # [1,1,257]
    # estimated_complex = in_mag * np.exp(1j * in_phase)  # [1,1,257]
    estimated_block = np.fft.irfft(estimated_complex)  # [1,1,512]
    estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')  # [1,1,512]
    estimated_block = np.squeeze(estimated_block)

    time_array.append(time.time() - start_time)
    out_file = np.concatenate([out_file, estimated_block[128:256]])

    time_array.append(time.time() - start_time)

audio = audio[:-256]
out_file = out_file[256:]

sf.write('TFRT3.wav', out_file, fs)

print('Processing Time [ms]:')
print(np.mean(np.stack(time_array)))
print(np.sum(np.stack(time_array)))
print("Encoder Processing Time : ", np.sum(np.stack(etime_array)))
print("LSTM Processing Time : ", np.sum(np.stack(ltime_array)))
print("Decoder Processing Time : ", np.sum(np.stack(dtime_array)))

print('Processing finished.')
# enhanced, fs = sf.read("./enhanced.wav")
enhanced, fs = sf.read("./TFRT3.wav")
# clean, fs = sf.read("./clean.wav")
clean, fs = sf.read("./clean3.wav")
# noisy, fs = sf.read("./noisy.wav")
noisy, fs = sf.read("./noisy3.wav")


def cal_pesq(y_true, y_pred):
    sr = 16000
    mode = "wb"
    pesq_score = get_pesq(sr, y_true, y_pred, mode)
    # pesq_score = get_pesq(sr, enhanced, clean, mode)
    return pesq_score


print("Enhanced Audio PESQ : ", cal_pesq(clean, enhanced))
print("Noisy Audio PESQ :    ", cal_pesq(clean, noisy))
