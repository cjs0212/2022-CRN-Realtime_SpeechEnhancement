import numpy as np
from pesq import pesq as get_pesq

# https://pypi.org/project/pesq/
def cal_pesq(enhanced, clean):
    sr = 16000
    mode = "wb"
    scores = []
    for i in range(len(enhanced)):
        pesq_score = get_pesq(sr, clean[i], enhanced[i], mode)
    if pesq_score > 0:
        scores.append(pesq_score)
    # mode = "nb" if clean_sr < 16000 else "wb"
    return scores

def pesq(y_true, y_pred):
    pesq_score = cal_pesq(y_pred.numpy(), y_true.numpy())
    if len(pesq_score) == 0:
        return 0
    return np.nanmean(pesq_score)