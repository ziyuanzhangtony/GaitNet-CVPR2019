from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch

def process_confusion_matrix(matrix, n_class, gt):
    # plt.imshow(matrix)
    # # plt.gray()
    # plt.show()

    matrix = np.reshape(matrix, (n_class * sum(gt)))

    def make_labels():
        squre_matrix = np.eye(n_class, n_class)
        matrix = []

        for i in range(n_class):
            for j in range(gt[i]):
                matrix.append(squre_matrix[i])

        matrix = np.asarray(matrix)
        return np.concatenate(matrix)

    labels = make_labels()
    labels = np.reshape(labels, (n_class * sum(gt)))
    fpr, tpr, _ = roc_curve(labels, matrix)
    roc_auc = auc(fpr, tpr)

    # plt.title('MRCNN-V2')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()


    return fpr, tpr, roc_auc


def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 3
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Alarm Rate/ False Positive Rate')
    plt.ylabel('True Accept Rate/ True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def find_idx(fpr, tpr, threthold=[0.01, 0.05, 0.1], ifround=True):
    outptut = []
    for i in threthold:
        item = fpr[fpr < i + 0.005].max()
        idx = np.where(fpr == item)
        val = tpr[idx][-1]
        if ifround:
            val = round(val, 2)
        outptut.append(val)
    return outptut


def calculate_cosine_similarity(a, b):
    score = 1 - spatial.distance.cosine(a, b)
    return score


def calculate_identication_rate_single(glrs, aprb, trueid, rank=1):
    scores = []
    for i in glrs:
        scores.append(calculate_cosine_similarity(i, aprb))
    max_val = max(scores)
    max_idx = scores.index(max_val)

    right, predicted = trueid, max_idx
    print(right, predicted)

    if max_idx in trueid:
        return 1, [right, predicted]
    else:
        return 0, [right, predicted]

def eval_lstm_roc(glr, prb, gt, n_test, networks, opt):
    netE, lstm = networks

    fg_glr = [netE(glr[i].cuda())[1].detach() for i in range(len(glr))]
    fg_glr = torch.stack(fg_glr, 0).view(len(fg_glr), n_test, opt.fg_dim)
    glr_vec = lstm(fg_glr)[1].detach().cpu().numpy()

    fg_prb = [netE(prb[i].cuda())[1].detach() for i in range(len(prb))]

    fg_prb = torch.stack(fg_prb, 0).view(len(fg_prb), -1, opt.fg_dim)
    prb_vec = lstm(fg_prb)[1].detach().cpu().numpy()
    # prb_vec = torch.mean(fg_prb, 0).detach().cpu().numpy()

    obj_arr = np.zeros((len(prb_vec), n_test), dtype=np.float32)
    for i in range(n_test):
        for j in range(len(prb_vec)):
            cs = calculate_cosine_similarity(glr_vec[i:i + 1, :],
                                             prb_vec[j:j + 1, :])
            obj_arr[j, i] = cs
    fpr, tpr, roc_auc = process_confusion_matrix(obj_arr, n_test, gt)
    return find_idx(fpr, tpr) #, obj_arr


def eval_lstm_cmc(glr, prb, networks, opt):
    netE, lstm = networks
    pb_vecs = []
    gr_vecs = []
    for pb in prb:
        fg_pb = [netE(pb[i].cuda())[1].detach() for i in range(len(pb))]
        fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
        pb_vec = lstm(fg_pb)[1].detach().cpu().numpy()
        pb_vecs.append(pb_vec)

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1].detach() for i in range(len(gr))]
        fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.fg_dim)
        gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()
        gr_vecs.append(gr_vec)

    scores_all = []
    for pb_idx, pv in enumerate(pb_vecs):
        scores_this_pv = []
        for gv_idx, gv in enumerate(gr_vecs):
            if opt.glr_views[gv_idx] != opt.prb_views[pb_idx]:
                score = []
                for i in range(len(pv)):
                    id = i
                    id_range = list(range(id, id + 1))
                    score.append(calculate_identication_rate_single(gv, pv[i], id_range)[0])
                score = sum(score) / float(len(score))
                scores_this_pv.append(score)
        scores_this_pv = sum(scores_this_pv) / float(len(scores_this_pv))
        scores_all.append(scores_this_pv)
    return scores_all

