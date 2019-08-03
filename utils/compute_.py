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


# def calculate_cosine_similarity(a, b):
#     # score = 1 - spatial.distance.cosine(a, b)
#     score = 1 - ((a - b)**2).mean()
#     return score
def cycle_finder(data):
    idx = []

    for i in range(1,20):
        a = data[i]
        max = 1 - spatial.distance.cosine(a, data[i+5])
        last = 0
        for j in range(i+5,len(data)):
            b = data[j]
            c = 1 - spatial.distance.cosine(a, b)
            if c > max:
                max = c
                last = j-i
        idx.append(last)
    return round(sum(idx)/len(idx))

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def calculate_cosine_similarity(a, b):
    cycle_length = 8
    # cycle_length = cycle_finder(a)
    n = 0
    for i in range(cycle_length, len(a), 1):
        a_ = a[i - cycle_length:i]
        max = 0
        for j in range(cycle_length, len(b), 1):
            b_ = b[j - cycle_length:j]
            # c = calculate_cosine_similarity_multidim(a_, b_)
            c = cos(a_,b_)
            c = torch.mean(c)
            if c.item() > max:
                max = c
        n += max
        # if max > n:
        #     n = max
    return n


# def calculate_cosine_similarity(a, b):
#     n = 0
#     for i in range(10, len(a),1):
#         a_ = a[i-10:i]
#         max = 0
#         for j in range(10, len(b),1):
#             b_ = b[j - 10:j]
#             c = calculate_cosine_similarity_multidim(a_,b_)
#             if c > max:
#                 max = c
#         n += max
#     return n

    # calculate_cosine_similarity_multidim(a[:20],b[:20])
    #
    #
    # n = 0
    # for i in a:
    #     max = 0
    #     for j in b:
    #         c = 1 - spatial.distance.cosine(i, j)
    #         if c > max:
    #             max = c
    #     n += max
    # # score = 1 - ((a - b)**2).mean()
    # return n


def calculate_identication_rate_single(glrs, aprb, trueid, rank=1):
    scores = []
    for i in glrs:
        scores.append(calculate_cosine_similarity(i, aprb))
    max_val = max(scores)
    max_idx = scores.index(max_val)

    true_val = scores[trueid[0]]

    print(trueid,true_val, max_idx, max_val)

    if max_idx in trueid:
        return 1, [trueid, max_idx]
    else:
        return 0, [trueid, max_idx]

def eval_roc(glr, prb, gt, n_test, networks, opt):
    netE, lstm = networks

    fg_glr = [netE(glr[i].cuda())[1] for i in range(len(glr))]
    # fg_glr = torch.stack(fg_glr, 0).view(len(fg_glr), n_test, opt.fg_dim)
    # glr_vec = lstm(fg_glr)[1].detach().cpu().numpy()
    glr_vec = lstm(fg_glr)[1]

    fg_prb = [netE(prb[i].cuda())[1] for i in range(len(prb))]

    # fg_prb = torch.stack(fg_prb, 0).view(len(fg_prb), -1, opt.fg_dim)
    # prb_vec = lstm(fg_prb)[1].detach().cpu().numpy()
    prb_vec = lstm(fg_prb)[1]
    # prb_vec = torch.mean(fg_prb, 0).detach().cpu().numpy()

    obj_arr = np.zeros((len(prb_vec), n_test), dtype=np.float32)
    for i in range(n_test):
        for j in range(len(prb_vec)):
            cs = calculate_cosine_similarity(glr_vec[i:i + 1, :],
                                             prb_vec[j:j + 1, :])
            obj_arr[j, i] = cs
    fpr, tpr, roc_auc = process_confusion_matrix(obj_arr, n_test, gt)
    return find_idx(fpr, tpr) #, obj_arr

def eval_cmc(glr, prb, networks, opt, glr_views, prb_views, is_same_view):
    netE, lstm = networks
    pb_vecs = []
    gr_vecs = []
    groundtruth_predicted = []
    for pb in prb:
        fg_pb = [netE(pb[i].cuda())[1] for i in range(len(pb))]
        # fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
        pb_vec = lstm(fg_pb)[1]
        # pb_vec = np.transpose(pb_vec, (1, 0, 2))
        pb_vec = pb_vec.transpose(1, 0)
        pb_vecs.append(pb_vec)

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1] for i in range(len(gr))]
        # fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.fg_dim)
        # gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()
        gr_vec = lstm(fg_gr)[1]
        # gr_vec = np.transpose(gr_vec, (1, 0, 2))
        gr_vec = gr_vec.transpose(1, 0)
        gr_vecs.append(gr_vec)

    scores_all = []
    for pb_idx, pv in enumerate(pb_vecs): #probe view
        scores_this_pv = []
        for gv_idx, gv in enumerate(gr_vecs): #gallerry view

            if glr_views[gv_idx] == prb_views[pb_idx]: # if not the same view
                if is_same_view:
                    scores = []
                    for i in range(len(pv)):
                        id = i
                        id_range = list(range(id, id + 1))
                        score, gt = calculate_identication_rate_single(gv, pv[i], id_range)
                        scores.append(score)
                        groundtruth_predicted.append(gt)
                    scores = sum(scores) / float(len(scores))
                    scores_this_pv.append(scores)

        scores_this_pv = sum(scores_this_pv) / float(len(scores_this_pv))
        scores_all.append(scores_this_pv)
    return scores_all,groundtruth_predicted

def calculate_cosine_similarity_multidim(a, b):
    max = 0
    score = 0
    for i in range(len(a)):
        score += 1 - spatial.distance.cosine(a[i], b[i])
        # score += 1 - ((a[i] - b[i])**2).mean()

    return score/len(a)

