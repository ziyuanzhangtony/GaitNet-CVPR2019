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
    # score = 1 - np.square(a-b).mean()
    score = 1 - spatial.distance.cosine(a, b)
    return score


def calculate_identication_rate_single(glrs, aprb, trueid, rank=1):
    scores = []
    for i in glrs:
        scores.append(calculate_cosine_similarity(i, aprb))
    max_val = max(scores)
    max_idx = scores.index(max_val)

    right, predicted = trueid, max_idx
    # print(right, predicted)

    if max_idx in trueid:
        return 1, [right, predicted]
    else:
        return 0, [right, predicted]


def calculate_identication_rate_two(glrs, aprb, trueid, rank=1):
    scores = []
    for i in glrs:
        score1 = calculate_cosine_similarity(i[:256], aprb[:256]) # lstm_final
        score2 = calculate_cosine_similarity(i[256:], aprb[256:]) # fs_mean
        scores.append(score1+score2)
    max_val = max(scores)
    max_idx = scores.index(max_val)

    right, predicted = trueid, max_idx
    # print(right, predicted)

    if max_idx in trueid:
        return 1, [right, predicted]
    else:
        return 0, [right, predicted]

def eval_roc(glr, prb, gt, n_test, networks, opt):
    netE, lstm = networks
    netE.eval()
    lstm.eval()

    fg_glr = [netE(glr[i].cuda())[1] for i in range(len(glr))]
    # fg_glr = torch.stack(fg_glr, 0).view(len(fg_glr), n_test, opt.fg_dim)
    glr_vec = lstm(fg_glr)[1].detach().cpu().numpy()

    fg_prb = [netE(prb[i].cuda())[1] for i in range(len(prb))]

    # fg_prb = torch.stack(fg_prb, 0).view(len(fg_prb), -1, opt.fg_dim)
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

def eval_roc_two(glr, prb, gt, n_test, networks, opt):
    netE, lstm = networks
    netE.eval()
    lstm.eval()

    # glr_vec = lstm(fg_glr)[1].detach().cpu().numpy()

    fd_glr = [netE(glr[i].cuda())[1] for i in range(len(glr))]
    # fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
    fg_glr = lstm(fd_glr)[1].detach().cpu().numpy()

    fs_glr = [netE(glr[i].cuda())[0][:, :128] for i in range(len(glr))]
    fs_glr = torch.stack(fs_glr, 0).mean(dim=0).detach().cpu().numpy()
    # fsd_glr = np.concatenate((fs_glr, fg_glr), 1)

    fd_prb = [netE(prb[i].cuda())[1] for i in range(len(prb))]
    # fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
    fg_prb = lstm(fd_prb)[1].detach().cpu().numpy()

    fs_prb = [netE(prb[i].cuda())[0][:, :128] for i in range(len(prb))]
    fs_prb = torch.stack(fs_prb, 0).mean(dim=0).detach().cpu().numpy()
    # fsd_prb = np.concatenate((fs_prb, fg_prb), 1)

    obj_arr = np.zeros((len(fg_prb), n_test), dtype=np.float32)
    for i in range(n_test):
        for j in range(len(fg_prb)):
            cs1 = calculate_cosine_similarity(fg_glr[i:i + 1, :],
                                             fg_prb[j:j + 1, :])
            cs2 = calculate_cosine_similarity(fs_glr[i:i + 1, :],
                                              fs_prb[j:j + 1, :])
            obj_arr[j, i] = (cs1+cs2)/2
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
        pb_vec = lstm(fg_pb)[1].detach().cpu().numpy()
        pb_vecs.append(pb_vec)

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1] for i in range(len(gr))]
        # fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.fg_dim)
        gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()
        gr_vecs.append(gr_vec)

    scores_all = []
    for pb_idx, pv in enumerate(pb_vecs): #probe view
        scores_this_pv = []
        for gv_idx, gv in enumerate(gr_vecs): #gallerry view

            if glr_views[gv_idx] == prb_views[pb_idx]: # if not the same view
                if is_same_view is False:
                    continue
            score = []
            for i in range(len(pv)):
                id = i
                id_range = list(range(id, id + 1))
                score.append(calculate_identication_rate_single(gv, pv[i], id_range)[0])
                groundtruth_predicted.append(calculate_identication_rate_single(gv, pv[i], id_range)[1])
            score = sum(score) / float(len(score))
            scores_this_pv.append(score)

        scores_this_pv = sum(scores_this_pv) / float(len(scores_this_pv))
        scores_all.append(scores_this_pv)
    return scores_all,groundtruth_predicted

def eval_cmc_two(glr, prb, networks, opt, glr_views, prb_views, is_same_view, n_glr_subj, n_prb_subj):
    netE, lstm = networks
    netE.eval()
    lstm.eval()
    pb_vecs = []
    gr_vecs = []
    groundtruth_predicted = []
    for pb in prb:
        fg_pb = [netE(pb[i].cuda())[1] for i in range(len(pb))]
        # fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
        pb_vec = lstm(fg_pb)[1].detach().cpu().numpy()

        fgs_pb = [netE(pb[i].cuda())[0][:,:128] for i in range(len(pb))]
        fgs_pb = torch.stack(fgs_pb,0).mean(dim=0).detach().cpu().numpy()
        pb_vecs.append(np.concatenate((pb_vec,fgs_pb),1))

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1] for i in range(len(gr))]
        # fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.fg_dim)
        gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()

        fgs_gr = [netE(gr[i].cuda())[0][:,:128] for i in range(len(gr))]
        fgs_gr = torch.stack(fgs_gr,0).mean(dim=0).detach().cpu().numpy()

        gr_vecs.append(np.concatenate((gr_vec,fgs_gr),1)) # combine fgd_final and fgs_mean

    scores_all = []
    for pb_idx, pv in enumerate(pb_vecs): #probe view
        scores_this_pv = []
        for gv_idx, gv in enumerate(gr_vecs): #gallerry view

            if glr_views[gv_idx] == prb_views[pb_idx]: # if not the same view
                if is_same_view is False:
                    continue
            score = []
            id_range = list(range(n_glr_subj))
            for i in range(len(pv)):
                s, gt = calculate_identication_rate_two(gv, pv[i], id_range)
                score.append(s)
                groundtruth_predicted.append(gt)
                # TO-DO MIGHT BUG HERE!!!!!!!!!!!!!!!
                if n_prb_subj == 1:
                    id_range = [i + 1 for i in id_range]
                else:
                    if i % n_prb_subj:
                        id_range = [i + n_glr_subj for i in id_range]
            score = sum(score) / float(len(score))
            scores_this_pv.append(score)
        if len(scores_this_pv) == 0:
            continue
        scores_this_pv = sum(scores_this_pv) / float(len(scores_this_pv))
        scores_all.append(scores_this_pv)
    return scores_all,groundtruth_predicted

def eval_cmc_two_avrage_on_gallery(glr, prb, networks, opt, glr_views, prb_views, is_same_view, n_glr_subj):
    netE, lstm = networks
    netE.eval()
    lstm.eval()
    pb_vecs = []
    gr_vecs = []
    groundtruth_predicted = []
    for pb in prb:
        fg_pb = [netE(pb[i].cuda())[1] for i in range(len(pb))]
        # fg_pb = torch.stack(fg_pb, 0).view(len(fg_pb), -1, opt.fg_dim)
        pb_vec = lstm(fg_pb)[1].detach().cpu().numpy()

        fgs_pb = [netE(pb[i].cuda())[0][:,:128] for i in range(len(pb))]
        fgs_pb = torch.stack(fgs_pb,0).mean(dim=0).detach().cpu().numpy()
        pb_vecs.append(np.concatenate((pb_vec,fgs_pb),1))

    for gr in glr:
        fg_gr = [netE(gr[i].cuda())[1] for i in range(len(gr))]
        # fg_gr = torch.stack(fg_gr, 0).view(len(fg_gr), -1, opt.fg_dim)
        gr_vec = lstm(fg_gr)[1].detach().cpu().numpy()

        fgs_gr = [netE(gr[i].cuda())[0][:,:128] for i in range(len(gr))]
        fgs_gr = torch.stack(fgs_gr,0).mean(dim=0).detach().cpu().numpy()

        gr_vecs.append(np.concatenate((gr_vec,fgs_gr),1)) # combine fgd_final and fgs_mean

    scores_all = []
    for gv_idx, gv in enumerate(gr_vecs):  # gallerry view
        scores_this_view = []
        for pb_idx, pv in enumerate(pb_vecs): #probe view
            if glr_views[gv_idx] == prb_views[pb_idx]:  # if the same view
                if is_same_view is False: # but wanna ignore
                    continue
            score = []
            for id in range(len(pv)): #for each subject
                id_range = list(range(id, id + n_glr_subj))
                s, gt = calculate_identication_rate_two(gv, pv[id], id_range)
                score.append(s)
                # groundtruth_predicted.append(gt)
            score = sum(score) / float(len(score)) # take avrage under this view
            scores_this_view.append(score)
        if len(scores_this_view) == 0:
            continue
        scores_this_view = sum(scores_this_view) / float(len(scores_this_view))
        scores_all.append(scores_this_view)
    return scores_all


def calculate_cosine_similarity_multidim(a, b):
    score = 0
    for i in range(len(a)):
        score += 1- spatial.distance.cosine(a[i], b[i])
    return score/len(a)

