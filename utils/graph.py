import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from imageio import imwrite

def adjust_white_balance(x):
    avgR = x[0, :, :].mean()
    avgG = x[1, :, :].mean()
    avgB = x[2, :, :].mean()

    avg = (avgB + avgG + avgR) / 3

    x[0, :, :] = np.minimum(x[0] * (avg / avgR), 1)
    x[1, :, :] = np.minimum(x[1] * (avg / avgG), 1)
    x[2, :, :] = np.minimum(x[2] * (avg / avgB), 1)

    return x


# def analogy_ha_hg_diff_subjects(row, col, networks, opt):
#     netE, netD = networks
#
#     # (B, L, C, H, W)
#     # (   B, C, H, W)
#     netE.eval()
#     netD.eval()
#
#     none = torch.zeros([1, 3, 64, 32]).cuda()
#
#     x_gs = torch.stack(row).cuda()
#     none = x_gs[:1]
#     # x_gs.fill_(0)
#     h_gs_x = netE(x_gs)[1]
#
#     nones = torch.stack([none[0]] * 5).cuda()
#     h_gs_nones = netE(nones)[1]
#
#     h_gs = [h_gs_nones[0],h_gs_x[1]]
#
#
#     x_as = torch.stack(col).cuda()
#     # x_as.fill_(0)
#
#     h_as = netE(x_as)[0]
#
#     # h_as.fill_(0)
#
#     # h_gs[0].fill_(0)
#     #
#     # h_gs[1].fill_(0)
#
#     # h_as = torch.zeros(3,288).cuda()
#
#     gene = [netD(torch.stack([i] * 5).cuda(), h_gs) for i in h_as]
#     row0 = torch.cat([none, x_gs])
#     rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
#     to_plot = torch.cat([row0] + rows)
#
#     img = make_grid(to_plot,6)
#     img = adjust_white_balance(img.detach().cpu())
#     save_image(img,'img_same_person.png',6)
#     # imwrite('img_fa.png',np.transpose(img,[1,2,0]))
#     return img

def analogy_ha_hg_diff_subjects(row, col, networks, opt):
    netE, netD = networks

    # (B, L, C, H, W)
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    # none = torch.zeros([1, 3, 64, 32]).cuda()

    x_gd = torch.stack(row).cuda()
    x_a = x_gd[1:2].clone()
    # x_a.fill_(0)

    x_gd = torch.stack([x_gd[0],x_gd[2]])
    # x_gd = x_gd[0:]
    # x_gd.fill_(0)

    h_gd_row = netE(x_gd)[1][1]
    # h_gd_row.fill_(0)

    x_gs = torch.stack(col).cuda()
    h_gs_col = netE(x_gs)[1][0]
    # h_gs_col.fill_(0)


    # nones = torch.stack([none[0]] * 5).cuda()
    # h_gs_nones = netE(nones)[1]

    # h_gs = [h_gs_nones[0],h_gs_x[1]]


    # x_as.fill_(0)

    # x_a = torch.stack([none[0]] * len(h_gs)).cuda()

    h_a = netE(x_a)[0]
    h_as = torch.stack([h_a[0]] * len(h_gd_row)).cuda()

    # h_as.fill_(0)

    gene = [netD(h_as, [torch.stack([i]*len(h_gd_row)).cuda(), h_gd_row]) for i in h_gs_col]
    row0 = torch.cat([x_a, x_gd])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_gs)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot,len(h_gd_row)+1)
    img = adjust_white_balance(img.detach().cpu())
    save_image(img,'img_same_person.png',len(h_gd_row)+1)
    # imwrite('img_fa.png',np.transpose(img,[1,2,0]))
    return img

def analogy_ha_hg_diff_subjects_two_parts(row, col, networks, opt):
    netE, netD = networks

    # (B, L, C, H, W)
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    none = torch.zeros([1, 3, opt.im_height, opt.im_width]).cuda()

    x_a = torch.stack(col+[none[0].cpu()]).cuda()
    h_a_col = netE(x_a)[0]
    h_a_col[-1].fill_(0)

    x_g = torch.stack(row+[none[0].cpu()]).cuda()
    h_g_row = netE(x_g)[1]
    h_g_row[-1].fill_(0)

    gene = [netD(torch.stack([i]*len(h_g_row)).cuda(), h_g_row) for i in h_a_col]
    row0 = torch.cat([none, x_g])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_a)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot,len(h_g_row)+1)
    img = adjust_white_balance(img.detach().cpu())
    save_image(img,'img_diff_person.png',len(h_g_row)+1)
    return img

def analogy_ha_hg_diff_subjects_two_parts_hgs(row, col, networks, opt):
    netE, netD = networks

    # (B, L, C, H, W)h_g_row
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    none = torch.zeros([1, 3, opt.im_height, opt.im_width]).cuda()

    x_a = torch.stack(col).cuda()
    h_a_col = netE(x_a)[0]
    h_a_col[:,128:].fill_(0) # only remove the hgs

    x_g = torch.stack(row).cuda()
    h_g_row = netE(x_g)[1]
    h_g_row.fill_(0)

    gene = [netD(torch.stack([i]*len(h_g_row)).cuda(), h_g_row) for i in h_a_col]
    row0 = torch.cat([none, x_g])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_a)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot,len(h_g_row)+1)
    img = adjust_white_balance(img.detach().cpu())
    save_image(img,'hgs_only.png',len(h_g_row)+1)
    # imwrite('img_fa.png',np.transpose(img,[1,2,0]))
    return img

def analogy_ha_hg_diff_subjects_two_parts_hgs_only(row, networks,opt):
    netE, netD = networks

    # (B, L, C, H, W)h_g_row
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    # none = torch.zeros([1, 3, 64, 32]).cuda()

    x = torch.stack(row).cuda()
    h_a = netE(x)[0]
    h_a[:,128:].fill_(0) # keep the hgs only

    h_g = netE(x)[1]
    # h_g = torch.stack([h_g[0]] * 5)
    h_g.fill_(0)

    gene = netD(h_a, h_g)
    to_plot = torch.cat([x, gene])

    img = make_grid(to_plot,len(h_a))
    img = adjust_white_balance(img.detach().cpu())
    save_image(img,'hgs_only.png',len(h_a))
    return img


def analogy_ha_hg_same_subjects(nm, cl, idx, networks, opt):
    netE, netD = networks



    # (B, L, C, H, W)
    # (   B, C, H, W)
    # netE.eval()
    # netD.eval()

    def rand_idx():
        return 0

    def rand_step():
        # return np.random.randint(0, opt.max_step)
        return 0

    none = torch.zeros([1, 3,opt.im_height, opt.im_width]).cuda()
    x_gs = torch.stack([i for i in [nm[idx][0], nm[idx][4], nm[idx][8], nm[idx][15]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(4,32).cuda()

    x_as = torch.stack([nm[idx][rand_step()], cl[idx][rand_step()]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.zeros(2,288).cuda()

    gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)

    img = make_grid(to_plot, 5)
    return img



def plot_anology(row, col, itr, networks, opt, writer):
    anology1 = analogy_ha_hg_diff_subjects_two_parts(row,col,networks, opt)
    anology2 = analogy_ha_hg_diff_subjects_two_parts_hgs_only(row,networks, opt)

    # anology2 = analogy_ha_hg_diff_subjects(test, networks, opt)
    # all = torch.cat([anology1, anology2], dim=1)
    # anology1 = adjust_white_balance(anology1.detach().cpu())
    # writer.add_image('diff_subjects', anology1, itr)

    # anology1 = analogy_ha_hg_same_subjects(data1, data2, 8, networks, opt)
    # anology2 = analogy_ha_hg_same_subjects(data1, data2, 9, networks, opt)
    # all = torch.cat([anology1, anology2], dim=1)
    # all = adjust_white_balance(all.detach())

    writer.add_image('ha_hg', anology1, itr)
    writer.add_image('ha_hg_nohgs', anology2, itr)


# def plot_tsne(data, itr, networks, opt):
