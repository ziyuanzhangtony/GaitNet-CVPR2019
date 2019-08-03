import random
import os
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from utils.dataloader import FVG
from utils.compute import *
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.modules_fvg_pami_tab6 import *
from utils.dataloader import get_training_batch
from torchvision.utils import make_grid,save_image

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/FVG-MRCNN/SEG/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--lstm_hidden_dim', type=int, default=128, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--clip_len', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=8, help='number of parallel data loading threads')
parser.add_argument('--num_train',type=int, default=136, help='')
parser.add_argument('--savedir', default='./runs')
parser.add_argument('--signature', default='136training-MRCNN-DIFFCOND-LSTMMEAN-WEIGHTDECAY')
opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)


# train_structure = {
#     'clip1': list(range(1, 12 + 1)),
#     'clip2': list(range(1, 12 + 1))
# }

train_structure = {
    'session1': list(range(1, 12 + 1)),
    'session2': list(range(1, 12 + 1)),
    'session3': list(range(1, 12 + 1)),
}

test_structure = {
    'gallery': ([90], ['nm'], [1]),
    'probe': ([90], ['cl'], [1]),
}


torch.cuda.set_device(opt.gpu)

# cudnn.deterministic = False0
cudnn.benchmark = True

#
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

# load latest module
module_save_path = os.path.join(opt.savedir, 'modules', opt.signature)
if os.path.exists(module_save_path):
    model_names = os.listdir(module_save_path)
    model_names = [f for f in model_names if f.endswith('.pickle')]
else:
    model_names = []
if len(model_names):
    model_names.sort(key=lambda a: int(a.split(".")[0]))
    loading_model_path = os.path.join(module_save_path, model_names[-1])
    itr = int(model_names[-1].split('.')[0])
else:
    loading_model_path = None
    itr = 0

os.makedirs('%s/modules/%s' % (opt.savedir, opt.signature), exist_ok=True)
os.makedirs('%s/gifs/%s' % (opt.savedir, opt.signature), exist_ok=True)
#################################################################################################################
# MODEL PROCESS
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netE = encoder(opt)
netD = decoder(opt)
lstm = lstm(opt)

netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)

if loading_model_path:
    checkpoint = torch.load(loading_model_path)
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])
    print('MODEL LOADING SUCCESSFULLY:', loading_model_path)

optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)

# optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999))

mse_loss = nn.MSELoss()
cse_loss = nn.CrossEntropyLoss()
l1_loss = nn.L1Loss()

netE.cuda()
netD.cuda()
lstm.cuda()
mse_loss.cuda()
cse_loss.cuda()

# #################################################################################################################
# DATASET PREPARATION
train_data = FVG(
    is_train_data=True,
    train_structure=train_structure,
    test_structure=test_structure,
    opt=opt
)
train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=True)

training_batch_generator = get_training_batch(train_loader)

test_data = FVG(
    is_train_data=False,
    train_structure=train_structure,
    test_structure=test_structure,
    opt=opt
)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=False,
                         pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)
# #################################################################################################################


#################################################################################################################

# def make_analogy_inter_subs(x, step, opt):
#     # (B, L, C, H, W)
#     # (   B, C, H, W)
#
#     none = torch.zeros([1, 3, opt.im_height, opt.im_width]).cuda()
#     x_gs = torch.stack([i for i in [x[0][step], x[1][step], x[2][step], x[3][step]]]).cuda()
#     h_gs = netE(x_gs)[1]
#     # h_gs = torch.zeros(4,32).cuda()
#
#     x_as = torch.stack([x[i][step] for i in [1, 2, 11]]).cuda()
#     # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()
#
#     h_as = netE(x_as)[0]
#     # h_as = torch.zeros(3,288).cuda()
#
#     gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
#     row0 = torch.cat([none, x_gs])
#     rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
#     to_plot = torch.cat([row0] + rows)
#
#     img = make_grid(to_plot, 5)
#     return img
#
#
# def make_analogy_intal_subs(nm, cl, idx):
#     # (B, L, C, H, W)
#     # (   B, C, H, W)
#     # netE.eval()
#     # netD.eval()
#
#     def rand_idx():
#         return 0
#
#     def rand_step():
#         # return np.random.randint(0, opt.max_step)
#         return 0
#
#     none = torch.zeros([1, 3, 64, 32]).cuda()
#     x_gs = torch.stack([i for i in [nm[idx][0], nm[idx][4], nm[idx][8], nm[idx][15]]]).cuda()
#     h_gs = netE(x_gs)[1]
#     # h_gs = torch.zeros(4,32).cuda()
#
#     x_as = torch.stack([nm[idx][rand_step()], cl[idx][rand_step()]]).cuda()
#     # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()
#
#     h_as = netE(x_as)[0]
#     # h_as = torch.zeros(2,288).cuda()
#
#     gene = [netD(torch.stack([i] * 4).cuda(), h_gs) for i in h_as]
#     row0 = torch.cat([none, x_gs])
#     rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
#     to_plot = torch.cat([row0] + rows)
#
#     img = make_grid(to_plot, 5)
#     return img
#
#
# def plot_anology(data, itr):
#     frames = []
#     for step in range(data.shape[1]):
#         # frame = data[:,step:step+1,:,:,:]
#         anology_frame = make_analogy_inter_subs(data, step, opt)
#         anology_frame = anology_frame.cpu().numpy()
#         anology_frame = np.transpose(anology_frame, (1, 2, 0))
#         # anology_frame = adjust_white_balance(anology_frame.cpu().numpy())
#         frames.append(anology_frame)
#     imageio.mimsave("{:s}/gifs/{:s}/{:d}.gif".format(opt.savedir, opt.signature, itr), frames)
#
#     # all = torch.cat([anology1, anology2], dim=1)
#
#     # writer.add_image('inter_sub', all, epoch)
#
#     # anology1 = make_analogy_intal_subs(data1, data2, 8)
#     # anology2 = make_analogy_intal_subs(data1, data2, 9)
#     # all = torch.cat([anology1, anology2], dim=1)
#     # all = adjust_white_balance(all.detach())
#     # writer.add_image('intral_sub', all, epoch)


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), torch.tensor(item), itr)


#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xn, Xc, l):

    Xn, Xc, Xmx = Xn.transpose(0, 1), Xc.transpose(0, 1), Xn.transpose(0, 1)
    hgs_n = []
    hgs_c = []

    self_rec_loss = 0
    for i in range(0, len(Xmx)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        # rp = torch.randperm(opt.batch_size).cuda()
        rdm = torch.LongTensor(1).random_(0, len(Xn))[0]
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xmx[rdm], Xmx[i]
        hamx0, hgmx0 = netE(xmx0)
        hamx1, hgmx1 = netE(xmx1)
        # ------------------------------------------------------------------
        xmx1_ = netD(hamx0, hgmx1)
        self_rec_loss += mse_loss(xmx1_, xmx1)
        # ------------------------------------------------------------------
        xn1, xc1 = Xn[i], Xc[i]
        _, hgn1 = netE(xn1)
        _, hgc1 = netE(xc1)

        hgs_n.append(hgn1)
        hgs_c.append(hgc1)

    hgs_n = torch.stack(hgs_n)
    hgs_n = torch.mean(hgs_n, 0)

    hgs_c = torch.stack(hgs_c)
    hgs_c = torch.mean(hgs_c, 0)

    # loss_out_haha = l1_loss(hgs_n,hgs_c) / 100
    loss_out_haha = mse_loss(hgs_n, hgs_c)
    # loss_out_haha.backward()
    # optimizerE.step()


    has_mx = []
    hgs_mx = []
    cse = 0
    for i in range(0, len(Xmx)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        # factor = (i / 5) ** 2 / 10
        # factor = torch.sigmoid(torch.tensor((i / 5) ** 2 / 10)).cuda()
        xmx = Xmx[i]
        hamx, hgmx = netE(xmx)
        has_mx.append(hamx)
        hgs_mx.append(hgmx)

    lstm_out_train, lstm_out_test, lstm_out_hidden = lstm(torch.stack(hgs_mx))
    cse += cse_loss(lstm_out_train, l) #* factor
    # cse /= opt.clip_len # mean

    lstm_predloss = 0
    for i in range(0, len(hgs_mx)-1):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        xmx_ = netD(has_mx[i], lstm_out_hidden[i])
        lstm_predloss += mse_loss(xmx_, Xmx[i+1])

        # lstm_predloss += mse_loss(lstm_out_hidden[i-1], hgs_mx[i])

    lstm_predloss = lstm_predloss/len(hgs_mx)

    # loss = self_rec_loss + loss_out_haha * 0.01 + cse * 0.1 + lstm_predloss
    loss = self_rec_loss + loss_out_haha * 0.01 + cse * 0.1
    # loss = cse * 0.1

    loss.backward()
    optimizerE.step()
    optimizerD.step()
    optimizerLstm.step()

    return [self_rec_loss.item(), loss_out_haha.item(), cse.item(), lstm_predloss.item()]

#################################################################################################################
# FUN TRAINING TIME !

debug_mode = False

proto_WS = np.load('testset_WS.npy', allow_pickle=True)
# random.shuffle(proto_WS[0])
# random.shuffle(proto_WS[1])
proto_WS = torch.tensor(proto_WS[0]).cuda(), torch.tensor(proto_WS[1]).cuda(), proto_WS[2]
proto_CB = np.load('testset_CB.npy', allow_pickle=True)
# random.shuffle(proto_CB[0])
# random.shuffle(proto_CB[1])
proto_CB = torch.tensor(proto_CB[0]).cuda(), torch.tensor(proto_CB[1]).cuda(), proto_CB[2]
proto_CL = np.load('testset_CL.npy', allow_pickle=True)
# random.shuffle(proto_CL[0])
# random.shuffle(proto_CL[1])
proto_CL = torch.tensor(proto_CL[0]).cuda(), torch.tensor(proto_CL[1]).cuda(), proto_CL[2]

# proto_WS = fvg.get_eval_data(False, 2, probe_session1=list(range(4, 9 + 1)), probe_session2=list(range(4, 6 + 1)))
# proto_CB = fvg.get_eval_data(False, 2, probe_session1=list(range(10, 12 + 1)), probe_session2=[])
# proto_CL = fvg.get_eval_data(False, 2, probe_session1=[], probe_session2=list(range(7, 9 + 1)))
# proto_CBG = fvg.get_eval_data(False, 2, probe_session1=list(range(10, 12 + 1)))


if not debug_mode:

    writer = SummaryWriter('%s/logs/%s' % (opt.savedir, opt.signature))

    while True:
        netE.train()
        netD.train()
        lstm.train()

        batch_cond1, batch_cond2, batch_mix, lb = next(training_batch_generator)

        losses = train_main(batch_cond1, batch_cond2, lb)
        write_tfboard(losses, itr, name='EDLoss')

        # ----------------EVAL()--------------------
        if itr % 10 == 0 and itr != 0:
            with torch.no_grad():
                netD.eval()
                netE.eval()
                lstm.eval()
                eval_WS = eval_roc(proto_WS[0], proto_WS[1], proto_WS[2], 90, [netE, lstm], opt)
                write_tfboard(eval_WS[:2], itr, name='WS')
                eval_CB = eval_roc(proto_CB[0], proto_CB[1], proto_CB[2], 90, [netE, lstm], opt)
                write_tfboard(eval_CB[:2], itr, name='CB')
                eval_CL = eval_roc(proto_CL[0], proto_CL[1], proto_CL[2], 90, [netE, lstm], opt)
                write_tfboard(eval_CL[:2], itr, name='CL')

                # plot_anology(batch_cond1, itr)
        # ----------------SAVE MODEL--------------------
        if itr % 200 == 0 and itr != 0:
            torch.save({
                'netD': netD.state_dict(),
                'netE': netE.state_dict(),
                'lstm': lstm.state_dict(),
            },
                '%s/modules/%s/%d.pickle' % (opt.savedir, opt.signature, itr), )

        itr += 1

else:
    writer = SummaryWriter('%s/logs/%s' % (opt.savedir, opt.signature+'-debug-lstm-last'))
    def experiment(glr, prb, gt, n_test, networks, opt):
        netE, lstm = networks

        fg_glr = [netE(glr[i].cuda())[1].detach() for i in range(len(glr))]
        fg_glr = torch.stack(fg_glr, 0).view(len(fg_glr), n_test, opt.fg_dim)
        glr_vec = lstm(fg_glr)[3].detach().cpu().numpy()

        fg_prb = [netE(prb[i].cuda())[1].detach() for i in range(len(prb))]
        fg_prb = torch.stack(fg_prb, 0).view(len(fg_prb), -1, opt.fg_dim)
        prb_vec = lstm(fg_prb)[3].detach().cpu().numpy()
        # prb_vec = torch.mean(fg_prb, 0).detach().cpu().numpy()
        for i in range(n_test):
                for j in range(len(fg_glr)):
                    if i < n_test - 1:
                        i_ = i + 1
                    else:
                        i_ = i
                    cs = calculate_cosine_similarity(glr_vec[30:31, i, :],
                                                     prb_vec[j:j + 1, i, :])
                    write_tfboard([cs], j, name=str(i))

    with torch.no_grad():
        netD.eval()
        netE.eval()
        lstm.eval()
        eval_WS = eval_roc(proto_WS[0], proto_WS[1], proto_WS[2], 90, [netE, lstm], opt)
        print(eval_WS)
        # write_tfboard(eval_WS[:2], itr, name='WS')
        eval_CB = eval_roc(proto_CB[0], proto_CB[1], proto_CB[2], 90, [netE, lstm], opt)
        print(eval_CB)
        # write_tfboard(eval_CB[:2], itr, name='CB')
        eval_CL = eval_roc(proto_CL[0], proto_CL[1], proto_CL[2], 90, [netE, lstm], opt)
        print(eval_CL)
        # write_tfboard(eval_CL[:2], itr, name='CL')
        # writer.close()
        # batch_cond1, batch_cond2, _ = next(testing_batch_generator)
        # plot_anology(batch_cond1, batch_cond2, itr)
        # experiment(proto_CB[0], proto_CB[1], proto_CB[2], 90, [netE, lstm], opt)

