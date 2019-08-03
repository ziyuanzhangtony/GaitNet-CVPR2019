import random
import os

import matplotlib
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
# from utils.dataloader import CASIAB_ALIGN as CASIAB
from utils.dataloader import FVG
from utils.compute import *
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.modules_casiab_cvpr2_tab2_hd import *
from utils.dataloader import get_training_batch
from utils.graph import *

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/FVG-MRCNN/SEG/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=320-64, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=64, help='size of the gait vector')
parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=128, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=64, help='the width of the input image to network')
parser.add_argument('--clip_len', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=8, help='number of parallel data loading threads')
parser.add_argument('--num_train',type=int, default=136, help='')
parser.add_argument('--savedir', default='./runs')
parser.add_argument('--signature', default='PAMI'
                                           '-136fortraining'
                                           '-NEW_MRCNN'
                                           '-AVG_LSTM_INI'
                                           '-128_64'
                                           '-max_pooling_7'
                                           '-256hidden'
                                           '-90_90'
                                           '-rate_decay_all'
                                           '-fgs_fgd_64d'
                                           '-test_len_50'
                                           '-with_factor'
                                           '-hgs_hgd_from_ha'
                                           '-HD')
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

cudnn.deterministic = True
# cudnn.benchmark = False

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
    itr = int(model_names[-1].split('_')[0])
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
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('LSTM') != -1:
        for i in m._parameters:
            if i.__class__.__name__.find('weight') != -1:
                i.data.normal_(0.0, 0.01)
            elif i.__class__.__name__.find('bias') != -1:
                i.bias.data.fill_(0)

netE = encoder(opt)
netD = decoder(opt)
lstm = lstm(opt)

netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)

if loading_model_path:
    checkpoint = torch.load(loading_model_path,'cuda:'+str(gpu_num))
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])
    print('MODEL LOADING SUCCESSFULLY:', loading_model_path)

# optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
# optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999))

schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE,500,0.9,)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD,500,0.9)
schedulerLSTM = torch.optim.lr_scheduler.StepLR(optimizerLstm,500,0.9)
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

def write_tfboard(vals,itr,name):
    for idx,item in enumerate(vals):
        writer.add_scalar('data/%s%d'%(name,idx), item, itr)

def return_clip(fg1, fg2, length):
    end_ = length
    max = calculate_cosine_similarity_multidim(fg1, fg2[0:length])
    for end in range(length, len(fg2)):
        start = end - length
        fg2_clip = fg2[start:end]
        if calculate_cosine_similarity_multidim(fg1, fg2_clip) > max:
            end_ = end
            max = calculate_cosine_similarity_multidim(fg1, fg2_clip)
    return (end_ - length, end_)

def alignment_and_clip(clip1, clip2, length, netE_use):
    start = np.random.randint(0, clip2.shape[1] - length)
    clip1_ = clip1[:, start:start + length, :, :, :]
    # start = np.random.randint(0, clip2.shape[1] - length)
    # clip2_ = clip2[:, start:start + length, :, :, :]

    clip1_fg = [netE_use(clip1_[i].cuda())[1].detach().cpu().numpy() for i in range(len(clip1_))]
    clip2_fg = [netE_use(clip2[i].cuda())[1].detach().cpu().numpy() for i in range(len(clip2))]
    clip2_ = []
    for sbi in range(len(clip1_)):
        start, end = return_clip(clip1_fg[sbi], clip2_fg[sbi], length)
        clip2_.append(clip2[sbi, start:end, :, :, :])
    clip2_ = torch.stack(clip2_)
    return clip1_, clip2_

def show_img(im1, im2, itr, opt):
    all = torch.cat([im1[0], im2[0]], dim=0)
    fname = '%s/analogy/%s/%d.png' % (opt.savedir, opt.signature, itr)
    save_image(all, fname, 20)

#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xa, Xb, l):
    Xa, Xb = Xa.transpose(0, 1), Xb.transpose(0, 1)
    hgd_a = []
    hgd_b = []
    self_rec_loss = 0
    hgs_loss = 0
    for i in range(0, len(Xa)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        rdm = torch.LongTensor(1).random_(0, len(Xa))[0]
        # ------------------------------------------------------------------
        xa0, xa1, xb1 = Xa[rdm], Xa[i], Xb[i]
        ha_a0, hg_a0 = netE(xa0)
        ha_a1, hg_a1 = netE(xa1)
        # ------------------------------------------------------------------
        # xrecon loss
        xa1_ = netD(ha_a0, hg_a1)
        self_rec_loss += mse_loss(xa1_, xa1)
        # ------------------------------------------------------------------
        # dyn gait sim loss
        ha_b1, hg_b1 = netE(xb1)
        hgd_a.append(hg_a1)
        hgd_b.append(hg_b1)
        # ------------------------------------------------------------------
        # sta gait sim loss
        hgs_loss += mse_loss(ha_a0[:,:128], ha_a1[:,:128])
        hgs_loss += mse_loss(ha_a1[:,:128], ha_b1[:,:128])
        hgs_loss += cse_loss(netE.fgs_clf(ha_a0[:,:128]),l)
    self_rec_loss /= len(Xa)
    self_rec_loss*=2000

    hgs_loss/=len(Xa)
    hgs_loss*=0.25
    # hgs_loss *= 10


    hgd_a = torch.stack(hgd_a)
    hgd_a = hgd_a.mean(dim=0)

    hgd_b = torch.stack(hgd_b)
    hgd_b = hgd_b.mean(dim=0)

    hgd_loss = mse_loss(hgd_a, hgd_b)
    hgd_loss *= 10


    hgs_a = []
    cse = 0
    for i in range(0, len(Xa)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()

        factor = (i / 5) ** 2 / 10
        # factor = torch.sigmoid(torch.tensor((i / 5) ** 2 / 10)).cuda()

        xa = Xa[i]
        hgs_a.append(netE(xa)[1])

        lstm_out = lstm(hgs_a)[0]
        cse += cse_loss(lstm_out, l) * factor


    cse /= opt.clip_len # mean
    # cse *= 0.1


    loss = self_rec_loss + hgd_loss + cse + hgs_loss
    # loss = self_rec_loss + hgd_loss + cse

    loss.backward()
    optimizerE.step()
    optimizerD.step()
    optimizerLstm.step()

    return [self_rec_loss.item(),
            hgd_loss.item(),
            cse.item(),
            hgs_loss.item()]

#################################################################################################################
# FUN TRAINING TIME !

debug_mode = False

# proto_WS = np.load('testset_WS.npy', allow_pickle=True)
# # random.shuffle(proto_WS[0])
# # random.shuffle(proto_WS[1])
# proto_WS = torch.tensor(proto_WS[0]).cuda(), torch.tensor(proto_WS[1]).cuda(), proto_WS[2]
# proto_CB = np.load('testset_CB.npy', allow_pickle=True)
# # random.shuffle(proto_CB[0])
# # random.shuffle(proto_CB[1])
# proto_CB = torch.tensor(proto_CB[0]).cuda(), torch.tensor(proto_CB[1]).cuda(), proto_CB[2]
# proto_CL = np.load('testset_CL.npy', allow_pickle=True)
# # random.shuffle(proto_CL[0])
# # random.shuffle(proto_CL[1])
# proto_CL = torch.tensor(proto_CL[0]).cuda(), torch.tensor(proto_CL[1]).cuda(), proto_CL[2]

# proto_WS = test_data.get_eval_format(2, list(range(4, 9 + 1)), list(range(4, 6 + 1)))
# proto_CB = test_data.get_eval_format(2, list(range(10, 12 + 1)), [])
# proto_CL = test_data.get_eval_format(2, [], list(range(7, 9 + 1)))
# proto_CBG = test_data.get_eval_format(2, list(range(10, 12 + 1)), [])
proto_ALL = test_data.get_eval_format_all()


if not debug_mode:

    writer = SummaryWriter('%s/logs/%s' % (opt.savedir, opt.signature))
    acc001_max = 0

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
                # eval_WS = eval_roc_two(proto_WS[0], proto_WS[1], proto_WS[2], 90, [netE, lstm], opt)
                # write_tfboard(eval_WS[:2], itr, name='WS')
                #
                # eval_CB = eval_roc_two(proto_CB[0], proto_CB[1], proto_CB[2], 90, [netE, lstm], opt)
                # write_tfboard(eval_CB[:2], itr, name='CB')
                #
                # eval_CL = eval_roc_two(proto_CL[0], proto_CL[1], proto_CL[2], 90, [netE, lstm], opt)
                # write_tfboard(eval_CL[:2], itr, name='CL')
                #
                # eval_CBG = eval_roc_two(proto_CBG[0], proto_CBG[1], proto_CBG[2], 90, [netE, lstm], opt)
                # write_tfboard(eval_CBG[:2], itr, name='CBG')

                # acc001 = [eval_WS[0],eval_CB[0],eval_CL[0],eval_CBG[0]]
                # acc001_mean = sum(acc001)/len(acc001)

                eval_ALL = eval_roc_two(proto_ALL[0], proto_ALL[1], proto_ALL[2], 90, [netE, lstm], opt)
                print(eval_ALL)
                acc001 = eval_ALL[0]
                acc001_mean = eval_ALL[0]

                print(itr)
                print(acc001)
                print(acc001_mean)
                print()
                if acc001_mean >= acc001_max:
                    torch.save({
                        'netD': netD.state_dict(),
                        'netE': netE.state_dict(),
                        'lstm': lstm.state_dict(),
                    },
                        '%s/modules/%s/%d_%d.pickle' % (opt.savedir, opt.signature, itr, int(acc001_mean * 1000)))
                    acc001_max = acc001_mean
                # plot_anology(batch_cond1, itr)
        itr += 1

else:
    with torch.no_grad():
        netD.eval()
        netE.eval()
        lstm.eval()
        # eval_WS = eval_roc_two(proto_WS[0], proto_WS[1], proto_WS[2], 90, [netE, lstm], opt)
        # write_tfboard(eval_WS[:2], itr, name='WS')

        # eval_CB = eval_roc_two(proto_CB[0], proto_CB[1], proto_CB[2], 90, [netE, lstm], opt)
        # write_tfboard(eval_CB[:2], itr, name='CB')

        # eval_CL = eval_roc_two(proto_CL[0], proto_CL[1], proto_CL[2], 90, [netE, lstm], opt)
        # write_tfboard(eval_CL[:2], itr, name='CL')

        # eval_CBG = eval_roc_two(proto_CBG[0], proto_CBG[1], proto_CBG[2], 90, [netE, lstm], opt)
        # write_tfboard(eval_CBG[:2], itr, name='CBG')

        eval_ALL = eval_roc_two(proto_ALL[0], proto_ALL[1], proto_ALL[2], 90, [netE, lstm], opt)
        print(eval_ALL)

        # acc001 = [eval_WS[0], eval_CB[0], eval_CL[0], eval_CBG[0]]
        # acc005 = [eval_WS[1], eval_CB[1], eval_CL[1], eval_CBG[1]]
        # acc001_mean = sum(acc001) / len(acc001)
        # acc005_mean = sum(acc005) / len(acc005)


        # print(itr)
        # print(acc001)
        # print(acc001_mean)
        # print()
        # print(acc005)
        # print(acc005_mean)

