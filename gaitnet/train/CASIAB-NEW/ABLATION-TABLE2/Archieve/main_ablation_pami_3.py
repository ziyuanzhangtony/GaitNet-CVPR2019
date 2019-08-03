import random
import os
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
# from utils.dataloader import CASIAB_ALIGN as CASIAB
from utils.dataloader import CASIAB
from utils.compute import *
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.Archieve.modules_casiab_pami3_tab2 import *
from utils.dataloader import get_training_batch
from torchvision.utils import save_image
#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    # default='/home/tony/Documents/CASIA-B/SEG',
                    # default='/home/tony/Documents/CASIA-B-/SEG',
                    # default='/home/tony/Research/CB-RGB-BS/',
                     default='/home/tony/Research/CB-SEG-MRCNN-V2/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=320-64, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=64, help='size of the gait vector')
parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--clip_len', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=8, help='number of parallel data loading threads')
parser.add_argument('--num_train',type=int, default=74, help='')
parser.add_argument('--savedir', default='./runs')
parser.add_argument('--signature', default='PAMI-74fortraining-NEW_MRCNN-NM_CL-AVG_LSTM_INI-64_32-max_pooling_3-256hidden-90_90-rate_decay_all-fgs_fgd-64d-test_len_50-')
opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)

train_structure = {
    'clip1': ([90],
              {'cl':list(range(1, 2 + 1)),
               'nm':list(range(1, 6 + 1))}
              ),
    'clip2': ([90],
          {'cl':list(range(1, 2 + 1)),
           'nm':list(range(1, 6 + 1))}
          ),
}

test_structure = {
    'gallery': ([90], ['nm'], [1]),
    'probe': ([90], ['cl'], [2]),
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
    checkpoint = torch.load(loading_model_path)
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])
    print('MODEL LOADING SUCCESSFULLY:', loading_model_path)

optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)

schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE,500,0.9)
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
train_data = CASIAB(
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

test_data = CASIAB(
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
def train_main(Xn, Xc, l, X2):

    Xn, Xc, X2 = Xn.transpose(0, 1), Xc.transpose(0, 1), X2.transpose(0, 1)
    hgs_n = []
    hgs_c = []
    hgs_2 = []
    # apperance_loss_same = 0
    # apperance_loss_diff = 0
    self_rec_loss = 0
    hgs_loss = 0
    for i in range(0, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        # rp = torch.randperm(opt.batch_size).cuda()
        rdm = torch.LongTensor(1).random_(0, len(Xn))[0]
        # rdm = i-1
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xn[rdm], Xn[i]
        hamx0, hgmx0 = netE(xmx0)
        hamx1, hgmx1 = netE(xmx1)
        ha2, hg2 = netE(X2[i])
        # apperance_loss_same += mse_loss(hamx0,hamx1)
        # apperance_loss_diff += mse_loss(hamx0, ha2)
        hgs_2.append(hg2)
        # ------------------------------------------------------------------
        xmx1_ = netD(hamx0, hgmx1)
        self_rec_loss += mse_loss(xmx1_, xmx1)
        # ------------------------------------------------------------------
        xn1, xc1 = Xn[i], Xc[i]
        _, hgn1 = netE(xn1)
        _, hgc1 = netE(xc1)


        hgs_n.append(hgn1[1])
        hgs_c.append(hgc1[1])
        # ------------------------------------------------------------------
        hgs_loss += mse_loss(hgmx0[0], hgmx1[0])
        hgs_loss += mse_loss(hgn1[0], hgc1[0])
        hgs_loss += cse_loss(netE.fgs_clf(hgn1[0]),l)

    hgs_loss/=len(Xn)

    hgs_n = torch.stack(hgs_n)
    hgs_n = torch.mean(hgs_n, 0)

    hgs_c = torch.stack(hgs_c)
    hgs_c = torch.mean(hgs_c, 0)

    # hgs_2 = torch.stack(hgs_2)
    # hgs_2 = torch.mean(hgs_2, 0)

    # loss_out_haha = l1_loss(hgs_n,hgs_c)
    loss_out_same = mse_loss(hgs_n, hgs_c)
    # loss_out_haha.backward()
    # optimizerE.step()
    # loss_out_diff = mse_loss(hgs_n, hgs_2)

    # out_same = netE.clf(hgs_n-hgs_c)
    # out_diff = netE.clf(hgs_n - hgs_2)
    # loss_out_same = cse_loss(out_same, torch.ones(16,dtype=torch.long).cuda())
    # loss_out_diff = cse_loss(out_diff, torch.zeros(16,dtype=torch.long).cuda())

    # trip_loss = max( mse_loss(hgs_n, hgs_c)-mse_loss(hgs_n, hgs_2) , mse_loss(hgs_n, hgs_c)-mse_loss(hgs_n, hgs_c))


    hgs_mx = []
    # fc = 0
    cse = 0
    for i in range(0, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()

        factor = (i / 5) ** 2 / 10
        # factor = torch.sigmoid(torch.tensor((i / 5) ** 2 / 10)).cuda()

        xmx = Xn[i]
        hgs_mx.append(netE(xmx)[1])

        lstm_out_mx = lstm(hgs_mx)[0]
        cse += cse_loss(lstm_out_mx, l) * factor


    cse /= opt.clip_len # mean

    # loss = self_rec_loss + loss_out_same * 0.01 + cse * 0.1

    loss = self_rec_loss + loss_out_same * 0.01 + cse * 0.1 + hgs_loss*0.1
    # loss =  loss_out_same+loss_out_diff + cse
    # loss = cse + self_rec_loss + loss_out_haha * 0.01

    loss.backward()
    optimizerE.step()
    optimizerD.step()
    optimizerLstm.step()

    return [self_rec_loss.item(), loss_out_same.item(), cse.item(),hgs_loss.item()]

#################################################################################################################
# FUN TRAINING TIME !
train_eval = train_data.get_eval_format()
test_eval = test_data.get_eval_format()

# train_glr = train_eval[0][0]
# train_prb = train_eval[1][0]
# train_glr = train_glr.transpose(0, 1)
# train_prb = train_prb.transpose(0, 1)
# train_glr, train_prb = alignment_and_clip(train_glr, train_prb, 60, netE)
# train_eval[0][0] = train_eval[0][0].transpose(0, 1)
# train_eval[1][0] = train_eval[1][0].transpose(0, 1)
#
# test_glr = test_eval[0][0]
# test_prb = test_eval[1][0]
# test_glr = test_glr.transpose(0, 1)
# test_prb = test_prb.transpose(0, 1)
# test_glr, test_prb = alignment_and_clip(test_glr, test_prb, 60, netE)
# test_eval[0][0] = test_eval[0][0].transpose(0, 1)
# test_eval[1][0] = test_eval[1][0].transpose(0, 1)

# with torch.no_grad():
#     netD.eval()
#     netE.eval()
#     lstm.eval()
#     # scores_cmc_cl_train, _ = eval_cmc(train_eval[0], train_eval[1],[netE, lstm], opt, [90], [90], True)
#     scores_cmc_cl_test, _ = eval_cmc(test_eval[0], test_eval[1], [netE, lstm], opt, [90], [90], True)
#     # write_tfboard(scores_cmc_cl_train, itr, name='train_accu_rank1_cl')
#     # write_tfboard(scores_cmc_cl_test, itr, name='test_accu_rank1_cl')
#
#     print()
#     print(itr)
#     print(scores_cmc_cl_test)
# # for param_group in optimizerE.param_groups:
# #     print(param_group['lr'])
#
# # print()
max_test_acc = 0
writer = SummaryWriter('%s/logs/%s'%(opt.savedir,opt.signature))
while True:
    netE.train()
    netD.train()
    lstm.train()

    batch_si1_c1, batch_si1_c2, lb1, batch_si2_c, lb2 = next(training_batch_generator)
    # batch_cond1, batch_cond2 = alignment_and_clip(batch_cond1, batch_cond2, 20, netE)
    schedulerE.step(itr)
    schedulerD.step(itr)
    schedulerLSTM.step(itr)

    losses = train_main(batch_si1_c1, batch_si1_c2, lb1, batch_si2_c )
    write_tfboard(losses,itr,name='EDLoss')

    # ----------------EVAL()--------------------
    if itr % 10 == 0:
        with torch.no_grad():
            netD.eval()
            netE.eval()
            lstm.eval()
            scores_cmc_cl_train, _ = eval_cmc(train_eval[0], train_eval[1],[netE, lstm], opt, [90], [90], True)
            scores_cmc_cl_test, _ = eval_cmc(test_eval[0], test_eval[1], [netE, lstm], opt, [90], [90], True)
            write_tfboard(scores_cmc_cl_train, itr, name='train_accu_rank1_cl')
            write_tfboard(scores_cmc_cl_test, itr, name='test_accu_rank1_cl')

            if scores_cmc_cl_test[0] > max_test_acc:
                torch.save({
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict(),
                    'lstm': lstm.state_dict(),
                },
                    '%s/modules/%s/%d_%d.pickle' % (opt.savedir, opt.signature, itr, int(scores_cmc_cl_test[0]*100)), )
                max_test_acc = scores_cmc_cl_test[0]

            print()
            print(itr)
            print(scores_cmc_cl_train,scores_cmc_cl_test)
            for param_group in optimizerE.param_groups:
                print(param_group['lr'])

            # print()
    #     # ----------------SAVE MODEL--------------------
    # if itr % 200 == 0:
    #     torch.save({
    #         'netD': netD.state_dict(),
    #         'netE': netE.state_dict(),
    #         'lstm':lstm.state_dict(),
    #         },
    #         '%s/modules/%s/%d.pickle'%(opt.savedir,opt.signature,itr),)

    itr+=1


