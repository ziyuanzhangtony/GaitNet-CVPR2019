import random
import os
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from utils.dataloader import CASIAB
from utils.compute import *
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.modules_casiab_tab2 import *
from utils.dataloader import get_training_batch
#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/CB-RGB-MRCNN/',
                    # default='/home/tony/Research/CB-RGB-BS/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--max_step', type=int, default=10, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=8, help='number of parallel data loading threads')
parser.add_argument('--num_train',type=int, default=74, help='')
parser.add_argument('--glr_views',type=list, default=[90], help='')
parser.add_argument('--prb_views',type=list, default=[90], help='')
parser.add_argument('--savedir', default='./runs')
parser.add_argument('--signature', default='old_code-new_data-padtopdown-newdataloader-with70padding-best-sigmoid_factor')
opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)
#################################################################################################################
#
torch.cuda.set_device(opt.gpu)

#
# cudnn.deterministic = True
cudnn.benchmark = True

#
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
bce_loss = nn.BCELoss()
cse_loss = nn.CrossEntropyLoss()
# trp_loss = nn.TripletMarginLoss(margin=2.0)
netE.cuda()
netD.cuda()
lstm.cuda()
mse_loss.cuda()
bce_loss.cuda()
cse_loss.cuda()
# trp_loss.cuda()

# l1_crit = nn.L1Loss(size_average=False)
# reg_loss = 0
# for param in netE.parameters():
#     reg_loss += l1_crit(param)
#
# factor = 0.0005
# loss = factor * reg_loss

# #################################################################################################################
# DATASET PREPARATION
train_data = CASIAB(
    is_train_data=True,
    data_root=opt.data_root,
    clip_len=opt.max_step,
    im_height=opt.im_height,
    im_width=opt.im_width,
    seed=opt.seed
)
train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=False,
                          pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader)

test_data = CASIAB(
    is_train_data=False,
    data_root=opt.data_root,
    clip_len=opt.max_step,
    im_height=opt.im_height,
    im_width=opt.im_width,
    seed=opt.seed
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

#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xn, Xc, l):
    # netE.zero_grad()
    # netD.zero_grad()
    # lstm.zero_grad()

    Xn, Xc = Xn.transpose(0, 1), Xc.transpose(0, 1)
    accu = []
    hgs_n = []
    hgs_c = []

    self_rec_loss = 0
    for i in range(0, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        # rp = torch.randperm(opt.batch_size).cuda()
        rdm = torch.LongTensor(1).random_(0, len(Xn))[0]
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xn[rdm], Xn[i]
        hamx0, hgmx0, emmx0 = netE(xmx0)
        hamx1, hgmx1, emmx1 = netE(xmx1)
        accu.append(hamx1)
        # ------------------------------------------------------------------
        xmx1_ = netD(hamx0, hgmx1)
        self_rec_loss += mse_loss(xmx1_, xmx1)
        # ------------------------------------------------------------------
        xn1, xc1 = Xn[i], Xc[i]
        han1, hgn1, _ = netE(xn1)
        hac1, hgc1, _ = netE(xc1)

        hgs_n.append(hgn1)
        hgs_c.append(hgc1)

    hgs_n = torch.stack(hgs_n)
    hgs_n = torch.mean(hgs_n, 0)

    hgs_c = torch.stack(hgs_c)
    hgs_c = torch.mean(hgs_c, 0)

    loss_out_haha = mse_loss(hgs_n, hgs_c) / 100

    hgs_mx = []
    # fc = 0
    cse = 0
    for i in range(0, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()

        factor = (i / 5) ** 2 / 10
        # factor = torch.sigmoid(torch.tensor((i / 5) ** 2 / 10)).cuda()

        # fc += factor
        # print(factor)
        # rp = torch.randperm(opt.batch_size).cuda()

        xmx = Xn[i]
        hgs_mx.append(netE(xmx)[1])
        lstm_out_mx = lstm(torch.stack(hgs_mx))[0]
        cse += cse_loss(lstm_out_mx, l) * factor
    cse /= opt.max_step
    cse *= 0.1

    loss = self_rec_loss + loss_out_haha + cse

    loss.backward()
    optimizerE.step()
    optimizerD.step()
    optimizerLstm.step()

    return [self_rec_loss.item(),
            loss_out_haha.item(),
            cse.item()]

#################################################################################################################
# FUN TRAINING TIME !
train_eval = test_data.get_eval_data(True)
test_eval = test_data.get_eval_data(False)
writer = SummaryWriter('%s/logs/%s'%(opt.savedir,opt.signature))
while True:
    netE.train()
    netD.train()
    lstm.train()

    im_nm, im_cl,lb = next(training_batch_generator1)
    # print(lb)

    losses1 = train_main(im_nm, im_cl,lb)
    write_tfboard(losses1,itr,name='EDLoss')

    # ----------------EVAL()--------------------
    if itr % 10 == 0:
        with torch.no_grad():
            netD.eval()
            netE.eval()
            lstm.eval()
            scores_cmc_cl_train = eval_cmc(train_eval[0], train_eval[1],[netE, lstm], opt, [90], [90], True)
            scores_cmc_cl_test = eval_cmc(test_eval[0], test_eval[1], [netE, lstm], opt, [90], [90], True)
            write_tfboard(scores_cmc_cl_train, itr, name='train_accu_rank1_cl')
            write_tfboard(scores_cmc_cl_test, itr, name='test_accu_rank1_cl')
            print(scores_cmc_cl_train,scores_cmc_cl_test)
            # print()
        # ----------------SAVE MODEL--------------------
    # if itr % 200 == 0:
    #     torch.save({
    #         'netD': netD.state_dict(),
    #         'netE': netE.state_dict(),
    #         'lstm':lstm.state_dict(),
    #         },
    #         '%s/modules/%s/%d.pickle'%(opt.savedir,opt.signature,itr),)

    itr+=1


