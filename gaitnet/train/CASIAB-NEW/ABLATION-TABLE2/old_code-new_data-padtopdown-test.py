import random
import os
import torch.optim as optim
from torchvision.utils import make_grid
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.dataloader import CASIAB
from utils.compute import *
import torch.backends.cudnn as cudnn
# cudnn.deterministic = True
cudnn.benchmark = True

torch.cuda.set_device(0)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#################################################################################################################

from utils.modules_casiab_tab2 import *

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
# gpu_num = int(input('Tell me the gpu you wanna use for this experiment:'))
# parser.add_argument('--gpu', type=int, default=gpu_num)
parser.add_argument('--siter', type=int, default=13000, help='number of itr to start with')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--data_root',
                    default='/home/tony/Research/CB-RGB-MRCNN/',
                    # default='/home/tony/Research/CB-RGB-BS/',
                     help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--fa_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--fg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--im_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--im_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--data_threads', type=int, default=2, help='number of parallel data loading threads')
# parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--num_train',type=int, default=74, help='')
parser.add_argument('--glr_views',type=list, default=[90], help='')
parser.add_argument('--prb_views',type=list, default=[90], help='')

time_now = 'old_code-new_data-padtopdown-olddataloader'
parser.add_argument('--signature', default=time_now)
parser.add_argument('--savedir', default='./runs')
opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
# os.makedirs('%s/analogy/%s'%(opt.savedir,opt.signature), exist_ok=True)
os.makedirs('%s/modules/%s'%(opt.savedir,opt.signature), exist_ok=True)
#################################################################################################################
# MODEL PROCESS
netE = encoder(opt)
netD = decoder(opt)
lstm = lstm(opt)
netE.apply(init_weights)
netD.apply(init_weights)
lstm.apply(init_weights)

optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)
optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=0.001)

# optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
# optimizerLstm = optim.Adam(lstm.parameters(), lr=opt.lr, betas=(0.9, 0.999))

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
cse_loss = nn.CrossEntropyLoss()
trp_loss = nn.TripletMarginLoss(margin=2.0)
netE.cuda()
netD.cuda()
lstm.cuda()
mse_loss.cuda()
bce_loss.cuda()
cse_loss.cuda()
trp_loss.cuda()

# #################################################################################################################
# DATASET PREPARATION
def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(),sequence[1].cuda(),sequence[2].cuda(),sequence[3].cuda()
            yield batch


train_data1 = CASIAB(
    is_train_data=True,
    data_root=opt.data_root,
    clip_len=opt.max_step,
    im_height=opt.im_height,
    im_width=opt.im_width,
    seed=opt.seed
)
train_loader1 = DataLoader(train_data1,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

training_batch_generator1 = get_training_batch(train_loader1)

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
                         drop_last=True,
                         pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)
#################################################################################################################

def make_analogy(x):
    # (B, L, C, H, W)
    # (   B, C, H, W)
    netE.eval()
    netD.eval()

    def rand_idx():
        return np.random.randint(0, opt.batch_size)
        # return 0

    def rand_step():
        return np.random.randint(0, opt.max_step)
        # return 0

    none = torch.zeros([1, 3, 64, 64]).cuda()
    x_gs = torch.stack([i for i in [x[0][rand_step()], x[0][rand_step()], x[0][rand_step()], x[0][rand_step()],
                                    x[rand_idx()][rand_step()], x[rand_idx()][rand_step()], x[rand_idx()][rand_step()],
                                    x[rand_idx()][rand_step()]]]).cuda()
    h_gs = netE(x_gs)[1]
    # h_gs = torch.zeros(8,20).cuda()

    x_as = torch.stack([x[i][rand_step()] for i in [0, rand_idx(), rand_idx(), rand_idx(), rand_idx()]]).cuda()
    # x_as = torch.stack([x[i][0] for i in [2, 2, 2, 2, 2]]).cuda()

    h_as = netE(x_as)[0]
    # h_as = torch.ones(5,128).cuda()


    gene = [netD(torch.stack([i] * 8).cuda(), h_gs) for i in h_as]
    row0 = torch.cat([none, x_gs])
    rows = [torch.cat([e.unsqueeze(0), gene[i]]) for i, e in enumerate(x_as)]
    to_plot = torch.cat([row0] + rows)


    img = make_grid(to_plot,9)
    return img

#################################################################################################################
# TRAINING FUNCTION DEFINE
def train_main(Xn, Xc, Xmx, l):
    l2 = l+ opt.num_train
    # l2 = l
    Xn, Xc, Xmx = Xn.transpose(0, 1), Xc.transpose(0, 1), Xmx.transpose(0, 1)
    accu = []
    hgs_n = []
    hgs_c = []

    self_rec_loss = 0
    for i in range(5, len(Xn)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        rp = torch.randperm(opt.batch_size).cuda()
        rdm = torch.LongTensor(1).random_(5, len(Xn))[0]
        # ------------------------------------------------------------------
        xmx0, xmx1 = Xmx[rdm], Xmx[i]
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

    # accu = torch.stack(accu)
    # accu = torch.mean(accu, 0)
    # out = clfer(accu, 1)
    # loss_out = cse_loss(out, l2) / 20

    hgs_n = torch.stack(hgs_n)
    hgs_n = torch.mean(hgs_n, 0)

    hgs_c = torch.stack(hgs_c)
    hgs_c = torch.mean(hgs_c, 0)

    # out_hgs_n = clfer(hgs_n,2)
    # loss_out_haha = cse_loss(out_haha, l) / 20
    loss_out_haha = mse_loss(hgs_n, hgs_c) / 100

    # xmx1_ = netD(hamx0,hgmx1)
    # cross_rec_loss = mse_loss(xmx1_,xmx1)

    # ------------------------------------------------------------------
    # xn0, xn1, xc0, xc1 = Xn[rdm], Xn[i], Xc[rdm], Xc[i]
    # # han0, hgn0 = netE(xn0)
    # han1, hgn1, emn1 = netE(xn1)
    # # hac0, hgc0 = netE(xc0)
    # hac1, hgc1, emc1 = netE(xc1)
    # ------------------------------------------------------------------

    # loss = loss_out + self_rec_loss + loss_out_haha
    loss = self_rec_loss + loss_out_haha

    loss.backward()
    optimizerE.step()
    optimizerD.step()

    # writer.add_image('xn1_', xn1_, itr)
    # writer.add_image('xc1_', xc1_, itr)
    return [loss.data.cpu().numpy(),
            loss_out_haha.data.cpu().numpy(),
            self_rec_loss.data.cpu().numpy()]


def train_lstm(x_n,x_c,x_mx,l):
    x_n = x_n.transpose(0, 1)
    x_c = x_c.transpose(0, 1)
    x_mx = x_mx.transpose(0, 1)
    cse=0
    mse = 0
    trp = 0
    hgs_nm = []
    hgs_cl = []
    hgs_mx = []
    for i in range(0, len(x_n)):
        netE.zero_grad()
        netD.zero_grad()
        lstm.zero_grad()
        factor = (i/5)**2/10

        # rp = torch.randperm(opt.batch_size).cuda()

        xmx = x_mx[i]
        hgs_mx.append(netE(xmx)[1])
        lstm_out_mx = lstm(torch.stack(hgs_mx))[0]
        cse += cse_loss(lstm_out_mx,Variable(l))*factor

        xn = x_n[i]
        hgs_nm.append(netE(xn)[1])
        hgs_nm_ = torch.stack(hgs_nm)
        lstm_out_n = lstm(hgs_nm_)[1]

        xc = x_c[i]
        hgs_cl.append(netE(xc)[1])
        lstm_out_c = lstm(torch.stack(hgs_cl))[1]
        mse += mse_loss(lstm_out_n,lstm_out_c.detach())*factor

        # trp += 0.1*trp_loss(out_n,out_c,out_c[rp,:])

    cse /= opt.max_step
    cse *= 0.1
    mse /= opt.max_step
    mse *= 0.1

    los = cse
    los.backward()
    optimizerLstm.step()
    optimizerE.step()
    return [cse.data.cpu().numpy(),mse.data.cpu().numpy()]

#################################################################################################################
# FUN TRAINING TIME !
train_eval = test_data.get_eval_data(True)
test_eval = test_data.get_eval_data(False)
# max_test = 0
# max_test_file = ''
# module_names = os.listdir('%s/modules/%s'%(opt.savedir,opt.signature))
# module_names = sorted( module_names, key=lambda a: int(a.split(".")[0]) )
total = 0
cnt = 0
for name_idx in range(10000,15000,200):
    name = '%d.pickle'%(name_idx)
    full_name = os.path.join('%s/modules/%s'%(opt.savedir,opt.signature),name)
    checkpoint = torch.load(full_name)
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    lstm.load_state_dict(checkpoint['lstm'])

    with torch.no_grad():
        netD.eval()
        netE.eval()
        lstm.eval()
        scores_cmc_cl_train = eval_cmc(train_eval[0], train_eval[1],[netE, lstm], opt, [90], [90], True)
        scores_cmc_cl_test = eval_cmc(test_eval[0], test_eval[1], [netE, lstm], opt, [90], [90], True)
        total += scores_cmc_cl_test [0]
        cnt+=1
        print(name,total,cnt)
        # if scores_cmc_cl_test[0] > max_test:
        #     max_test =scores_cmc_cl_test[0]
        #     max_test_file = name
        # print(name,scores_cmc_cl_train,scores_cmc_cl_test,max_test)
print(total/cnt)




