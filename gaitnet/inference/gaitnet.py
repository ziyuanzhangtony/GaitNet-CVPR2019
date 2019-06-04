import numpy as np
import torch
import time
from torchvision import transforms
import torch.nn as nn
import argparse
class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1
                      ),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)
class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 4, 2, 1, ),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)
class encoder(nn.Module):
    def __init__(self, nc=3):
        super(encoder, self).__init__()
        self.em_dim = opt.em_dim
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),

            # nn.Conv1d(nf * 8, self.em_dim, 4, 1, 0),
            # nn.BatchNorm2d(self.em_dim),
        )

        self.flatten = nn.Sequential(
            nn.Linear(nf * 8 * 2 * 4,self.em_dim),
            nn.BatchNorm1d(self.em_dim),
        )

        self.ha_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.em_dim, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, opt.ha_dim),
            nn.BatchNorm1d(opt.ha_dim)
        )

        self.hg_fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.em_dim, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, self.em_dim // 2),
            nn.BatchNorm1d(self.em_dim // 2),

            nn.LeakyReLU(),
            nn.Linear(self.em_dim // 2, opt.hg_dim),
            nn.BatchNorm1d(opt.hg_dim)
        )

    def forward(self, input):
        embedding = self.main(input).view(-1, 64 * 8 * 2 * 4)
        embedding = self.flatten(embedding)
        ha, hg = torch.split(embedding, [opt.ha_dim, opt.hg_dim], dim=1)
        return ha, hg, embedding
class decoder(nn.Module):
    def __init__(self, nc=3):
        super(decoder, self).__init__()
        nf = 64
        self.em_dim = opt.em_dim

        self.trans = nn.Sequential(
            nn.Linear(self.em_dim, nf * 8 * 2 * 4),
            nn.BatchNorm1d(nf * 8 * 2 * 4),
        )

        self.main = nn.Sequential(
            # nn.ConvTranspose2d(self.em_dim, nf * 8, 4, 1, 0),
            # nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2),
            vgg_layer(nf * 8, nf * 8),

            dcgan_upconv(nf * 8, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_upconv(nf * 4, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_upconv(nf * 2, nf),
            vgg_layer(nf, nf),

            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
            # because image pixels are from 0-1, 0-255
        )



    def forward(self, ha, hg):
        hidden = torch.cat([ha, hg], 1).view(-1, opt.em_dim)
        small = self.trans(hidden).view(-1, 64 * 8, 4, 2)
        img = self.main(small)
        return img
class lstm(nn.Module):
    def __init__(self, hidden_dim=128, tagset_size=226):
        super(lstm, self).__init__()
        self.source_dim = opt.hg_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.lens = 0
        self.lstm = nn.LSTM(self.source_dim, hidden_dim, 3)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

        )
        self.main = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hidden_dim, tagset_size),
            nn.BatchNorm1d(tagset_size),
        )

    def forward(self, batch):
        lens = batch.shape[0]
        lstm_out, _ = self.lstm(batch.view(lens, -1, self.source_dim))
        lstm_out_test = self.fc1(torch.mean(lstm_out.view(lens, -1, self.hidden_dim), 0))
        # lstm_out_test = self.fc1(lstm_out.view(lens,-1,self.hidden_dim)[-1])
        # lstm_out_test = torch.mean(batch.view(lens,-1,64),0)
        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)
        return lstm_out_train, lstm_out_test, lstm_out
#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()

# print(torch.__version__)

is_cuda = torch.cuda.is_available()

parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--em_dim', type=int, default=320, help='size of the pose vector')
parser.add_argument('--ha_dim', type=int, default=288, help='size of the appearance vector')
parser.add_argument('--hg_dim', type=int, default=32, help='size of the gait vector')
parser.add_argument('--image_width', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
opt = parser.parse_args()
netE = encoder()
netD = decoder()
lstm = lstm()
if is_cuda:
    checkpoint = torch.load('gaitnet/model.pickle')
else:
    checkpoint = torch.load('gaitnet/model.pickle', map_location='cpu')
netE.load_state_dict(checkpoint['netE'])
netD.load_state_dict(checkpoint['netD'])
lstm.load_state_dict(checkpoint['lstm'])

netD.eval()
netE.eval()
lstm.eval()

if is_cuda:
    netE.cuda()
    netD.cuda()
    lstm.cuda()

def process_video(video):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((opt.image_height, opt.image_width)),
        transforms.ToTensor()
    ])

    shape = [len(video), 3, opt.image_height, opt.image_width]
    data = np.zeros(shape, np.float32)
    for i in range(len(video)):
            img = video[i]
            img = transform(img)
            data[i] = img

    return torch.from_numpy(data)

def main(video):
    video = process_video(video)
    start = time.time()
    if is_cuda:
        video = video.cuda()
    hp_glr = netE(video)[1]
    glr_vec = lstm(hp_glr)[1].detach().cpu().numpy()
    print("GAITNET TIME:",time.time()-start)
    print("GAITNET FPS:", len(video)/(time.time()-start))
    return glr_vec

