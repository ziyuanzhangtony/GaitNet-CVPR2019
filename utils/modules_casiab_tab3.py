import torch
from utils.basic_networks import *

class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        self.em_dim = opt.em_dim
        nc = 3
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            # vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            # vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            # vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            # vgg_layer(nf * 8, nf * 8),

            # vgg_layer(nf * 8, nf * 8),
            # vgg_layer(nf * 8, nf * 8),

            # nn.Conv1d(nf * 8, self.em_dim, 4, 1, 0),
            # nn.BatchNorm2d(self.em_dim),
        )

        self.flatten = nn.Sequential(
            nn.Linear(nf * 8 * 2 * 4,self.em_dim),
            nn.BatchNorm1d(self.em_dim),
        )

    def forward(self, input):
        embedding = self.main(input).view(-1, 64 * 8 * 2 * 4)
        embedding = self.flatten(embedding)
        fa, fg = torch.split(embedding, [self.opt.fa_dim, self.opt.fg_dim], dim=1)
        return fa, fg, embedding
class decoder(nn.Module):
    def __init__(self, opt):
        super(decoder, self).__init__()
        self.opt = opt
        nc = 3
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
            # vgg_layer(nf * 8, nf * 8),

            dcgan_upconv(nf * 8, nf * 4),
            # vgg_layer(nf * 4, nf * 4),

            dcgan_upconv(nf * 4, nf * 2),
            # vgg_layer(nf * 2, nf * 2),

            dcgan_upconv(nf * 2, nf),
            # vgg_layer(nf, nf),

            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
            # because image pixels are from 0-1, 0-255
        )



    def forward(self, fa, fg):
        hidden = torch.cat([fa, fg], 1).view(-1, self.opt.em_dim)
        small = self.trans(hidden).view(-1, 64 * 8, 4, 2)
        img = self.main(small)
        return img
class lstm(nn.Module):
    def __init__(self, opt):
        super(lstm, self).__init__()
        hidden_dim = 128
        tagset_size = 100
        self.source_dim = opt.fg_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.lens = 0
        self.lstm = nn.LSTM(self.source_dim, hidden_dim, 3)
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.main = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hidden_dim, tagset_size),
            nn.BatchNorm1d(tagset_size),
        )
    def forward(self, batch):
        lens = batch.shape[0]
        lstm_out, _ = self.lstm(batch.view(lens,-1,self.source_dim))
        lstm_out_test = self.fc1(torch.mean(lstm_out.view(lens,-1,self.hidden_dim),0))

        # lstm_out_test = nn.functional.normalize(lstm_out_test, p=2)

        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)
        return lstm_out_train,lstm_out_test