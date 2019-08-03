import torch
from utils.basic_networks import *
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, opt):
        super(encoder, self).__init__()
        self.opt = opt
        self.em_dim = opt.em_dim
        nc = 3
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
        )
        self.flatten = nn.Sequential(
            nn.Linear(nf * 8 * 2 * 4,self.em_dim),
            nn.BatchNorm1d(self.em_dim),
        )
    def forward(self, input):
        embedding = self.main(input).view(-1, 64 * 8 * 2 * 4)
        embedding = self.flatten(embedding)
        fa, fg = torch.split(embedding, [self.opt.fa_dim, self.opt.fg_dim], dim=1)
        # fg = F.softmax(fg,1)
        return fa, fg

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
    def forward(self, fa, fg):
        hidden = torch.cat([fa, fg], 1).view(-1, self.opt.em_dim)
        small = self.trans(hidden).view(-1, 64 * 8, 4, 2)
        img = self.main(small)
        return img

class lstm(nn.Module):
    def __init__(self, opt):
        super(lstm, self).__init__()
        self.source_dim = opt.fg_dim
        self.hidden_dim = opt.lstm_hidden_dim
        self.tagset_size = opt.num_train
        # self.lstm = nn.LSTM(self.source_dim, self.hidden_dim, 3, bidirectional=True)
        self.lstm = nn.LSTM(self.source_dim, self.hidden_dim, 3)

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            # nn.Linear(self.source_dim, self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.match = nn.Sequential(
            nn.Linear(self.source_dim, self.source_dim),
            nn.BatchNorm1d(self.source_dim),
            nn.LeakyReLU(0.2),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.source_dim, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.main = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.BatchNorm1d(self.tagset_size),
            # nn.Softmax(dim=1),
        )
    def forward(self, batch):
        num_frames = batch.shape[0]
        batch_size = batch.shape[1]

        lstm_out, hidden = self.lstm(batch)

        # attention_scales = [self.attention(i) for i in batch]

        # attention_scales = [torch.split(i,[self.hidden_dim,1],dim=1)[1] for i in lstm_out]
        # lstm_out_128 = [torch.split(i,[self.hidden_dim,1],dim=1)[0] for i in lstm_out]
        # attention_scales = torch.stack(attention_scales)
        # attention_scales = attention_scales[:,:,0]
        # attention_scales = F.softmax(attention_scales,0)

        # lstm_out_128 = torch.stack(lstm_out_128)

        # attention_result = attention_scales*batch
        # attention_result = torch.sum(attention_result,0)

        # lstm_out_test = self.fc1(lstm_out.view(num_frames,-1,self.hidden_dim)[-1])
        # lstm_out_test = lstm_out.view(num_frames,-1,self.hidden_dim)[-1]


        # lstm_out_test = self.fc1(lstm_out[-1])
        # lstm_out_test = torch.mean(lstm_out.view(num_frames,-1,self.hidden_dim),0)

        pool = torch.mean(lstm_out,dim=0)
        # pool = torch.max(lstm_out, dim=0)[0]
        # pool = lstm_out[-1]

        lstm_out_test = self.fc1(pool)

        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)

        lstm_out_ = [self.match(i) for i in batch]

        return lstm_out_train, lstm_out_test, lstm_out_