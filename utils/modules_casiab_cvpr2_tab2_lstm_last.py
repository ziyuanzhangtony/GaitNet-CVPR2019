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
            vgg_layer(nc, nf), # 64x32
            nn.AdaptiveMaxPool2d((32, 16)),

            vgg_layer(nf, nf*4), # 32x16
            nn.AdaptiveMaxPool2d((16, 8)),

            vgg_layer(nf*4, nf*8), # 16x8
            nn.AdaptiveMaxPool2d((4, 2)),
            # dcgan_conv(nf * 8, nf * 8),
        )
        self.flatten = nn.Sequential(
            nn.Linear(nf * 8 * 2 * 4,self.em_dim),
            nn.BatchNorm1d(self.em_dim),
        )

        self.fgs_clf = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.opt.num_train),
            nn.BatchNorm1d(self.opt.num_train),
        )

    def forward(self, input):
        embedding = self.main(input).view(-1, 64 * 8 * 2 * 4)
        embedding = self.flatten(embedding)
        fa, fg = torch.split(embedding, [self.opt.fa_dim, self.opt.fg_dim], dim=1)
        # fg = torch.nn.functional.leaky_relu(fg,0.2)
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

            # dcgan_upconv(nf * 8, nf * 8),

            dcgan_upconv(nf * 8, nf * 4),
            dcgan_upconv(nf * 4, nf * 2),
            dcgan_upconv(nf * 2, nf),
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


        self.lstm = nn.LSTM(self.source_dim, self.hidden_dim, 3, )
        # bidirectional = True

        # self.lstm =  nn.ModuleDict( {
        #     'lstm1': nn.LSTM(self.source_dim, self.hidden_dim,1),
        #     'lstm1_bn': nn.Sequential(
        #         nn.BatchNorm1d(self.hidden_dim),
        #         # nn.LeakyReLU(0.2),
        #     ),
        #
        #
        #     'lstm2':nn.LSTM(self.hidden_dim, self.hidden_dim, 1),
        #     'lstm2_bn': nn.Sequential(
        #         nn.BatchNorm1d(self.hidden_dim),
        #         # nn.LeakyReLU(0.2),
        #     ),
        #
        #     'lstm3':nn.LSTM(self.hidden_dim, self.hidden_dim, 1),
        #     'lstm3_bn': nn.Sequential(
        #         nn.BatchNorm1d(self.hidden_dim),
        #         # nn.LeakyReLU(0.2),
        #     ),
        # }
        # )



        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.main = nn.Sequential(
            # nn.LeakyReLU(0.2),
            # nn.Dropout(),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.BatchNorm1d(self.tagset_size),
            # nn.Softmax(dim=1),
        )
    def forward(self, batch):
        num_frames = len(batch)
        try:
            batch = torch.stack(batch)
        except:
            print("a Tensor already")
        # batch = batch[1:num_frames+1] - batch[0:num_frames]

        lstm_out, _ = self.lstm(batch)

        # lstm_out1, _ = self.lstm['lstm1'](batch)
        # lstm_out1 = self.lstm['lstm1_bn'](lstm_out1.permute(0,2,1))
        #
        # lstm_out2, _ = self.lstm['lstm2'](lstm_out1.permute(0,2,1))
        # lstm_out2 = self.lstm['lstm2_bn'](lstm_out2.permute(0,2,1))
        #
        # lstm_out3, _ = self.lstm['lstm3'](lstm_out2.permute(0,2,1))
        # lstm_out3 = self.lstm['lstm3_bn'](lstm_out3.permute(0,2,1))
        # lstm_out3 = lstm_out3.permute(0,2,1)

        # lstm_out_test = self.fc1(lstm_out.view(num_frames,-1,self.hidden_dim)[-1])
        # lstm_out_test = lstm_out.view(num_frames,-1,self.hidden_dim)[-1]


        lstm_out_test = self.fc1(lstm_out[-1])
        # lstm_out_test = torch.mean(lstm_out.view(num_frames,-1,self.hidden_dim),0)


        lstm_out_train = self.main(lstm_out_test).view(-1, self.tagset_size)

        # batch_mean = batch.view(num_frames, -1, self.source_dim).mean(dim=0)

        return lstm_out_train, lstm_out_test, lstm_out